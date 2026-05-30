import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATConv, global_mean_pool
except ImportError as exc:
    raise ImportError("torch_geometric is required for the structure view GAT encoder.") from exc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LEGACY_RAG_ROOT = os.path.abspath(os.path.join(ROOT, "..", "rag"))
if LEGACY_RAG_ROOT not in sys.path:
    sys.path.insert(0, LEGACY_RAG_ROOT)

from retrieval_utils import TimeBucketManager, load_full_dataset  # noqa: E402


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


VIEW_NAMES = ("sem", "str", "traj")
VIEW_TO_INDEX = {name: idx for idx, name in enumerate(VIEW_NAMES)}


def parse_active_views(value) -> Tuple[str, ...]:
    if value is None:
        return VIEW_NAMES
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        raw = str(value).replace("+", ",").split(",")
    views = []
    for item in raw:
        name = str(item).strip().lower()
        if not name:
            continue
        aliases = {
            "semantic": "sem",
            "structure": "str",
            "struct": "str",
            "graph": "str",
            "trajectory": "traj",
        }
        name = aliases.get(name, name)
        if name not in VIEW_TO_INDEX:
            raise argparse.ArgumentTypeError(
                f"Unsupported view {name!r}; choose a comma-separated subset of {','.join(VIEW_NAMES)}"
            )
        if name not in views:
            views.append(name)
    if not views:
        raise argparse.ArgumentTypeError("active_views must contain at least one view.")
    return tuple(views)


def active_view_mask(active_views: Tuple[str, ...], device=None) -> torch.Tensor:
    mask = torch.zeros(len(VIEW_NAMES), dtype=torch.bool, device=device)
    for name in active_views:
        mask[VIEW_TO_INDEX[name]] = True
    return mask


def load_json_or_jsonl(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    if not text:
        return []
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "<question>:" in line and "<answer>:" in line:
            q_part, a_part = line.split("<answer>:", 1)
            rows.append({"question": q_part.replace("<question>:", "").strip(), "answer": a_part.strip()})
        else:
            rows.append(json.loads(line))
    return rows


def parse_qa_sample(question: str) -> Tuple[List[dict], Optional[int], Optional[int], Optional[int]]:
    traj = []
    section_patterns = [
        r"\[Current trajectory'?s? check-in sequence\]:\s*(.*?)\s*\[Historical check-in sequences\]:",
        r"\[Current trajectory'?s? check-in sequence\]:\s*(.*?)\s*Given the data,",
    ]
    traj_text = None
    for pat in section_patterns:
        m = re.search(pat, question, re.DOTALL)
        if m:
            traj_text = m.group(1)
            break
    if traj_text is None:
        traj_text = question.split("Given the data,")[0]

    visit_pat = (
        r"At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+"
        r"user\s+\d+\s+visited\s+POI\s+id\s+(\d+)\s+"
        r"which\s+is\s+a\s+(.+?)\s+with\s+Category\s+id\s+(\d+)"
    )
    for m in re.finditer(visit_pat, traj_text):
        ts, poi_id, cat_name, cat_id = m.groups()
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        traj.append(
            {
                "poi_id": int(poi_id),
                "category": cat_name.strip(),
                "category_id": int(cat_id),
                "epoch": int(dt.timestamp()),
                "hour": int(dt.hour),
                "dow": int(dt.weekday()),
            }
        )

    # Fallback: parse GSM8K-style current trajectory block:
    # "YYYY-mm-dd HH:MM:SS | poi_id | poi_category"
    if not traj:
        current_block = ""
        mcur = re.search(
            r"the most recent entries\s*\(time,\s*poi_id,\s*poi category\):\s*(.*?)\s*</current>",
            question,
            re.IGNORECASE | re.DOTALL,
        )
        if mcur:
            current_block = mcur.group(1)
        else:
            mcur2 = re.search(r"<current>\s*(.*?)\s*Given the data,", question, re.IGNORECASE | re.DOTALL)
            if mcur2:
                current_block = mcur2.group(1)

        entry_pat = re.compile(
            r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\|\s*(\d+)\s*\|\s*"
            r"(.+?)(?=\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\s*\||\s*$)",
            re.DOTALL,
        )
        for m in entry_pat.finditer(current_block):
            ts, poi_id, cat_name = m.groups()
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            traj.append(
                {
                    "poi_id": int(poi_id),
                    "category": cat_name.strip(),
                    # Category id is optional in gsm8k-style text and unused by this script.
                    "category_id": -1,
                    "epoch": int(dt.timestamp()),
                    "hour": int(dt.hour),
                    "dow": int(dt.weekday()),
                }
            )

    target_epoch = None
    target_hour = None
    target_dow = None
    target_pat = r"Given the data,\s+At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+Which"
    tm = re.search(target_pat, question)
    if tm is None:
        # GSM8K-style lowercase "at/which"
        tm = re.search(
            r"Given the data,\s+at\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+which",
            question,
            re.IGNORECASE,
        )
    if tm:
        try:
            dt = datetime.strptime(tm.group(1), "%Y-%m-%d %H:%M:%S")
            target_epoch = int(dt.timestamp())
            target_hour = int(dt.hour)
            target_dow = int(dt.weekday())
        except ValueError:
            pass
    return traj, target_epoch, target_hour, target_dow


def parse_answer_poi(answer: str) -> Optional[int]:
    m = re.search(r"POI\s+id\s+(\d+)", answer, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r"\b(\d+)\b", answer)
    return int(m.group(1)) if m else None


def bucket_from_time(tm: TimeBucketManager, hour: int, dow: int) -> int:
    return int(tm.get_bucket(int(hour) % 24, int(dow) >= 5))


def load_feature_cache(cache_dir: str, all_pois: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    sem_path = os.path.join(cache_dir, "poi_sem_vectors.npy")
    geo_path = os.path.join(cache_dir, "poi_geo_features.npy")
    poi_path = os.path.join(cache_dir, "poi_id_list.json")
    if not os.path.exists(sem_path) or not os.path.exists(geo_path) or not os.path.exists(poi_path):
        raise FileNotFoundError(
            f"Feature cache must contain poi_sem_vectors.npy, poi_geo_features.npy, poi_id_list.json: {cache_dir}"
        )
    with open(poi_path, "r", encoding="utf-8") as f:
        cached_pois = [int(x) for x in json.load(f)]
    if cached_pois != [int(x) for x in all_pois]:
        raise RuntimeError("Feature cache POI order does not match this dataset. Rebuild cache for the same dataset.")
    sem = torch.tensor(np.load(sem_path).astype(np.float32), dtype=torch.float32)
    geo = torch.tensor(np.load(geo_path).astype(np.float32), dtype=torch.float32)
    return sem, geo


class TemporalTransitionIndex:
    def __init__(self):
        self.edges = defaultdict(list)
        self.out_times = defaultdict(list)
        self.in_times = defaultdict(list)

    def add(self, epoch: int, bucket: int, src: int, dst: int):
        self.edges[(bucket, src, dst)].append(int(epoch))
        self.out_times[(bucket, src)].append((int(epoch), int(dst)))
        self.in_times[(bucket, dst)].append(int(epoch))

    def build(self):
        for key in self.edges:
            self.edges[key].sort()
        for key in self.out_times:
            self.out_times[key].sort(key=lambda x: x[0])
        for key in self.in_times:
            self.in_times[key].sort()

    @staticmethod
    def _count_before(sorted_epochs: List[int], cutoff: Optional[int]) -> int:
        if cutoff is None:
            return len(sorted_epochs)
        import bisect

        return bisect.bisect_left(sorted_epochs, int(cutoff))

    def edge_count(self, bucket: int, src: int, dst: int, cutoff: Optional[int]) -> int:
        return self._count_before(self.edges.get((bucket, src, dst), []), cutoff)

    def neighbors(self, bucket: int, src: int, cutoff: Optional[int]) -> Dict[int, int]:
        entries = self.out_times.get((bucket, src), [])
        out = defaultdict(int)
        for epoch, dst in entries:
            if cutoff is not None and epoch >= cutoff:
                break
            out[int(dst)] += 1
        return dict(out)

    def node_stats(self, bucket: int, node: int, cutoff: Optional[int]) -> Tuple[int, int, int, int]:
        out_nb = self.neighbors(bucket, node, cutoff)
        out_cnt = sum(out_nb.values())
        in_cnt = self._count_before(self.in_times.get((bucket, node), []), cutoff)
        return out_cnt, in_cnt, len(out_nb), max(out_nb.values()) if out_nb else 0


class SubgraphBuilder:
    def __init__(self, index: TemporalTransitionIndex, max_nodes: int = 20):
        self.index = index
        self.max_nodes = max_nodes

    def build(self, center: int, bucket: int, cutoff: Optional[int]) -> Data:
        nb = self.index.neighbors(bucket, center, cutoff)
        nodes = [int(center)]
        for dst, _ in sorted(nb.items(), key=lambda x: (-x[1], x[0]))[: self.max_nodes - 1]:
            if dst != center:
                nodes.append(int(dst))

        nmap = {p: i for i, p in enumerate(nodes)}
        srcs, dsts, weights = [], [], []
        for src in nodes:
            for dst in nodes:
                w = self.index.edge_count(bucket, src, dst, cutoff)
                if w > 0:
                    srcs.append(nmap[src])
                    dsts.append(nmap[dst])
                    weights.append(float(math.log1p(w)))
        if not srcs:
            srcs, dsts, weights = [0], [0], [1.0]

        stats = np.zeros((len(nodes), 4), dtype=np.float32)
        for i, node in enumerate(nodes):
            stats[i] = np.asarray(self.index.node_stats(bucket, node, cutoff), dtype=np.float32)
        stats = stats / (stats.max(axis=0, keepdims=True) + 1e-6)

        return Data(
            poi_idx=torch.tensor(nodes, dtype=torch.long),
            graph_stats=torch.tensor(stats, dtype=torch.float32),
            edge_index=torch.tensor([srcs, dsts], dtype=torch.long),
            edge_weight=torch.tensor(weights, dtype=torch.float32),
            time_bucket=torch.tensor([int(bucket)], dtype=torch.long),
            num_nodes=len(nodes),
        )


@dataclass
class RetrievalSample:
    query_poi_indices: List[int]
    query_hours: List[int]
    query_dows: List[int]
    target_poi_idx: int
    target_category_idx: int
    current_hour: int
    current_dow: int
    bucket: int
    target_epoch: int
    graph: Data


def build_transition_index(dataset, p2i: Dict[int, int], tm: TimeBucketManager) -> TemporalTransitionIndex:
    idx = TemporalTransitionIndex()
    for traj in dataset.all_trajectories.values():
        visits = getattr(traj, "visits", [])
        for i in range(len(visits) - 1):
            src_v = visits[i]
            dst_v = visits[i + 1]
            src = p2i.get(int(src_v.poi_id), -1)
            dst = p2i.get(int(dst_v.poi_id), -1)
            if src < 0 or dst < 0:
                continue
            idx.add(int(dst_v.epoch), int(tm.get_bucket_from_visit(dst_v)), src, dst)
    idx.build()
    return idx


def build_samples(
    qa_path: str,
    p2i: Dict[int, int],
    poi_category_map: Dict[int, int],
    tm: TimeBucketManager,
    graph_builder: SubgraphBuilder,
    max_seq_len: int,
    max_samples: int,
) -> List[RetrievalSample]:
    rows = load_json_or_jsonl(qa_path)
    samples = []
    for row in tqdm(rows, desc=f"parse {os.path.basename(qa_path)}"):
        traj, target_epoch, target_hour, target_dow = parse_qa_sample(row.get("question", ""))
        target_poi = parse_answer_poi(row.get("answer", ""))
        if not traj or target_poi is None:
            continue
        if target_epoch is None or target_hour is None or target_dow is None:
            last = traj[-1]
            target_epoch = int(last["epoch"])
            target_hour = int(last["hour"])
            target_dow = int(last["dow"])

        target_idx = p2i.get(int(target_poi), -1)
        if target_idx < 0:
            continue
        target_category_idx = int(poi_category_map.get(int(target_poi), -1))
        if target_category_idx < 0:
            continue

        visits = traj[-max_seq_len:] if max_seq_len > 0 else traj
        poi_indices = [p2i.get(int(v["poi_id"]), -1) for v in visits]
        if any(x < 0 for x in poi_indices):
            continue
        hours = [int(v["hour"]) % 24 for v in visits]
        dows = [int(v["dow"]) % 7 for v in visits]
        bucket = bucket_from_time(tm, target_hour, target_dow)
        graph = graph_builder.build(center=poi_indices[-1], bucket=bucket, cutoff=target_epoch)
        samples.append(
            RetrievalSample(
                query_poi_indices=poi_indices,
                query_hours=hours,
                query_dows=dows,
                target_poi_idx=target_idx,
                target_category_idx=target_category_idx,
                current_hour=int(target_hour) % 24,
                current_dow=int(target_dow) % 7,
                bucket=bucket,
                target_epoch=int(target_epoch),
                graph=graph,
            )
        )
        if max_samples and len(samples) >= max_samples:
            break
    print(f"Built {len(samples)} samples from {qa_path}")
    return samples


class MultiViewDataset(Dataset):
    def __init__(self, samples: List[RetrievalSample]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[RetrievalSample]) -> dict:
    max_len = max(len(x.query_poi_indices) for x in batch)

    def pad(values, pad_value=0):
        return F.pad(torch.tensor(values, dtype=torch.long), (0, max_len - len(values)), value=pad_value)

    return {
        "poi_seq": torch.stack([pad(x.query_poi_indices) for x in batch]),
        "hour_seq": torch.stack([pad(x.query_hours) for x in batch]),
        "dow_seq": torch.stack([pad(x.query_dows) for x in batch]),
        "lengths": torch.tensor([len(x.query_poi_indices) for x in batch], dtype=torch.long),
        "last_poi": torch.tensor([x.query_poi_indices[-1] for x in batch], dtype=torch.long),
        "current_hour": torch.tensor([x.current_hour for x in batch], dtype=torch.long),
        "current_dow": torch.tensor([x.current_dow for x in batch], dtype=torch.long),
        "bucket": torch.tensor([x.bucket for x in batch], dtype=torch.long),
        "target": torch.tensor([x.target_poi_idx for x in batch], dtype=torch.long),
        "target_category": torch.tensor([x.target_category_idx for x in batch], dtype=torch.long),
        "graph": Batch.from_data_list([x.graph for x in batch]),
    }


class POIEncoder(nn.Module):
    def __init__(self, sem_vectors: torch.Tensor, geo_features: torch.Tensor, hidden_size: int, id_dim: int = 128):
        super().__init__()
        num_pois, sem_dim = sem_vectors.shape
        geo_dim = geo_features.shape[1]
        self.register_buffer("sem_vectors", sem_vectors)
        self.register_buffer("geo_features", geo_features)
        self.id_embedding = nn.Embedding(num_pois, id_dim)
        self.sem_proj = nn.Sequential(nn.Linear(sem_dim, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.geo_proj = nn.Sequential(nn.Linear(geo_dim, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.id_proj = nn.Sequential(nn.Linear(id_dim, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.fuse = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, poi_idx: torch.Tensor) -> torch.Tensor:
        sem = self.sem_proj(self.sem_vectors[poi_idx])
        geo = self.geo_proj(self.geo_features[poi_idx])
        pid = self.id_proj(self.id_embedding(poi_idx))
        return F.normalize(self.fuse(torch.cat([pid, sem, geo], dim=-1)), dim=-1)

    def all_embeddings(self) -> torch.Tensor:
        idx = torch.arange(self.sem_vectors.size(0), device=self.sem_vectors.device)
        return self.forward(idx)


class SemanticViewEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_buckets: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hour_emb = nn.Embedding(24, hidden_size)
        self.dow_emb = nn.Embedding(7, hidden_size)
        self.bucket_emb = nn.Embedding(num_buckets, hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, last_poi_emb, hours, dows, buckets):
        x = torch.cat([last_poi_emb, self.hour_emb(hours), self.dow_emb(dows), self.bucket_emb(buckets)], dim=-1)
        return F.normalize(self.mlp(x), dim=-1)


class TrajectoryTransformerEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hour_emb = nn.Embedding(24, hidden_size)
        self.dow_emb = nn.Embedding(7, hidden_size)
        self.in_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))

    def forward(self, poi_emb_seq, hours, dows, lengths):
        x = self.in_proj(torch.cat([poi_emb_seq, self.hour_emb(hours), self.dow_emb(dows)], dim=-1))
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        h = self.encoder(x, src_key_padding_mask=mask)
        last_idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        last = h.gather(1, last_idx).squeeze(1)
        return F.normalize(self.out(last), dim=-1)


class StructureGATEncoder(nn.Module):
    def __init__(self, hidden_size: int, graph_stat_dim: int = 4, num_buckets: int = 8, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.bucket_emb = nn.Embedding(num_buckets, hidden_size)
        self.stat_proj = nn.Sequential(nn.Linear(graph_stat_dim, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.input_proj = nn.Sequential(nn.Linear(hidden_size * 3, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.gat1 = GATConv(hidden_size, hidden_size // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_size, hidden_size, heads=1, dropout=dropout)
        self.out = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))

    def forward(self, graph: Batch, node_poi_emb: torch.Tensor, return_nodes: bool = False):
        bucket = graph.time_bucket[graph.batch]
        x = torch.cat([node_poi_emb, self.stat_proj(graph.graph_stats), self.bucket_emb(bucket)], dim=-1)
        h = self.input_proj(x)
        h = F.elu(self.gat1(h, graph.edge_index))
        h = self.gat2(h, graph.edge_index)
        if return_nodes:
            node_h = self.out(h)
            graph_h = F.normalize(self.out(global_mean_pool(h, graph.batch)), dim=-1)
            return graph_h, node_h
        return F.normalize(self.out(global_mean_pool(h, graph.batch)), dim=-1)


def pad_graph_nodes(node_h: torch.Tensor, batch_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(batch_index.max().item()) + 1
    counts = torch.bincount(batch_index, minlength=batch_size)
    max_nodes = int(counts.max().item())
    hidden = node_h.size(-1)
    padded = node_h.new_zeros(batch_size, max_nodes, hidden)
    mask = torch.ones(batch_size, max_nodes, device=node_h.device, dtype=torch.bool)
    for graph_id in range(batch_size):
        node_idx = (batch_index == graph_id).nonzero(as_tuple=False).squeeze(-1)
        n = int(node_idx.numel())
        if n > 0:
            padded[graph_id, :n] = node_h[node_idx]
            mask[graph_id, :n] = False
    return padded, mask


class ContextAlignedGraphEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1, use_residual: bool = True):
        super().__init__()
        self.use_residual = use_residual
        self.entity_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))
        self.traj_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))
        self.node_proj = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size))
        self.dropout = nn.Dropout(dropout)
        self.residual_gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.residual_gate[-2].bias, -2.0)

    @staticmethod
    def _single_head_cross_attn(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor):
        scores = torch.sum(keys * query.unsqueeze(1), dim=-1) / math.sqrt(query.size(-1))
        scores = scores.masked_fill(mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        ctx = torch.sum(attn.unsqueeze(-1) * values, dim=1)
        return ctx, attn

    def forward(
        self,
        graph_vec: torch.Tensor,
        node_tokens: torch.Tensor,
        node_mask: torch.Tensor,
        entity_query: torch.Tensor,
        traj_query: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        entity_q = self.entity_proj(entity_query)
        traj_q = self.traj_proj(traj_query)
        node_kv = self.node_proj(node_tokens)
        entity_ctx, entity_attn = self._single_head_cross_attn(entity_q, node_kv, node_kv, node_mask)
        traj_ctx, traj_attn = self._single_head_cross_attn(traj_q, node_kv, node_kv, node_mask)
        aligned = self.dropout(entity_ctx + traj_ctx)
        if self.use_residual:
            gate_in = torch.cat([graph_vec, aligned, graph_vec * aligned], dim=-1)
            gate = self.residual_gate(gate_in)
            updated_graph = F.normalize(graph_vec + gate * aligned, dim=-1)
        else:
            gate = torch.ones(aligned.size(0), 1, device=aligned.device, dtype=aligned.dtype)
            updated_graph = F.normalize(aligned, dim=-1)
        return updated_graph, {
            "graph_align_gate": gate,
            "entity_graph_attn": entity_attn,
            "traj_graph_attn": traj_attn,
        }


class GatedFusion(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, z_sem, z_str, z_traj, active_mask: Optional[torch.Tensor] = None):
        logits = self.gate(torch.cat([z_sem, z_str, z_traj], dim=-1))
        if active_mask is not None:
            logits = logits.masked_fill(~active_mask.view(1, -1), -1e9)
        gate = F.softmax(logits, dim=-1)
        z = gate[:, 0:1] * z_sem + gate[:, 1:2] * z_str + gate[:, 2:3] * z_traj
        return F.normalize(z, dim=-1), gate


class CrossViewRelevanceFusion(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.view_type = nn.Parameter(torch.zeros(3, hidden_size))
        nn.init.normal_(self.view_type, mean=0.0, std=0.02)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.shared_relevance = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, z_sem, z_str, z_traj, active_mask: Optional[torch.Tensor] = None):
        tokens = torch.stack([z_sem, z_str, z_traj], dim=1)
        tokens = tokens + self.view_type.unsqueeze(0)
        key_padding_mask = None
        if active_mask is not None:
            key_padding_mask = (~active_mask).view(1, -1).expand(tokens.size(0), -1)
        attn_out, attn = self.cross_attn(tokens, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=True)
        tokens = self.norm1(tokens + self.dropout(attn_out))
        tokens = self.norm2(tokens + self.dropout(self.ffn(tokens)))

        context = tokens.mean(dim=1, keepdim=True).expand_as(tokens)
        token_rel = self.shared_relevance(tokens)
        context_rel = self.shared_relevance(context)
        gate_in = torch.cat(
            [
                token_rel,
                context_rel,
                token_rel * context_rel,
                torch.abs(token_rel - context_rel),
            ],
            dim=-1,
        )
        logits = self.gate(gate_in).squeeze(-1)
        if active_mask is not None:
            logits = logits.masked_fill(~active_mask.view(1, -1), -1e9)
        gate = F.softmax(logits, dim=-1)
        z = torch.sum(gate.unsqueeze(-1) * tokens, dim=1)
        return F.normalize(z, dim=-1), gate, attn


class MultiViewPOIRetriever(nn.Module):
    def __init__(
        self,
        sem_vectors: torch.Tensor,
        geo_features: torch.Tensor,
        hidden_size: int,
        dropout: float,
        num_categories: int,
        fusion_type: str = "gated",
        fusion_heads: int = 4,
        graph_alignment: str = "none",
        graph_align_heads: int = 4,
        graph_align_residual: bool = True,
        active_views: Tuple[str, ...] = VIEW_NAMES,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.active_views = parse_active_views(active_views)
        self.register_buffer("active_view_mask", active_view_mask(self.active_views), persistent=False)
        self.poi_encoder = POIEncoder(sem_vectors, geo_features, hidden_size)
        self.semantic_encoder = SemanticViewEncoder(hidden_size, dropout=dropout)
        self.trajectory_encoder = TrajectoryTransformerEncoder(hidden_size, dropout=dropout)
        self.structure_encoder = StructureGATEncoder(hidden_size, dropout=dropout)
        self.fusion_type = fusion_type
        self.graph_alignment = graph_alignment
        self.category_head = nn.Sequential(
            nn.LayerNorm(hidden_size * 3),
            nn.Linear(hidden_size * 3, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_categories),
        )
        if graph_alignment == "none":
            self.graph_aligner = None
        elif graph_alignment == "context_cross_attn":
            self.graph_aligner = ContextAlignedGraphEncoder(
                hidden_size,
                num_heads=graph_align_heads,
                dropout=dropout,
                use_residual=graph_align_residual,
            )
        else:
            raise ValueError(f"Unsupported graph_alignment: {graph_alignment}")
        if fusion_type == "gated":
            self.fusion = GatedFusion(hidden_size, dropout=dropout)
        elif fusion_type == "cross_attn":
            self.fusion = CrossViewRelevanceFusion(hidden_size, num_heads=fusion_heads, dropout=dropout)
        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, batch: dict):
        poi_seq = batch["poi_seq"]
        batch_size = poi_seq.size(0)
        device = poi_seq.device
        zero = self.poi_encoder.id_embedding.weight.new_zeros(batch_size, self.hidden_size)
        z_seq = None
        z_last = None
        graph = batch["graph"]

        if "sem" in self.active_views or "str" in self.active_views:
            z_last = self.poi_encoder(batch["last_poi"])
        if "traj" in self.active_views:
            z_seq = self.poi_encoder(poi_seq)

        z_sem = (
            self.semantic_encoder(z_last, batch["current_hour"], batch["current_dow"], batch["bucket"])
            if "sem" in self.active_views
            else zero
        )
        z_traj = (
            self.trajectory_encoder(z_seq, batch["hour_seq"], batch["dow_seq"], batch["lengths"])
            if "traj" in self.active_views
            else zero
        )
        graph_align_info = None
        if "str" in self.active_views:
            node_poi_emb = self.poi_encoder(graph.poi_idx)
            if self.graph_aligner is not None:
                z_str, node_h = self.structure_encoder(graph, node_poi_emb, return_nodes=True)
                node_tokens, node_mask = pad_graph_nodes(node_h, graph.batch)
                align_entity_query = z_sem if "sem" in self.active_views else zero
                align_traj_query = z_traj if "traj" in self.active_views else zero
                z_str, graph_align_info = self.graph_aligner(
                    z_str, node_tokens, node_mask, align_entity_query, align_traj_query
                )
            else:
                z_str = self.structure_encoder(graph, node_poi_emb)
        else:
            z_str = zero
        if self.fusion_type == "cross_attn":
            z_fused, gate, view_attn = self.fusion(z_sem, z_str, z_traj, self.active_view_mask.to(device))
        else:
            z_fused, gate = self.fusion(z_sem, z_str, z_traj, self.active_view_mask.to(device))
            view_attn = None
        category_input = torch.cat([z_sem, z_str, z_traj], dim=-1)
        return {
            "sem": z_sem,
            "str": z_str,
            "traj": z_traj,
            "fused": z_fused,
            "category_logits": self.category_head(category_input),
            "gate": gate,
            "view_attn": view_attn,
            "graph_align": graph_align_info,
        }


def retrieval_ce_loss(z: torch.Tensor, all_poi: torch.Tensor, target: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z @ all_poi.t() / temperature
    return F.cross_entropy(logits, target)


def hard_negative_margin_loss(
    z: torch.Tensor,
    all_poi: torch.Tensor,
    target: torch.Tensor,
    margin: float,
    hard_topk: int,
) -> torch.Tensor:
    if hard_topk <= 0:
        return z.new_tensor(0.0)
    scores = z @ all_poi.t()
    pos_scores = scores.gather(1, target.view(-1, 1))
    neg_scores = scores.masked_fill(
        F.one_hot(target, num_classes=scores.size(1)).bool(),
        float("-inf"),
    )
    k = min(int(hard_topk), max(1, scores.size(1) - 1))
    hard_scores = neg_scores.topk(k, dim=-1).values
    return F.relu(float(margin) + hard_scores - pos_scores).mean()


def contrastive_loss(a: torch.Tensor, b: torch.Tensor, temperature: float) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def compute_loss(
    model: MultiViewPOIRetriever,
    out: dict,
    target: torch.Tensor,
    target_category: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, dict]:
    active_views = getattr(model, "active_views", VIEW_NAMES)
    loss_category = F.cross_entropy(out["category_logits"], target_category)
    category_pred = out["category_logits"].argmax(dim=-1)
    category_acc = (category_pred == target_category).float().mean()
    if args.category_only:
        loss_fused = out["fused"].new_tensor(0.0)
        loss_traj = out["fused"].new_tensor(0.0)
        loss_sem = out["fused"].new_tensor(0.0)
        loss_str = out["fused"].new_tensor(0.0)
        loss_align = out["fused"].new_tensor(0.0)
        loss_rank = out["fused"].new_tensor(0.0)
        loss = args.category_weight * loss_category
    else:
        all_poi = model.poi_encoder.all_embeddings()
        zero = out["fused"].new_tensor(0.0)
        loss_fused = retrieval_ce_loss(out["fused"], all_poi, target, args.temperature)
        loss_traj = retrieval_ce_loss(out["traj"], all_poi, target, args.temperature) if "traj" in active_views else zero
        loss_sem = retrieval_ce_loss(out["sem"], all_poi, target, args.temperature) if "sem" in active_views else zero
        loss_str = retrieval_ce_loss(out["str"], all_poi, target, args.temperature) if "str" in active_views else zero
        align_terms = []
        if "sem" in active_views and "traj" in active_views:
            align_terms.append(contrastive_loss(out["sem"], out["traj"], args.align_temperature))
        if "str" in active_views and "traj" in active_views:
            align_terms.append(contrastive_loss(out["str"], out["traj"], args.align_temperature))
        if "sem" in active_views and "str" in active_views:
            align_terms.append(0.5 * contrastive_loss(out["sem"], out["str"], args.align_temperature))
        loss_align = sum(align_terms) if align_terms else zero
        loss_rank = hard_negative_margin_loss(
            out["fused"],
            all_poi,
            target,
            args.rank_margin,
            args.hard_topk,
        )
        loss = (
            loss_fused
            + args.traj_weight * loss_traj
            + args.sem_weight * loss_sem
            + args.str_weight * loss_str
            + args.align_weight * loss_align
            + args.rank_weight * loss_rank
            + args.category_weight * loss_category
        )
    logs = {
        "loss": float(loss.detach().item()),
        "loss_fused": float(loss_fused.detach().item()),
        "loss_traj": float(loss_traj.detach().item()),
        "loss_sem": float(loss_sem.detach().item()),
        "loss_str": float(loss_str.detach().item()),
        "loss_align": float(loss_align.detach().item()),
        "loss_rank": float(loss_rank.detach().item()),
        "loss_category": float(loss_category.detach().item()),
        "category_acc": float(category_acc.detach().item()),
        "gate_sem": float(out["gate"][:, 0].detach().mean().item()),
        "gate_str": float(out["gate"][:, 1].detach().mean().item()),
        "gate_traj": float(out["gate"][:, 2].detach().mean().item()),
    }
    if out.get("view_attn") is not None:
        attn = out["view_attn"].detach()
        eye = torch.eye(attn.size(-1), device=attn.device, dtype=torch.bool).unsqueeze(0)
        logs["view_attn_diag"] = float(attn.masked_select(eye).mean().item())
        logs["view_attn_cross"] = float(attn.masked_select(~eye).mean().item())
    if out.get("graph_align") is not None:
        graph_align = out["graph_align"]
        logs["graph_align_gate"] = float(graph_align["graph_align_gate"].detach().mean().item())
        logs["entity_graph_attn_max"] = float(graph_align["entity_graph_attn"].detach().max(dim=-1).values.mean().item())
        logs["traj_graph_attn_max"] = float(graph_align["traj_graph_attn"].detach().max(dim=-1).values.mean().item())
    return loss, logs


@torch.no_grad()
def evaluate(model: MultiViewPOIRetriever, loader: DataLoader, device: torch.device, top_ks=(1, 5, 10, 20)) -> dict:
    model.eval()
    hit = {k: 0 for k in top_ks}
    total = 0
    category_hit = {k: 0 for k in top_ks}
    all_poi = model.poi_encoder.all_embeddings()
    for batch in loader:
        batch = move_batch(batch, device)
        out = model(batch)
        scores = out["fused"] @ all_poi.t()
        top = scores.topk(max(top_ks), dim=-1).indices
        target = batch["target"]
        cat_top = out["category_logits"].topk(max(top_ks), dim=-1).indices
        target_category = batch["target_category"]
        for i in range(target.size(0)):
            total += 1
            pred = top[i].tolist()
            tgt = int(target[i].item())
            cat_pred = cat_top[i].tolist()
            tgt_cat = int(target_category[i].item())
            for k in top_ks:
                if tgt in pred[:k]:
                    hit[k] += 1
                if tgt_cat in cat_pred[:k]:
                    category_hit[k] += 1
    metrics = {f"recall@{k}": hit[k] / max(total, 1) for k in top_ks}
    metrics.update({f"category_top@{k}": category_hit[k] / max(total, 1) for k in top_ks})
    metrics["total"] = total
    return metrics


def move_batch(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if hasattr(v, "to"):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def train(args):
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    tm = TimeBucketManager()

    dataset = load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa)
    all_pois = sorted(int(x) for x in dataset.poi_dict.keys())
    p2i = {pid: i for i, pid in enumerate(all_pois)}
    category_ids = sorted(int(x) for x in dataset.category_id_to_name.keys())
    cat2i = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
    poi_category_map = {int(pid): cat2i[int(info.category_id)] for pid, info in dataset.poi_dict.items() if int(info.category_id) in cat2i}
    sem_vectors, geo_features = load_feature_cache(args.feature_cache, all_pois)

    transition_index = build_transition_index(dataset, p2i, tm)
    graph_builder = SubgraphBuilder(transition_index, max_nodes=args.max_graph_nodes)
    train_samples = build_samples(args.train_qa, p2i, poi_category_map, tm, graph_builder, args.max_seq_len, args.max_train_samples)
    val_samples = build_samples(args.test_qa, p2i, poi_category_map, tm, graph_builder, args.max_seq_len, args.max_val_samples)
    if not train_samples:
        raise RuntimeError("No train samples were built.")
    if not val_samples:
        raise RuntimeError("No validation samples were built.")

    train_loader = DataLoader(
        MultiViewDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        MultiViewDataset(val_samples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    train_eval_loader = DataLoader(
        MultiViewDataset(train_samples),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    model = MultiViewPOIRetriever(
        sem_vectors,
        geo_features,
        args.hidden_size,
        args.dropout,
        num_categories=len(cat2i),
        fusion_type=args.fusion_type,
        fusion_heads=args.fusion_heads,
        graph_alignment=args.graph_alignment,
        graph_align_heads=args.graph_align_heads,
        active_views=args.active_views,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    best_recall20 = -1.0
    best_epoch = 0
    no_improve_epochs = 0
    history = []
    print(f"[target-poi-multiview] train={len(train_samples)} val={len(val_samples)} pois={len(all_pois)}")
    print(
        f"[target-poi-multiview] hidden={args.hidden_size} "
        f"fusion={args.fusion_type} active_views={','.join(model.active_views)} "
        f"graph_alignment={args.graph_alignment} "
        f"align_weight={args.align_weight}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = defaultdict(float)
        steps = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}"):
            batch = move_batch(batch, device)
            opt.zero_grad()
            out = model(batch)
            loss, logs = compute_loss(model, out, batch["target"], batch["target_category"], args)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            for k, v in logs.items():
                sums[k] += float(v)
            steps += 1
        sched.step()

        train_logs = {k: v / max(1, steps) for k, v in sums.items()}
        val_metrics = evaluate(model, val_loader, device=device)
        train_metrics = {}
        if args.report_train_recall:
            train_metrics_raw = evaluate(model, train_eval_loader, device=device)
            train_metrics = {f"train_{k}": v for k, v in train_metrics_raw.items() if k != "total"}
        row = {"epoch": epoch, **train_logs, **val_metrics}
        row.update(train_metrics)
        history.append(row)
        msg = (
            f"[target-poi-multiview] epoch={epoch} "
            f"loss={row['loss']:.4f} fused={row['loss_fused']:.4f} "
            f"rank={row['loss_rank']:.4f} align={row['loss_align']:.4f} "
            f"cat={row['loss_category']:.4f} "
            f"cat_top@1={row['category_top@1']:.4f} "
            f"cat_top@5={row['category_top@5']:.4f} "
            f"cat_top@10={row['category_top@10']:.4f} "
            f"cat_top@20={row['category_top@20']:.4f} "
            f"recall@1={row['recall@1']:.4f} recall@5={row['recall@5']:.4f} "
            f"recall@10={row['recall@10']:.4f} recall@20={row['recall@20']:.4f} "
            f"gate=({row['gate_sem']:.2f},{row['gate_str']:.2f},{row['gate_traj']:.2f})"
        )
        if args.report_train_recall:
            msg += (
                f" train_recall@1={row['train_recall@1']:.4f}"
                f" train_recall@5={row['train_recall@5']:.4f}"
                f" train_recall@10={row['train_recall@10']:.4f}"
                f" train_recall@20={row['train_recall@20']:.4f}"
                f" train_cat_top@1={row['train_category_top@1']:.4f}"
                f" train_cat_top@5={row['train_category_top@5']:.4f}"
                f" train_cat_top@10={row['train_category_top@10']:.4f}"
                f" train_cat_top@20={row['train_category_top@20']:.4f}"
            )
        print(msg)

        if row["recall@20"] > best_recall20 + args.early_stop_min_delta:
            best_recall20 = row["recall@20"]
            best_epoch = epoch
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(args.save_dir, "model.pth"))
        else:
            no_improve_epochs += 1

        if args.early_stop_patience > 0 and no_improve_epochs >= args.early_stop_patience:
            print(
                f"[target-poi-multiview] early stop at epoch={epoch}; "
                f"best_epoch={best_epoch} best_recall@20={best_recall20:.4f} "
                f"patience={args.early_stop_patience} min_delta={args.early_stop_min_delta}"
            )
            break

    with open(os.path.join(args.save_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.save_dir, "poi_id_list.json"), "w", encoding="utf-8") as f:
        json.dump(all_pois, f, ensure_ascii=False)
    torch.save(vars(args), os.path.join(args.save_dir, "config.pth"))
    print(f"[target-poi-multiview] best recall@20={best_recall20:.4f}; saved to {args.save_dir}")


def build_parser():
    default_dataset = os.getenv("DATASET_NAME", "nyc")
    default_data_dir = f"./datasets/{default_dataset}/preprocessed"
    default_save_dir = f"./rag/target_poi_multiview/artifacts_dropout04_wd3e3_{default_dataset}_gsm8k"
    p = argparse.ArgumentParser(description="Direct target-POI retrieval with semantic/structure/trajectory views.")
    p.add_argument("--dataset_name", default=default_dataset)
    p.add_argument("--train_csv", default=f"{default_data_dir}/train_sample.csv")
    p.add_argument("--test_csv", default=f"{default_data_dir}/test_sample_with_traj.csv")
    p.add_argument("--train_qa", default="./datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k.json")
    p.add_argument("--test_qa", default="./datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt")
    p.add_argument("--feature_cache", default=f"./rag/feature_cache/bert_{default_dataset}")
    p.add_argument("--save_dir", default=default_save_dir)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.6)
    p.add_argument("--fusion_type", choices=["gated", "cross_attn"], default="gated")
    p.add_argument(
        "--active_views",
        type=parse_active_views,
        default=parse_active_views("sem,str,traj"),
        help="Comma-separated active query views for ablation. Choices: sem,str,traj.",
    )
    p.add_argument("--fusion_heads", type=int, default=4)
    p.add_argument("--graph_alignment", choices=["none", "context_cross_attn"], default="none")
    p.add_argument("--graph_align_heads", type=int, default=4)
    p.add_argument("--max_seq_len", type=int, default=20)
    p.add_argument("--max_graph_nodes", type=int, default=20)
    p.add_argument("--max_train_samples", type=int, default=0)
    p.add_argument("--max_val_samples", type=int, default=0)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=3e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--align_temperature", type=float, default=0.07)
    p.add_argument("--traj_weight", type=float, default=0.12)
    p.add_argument("--sem_weight", type=float, default=0.06)
    p.add_argument("--str_weight", type=float, default=0.06)
    p.add_argument("--align_weight", type=float, default=0.0)
    p.add_argument("--rank_weight", type=float, default=0.0)
    p.add_argument("--rank_margin", type=float, default=0.1)
    p.add_argument("--hard_topk", type=int, default=20)
    p.add_argument("--category_weight", type=float, default=0)
    p.add_argument("--category_only", type=str2bool, default=False)
    p.add_argument("--report_train_recall", type=str2bool, default=True, help="Whether to compute and report recall on the training set.")
    p.add_argument("--early_stop_patience", type=int, default=10, help="Stop if val recall@20 does not improve for N epochs; 0 disables.")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum val recall@20 improvement to reset patience.")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
