import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import train_gsm8k as mv  # noqa: E402


ENTRY_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\|\s*(\d+)\s*\|\s*([^\n<]+)"
)


def load_rows(path):
    lower = path.lower()
    if lower.endswith(".json"):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    text = Path(path).read_text(encoding="utf-8")
    if "<question>:" in text and "<answer>:" in text:
        rows = []
        pattern = re.compile(r"<question>:\s*(.*?)\s*<answer>:\s*(.*?)(?=\n<question>:|\Z)", re.DOTALL)
        for match in pattern.finditer(text):
            rows.append({"question": match.group(1).strip(), "answer": match.group(2).strip()})
        if rows:
            return rows
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "<answer>:" in line:
            q, a = line.split("<answer>:", 1)
            rows.append({"question": q.replace("<question>:", "").strip(), "answer": a.strip()})
        else:
            rows.append(json.loads(line))
    return rows


def load_poi_info(train_csv, test_csv, poi_info_csv=None):
    info = defaultdict(lambda: {"lat": None, "lng": None, "category": "Unknown", "cat_id": None})
    for csv_path in [train_csv, test_csv]:
        if not csv_path or not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path)
        for row in df.itertuples(index=False):
            data = row._asdict()
            pid = int(data["PoiId"])
            info[pid]["lat"] = float(data["Latitude"])
            info[pid]["lng"] = float(data["Longitude"])
            info[pid]["category"] = str(data["PoiCategoryName"]).strip() or "Unknown"
            info[pid]["cat_id"] = int(data["PoiCategoryId"])
    if poi_info_csv and os.path.exists(poi_info_csv):
        df = pd.read_csv(poi_info_csv)
        for row in df.itertuples(index=False):
            data = row._asdict()
            pid = int(data["poi_id"])
            info[pid]["lat"] = float(data["latitude"])
            info[pid]["lng"] = float(data["longitude"])
            if "poi_catid" in data:
                info[pid]["cat_id"] = int(data["poi_catid"])
    return info


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(a))


def distance_from_last(pid, last_poi, poi_info):
    if last_poi is None:
        return None
    last = poi_info.get(int(last_poi), {})
    cur = poi_info.get(int(pid), {})
    if last.get("lat") is None or cur.get("lat") is None:
        return None
    return haversine_km(last["lat"], last["lng"], cur["lat"], cur["lng"])


def format_distance_km(distance_km):
    if distance_km is None:
        return "NA"
    return f"{float(distance_km):.2f}"


def add_unique(out, seen, cand, final_k):
    pid = int(cand["poi_id"])
    if pid in seen:
        return False
    out.append(cand)
    seen.add(pid)
    return len(out) >= final_k


def parse_context(question):
    traj, target_epoch, target_hour, target_dow = mv.parse_qa_sample(question)
    if not traj:
        return None
    if target_epoch is None or target_hour is None or target_dow is None:
        last = traj[-1]
        target_epoch = int(last["epoch"])
        target_hour = int(last["hour"])
        target_dow = int(last["dow"])
    return traj, int(target_epoch), int(target_hour), int(target_dow)


def history_candidates(question, seen, k):
    # Rank by frequency first, then by most recent occurrence in the visible prompt.
    entries = [(int(pid), idx) for idx, (_ts, pid, _cat) in enumerate(ENTRY_RE.findall(question))]
    counts = Counter(pid for pid, _idx in entries)
    last_pos = {}
    for pid, idx in entries:
        last_pos[pid] = idx
    ranked = sorted(counts, key=lambda pid: (-counts[pid], -last_pos.get(pid, -1), pid))
    out = []
    for pid in ranked:
        if pid in seen:
            continue
        out.append({"poi_id": pid, "source": "history", "history_count": int(counts[pid])})
        if len(out) >= k:
            break
    return out


def geo_candidates(last_poi, all_pois, poi_info, seen, k):
    rows = []
    for pid in all_pois:
        pid = int(pid)
        if pid == last_poi or pid in seen:
            continue
        dist = distance_from_last(pid, last_poi, poi_info)
        if dist is None:
            continue
        rows.append((dist, pid))
    rows.sort(key=lambda x: (x[0], x[1]))
    return [{"poi_id": pid, "source": "geo_nearest", "distance_km": float(dist)} for dist, pid in rows[:k]]


def transition_candidates(transition_index, bucket, last_idx, idx_to_poi, seen, k, cutoff):
    counts = transition_index.neighbors(int(bucket), int(last_idx), int(cutoff) if cutoff is not None else None)
    ranked = sorted(counts.items(), key=lambda x: (-x[1], idx_to_poi[int(x[0])]))
    out = []
    for dst_idx, cnt in ranked:
        pid = int(idx_to_poi[int(dst_idx)])
        if pid in seen:
            continue
        out.append({"poi_id": pid, "source": "transition", "transition_count": int(cnt)})
        if len(out) >= k:
            break
    return out


def _minmax(values):
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def rerank_candidates(candidates, row, ctx, poi_info, transition_index, p2i, args):
    if not getattr(args, "rerank_candidates", False):
        return candidates

    retrieval_scores = [
        float(c.get("retrieval_score", 0.0))
        for c in candidates
        if "retrieval_score" in c
    ]
    fallback_retrieval = min(retrieval_scores) if retrieval_scores else 0.0
    transition_counts = transition_index.neighbors(
        int(ctx["bucket"]),
        int(ctx["last_idx"]),
        int(ctx["target_epoch"]) if ctx.get("target_epoch") is not None else None,
    )
    history_count = Counter(int(pid) for _ts, pid, _cat in ENTRY_RE.findall(row.get("question", "")))

    raw_retrieval = [float(c.get("retrieval_score", fallback_retrieval)) for c in candidates]
    raw_transition = [
        math.log1p(float(transition_counts.get(p2i.get(int(c["poi_id"]), -1), 0.0)))
        for c in candidates
    ]
    raw_history = [math.log1p(float(history_count.get(int(c["poi_id"]), 0.0))) for c in candidates]
    raw_geo = []
    for c in candidates:
        dist = distance_from_last(int(c["poi_id"]), int(ctx["last_poi"]), poi_info)
        raw_geo.append(0.0 if dist is None else 1.0 / (1.0 + float(dist)))

    retrieval = _minmax(raw_retrieval)
    transition = _minmax(raw_transition)
    history = _minmax(raw_history)
    geo = _minmax(raw_geo)

    out = []
    for original_rank, (cand, r, t, h, g) in enumerate(
        zip(candidates, retrieval, transition, history, geo),
        start=1,
    ):
        cand = dict(cand)
        cand["mixed_original_rank"] = original_rank
        cand["rerank_score"] = (
            args.rerank_retriever_weight * r
            + args.rerank_transition_weight * t
            + args.rerank_history_weight * h
            + args.rerank_geo_weight * g
        )
        out.append(cand)
    out.sort(key=lambda x: (-float(x["rerank_score"]), int(x["mixed_original_rank"])))
    return out


def build_export_samples(raw_rows, p2i, poi_category_map, tm, graph_builder, max_seq_len):
    samples, raw_indices, contexts = [], [], {}
    skipped = []
    for raw_idx, row in enumerate(tqdm(raw_rows, desc="build samples")):
        ctx = parse_context(row.get("question", ""))
        target_poi = mv.parse_answer_poi(row.get("answer", ""))
        if ctx is None or target_poi is None:
            skipped.append(raw_idx)
            continue
        traj, target_epoch, target_hour, target_dow = ctx
        visits = traj[-max_seq_len:] if max_seq_len > 0 else traj
        poi_indices = [p2i.get(int(v["poi_id"]), -1) for v in visits]
        if any(x < 0 for x in poi_indices):
            skipped.append(raw_idx)
            continue
        target_idx = p2i.get(int(target_poi), -1)
        target_category_idx = int(poi_category_map.get(int(target_poi), -1))
        if target_idx < 0 or target_category_idx < 0:
            skipped.append(raw_idx)
            continue
        bucket = mv.bucket_from_time(tm, target_hour, target_dow)
        graph = graph_builder.build(center=poi_indices[-1], bucket=bucket, cutoff=target_epoch)
        samples.append(
            mv.RetrievalSample(
                query_poi_indices=poi_indices,
                query_hours=[int(v["hour"]) % 24 for v in visits],
                query_dows=[int(v["dow"]) % 7 for v in visits],
                target_poi_idx=target_idx,
                target_category_idx=target_category_idx,
                current_hour=int(target_hour) % 24,
                current_dow=int(target_dow) % 7,
                bucket=bucket,
                target_epoch=int(target_epoch),
                graph=graph,
            )
        )
        raw_indices.append(raw_idx)
        contexts[raw_idx] = {
            "last_poi": int(traj[-1]["poi_id"]),
            "last_idx": int(poi_indices[-1]),
            "bucket": int(bucket),
            "target_epoch": int(target_epoch),
            "target_poi": int(target_poi),
        }
    return samples, raw_indices, contexts, skipped


@torch.no_grad()
def retrieve_candidates(model, loader, idx_to_poi, device, top_k):
    model.eval()
    all_poi = model.poi_encoder.all_embeddings()
    rows = []
    for batch in tqdm(loader, desc="retrieve"):
        batch = mv.move_batch(batch, device)
        scores = model(batch)["fused"] @ all_poi.t()
        vals, inds = scores.topk(top_k, dim=-1)
        for b in range(inds.size(0)):
            rows.append(
                [
                    {
                        "poi_id": int(idx_to_poi[int(inds[b, j].item())]),
                        "source": "retriever",
                        "retrieval_rank": int(j + 1),
                        "retrieval_score": float(vals[b, j].item()),
                    }
                    for j in range(inds.size(1))
                ]
            )
    return rows


def build_mixed_candidates(row, retrieval_rows, ctx, all_pois, poi_info, transition_index, idx_to_poi, args):
    out, seen = [], set()
    for cand in retrieval_rows[: args.retriever_k]:
        if add_unique(out, seen, dict(cand), args.final_k):
            return out
    for cand in transition_candidates(
        transition_index, ctx["bucket"], ctx["last_idx"], idx_to_poi, seen, args.transition_k, ctx["target_epoch"]
    ):
        if add_unique(out, seen, cand, args.final_k):
            return out
    for cand in geo_candidates(ctx["last_poi"], all_pois, poi_info, seen, args.geo_k):
        if add_unique(out, seen, cand, args.final_k):
            return out
    for cand in history_candidates(row.get("question", ""), seen, args.history_k):
        if add_unique(out, seen, cand, args.final_k):
            return out
    for cand in retrieval_rows:
        cand = dict(cand)
        cand["source"] = "retriever_fill"
        if add_unique(out, seen, cand, args.final_k):
            return out
    return out


def candidate_block(candidates, last_poi, poi_info):
    lines = [
        "<candidates>",
        "The following candidate POIs are provided as reference. Each candidate POI is formatted as (poi_id, poi category, d=distance from the last visited POI "
        + f"{last_poi}):",
        "",
    ]
    for cand in candidates:
        pid = int(cand["poi_id"])
        category = poi_info.get(pid, {}).get("category") or "Unknown"
        dist = distance_from_last(pid, last_poi, poi_info)
        lines.append(f"{pid} | {category} | d={format_distance_km(dist)}")
    lines.append("</candidates>")
    return "\n".join(lines)


def rewrite_question(question, candidates, last_poi, poi_info):
    q = re.sub(
        r"^You will be given history and current trajectory data of a user from NYC\.",
        "You will be given history trajectory data, current trajectory data, and candidate POIs of a user from NYC.",
        question.strip(),
        flags=re.IGNORECASE,
    )
    q = re.sub(r"\s*<candidates>.*?</candidates>\s*", " ", q, flags=re.IGNORECASE | re.DOTALL).strip()
    block = candidate_block(candidates, last_poi, poi_info)
    m = re.search(r"\s*Given the data,", q, flags=re.IGNORECASE)
    if m:
        return q[: m.start()].rstrip() + "\n\n" + block + "\n\n" + q[m.start() :].lstrip()
    return q.rstrip() + "\n\n" + block


def write_rows(path, rows):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if str(path).lower().endswith(".json"):
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        lines = [f"<question>: {row['question']} <answer>: {row['answer']}" for row in rows]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_split(name, raw_rows, raw_indices, retrieval, contexts, all_pois, poi_info, transition_index, idx_to_poi, p2i, args):
    retrieval_by_raw = {raw_idx: rows for raw_idx, rows in zip(raw_indices, retrieval)}
    out_rows = []
    hit = {10: 0, 20: 0}
    source_counts = Counter()
    lengths = []
    for idx, row in enumerate(tqdm(raw_rows, desc=f"rewrite {name}")):
        if idx not in retrieval_by_raw or idx not in contexts:
            out_rows.append(dict(row))
            continue
        cands = build_mixed_candidates(
            row, retrieval_by_raw[idx], contexts[idx], all_pois, poi_info, transition_index, idx_to_poi, args
        )
        cands = rerank_candidates(cands, row, contexts[idx], poi_info, transition_index, p2i, args)
        target = mv.parse_answer_poi(row.get("answer", ""))
        cand_ids = [int(c["poi_id"]) for c in cands]
        if target in cand_ids[:10]:
            hit[10] += 1
        if target in cand_ids[:20]:
            hit[20] += 1
        source_counts.update(str(c.get("source", "unknown")) for c in cands)
        lengths.append(len(cands))
        out_rows.append(
            {
                "question": rewrite_question(row["question"], cands, contexts[idx]["last_poi"], poi_info),
                "answer": row["answer"],
            }
        )
    total = len(raw_rows)
    stats = {
        "samples": total,
        "rewritten": len(lengths),
        "avg_candidates": sum(lengths) / max(len(lengths), 1),
        "target_in_candidates@10": hit[10] / max(total, 1),
        "target_in_candidates@20": hit[20] / max(total, 1),
        "source_counts": dict(source_counts),
    }
    return out_rows, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", default="nyc")
    p.add_argument("--train_csv", default="./datasets/nyc/preprocessed/train_sample.csv")
    p.add_argument("--test_csv", default="./datasets/nyc/preprocessed/test_sample_with_traj.csv")
    p.add_argument("--train_qa", default="./datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k.json")
    p.add_argument("--test_qa", default="./datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt")
    p.add_argument("--feature_cache", default="./rag/feature_cache/bert_nyc")
    p.add_argument("--artifact_dir", default="./rag/target_poi_multiview/artifacts_gsm8k_fusedvec_v1")
    p.add_argument("--poi_info_csv", default="./datasets/nyc/preprocessed/poi_info.csv")
    p.add_argument("--output_train", default="./datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_candidates_mix10trans4geo3hist3.json")
    p.add_argument("--output_test", default="./datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_candidates_mix10trans4geo3hist3.txt")
    p.add_argument("--output_stats", default="./datasets/NYC/preprocessed/gsm8k_candidates_mix10trans4geo3hist3_stats.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_seq_len", type=int, default=20)
    p.add_argument("--max_graph_nodes", type=int, default=20)
    p.add_argument("--retrieval_top_k", type=int, default=100)
    p.add_argument("--retriever_k", type=int, default=10)
    p.add_argument("--transition_k", type=int, default=4)
    p.add_argument("--geo_k", type=int, default=3)
    p.add_argument("--history_k", type=int, default=3)
    p.add_argument("--final_k", type=int, default=20)
    p.add_argument("--rerank_candidates", action="store_true")
    p.add_argument("--rerank_retriever_weight", type=float, default=1.0)
    p.add_argument("--rerank_transition_weight", type=float, default=0.0)
    p.add_argument("--rerank_history_weight", type=float, default=0.0)
    p.add_argument("--rerank_geo_weight", type=float, default=0.0)
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir)
    cfg = torch.load(artifact_dir / "config.pth", map_location="cpu")
    args.max_seq_len = int(cfg.get("max_seq_len", args.max_seq_len))
    args.max_graph_nodes = int(cfg.get("max_graph_nodes", args.max_graph_nodes))
    hidden_size = int(cfg.get("hidden_size", 96))
    dropout = float(cfg.get("dropout", 0.55))
    fusion_type = str(cfg.get("fusion_type", "gated"))
    fusion_heads = int(cfg.get("fusion_heads", 4))
    graph_alignment = str(cfg.get("graph_alignment", "none"))
    graph_align_heads = int(cfg.get("graph_align_heads", 4))
    active_views = mv.parse_active_views(cfg.get("active_views", "sem,str,traj"))

    device = torch.device(args.device)
    train_rows = load_rows(args.train_qa)
    test_rows = load_rows(args.test_qa)
    poi_info = load_poi_info(args.train_csv, args.test_csv, args.poi_info_csv)

    tm = mv.TimeBucketManager()
    dataset = mv.load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa)
    all_pois = sorted(int(x) for x in dataset.poi_dict.keys())
    p2i = {pid: i for i, pid in enumerate(all_pois)}
    idx_to_poi = {i: pid for pid, i in p2i.items()}
    category_ids = sorted(int(x) for x in dataset.category_id_to_name.keys())
    cat2i = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
    poi_category_map = {
        int(pid): cat2i[int(info.category_id)]
        for pid, info in dataset.poi_dict.items()
        if int(info.category_id) in cat2i
    }
    sem_vectors, geo_features = mv.load_feature_cache(args.feature_cache, all_pois)
    transition_index = mv.build_transition_index(dataset, p2i, tm)
    graph_builder = mv.SubgraphBuilder(transition_index, max_nodes=args.max_graph_nodes)

    train_samples, train_raw_indices, train_ctx, train_skipped = build_export_samples(
        train_rows, p2i, poi_category_map, tm, graph_builder, args.max_seq_len
    )
    test_samples, test_raw_indices, test_ctx, test_skipped = build_export_samples(
        test_rows, p2i, poi_category_map, tm, graph_builder, args.max_seq_len
    )
    train_loader = DataLoader(
        mv.MultiViewDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=mv.collate_fn,
    )
    test_loader = DataLoader(
        mv.MultiViewDataset(test_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=mv.collate_fn,
    )

    model = mv.MultiViewPOIRetriever(
        sem_vectors,
        geo_features,
        hidden_size,
        dropout,
        num_categories=len(cat2i),
        fusion_type=fusion_type,
        fusion_heads=fusion_heads,
        graph_alignment=graph_alignment,
        graph_align_heads=graph_align_heads,
        graph_align_residual=bool(cfg.get("graph_align_residual", True)),
        active_views=active_views,
    ).to(device)
    model.load_state_dict(torch.load(artifact_dir / "model.pth", map_location=device))

    train_ret = retrieve_candidates(model, train_loader, idx_to_poi, device, args.retrieval_top_k)
    test_ret = retrieve_candidates(model, test_loader, idx_to_poi, device, args.retrieval_top_k)

    train_out, train_stats = process_split(
        "train", train_rows, train_raw_indices, train_ret, train_ctx, all_pois, poi_info, transition_index, idx_to_poi, p2i, args
    )
    test_out, test_stats = process_split(
        "test", test_rows, test_raw_indices, test_ret, test_ctx, all_pois, poi_info, transition_index, idx_to_poi, p2i, args
    )
    write_rows(args.output_train, train_out)
    write_rows(args.output_test, test_out)

    stats = {
        "artifact_dir": str(artifact_dir),
        "active_views": list(active_views),
        "strategy": (
            f"retriever{args.retriever_k} + transition{args.transition_k} + geo{args.geo_k} "
            f"+ history{args.history_k}; fill from retriever top{args.retrieval_top_k}; "
            f"rerank={bool(args.rerank_candidates)} "
            f"weights=(retriever={args.rerank_retriever_weight}, "
            f"transition={args.rerank_transition_weight}, "
            f"history={args.rerank_history_weight}, geo={args.rerank_geo_weight})"
        ),
        "train_skipped": train_skipped,
        "test_skipped": test_skipped,
        "train": train_stats,
        "test": test_stats,
        "outputs": {
            "train": args.output_train,
            "test": args.output_test,
            "stats": args.output_stats,
        },
    }
    Path(args.output_stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
