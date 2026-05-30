import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import train as mv  # noqa: E402


VISIT_RE = re.compile(
    r"At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+"
    r"user\s+\d+\s+visited\s+POI\s+id\s+(\d+)\s+"
    r"which\s+is\s+a\s+(.+?)\s+with\s+Category\s+id\s+(\d+)"
)


def load_rows(path):
    if path.lower().endswith(".json"):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if "<answer>:" in line:
            q, a = line.split("<answer>:", 1)
            rows.append({"question": q.replace("<question>:", "").strip(), "answer": a.strip()})
        else:
            rows.append(json.loads(line))
    return rows


def strip_existing_candidate_text(question):
    marker = "Use the following candidate POIs as supplementary references"
    idx = question.find(marker)
    return question[:idx].rstrip() if idx >= 0 else question


def parse_target(answer):
    return mv.parse_answer_poi(answer)


def parse_last_poi(question):
    traj, _target_epoch, _target_hour, _target_dow = mv.parse_qa_sample(question)
    return int(traj[-1]["poi_id"]) if traj else None


def load_poi_info(poi_info_csv, poi_desc_csv, train_rows, test_rows):
    poi_info = defaultdict(lambda: {"lat": None, "lng": None, "category": "Unknown", "cat_id": None})
    if poi_info_csv and os.path.exists(poi_info_csv):
        df = pd.read_csv(poi_info_csv)
        for row in df.itertuples(index=False):
            data = row._asdict()
            pid = int(data["poi_id"])
            poi_info[pid]["lat"] = float(data["latitude"])
            poi_info[pid]["lng"] = float(data["longitude"])
            if "poi_catid" in data:
                poi_info[pid]["cat_id"] = int(data["poi_catid"])
    if poi_desc_csv and os.path.exists(poi_desc_csv):
        df = pd.read_csv(poi_desc_csv)
        for row in df.itertuples(index=False):
            data = row._asdict()
            pid = int(data["poi_id"])
            poi_info[pid]["category"] = str(data.get("PoiCategoryName", "Unknown")).strip() or "Unknown"
    for rows in (train_rows, test_rows):
        for item in rows:
            text = item.get("question", "") + " " + item.get("answer", "")
            for _ts, poi_id, category, cat_id in VISIT_RE.findall(text):
                pid = int(poi_id)
                poi_info[pid]["category"] = category.strip()
                poi_info[pid]["cat_id"] = int(cat_id)
    return poi_info


def haversine_km(lat1, lon1, lat2, lon2):
    radius = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return radius * 2 * math.asin(math.sqrt(a))


def distance_from_last(pid, last_poi, poi_info):
    last = poi_info.get(int(last_poi), {}) if last_poi is not None else {}
    info = poi_info.get(int(pid), {})
    if last.get("lat") is None or info.get("lat") is None:
        return None
    return haversine_km(last["lat"], last["lng"], info["lat"], info["lng"])


def format_distance(distance_km):
    if distance_km is None:
        return "NA"
    if distance_km < 1.0:
        return f"{distance_km * 1000:.0f}m"
    if distance_km < 10.0:
        return f"{distance_km:.1f}km"
    return f"{distance_km:.0f}km"


def build_candidate_text(candidate_rows, last_poi, poi_info):
    parts = []
    for cand in candidate_rows:
        pid = int(cand["poi_id"])
        info = poi_info.get(pid, {})
        category = info.get("category") or "Unknown"
        dist = distance_from_last(pid, last_poi, poi_info)
        rank = cand.get("rank", "NA")
        score_norm = cand.get("score_norm", None)
        if isinstance(score_norm, (int, float)):
            score_text = f"{float(score_norm):.3f}"
        else:
            score_text = "NA"
        parts.append(
            f"#{rank} POI {pid}(score={score_text}, {category}, {format_distance(dist)})"
        )
    prefix = (
        "Use the following candidate POIs as supplementary references to refine your prediction. "
        "Each candidate is shown as rank POI id(score, category, distance). "
        f"Distance is measured from the last visited POI {last_poi} in the current trajectory. "
    )
    return prefix + "Candidate POIs:[" + ", ".join(parts) + "]."


def append_candidates(question, candidate_rows, last_poi, poi_info):
    base = strip_existing_candidate_text(question).rstrip()
    sep = " " if base.endswith(".") else ". "
    return base + sep + build_candidate_text(candidate_rows, last_poi, poi_info)


def build_export_samples(raw_rows, p2i, poi_info, tm, graph_builder, max_seq_len):
    samples, raw_indices, skipped = [], [], []
    for idx, row in enumerate(tqdm(raw_rows, desc="build export samples")):
        question = row.get("question", "")
        traj, target_epoch, target_hour, target_dow = mv.parse_qa_sample(question)
        if not traj:
            skipped.append(idx)
            continue
        if target_epoch is None or target_hour is None or target_dow is None:
            last = traj[-1]
            target_epoch = int(last["epoch"])
            target_hour = int(last["hour"])
            target_dow = int(last["dow"])
        visits = traj[-max_seq_len:] if max_seq_len > 0 else traj
        poi_indices = [p2i.get(int(v["poi_id"]), -1) for v in visits]
        if any(x < 0 for x in poi_indices):
            skipped.append(idx)
            continue
        target = parse_target(row.get("answer", ""))
        target_idx = p2i.get(int(target), 0) if target is not None else 0
        target_category_idx = 0
        if target is not None:
            info = poi_info.get(int(target), {})
            cat_id = info.get("cat_id")
            if cat_id is not None:
                target_category_idx = int(cat_id)
        bucket = mv.bucket_from_time(tm, target_hour, target_dow)
        graph = graph_builder.build(center=poi_indices[-1], bucket=bucket, cutoff=int(target_epoch))
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
        raw_indices.append(idx)
    return samples, raw_indices, skipped


def build_time_profile(dataset, p2i):
    hour_cnt = np.ones((len(p2i), 24), dtype=np.float32)
    dow_cnt = np.ones((len(p2i), 7), dtype=np.float32)
    for traj in dataset.all_trajectories.values():
        for v in getattr(traj, "visits", []):
            idx = p2i.get(int(v.poi_id), None)
            if idx is None:
                continue
            hour_cnt[idx, int(v.hour) % 24] += 1.0
            dow_cnt[idx, int(v.day_of_week) % 7] += 1.0
    hour_prob = hour_cnt / hour_cnt.sum(axis=1, keepdims=True)
    dow_prob = dow_cnt / dow_cnt.sum(axis=1, keepdims=True)
    return hour_prob, dow_prob


@torch.no_grad()
def retrieve_candidates(model, loader, idx_to_poi, device, top_k):
    model.eval()
    all_poi = model.poi_encoder.all_embeddings()
    rows = []
    query_vecs = []
    for batch in tqdm(loader, desc="retrieve"):
        batch = mv.move_batch(batch, device)
        out = model(batch)
        scores = out["fused"] @ all_poi.t()
        vals, inds = scores.topk(top_k, dim=-1)
        q = out["fused"].detach().cpu().numpy()
        query_vecs.append(q)
        for b in range(inds.size(0)):
            rows.append(
                [
                    {
                        "poi_id": int(idx_to_poi[int(inds[b, j].item())]),
                        "score": float(vals[b, j].item()),
                        "retrieval_rank": int(j + 1),
                    }
                    for j in range(inds.size(1))
                ]
            )
    return rows, np.concatenate(query_vecs, axis=0), all_poi.detach().cpu().numpy()


def mmr_select(cand_indices, query_sim, doc_sim, k, lamb):
    selected = []
    remaining = list(cand_indices)
    while remaining and len(selected) < k:
        best = None
        best_score = -1e9
        for idx in remaining:
            if selected:
                max_sim = max(float(doc_sim[idx, s]) for s in selected)
            else:
                max_sim = 0.0
            score = float(lamb) * float(query_sim[idx]) - (1.0 - float(lamb)) * max_sim
            if score > best_score:
                best_score = score
                best = idx
        selected.append(best)
        remaining.remove(best)
    return selected


def score_and_rank_final(final_rows, query_vec, all_embed, p2i):
    raw_scores = []
    for cand in final_rows:
        pid = int(cand["poi_id"])
        score = float(np.dot(query_vec, all_embed[p2i[pid]]))
        cand["score_raw_global"] = score
        raw_scores.append(score)

    lo = min(raw_scores) if raw_scores else 0.0
    hi = max(raw_scores) if raw_scores else 0.0
    denom = hi - lo
    for i, cand in enumerate(final_rows, start=1):
        if denom <= 1e-12:
            norm = 1.0 if cand["score_raw_global"] == hi else 0.0
        else:
            norm = (cand["score_raw_global"] - lo) / denom
            norm = max(0.0, min(1.0, norm))
        cand["score_norm"] = float(norm)
        cand["rank"] = i
    return final_rows


def build_layered(row, raw_idx, retrieval_rows, query_vec, all_embed, p2i, poi_info, hour_prob, dow_prob, args, is_train=False):
    rng = random.Random(args.random_seed + int(raw_idx))
    last_poi = parse_last_poi(row.get("question", ""))
    _traj, _epoch, target_hour, target_dow = mv.parse_qa_sample(row.get("question", ""))
    if target_hour is None:
        target_hour = 12
    if target_dow is None:
        target_dow = 0

    rows = [dict(c) for c in retrieval_rows[: args.retrieval_top_k]]
    scores = np.array([float(c["score"]) for c in rows], dtype=np.float32)
    if len(scores) == 0:
        return []
    smin, smax = float(scores.min()), float(scores.max())
    denom = max(smax - smin, 1e-12)
    sim_norm = (scores - smin) / denom

    poi_idx = [p2i[int(c["poi_id"])] for c in rows]
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    emb = all_embed[poi_idx]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    query_sim = np.clip(emb @ q, -1.0, 1.0)
    doc_sim = np.clip(emb @ emb.T, -1.0, 1.0)

    dist_pref = []
    for c in rows:
        d = distance_from_last(int(c["poi_id"]), last_poi, poi_info)
        if d is None:
            dist_pref.append(0.5)
        elif d <= 2.0:
            dist_pref.append(1.0)
        elif d <= 8.0:
            dist_pref.append(0.7)
        else:
            dist_pref.append(0.3)
    dist_pref = np.asarray(dist_pref, dtype=np.float32)

    tscore = []
    for idx in poi_idx:
        tscore.append(0.5 * float(hour_prob[idx, int(target_hour) % 24]) + 0.5 * float(dow_prob[idx, int(target_dow) % 7]))
    tscore = np.asarray(tscore, dtype=np.float32)
    tmin, tmax = float(tscore.min()), float(tscore.max())
    tscore = (tscore - tmin) / max(tmax - tmin, 1e-12)

    base = 0.55 * sim_norm + 0.2 * dist_pref + 0.15 * tscore + 0.1 * np.clip(query_sim, 0.0, 1.0)

    selected = []
    seen = set()

    high_k = int(args.high_k)
    mid_k = int(args.mid_k)
    far_k = int(args.far_k)
    tail_k = int(args.tail_k)

    # band1: high relevance
    if is_train and args.train_random_high:
        pool_n = min(args.high_random_pool_k, len(rows))
        top_pool = list(range(pool_n))
        rng.shuffle(top_pool)
        top8_idx = sorted(top_pool[: min(high_k, len(rows))], key=lambda i: int(rows[i].get("retrieval_rank", 10**9)))
    else:
        top8_idx = list(np.argsort(-base)[: min(high_k, len(rows))])
    for i in top8_idx:
        pid = int(rows[i]["poi_id"])
        if pid in seen:
            continue
        r = dict(rows[i])
        r["source"] = "high_rel"
        selected.append(r)
        seen.add(pid)

    # band2: diverse
    mid_pool = [i for i in range(len(rows)) if 8 <= i < min(80, len(rows)) and int(rows[i]["poi_id"]) not in seen]
    mid_sel = mmr_select(mid_pool, query_sim, doc_sim, k=mid_k, lamb=args.mmr_lambda)
    for i in mid_sel:
        pid = int(rows[i]["poi_id"])
        if pid in seen:
            continue
        r = dict(rows[i])
        r["source"] = "mid_diverse"
        selected.append(r)
        seen.add(pid)

    # band3: far contrast
    far_pool = []
    for i in range(min(40, len(rows)), min(200, len(rows))):
        pid = int(rows[i]["poi_id"])
        if pid in seen:
            continue
        d = distance_from_last(pid, last_poi, poi_info)
        if d is not None and d >= args.far_min_km:
            far_pool.append(i)
    far_pool = sorted(far_pool, key=lambda i: (-base[i], i))
    for i in far_pool[:far_k]:
        pid = int(rows[i]["poi_id"])
        if pid in seen:
            continue
        r = dict(rows[i])
        r["source"] = "far_contrast"
        selected.append(r)
        seen.add(pid)

    # band4: tail hard negatives
    tail_pool = [i for i in range(min(200, len(rows)), len(rows)) if int(rows[i]["poi_id"]) not in seen]
    rng.shuffle(tail_pool)
    for i in tail_pool[:tail_k]:
        pid = int(rows[i]["poi_id"])
        if pid in seen:
            continue
        r = dict(rows[i])
        r["source"] = "tail_hard"
        selected.append(r)
        seen.add(pid)

    # fill to 20 by remaining best base
    if len(selected) < args.top_k:
        remain = [i for i in np.argsort(-base).tolist() if int(rows[i]["poi_id"]) not in seen]
        for i in remain:
            pid = int(rows[i]["poi_id"])
            if pid in seen:
                continue
            r = dict(rows[i])
            r["source"] = "fill_rel"
            selected.append(r)
            seen.add(pid)
            if len(selected) >= args.top_k:
                break

    out = selected[: args.top_k]
    return score_and_rank_final(out, query_vec, all_embed, p2i)


def build_augmented_rows(raw_rows, candidate_map, poi_info):
    out = []
    stats = {
        "samples": len(raw_rows),
        "contains_target": 0,
        "contains_target_ratio": 0.0,
        "bad_candidate_size": 0,
        "source_counts": defaultdict(int),
    }
    for raw_idx, item in enumerate(raw_rows):
        cands = candidate_map.get(raw_idx, [])
        ids = [int(c["poi_id"]) for c in cands]
        target = parse_target(item.get("answer", ""))
        if target in ids:
            stats["contains_target"] += 1
        if len(ids) != 20:
            stats["bad_candidate_size"] += 1
        for cand in cands:
            stats["source_counts"][cand.get("source", "unknown")] += 1
        last_poi = parse_last_poi(item.get("question", ""))
        question = append_candidates(item["question"], cands, last_poi, poi_info) if cands else item["question"]
        out.append({"question": question, "answer": item["answer"], "retrieval_candidates_target_poi_multiview": cands})
    stats["contains_target_ratio"] = stats["contains_target"] / len(raw_rows) if raw_rows else 0.0
    stats["source_counts"] = dict(stats["source_counts"])
    return out, stats


def save_json(rows, path):
    Path(path).write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def save_txt(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in rows:
            q = item["question"]
            a = item["answer"]
            if not q.startswith("<question>:"):
                q = "<question>: " + q
            if not a.startswith("<answer>:"):
                a = "<answer>: " + a
            f.write(q + " " + a + "\n")


def main():
    default_dataset = os.getenv("DATASET_NAME", "nyc")
    default_data_dir = f"./datasets/{default_dataset}/preprocessed"
    default_tag = "dst_time_mmrbands8_6_4_2_noleak"
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default=default_dataset)
    parser.add_argument("--candidate_tag", default=default_tag)
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--train_qa", default=None)
    parser.add_argument("--test_qa", default=None)
    parser.add_argument("--test_qa_for_dataset", default=None)
    parser.add_argument("--poi_info", default=None)
    parser.add_argument("--poi_desc", default=None)
    parser.add_argument("--feature_cache", default=f"./rag/feature_cache/bert_{default_dataset}")
    parser.add_argument("--artifact_dir", default=f"./rag/target_poi_multiview/artifacts_dropout04_wd3e3_{default_dataset}")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--train_output", default=None)
    parser.add_argument("--test_output", default=None)
    parser.add_argument("--stats_output", default=None)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--high_k", type=int, default=10)
    parser.add_argument("--mid_k", type=int, default=4)
    parser.add_argument("--far_k", type=int, default=4)
    parser.add_argument("--tail_k", type=int, default=2)
    parser.add_argument("--retrieval_top_k", type=int, default=500)
    parser.add_argument("--far_min_km", type=float, default=10.0)
    parser.add_argument("--mmr_lambda", type=float, default=0.68)
    parser.add_argument("--train_random_high", action="store_true")
    parser.add_argument("--high_random_pool_k", type=int, default=40)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--max_graph_nodes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = f"./datasets/{args.dataset_name}/preprocessed"
    tag = args.candidate_tag
    args.train_csv = args.train_csv or f"{data_dir}/train_sample.csv"
    args.test_csv = args.test_csv or f"{data_dir}/test_sample_with_traj.csv"
    args.train_qa = args.train_qa or f"{data_dir}/train_qa_pairs_kqt.json"
    args.test_qa = args.test_qa or f"{data_dir}/test_qa_pairs_kqt.txt"
    args.test_qa_for_dataset = args.test_qa_for_dataset or f"{data_dir}/test_qa_pairs_kqt.json"
    args.poi_info = args.poi_info or f"{data_dir}/poi_info.csv"
    args.poi_desc = args.poi_desc or f"{data_dir}/poi_desc_with_id.csv"
    args.train_output = args.train_output or f"{data_dir}/train_{tag}.json"
    args.test_output = args.test_output or f"{data_dir}/test_{tag}.txt"
    args.stats_output = args.stats_output or f"{data_dir}/target_poi_multiview_{tag}_stats.json"

    train_rows = load_rows(args.train_qa)
    test_rows = load_rows(args.test_qa)
    poi_info = load_poi_info(args.poi_info, args.poi_desc, train_rows, test_rows)

    artifact_dir = Path(args.artifact_dir)
    model_path = args.model_path or str(artifact_dir / "model.pth")
    config_path = artifact_dir / "config.pth"
    if config_path.exists():
        cfg = torch.load(config_path, map_location="cpu")
        args.hidden_size = int(cfg.get("hidden_size", args.hidden_size))
        args.dropout = float(cfg.get("dropout", args.dropout))
        args.fusion_type = str(cfg.get("fusion_type", "gated"))
        args.fusion_heads = int(cfg.get("fusion_heads", 4))
        args.graph_alignment = str(cfg.get("graph_alignment", "none"))
        args.graph_align_heads = int(cfg.get("graph_align_heads", 4))
        args.graph_align_residual = bool(cfg.get("graph_align_residual", True))
        args.max_seq_len = int(cfg.get("max_seq_len", args.max_seq_len))
        args.max_graph_nodes = int(cfg.get("max_graph_nodes", args.max_graph_nodes))

    device = torch.device(args.device)
    tm = mv.TimeBucketManager()
    ds = mv.load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa_for_dataset)
    all_pois = sorted(int(x) for x in ds.poi_dict.keys())
    p2i = {pid: i for i, pid in enumerate(all_pois)}
    idx_to_poi = {i: pid for pid, i in p2i.items()}
    category_ids = sorted(int(x) for x in ds.category_id_to_name.keys())
    cat2i = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
    sem, geo = mv.load_feature_cache(args.feature_cache, all_pois)
    transition_index = mv.build_transition_index(ds, p2i, tm)
    graph_builder = mv.SubgraphBuilder(transition_index, max_nodes=args.max_graph_nodes)
    hour_prob, dow_prob = build_time_profile(ds, p2i)

    train_samples, train_raw_indices, train_skipped = build_export_samples(train_rows, p2i, poi_info, tm, graph_builder, args.max_seq_len)
    test_samples, test_raw_indices, test_skipped = build_export_samples(test_rows, p2i, poi_info, tm, graph_builder, args.max_seq_len)

    model = mv.MultiViewPOIRetriever(
        sem,
        geo,
        args.hidden_size,
        args.dropout,
        num_categories=len(cat2i),
        fusion_type=getattr(args, "fusion_type", "gated"),
        fusion_heads=getattr(args, "fusion_heads", 4),
        graph_alignment=getattr(args, "graph_alignment", "none"),
        graph_align_heads=getattr(args, "graph_align_heads", 4),
        graph_align_residual=getattr(args, "graph_align_residual", True),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    train_loader = DataLoader(mv.MultiViewDataset(train_samples), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mv.collate_fn)
    test_loader = DataLoader(mv.MultiViewDataset(test_samples), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mv.collate_fn)
    train_retrieval, train_qvec, all_embed = retrieve_candidates(model, train_loader, idx_to_poi, device, args.retrieval_top_k)
    test_retrieval, test_qvec, _all_embed = retrieve_candidates(model, test_loader, idx_to_poi, device, args.retrieval_top_k)

    train_candidate_map = {}
    for j, raw_idx in enumerate(train_raw_indices):
        train_candidate_map[raw_idx] = build_layered(
            train_rows[raw_idx], raw_idx, train_retrieval[j], train_qvec[j], all_embed, p2i, poi_info, hour_prob, dow_prob, args, is_train=True
        )
    test_candidate_map = {}
    for j, raw_idx in enumerate(test_raw_indices):
        test_candidate_map[raw_idx] = build_layered(
            test_rows[raw_idx], raw_idx, test_retrieval[j], test_qvec[j], all_embed, p2i, poi_info, hour_prob, dow_prob, args, is_train=False
        )

    train_out, train_stats = build_augmented_rows(train_rows, train_candidate_map, poi_info)
    test_out, test_stats = build_augmented_rows(test_rows, test_candidate_map, poi_info)
    train_stats["retrieved_rows"] = len(train_candidate_map)
    train_stats["skipped_rows"] = len(train_skipped)
    test_stats["retrieved_rows"] = len(test_candidate_map)
    test_stats["skipped_rows"] = len(test_skipped)

    save_json(train_out, args.train_output)
    save_txt(test_out, args.test_output)
    stats = {
        "rule": "no-leak bands: high_k + mid_k + far_k + tail_k (MMR in mid band)",
        "model_path": model_path,
        "artifact_dir": str(artifact_dir),
        "random_seed": args.random_seed,
        "retrieval_top_k": args.retrieval_top_k,
        "high_k": args.high_k,
        "mid_k": args.mid_k,
        "far_k": args.far_k,
        "tail_k": args.tail_k,
        "mmr_lambda": args.mmr_lambda,
        "train_random_high": args.train_random_high,
        "high_random_pool_k": args.high_random_pool_k,
        "far_min_km": args.far_min_km,
        "train": train_stats,
        "test": test_stats,
        "outputs": {"train": args.train_output, "test": args.test_output},
    }
    save_json(stats, args.stats_output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
