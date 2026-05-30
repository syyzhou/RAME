import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import export_layered_candidates as base  # noqa: E402
import train as mv  # noqa: E402


def history_category_keys(question, poi_info, max_categories):
    traj, _target_epoch, _target_hour, _target_dow = mv.parse_qa_sample(question)
    counts = Counter()
    for visit in traj:
        key = base.poi_category_key(int(visit["poi_id"]), poi_info)
        if key is not None:
            counts[key] += 1
    return [key for key, _count in counts.most_common(max_categories)]


def parse_query_context(row, tm):
    traj, target_epoch, target_hour, target_dow = mv.parse_qa_sample(row.get("question", ""))
    if not traj:
        return None, None, None, None
    if target_epoch is None or target_hour is None or target_dow is None:
        last = traj[-1]
        target_epoch = int(last["epoch"])
        target_hour = int(last["hour"])
        target_dow = int(last["dow"])
    bucket = mv.bucket_from_time(tm, int(target_hour), int(target_dow))
    return int(traj[-1]["poi_id"]), int(target_epoch), int(bucket), traj


def retrieval_rank_map(retrieval_rows):
    return {int(c["poi_id"]): int(c.get("retrieval_rank", 10**9)) for c in retrieval_rows}


def build_easy_negatives(last_poi, poi_info, all_pois, seen, retrieval_ranks, history_keys, top_keys, args, rng):
    avoid_keys = set(history_keys) | set(top_keys)
    pool = []
    for pid in all_pois:
        pid = int(pid)
        if pid == last_poi or pid in seen:
            continue
        rank = retrieval_ranks.get(pid, 10**9)
        if rank <= args.easy_exclude_top_rank:
            continue
        key = base.poi_category_key(pid, poi_info)
        if key in avoid_keys:
            continue
        dist = base.distance_from_last(pid, last_poi, poi_info)
        if dist is None or dist < args.easy_min_km:
            continue
        pool.append((dist, rank, pid))
    rng.shuffle(pool)
    selected = sorted(pool[: max(args.easy_k * 40, args.easy_k)], key=lambda x: (-x[0], -x[1], x[2]))[: args.easy_k]
    return [
        {
            "poi_id": pid,
            "source": "easy_far_diverse_negative",
            "distance_km": float(dist),
            "retrieval_rank": None if rank >= 10**9 else int(rank),
        }
        for dist, rank, pid in selected
    ]


def build_contrast_negatives(
    row,
    last_poi,
    target_epoch,
    bucket,
    poi_info,
    all_pois,
    seen,
    retrieval_ranks,
    transition_index,
    p2i,
    idx_to_poi,
    history_keys,
    args,
    rng,
):
    pool = []
    last_idx = p2i.get(int(last_poi), -1)
    if last_idx >= 0:
        for dst_idx, count in transition_index.neighbors(bucket, last_idx, target_epoch).items():
            pid = int(idx_to_poi[int(dst_idx)])
            if pid == last_poi or pid in seen:
                continue
            rank = retrieval_ranks.get(pid, 10**9)
            if rank <= args.contrast_exclude_top_rank:
                continue
            dist = base.distance_from_last(pid, last_poi, poi_info)
            key = base.poi_category_key(pid, poi_info)
            pool.append((0, -int(count), dist if dist is not None else 9999.0, rank, pid, key))

    for pid in all_pois:
        pid = int(pid)
        if pid == last_poi or pid in seen:
            continue
        rank = retrieval_ranks.get(pid, 10**9)
        if rank <= args.contrast_exclude_top_rank:
            continue
        dist = base.distance_from_last(pid, last_poi, poi_info)
        if dist is None or dist > args.contrast_max_km:
            continue
        key = base.poi_category_key(pid, poi_info)
        is_history_like = key in set(history_keys)
        pool.append((1 if is_history_like else 2, 0, dist, rank, pid, key))

    rng.shuffle(pool)
    selected = sorted(pool[: max(args.contrast_k * 60, args.contrast_k)], key=lambda x: (x[0], x[1], x[2], -x[3], x[4]))
    out = []
    local_seen = set()
    for _kind, neg_count, dist, rank, pid, key in selected:
        if pid in local_seen:
            continue
        local_seen.add(pid)
        out.append(
            {
                "poi_id": pid,
                "source": "context_contrast_negative",
                "distance_km": None if dist == 9999.0 else float(dist),
                "retrieval_rank": None if rank >= 10**9 else int(rank),
                "history_category_match": bool(key in set(history_keys)),
                "transition_count": int(-neg_count),
            }
        )
        if len(out) >= args.contrast_k:
            break
    return out


def merge_candidates(strong, supplements, top_k, rng):
    supplements = supplements[: max(0, top_k - len(strong))]
    rng.shuffle(supplements)
    sup_positions = set(rng.sample(range(top_k), len(supplements))) if supplements else set()
    out = []
    strong_idx = 0
    sup_idx = 0
    for pos in range(top_k):
        if pos in sup_positions and sup_idx < len(supplements):
            out.append(supplements[sup_idx])
            sup_idx += 1
        elif strong_idx < len(strong):
            out.append(strong[strong_idx])
            strong_idx += 1
        elif sup_idx < len(supplements):
            out.append(supplements[sup_idx])
            sup_idx += 1
    return out[:top_k]


def build_nooracle_diverse(row, raw_idx, retrieval_rows, all_pois, poi_info, transition_index, p2i, idx_to_poi, tm, args, use_top12=False):
    rng = random.Random(args.random_seed + int(raw_idx))
    last_poi, target_epoch, bucket, _traj = parse_query_context(row, tm)
    if last_poi is None:
        return []

    top20 = [dict(c) for c in retrieval_rows[:20]]
    if use_top12:
        selected_top = sorted(top20[: args.strong_k], key=lambda x: int(x.get("retrieval_rank", 10**9)))
    else:
        selected_top = top20[:]
        rng.shuffle(selected_top)
        selected_top = sorted(selected_top[: args.strong_k], key=lambda x: int(x.get("retrieval_rank", 10**9)))

    out, seen = [], set()
    for cand in selected_top:
        cand["source"] = "retrieved_strong"
        base.add_unique(out, seen, cand)

    ranks = retrieval_rank_map(retrieval_rows)
    top_keys = base.retrieval_top_category_keys(
        top20,
        poi_info,
        top_n=args.category_reference_top_k,
        max_categories=args.category_reference_max_categories,
    )
    hist_keys = history_category_keys(row.get("question", ""), poi_info, args.history_reference_max_categories)

    easy_candidates = []
    for cand in build_easy_negatives(last_poi, poi_info, all_pois, seen, ranks, hist_keys, top_keys, args, rng):
        if int(cand["poi_id"]) not in seen:
            easy_candidates.append(cand)
            seen.add(int(cand["poi_id"]))

    contrast_candidates = []
    for cand in build_contrast_negatives(
        row,
        last_poi,
        target_epoch,
        bucket,
        poi_info,
        all_pois,
        seen,
        ranks,
        transition_index,
        p2i,
        idx_to_poi,
        hist_keys,
        args,
        rng,
    ):
        if int(cand["poi_id"]) not in seen:
            contrast_candidates.append(cand)
            seen.add(int(cand["poi_id"]))

    tail_candidates = []
    for cand in base.build_tail_candidates(retrieval_rows, seen, args.tail_k, rng, args.tail_min_rank):
        if int(cand["poi_id"]) not in seen:
            tail_candidates.append(cand)
            seen.add(int(cand["poi_id"]))

    fill_candidates = []
    supplements_count = len(easy_candidates) + len(contrast_candidates) + len(tail_candidates)
    if len(out) + supplements_count < args.top_k:
        for cand in base.random_fill(all_pois, seen, args.top_k - len(out) - supplements_count, rng):
            cand["source"] = "random_fill_nooracle"
            if int(cand["poi_id"]) not in seen:
                fill_candidates.append(cand)
                seen.add(int(cand["poi_id"]))

    merged = merge_candidates(
        out,
        easy_candidates + contrast_candidates + tail_candidates + fill_candidates,
        args.top_k,
        rng,
    )
    if len(merged) < args.top_k:
        for cand in base.random_fill(all_pois, seen, args.top_k - len(merged), rng):
            cand["source"] = "random_fill_nooracle"
            merged.append(cand)
            if len(merged) >= args.top_k:
                break

    for i, cand in enumerate(merged[: args.top_k], start=1):
        cand["rank"] = i
    return merged[: args.top_k]


def main():
    default_dataset = os.getenv("DATASET_NAME", "nyc")
    default_data_dir = f"./datasets/{default_dataset}/preprocessed"
    default_tag = "dst_time_top12_easy3_contrast3_tail2_nooracle"

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
    parser.add_argument("--retrieval_top_k", type=int, default=200)
    parser.add_argument("--strong_k", type=int, default=12)
    parser.add_argument("--easy_k", type=int, default=3)
    parser.add_argument("--contrast_k", type=int, default=3)
    parser.add_argument("--tail_k", type=int, default=2)
    parser.add_argument("--tail_min_rank", type=int, default=101)
    parser.add_argument("--easy_min_km", type=float, default=10.0)
    parser.add_argument("--easy_exclude_top_rank", type=int, default=50)
    parser.add_argument("--contrast_max_km", type=float, default=2.0)
    parser.add_argument("--contrast_exclude_top_rank", type=int, default=20)
    parser.add_argument("--category_reference_top_k", type=int, default=5)
    parser.add_argument("--category_reference_max_categories", type=int, default=2)
    parser.add_argument("--history_reference_max_categories", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--max_graph_nodes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_use_top12", action="store_true")
    parser.add_argument("--test_use_top12", action="store_true")
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

    train_rows = base.load_rows(args.train_qa)
    test_rows = base.load_rows(args.test_qa)
    poi_info = base.load_poi_info(args.poi_info, args.poi_desc, train_rows, test_rows)

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
    sem, geo = mv.load_feature_cache(args.feature_cache, all_pois)
    transition_index = mv.build_transition_index(ds, p2i, tm)
    graph_builder = mv.SubgraphBuilder(transition_index, max_nodes=args.max_graph_nodes)

    train_samples, train_raw_indices, train_skipped = base.build_export_samples(train_rows, p2i, tm, graph_builder, args.max_seq_len)
    test_samples, test_raw_indices, test_skipped = base.build_export_samples(test_rows, p2i, tm, graph_builder, args.max_seq_len)

    model = mv.MultiViewPOIRetriever(
        sem,
        geo,
        args.hidden_size,
        args.dropout,
        fusion_type=getattr(args, "fusion_type", "gated"),
        fusion_heads=getattr(args, "fusion_heads", 4),
        graph_alignment=getattr(args, "graph_alignment", "none"),
        graph_align_heads=getattr(args, "graph_align_heads", 4),
        graph_align_residual=getattr(args, "graph_align_residual", True),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    train_loader = DataLoader(mv.MultiViewDataset(train_samples), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mv.collate_fn)
    test_loader = DataLoader(mv.MultiViewDataset(test_samples), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mv.collate_fn)
    train_retrieval = base.retrieve_candidates(model, train_loader, idx_to_poi, device, args.retrieval_top_k)
    test_retrieval = base.retrieve_candidates(model, test_loader, idx_to_poi, device, args.retrieval_top_k)

    train_candidate_map = {
        raw_idx: build_nooracle_diverse(
            train_rows[raw_idx],
            raw_idx,
            cands,
            all_pois,
            poi_info,
            transition_index,
            p2i,
            idx_to_poi,
            tm,
            args,
            use_top12=args.train_use_top12,
        )
        for raw_idx, cands in zip(train_raw_indices, train_retrieval)
    }
    test_candidate_map = {
        raw_idx: build_nooracle_diverse(
            test_rows[raw_idx],
            raw_idx,
            cands,
            all_pois,
            poi_info,
            transition_index,
            p2i,
            idx_to_poi,
            tm,
            args,
            use_top12=args.test_use_top12,
        )
        for raw_idx, cands in zip(test_raw_indices, test_retrieval)
    }

    train_out, train_stats = base.build_augmented_rows(train_rows, train_candidate_map, poi_info)
    test_out, test_stats = base.build_augmented_rows(test_rows, test_candidate_map, poi_info)
    train_stats["retrieved_rows"] = len(train_candidate_map)
    train_stats["skipped_rows"] = len(train_skipped)
    test_stats["retrieved_rows"] = len(test_candidate_map)
    test_stats["skipped_rows"] = len(test_skipped)

    base.save_json(train_out, args.train_output)
    base.save_txt(test_out, args.test_output)
    stats = {
        "rule": (
            "no-oracle diverse candidates: top12 retriever candidates + easy3 far/diverse negatives "
            "+ contrast3 local/history/transition negatives + tail2 low-score retriever negatives. "
            "No target POI/category or answer-derived category is used when selecting negative candidates."
        ),
        "model_path": model_path,
        "artifact_dir": str(artifact_dir),
        "random_seed": args.random_seed,
        "retrieval_top_k": args.retrieval_top_k,
        "easy_min_km": args.easy_min_km,
        "contrast_max_km": args.contrast_max_km,
        "train": train_stats,
        "test": test_stats,
        "outputs": {"train": args.train_output, "test": args.test_output},
    }
    base.save_json(stats, args.stats_output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
