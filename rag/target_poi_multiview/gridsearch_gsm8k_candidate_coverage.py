#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import train_gsm8k as mv  # noqa: E402
import export_gsm8k_mixed_candidate_prompts as exp  # noqa: E402


def parse_int_list(text):
    return [int(x) for x in str(text).split(",") if str(x).strip()]


def build_model_and_retrieve(args, split_name, test_csv, test_qa):
    artifact_dir = Path(args.artifact_dir)
    cfg = torch.load(artifact_dir / "config.pth", map_location="cpu")
    max_seq_len = int(cfg.get("max_seq_len", args.max_seq_len))
    max_graph_nodes = int(cfg.get("max_graph_nodes", args.max_graph_nodes))
    hidden_size = int(cfg.get("hidden_size", 96))
    dropout = float(cfg.get("dropout", 0.55))
    fusion_type = str(cfg.get("fusion_type", "gated"))
    fusion_heads = int(cfg.get("fusion_heads", 4))
    graph_alignment = str(cfg.get("graph_alignment", "none"))
    graph_align_heads = int(cfg.get("graph_align_heads", 4))
    active_views = mv.parse_active_views(cfg.get("active_views", "sem,str,traj"))

    raw_rows = exp.load_rows(test_qa)
    poi_info = exp.load_poi_info(args.train_csv, test_csv, args.poi_info_csv)
    tm = mv.TimeBucketManager()
    dataset = mv.load_full_dataset(args.train_csv, test_csv, args.train_qa, test_qa)
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
    graph_builder = mv.SubgraphBuilder(transition_index, max_nodes=max_graph_nodes)
    samples, raw_indices, contexts, skipped = exp.build_export_samples(
        raw_rows, p2i, poi_category_map, tm, graph_builder, max_seq_len
    )
    loader = DataLoader(
        mv.MultiViewDataset(samples),
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
    ).to(args.device)
    model.load_state_dict(torch.load(artifact_dir / "model.pth", map_location=args.device))
    retrieval = exp.retrieve_candidates(model, loader, idx_to_poi, args.device, args.retrieval_top_k)

    return {
        "name": split_name,
        "raw_rows": raw_rows,
        "raw_indices": raw_indices,
        "contexts": contexts,
        "retrieval_by_raw": {raw_idx: rows for raw_idx, rows in zip(raw_indices, retrieval)},
        "all_pois": all_pois,
        "poi_info": poi_info,
        "transition_index": transition_index,
        "idx_to_poi": idx_to_poi,
        "skipped": skipped,
    }


def build_pools(split, args):
    pools = {}
    max_transition = max(args.transition_grid) if args.transition_grid else 0
    max_history = max(args.history_grid) if args.history_grid else 0
    max_geo = max(args.geo_grid) if args.geo_grid else 0
    max_retriever = max(args.retriever_grid) if args.retriever_grid else 0
    max_retriever = max(max_retriever, args.final_k)

    for raw_idx in tqdm(split["raw_indices"], desc=f"build pools {split['name']}"):
        row = split["raw_rows"][raw_idx]
        ctx = split["contexts"][raw_idx]
        retrieval_rows = split["retrieval_by_raw"][raw_idx]
        seen = set()
        transition = exp.transition_candidates(
            split["transition_index"],
            ctx["bucket"],
            ctx["last_idx"],
            split["idx_to_poi"],
            seen,
            max_transition,
            ctx["target_epoch"],
        )
        geo = exp.geo_candidates(ctx["last_poi"], split["all_pois"], split["poi_info"], seen, max_geo)
        history = exp.history_candidates(row.get("question", ""), seen, max_history)
        pools[raw_idx] = {
            "target": mv.parse_answer_poi(row.get("answer", "")),
            "retriever": retrieval_rows[:max_retriever],
            "transition": transition,
            "geo": geo,
            "history": history,
            "retriever_fill": retrieval_rows,
        }
    return pools


def add_unique_ids(out, seen, candidates, quota, final_k):
    for cand in candidates[:quota]:
        pid = int(cand["poi_id"])
        if pid in seen:
            continue
        out.append(pid)
        seen.add(pid)
        if len(out) >= final_k:
            return True
    return False


def build_candidate_ids(pool, recipe, final_k):
    out, seen = [], set()
    if add_unique_ids(out, seen, pool["retriever"], recipe["retriever_k"], final_k):
        return out
    if add_unique_ids(out, seen, pool["transition"], recipe["transition_k"], final_k):
        return out
    if add_unique_ids(out, seen, pool["geo"], recipe["geo_k"], final_k):
        return out
    if add_unique_ids(out, seen, pool["history"], recipe["history_k"], final_k):
        return out
    add_unique_ids(out, seen, pool["retriever_fill"], 10**9, final_k)
    return out


def evaluate_recipe(pools, raw_indices, recipe, final_k):
    hits = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
    lengths = []
    source_short = {
        "retriever_k": "r",
        "transition_k": "t",
        "history_k": "h",
        "geo_k": "g",
    }
    del source_short
    for raw_idx in raw_indices:
        pool = pools[raw_idx]
        target = pool["target"]
        ids = build_candidate_ids(pool, recipe, final_k)
        lengths.append(len(ids))
        for k in hits:
            if target in ids[: min(k, final_k)]:
                hits[k] += 1
    total = len(raw_indices)
    result = dict(recipe)
    result.update({
        "avg_candidates": sum(lengths) / max(len(lengths), 1),
        "coverage@1": hits[1] / max(total, 1),
        "coverage@5": hits[5] / max(total, 1),
        "coverage@10": hits[10] / max(total, 1),
        "coverage@20": hits[20] / max(total, 1),
        "coverage@50": hits[50] / max(total, 1),
    })
    if final_k >= 100:
        result["coverage@100"] = hits[100] / max(total, 1)
    return result


def build_candidate_ids_from_sequence(pool, source_sequence, final_k):
    out, seen = [], set()
    cursors = {source: 0 for source in ("retriever", "transition", "geo", "history")}
    for source in source_sequence:
        rows = pool.get(source, [])
        cursor = cursors.get(source, 0)
        added = False
        while cursor < len(rows):
            pid = int(rows[cursor]["poi_id"])
            cursor += 1
            if pid in seen:
                continue
            out.append(pid)
            seen.add(pid)
            added = True
            break
        cursors[source] = cursor
        if not added:
            fill_cursor = cursors.get("retriever", 0)
            rows = pool.get("retriever", [])
            while fill_cursor < len(rows):
                pid = int(rows[fill_cursor]["poi_id"])
                fill_cursor += 1
                if pid in seen:
                    continue
                out.append(pid)
                seen.add(pid)
                break
            cursors["retriever"] = fill_cursor
        if len(out) >= final_k:
            break
    return out


def evaluate_sequence(pools, raw_indices, source_sequence, final_k):
    hits = {1: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
    lengths = []
    for raw_idx in raw_indices:
        pool = pools[raw_idx]
        target = pool["target"]
        ids = build_candidate_ids_from_sequence(pool, source_sequence, final_k)
        lengths.append(len(ids))
        for k in hits:
            if target in ids[: min(k, final_k)]:
                hits[k] += 1
    total = len(raw_indices)
    result = {
        "avg_candidates": sum(lengths) / max(len(lengths), 1),
        "coverage@1": hits[1] / max(total, 1),
        "coverage@5": hits[5] / max(total, 1),
        "coverage@10": hits[10] / max(total, 1),
        "coverage@20": hits[20] / max(total, 1),
        "coverage@50": hits[50] / max(total, 1),
    }
    if final_k >= 100:
        result["coverage@100"] = hits[100] / max(total, 1)
    return result


def greedy_source_sequence(pools, raw_indices, final_k, sources):
    sequence = []
    trace = []
    for step in tqdm(range(final_k), desc="greedy val"):
        best_source = None
        best_result = None
        for source in sources:
            trial = sequence + [source]
            result = evaluate_sequence(pools, raw_indices, trial, final_k=len(trial))
            key = (
                result.get("coverage@50", 0.0),
                result.get("coverage@20", 0.0),
                result.get("coverage@10", 0.0),
                result.get("coverage@5", 0.0),
                result.get("coverage@1", 0.0),
            )
            if best_result is None:
                best_source = source
                best_result = result
                best_key = key
            elif key > best_key:
                best_source = source
                best_result = result
                best_key = key
        sequence.append(best_source)
        trace.append({"position": step + 1, "source": best_source, **best_result})
    return sequence, trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact_dir", default="./rag/target_poi_multiview/gsm8k_final_val")
    p.add_argument("--feature_cache", default="./rag/feature_cache/bert_nyc")
    p.add_argument("--train_csv", default="./datasets/nyc/preprocessed/train_sample.csv")
    p.add_argument("--train_qa", default="./datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k.json")
    p.add_argument("--val_csv", default="./datasets/NYC/preprocessed/validate_sample_with_traj.csv")
    p.add_argument("--val_qa", default="./datasets/NYC/preprocessed/validate_qa_pairs_llm4poi_gsm8k.txt")
    p.add_argument("--test_csv", default="./datasets/nyc/preprocessed/test_sample_with_traj.csv")
    p.add_argument("--test_qa", default="./datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt")
    p.add_argument("--poi_info_csv", default="./datasets/nyc/preprocessed/poi_info.csv")
    p.add_argument("--output", default="./datasets/NYC/preprocessed/gsm8k_final_valartifact_top50_quota_gridsearch_results.json")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_seq_len", type=int, default=20)
    p.add_argument("--max_graph_nodes", type=int, default=20)
    p.add_argument("--retrieval_top_k", type=int, default=300)
    p.add_argument("--final_k", type=int, default=50)
    p.add_argument("--retriever_grid", default="10,15,20,25,30,35,40,50")
    p.add_argument("--transition_grid", default="0,5,10,15,20")
    p.add_argument("--history_grid", default="0,5,10,15,20")
    p.add_argument("--geo_grid", default="0,3,5,8,10")
    p.add_argument("--mode", choices=["grid", "greedy"], default="grid")
    p.add_argument("--greedy_sources", default="retriever,transition,geo,history")
    p.add_argument("--transition_pool_k", type=int, default=80)
    p.add_argument("--history_pool_k", type=int, default=80)
    p.add_argument("--geo_pool_k", type=int, default=80)
    p.add_argument("--no_save", action="store_true")
    args = p.parse_args()

    args.device = torch.device(args.device)
    args.retriever_grid = parse_int_list(args.retriever_grid)
    args.transition_grid = parse_int_list(args.transition_grid)
    args.history_grid = parse_int_list(args.history_grid)
    args.geo_grid = parse_int_list(args.geo_grid)
    if args.mode == "greedy":
        args.retriever_grid = [max(args.final_k, args.retrieval_top_k)]
        args.transition_grid = [args.transition_pool_k]
        args.history_grid = [args.history_pool_k]
        args.geo_grid = [args.geo_pool_k]

    val_split = build_model_and_retrieve(args, "val", args.val_csv, args.val_qa)
    test_split = build_model_and_retrieve(args, "test", args.test_csv, args.test_qa)
    val_pools = build_pools(val_split, args)
    test_pools = build_pools(test_split, args)

    if args.mode == "greedy":
        sources = [x.strip() for x in args.greedy_sources.split(",") if x.strip()]
        sequence, trace = greedy_source_sequence(val_pools, val_split["raw_indices"], args.final_k, sources)
        val_best = evaluate_sequence(val_pools, val_split["raw_indices"], sequence, args.final_k)
        test_best = evaluate_sequence(test_pools, test_split["raw_indices"], sequence, args.final_k)
        output = {
            "artifact_dir": args.artifact_dir,
            "mode": "greedy_source_sequence",
            "final_k": args.final_k,
            "retrieval_top_k": args.retrieval_top_k,
            "selection_metric": "validation-only greedy source allocation; maximize cumulative coverage at each position",
            "sources": sources,
            "val_rows": len(val_split["raw_indices"]),
            "test_rows": len(test_split["raw_indices"]),
            "val_skipped": val_split["skipped"],
            "test_skipped": test_split["skipped"],
            "source_sequence": sequence,
            "source_counts": {source: sequence.count(source) for source in sources},
            "best_val": val_best,
            "test_with_best_val_sequence": test_best,
            "trace": trace,
        }
        if not args.no_save:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    recipes = []
    for r in args.retriever_grid:
        for t in args.transition_grid:
            for h in args.history_grid:
                for g in args.geo_grid:
                    recipes.append({
                        "retriever_k": r,
                        "transition_k": t,
                        "history_k": h,
                        "geo_k": g,
                    })

    val_results = [
        evaluate_recipe(val_pools, val_split["raw_indices"], recipe, args.final_k)
        for recipe in tqdm(recipes, desc="grid val")
    ]
    val_results.sort(
        key=lambda x: (
            x["coverage@50"],
            x["coverage@20"],
            x["coverage@10"],
            x["coverage@5"],
            -x["retriever_k"] - x["transition_k"] - x["history_k"] - x["geo_k"],
        ),
        reverse=True,
    )
    best_recipe = {
        "retriever_k": val_results[0]["retriever_k"],
        "transition_k": val_results[0]["transition_k"],
        "history_k": val_results[0]["history_k"],
        "geo_k": val_results[0]["geo_k"],
    }
    test_best = evaluate_recipe(test_pools, test_split["raw_indices"], best_recipe, args.final_k)

    output = {
        "artifact_dir": args.artifact_dir,
        "final_k": args.final_k,
        "retrieval_top_k": args.retrieval_top_k,
        "selection_metric": "maximize validation coverage@50; tie-break @20/@10/@5",
        "num_grid": len(recipes),
        "val_rows": len(val_split["raw_indices"]),
        "test_rows": len(test_split["raw_indices"]),
        "val_skipped": val_split["skipped"],
        "test_skipped": test_split["skipped"],
        "best_val": val_results[0],
        "test_with_best_val_recipe": test_best,
        "top20_val": val_results[:20],
    }
    if not args.no_save:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
