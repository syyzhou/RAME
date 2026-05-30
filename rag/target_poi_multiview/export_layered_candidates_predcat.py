import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

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
ANSWER_RE = re.compile(r"will visit POI id\s+(\d+)\.([^\.\n]+)", re.IGNORECASE)


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
            m = ANSWER_RE.search(item.get("answer", ""))
            if m:
                poi_info[int(m.group(1))]["category"] = m.group(2).strip()
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
        parts.append(f"POI {pid}({category}, {format_distance(dist)})")
    prefix = (
        "Use the following candidate POIs as supplementary references to refine your prediction. "
        "Each candidate is shown as POI id(category, distance). "
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
                        "score": float(vals[b, j].item()),
                        "retrieval_rank": int(j + 1),
                    }
                    for j in range(inds.size(1))
                ]
            )
    return rows


def add_unique(out, seen, cand):
    pid = int(cand["poi_id"])
    if pid in seen:
        return False
    out.append(cand)
    seen.add(pid)
    return True


def build_category_index(poi_info, all_pois):
    by_cat = defaultdict(list)
    for pid in all_pois:
        info = poi_info.get(int(pid), {})
        key = info.get("cat_id")
        if key is None:
            key = info.get("category") or "Unknown"
        by_cat[key].append(int(pid))
    return by_cat


def poi_category_key(poi_id, poi_info):
    info = poi_info.get(int(poi_id), {}) if poi_id is not None else {}
    return info.get("cat_id") if info.get("cat_id") is not None else (info.get("category") or "Unknown")


def predicted_category_distribution(retrieval_rows, poi_info, top_n=50, max_categories=3):
    """Estimate a proxy category distribution from retriever top-k results.

    This is no-oracle: it only uses retrieved candidates and their scores/ranks.
    The distribution is used as a soft prior, not as a hard filter.
    """
    rows = retrieval_rows[:top_n]
    if not rows:
        return {}, []

    scores = [float(c.get("score", 0.0)) for c in rows]
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    z = sum(exp_scores) or 1.0

    category_scores = defaultdict(float)
    first_rank = {}
    for cand, prob in zip(rows, exp_scores):
        key = poi_category_key(cand.get("poi_id"), poi_info)
        rank = int(cand.get("retrieval_rank", len(first_rank) + 1))
        # score probability is primary; reciprocal rank stabilizes when raw scores are close.
        category_scores[key] += (prob / z) + 0.05 / max(rank, 1)
        first_rank[key] = min(first_rank.get(key, rank), rank)

    total = sum(category_scores.values()) or 1.0
    dist = {k: v / total for k, v in category_scores.items()}
    ranked_keys = sorted(dist, key=lambda k: (-dist[k], first_rank.get(k, 10**9), str(k)))
    if max_categories > 0:
        ranked_keys = ranked_keys[:max_categories]
    return dist, ranked_keys


def normalize_retrieval_scores(rows):
    if not rows:
        return {}
    vals = [float(c.get("score", 0.0)) for c in rows]
    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-12:
        return {int(c["poi_id"]): 1.0 for c in rows}
    return {int(c["poi_id"]): (float(c.get("score", 0.0)) - lo) / (hi - lo) for c in rows}


def build_cat_rerank_candidates(retrieval_rows, poi_info, seen, k, pool_k, pred_top_n, max_categories, category_lambda):
    """Select category-consistent candidates from retriever top pool via soft reranking.

    Unlike the previous far_same_category branch, this never samples from the global
    category pool. Every selected POI must already be in retriever top `pool_k`, so
    it remains query-relevant.
    """
    if k <= 0:
        return []
    pool = [dict(c) for c in retrieval_rows[:pool_k] if int(c["poi_id"]) not in seen]
    cat_dist, pred_keys = predicted_category_distribution(
        retrieval_rows, poi_info, top_n=pred_top_n, max_categories=max_categories
    )
    if not pool:
        return []

    norm_scores = normalize_retrieval_scores(pool)
    pred_key_set = set(pred_keys)
    scored = []
    for cand in pool:
        pid = int(cand["poi_id"])
        key = poi_category_key(pid, poi_info)
        # Soft prior: do not hard-filter categories; non-predicted categories keep prior 0.
        cat_prior = cat_dist.get(key, 0.0)
        rerank_score = category_lambda * norm_scores.get(pid, 0.0) + (1.0 - category_lambda) * cat_prior
        # Prefer predicted categories only as a tie-break signal, not as a hard rule.
        scored.append((rerank_score, int(key in pred_key_set), -int(cand.get("retrieval_rank", 10**9)), pid, cand))

    scored.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    out = []
    for rerank_score, _is_pred, _neg_rank, _pid, cand in scored[:k]:
        cand["source"] = "cat_soft_rerank"
        cand["cat_rerank_score"] = float(rerank_score)
        cand["predicted_category_prior"] = float(cat_dist.get(poi_category_key(cand["poi_id"], poi_info), 0.0))
        out.append(cand)
    return out


def retrieval_category_keys(retrieval_rows, poi_info, top_n=100, max_categories=0):
    keys = []
    for cand in retrieval_rows[:top_n]:
        key = poi_category_key(cand.get("poi_id"), poi_info)
        if key not in keys:
            keys.append(key)
        if max_categories and len(keys) >= max_categories:
            break
    return keys


def build_far_mismatch_from_retrieval(retrieval_rows, last_poi, poi_info, seen, k, min_km, reference_top_k, max_reference_categories, pool_k):
    """Pick far contrastive negatives from retrieved candidates, not from all POIs."""
    if k <= 0:
        return []
    reference_keys = set(retrieval_category_keys(
        retrieval_rows, poi_info, top_n=reference_top_k, max_categories=max_reference_categories
    ))
    pool = []
    fallback = []
    for cand in retrieval_rows[:pool_k]:
        pid = int(cand["poi_id"])
        if pid == last_poi or pid in seen:
            continue
        key = poi_category_key(pid, poi_info)
        if key in reference_keys:
            continue
        dist = distance_from_last(pid, last_poi, poi_info)
        item = (dist if dist is not None else -1.0, -int(cand.get("retrieval_rank", 10**9)), pid, dict(cand))
        fallback.append(item)
        if dist is not None and dist >= min_km:
            pool.append(item)
    ranked = sorted(pool or fallback, key=lambda x: (-x[0], x[1], x[2]))
    out = []
    for dist, _neg_rank, _pid, cand in ranked[:k]:
        cand["source"] = "far_category_mismatch"
        if dist is not None and dist >= 0:
            cand["distance_km"] = float(dist)
        out.append(cand)
    return out


def merge_candidates(strong, cat_candidates, mismatch_candidates, tail_candidates, fill_candidates, top_k, rng, random_insert=False):
    """Default: stable order = strong first, then useful supplements.

    Set --random_insert_supplements only if you deliberately want prompt position
    augmentation during training.
    """
    strong = strong[:top_k]
    supplements = cat_candidates + mismatch_candidates + tail_candidates + fill_candidates
    supplements = supplements[: max(0, top_k - len(strong))]
    if not random_insert:
        return (strong + supplements)[:top_k]

    rng.shuffle(supplements)
    num_sup = len(supplements)
    sup_positions = set(rng.sample(range(top_k), num_sup)) if num_sup > 0 else set()
    merged = []
    i_strong = 0
    i_sup = 0
    for pos in range(top_k):
        if pos in sup_positions and i_sup < len(supplements):
            merged.append(supplements[i_sup])
            i_sup += 1
        elif i_strong < len(strong):
            merged.append(strong[i_strong])
            i_strong += 1
        elif i_sup < len(supplements):
            merged.append(supplements[i_sup])
            i_sup += 1
    return merged[:top_k]


def build_tail_candidates(retrieval_rows, seen, k, rng, min_rank):
    tail = [dict(c) for c in retrieval_rows if int(c.get("retrieval_rank", 0)) >= min_rank and int(c["poi_id"]) not in seen]
    rng.shuffle(tail)
    out = []
    for cand in tail:
        cand["source"] = "low_score_tail"
        out.append(cand)
        if len(out) >= k:
            break
    return out


def random_fill(all_pois, seen, k, rng):
    pool = [pid for pid in all_pois if pid not in seen]
    rng.shuffle(pool)
    return [{"poi_id": pid, "source": "random_fill"} for pid in pool[:k]]


def merge_random_inserts(strong, far_candidates, mismatch_candidates, tail_candidates, fill_candidates, top_k, rng):
    """Preserve retrieval order, and randomly insert all supplementary candidates."""
    strong = strong[:top_k]
    supplements = far_candidates + mismatch_candidates + tail_candidates + fill_candidates
    supplements = supplements[: max(0, top_k - len(strong))]
    rng.shuffle(supplements)

    num_sup = len(supplements)
    sup_positions = set(rng.sample(range(top_k), num_sup)) if num_sup > 0 else set()

    merged = []
    i_strong = 0
    i_sup = 0
    for pos in range(top_k):
        use_sup = pos in sup_positions and i_sup < len(supplements)
        if use_sup:
            merged.append(supplements[i_sup])
            i_sup += 1
        elif i_strong < len(strong):
            merged.append(strong[i_strong])
            i_strong += 1
        elif i_sup < len(supplements):
            merged.append(supplements[i_sup])
            i_sup += 1
    return merged[:top_k]


def build_layered(row, raw_idx, retrieval_rows, all_pois, poi_info, category_index, args, use_top12=False):
    rng = random.Random(args.random_seed + int(raw_idx))
    last_poi = parse_last_poi(row.get("question", ""))
    if getattr(args, "candidate_strategy", "legacy") == "final20":
        return build_final20(row, raw_idx, retrieval_rows, all_pois, poi_info, category_index, args)
    top20 = [dict(c) for c in retrieval_rows[:20]]

    # Strong candidates: keep the highest-ranked retrieved POIs. By default this is
    # deterministic for both train/test. Use --train_shuffle_strong for augmentation.
    if args.train_shuffle_strong and not use_top12:
        selected_top = top20[:]
        rng.shuffle(selected_top)
        selected_top = selected_top[: args.strong_k]
        selected_top = sorted(selected_top, key=lambda x: int(x.get("retrieval_rank", 10**9)))
    else:
        selected_top = sorted(top20[: args.strong_k], key=lambda x: int(x.get("retrieval_rank", 10**9)))

    out, seen = [], set()
    for cand in selected_top:
        cand["source"] = "retrieved_strong"
        add_unique(out, seen, cand)

    cat_candidates = []
    for cand in build_cat_rerank_candidates(
        retrieval_rows,
        poi_info,
        seen,
        args.cat_rerank_k,
        args.cat_rerank_pool_k,
        args.predicted_category_top_k,
        args.predicted_category_max_categories,
        args.category_lambda,
    ):
        if int(cand["poi_id"]) not in seen:
            cat_candidates.append(cand)
            seen.add(int(cand["poi_id"]))

    mismatch_candidates = []
    for cand in build_far_mismatch_from_retrieval(
        retrieval_rows,
        last_poi,
        poi_info,
        seen,
        args.far_mismatch_k,
        args.far_min_km,
        args.category_reference_top_k,
        args.category_reference_max_categories,
        args.cat_rerank_pool_k,
    ):
        if int(cand["poi_id"]) not in seen:
            mismatch_candidates.append(cand)
            seen.add(int(cand["poi_id"]))

    tail_candidates = []
    for cand in build_tail_candidates(retrieval_rows, seen, args.tail_k, rng, args.tail_min_rank):
        if int(cand["poi_id"]) not in seen:
            tail_candidates.append(cand)
            seen.add(int(cand["poi_id"]))

    supplements_count = len(cat_candidates) + len(mismatch_candidates) + len(tail_candidates)
    fill_candidates = []
    if len(out) + supplements_count < args.top_k:
        # Prefer retriever leftovers over global random fill to preserve query relevance.
        for cand in retrieval_rows[: args.retrieval_top_k]:
            pid = int(cand["poi_id"])
            if pid not in seen:
                tmp = dict(cand)
                tmp["source"] = "retrieval_fill"
                fill_candidates.append(tmp)
                seen.add(pid)
            if len(out) + supplements_count + len(fill_candidates) >= args.top_k:
                break

    if len(out) + supplements_count + len(fill_candidates) < args.top_k:
        for cand in random_fill(all_pois, seen, args.top_k - len(out) - supplements_count - len(fill_candidates), rng):
            if int(cand["poi_id"]) not in seen:
                fill_candidates.append(cand)
                seen.add(int(cand["poi_id"]))

    merged = merge_candidates(
        out[: args.top_k],
        cat_candidates,
        mismatch_candidates,
        tail_candidates,
        fill_candidates,
        args.top_k,
        rng,
        random_insert=args.random_insert_supplements,
    )

    out = merged[: args.top_k]
    for i, cand in enumerate(out, start=1):
        cand["rank"] = i
    return out


def build_final20(row, raw_idx, retrieval_rows, all_pois, poi_info, category_index, args):
    """Final20 strategy used consistently for both train/test.

    Composition:
    - base: top `final20_base_k` from POI retriever
    - cross: in retriever top pool AND category in predicted top categories
    - near: in predicted top categories with distance in [near_min_km, near_max_km]
    - fill: retriever leftovers then random fill
    """
    rng = random.Random(args.random_seed + int(raw_idx))
    last_poi = parse_last_poi(row.get("question", ""))
    out, seen = [], set()

    # 1) strong base from retriever ranking
    for cand in retrieval_rows[: args.final20_base_k]:
        c = dict(cand)
        c["source"] = "retrieved_strong"
        add_unique(out, seen, c)

    # 2) predicted top categories from retrieval evidence (no oracle)
    cat_dist, pred_keys = predicted_category_distribution(
        retrieval_rows,
        poi_info,
        top_n=args.predicted_category_top_k,
        max_categories=args.predicted_category_max_categories,
    )
    pred_key_set = set(pred_keys)

    # 3) cross pool: POIs in retrieval pool whose category is in predicted categories
    cross = []
    for cand in retrieval_rows[: args.final20_pool_k]:
        pid = int(cand["poi_id"])
        if pid in seen:
            continue
        key = poi_category_key(pid, poi_info)
        if key not in pred_key_set:
            continue
        c = dict(cand)
        c["source"] = "cross_pool"
        c["predicted_category_prior"] = float(cat_dist.get(key, 0.0))
        cross.append(c)
    for c in cross[: args.final20_cross_k]:
        add_unique(out, seen, c)

    # 4) near-category picks: predicted categories + distance constraint
    near = []
    if last_poi is not None:
        for cand in retrieval_rows[: args.final20_pool_k]:
            pid = int(cand["poi_id"])
            if pid in seen:
                continue
            key = poi_category_key(pid, poi_info)
            if key not in pred_key_set:
                continue
            dist = distance_from_last(pid, last_poi, poi_info)
            if dist is None:
                continue
            if dist < args.final20_near_min_km or dist > args.final20_near_max_km:
                continue
            near.append((dist, int(cand.get("retrieval_rank", 10**9)), dict(cand)))
    near.sort(key=lambda x: (x[0], x[1]))
    for dist, _rank, cand in near[: args.final20_near_k]:
        cand["source"] = "near_category"
        cand["distance_km"] = float(dist)
        add_unique(out, seen, cand)

    # 5) fill from retrieval leftovers, then random fill if needed
    if len(out) < args.top_k:
        for cand in retrieval_rows[: args.retrieval_top_k]:
            pid = int(cand["poi_id"])
            if pid in seen:
                continue
            c = dict(cand)
            c["source"] = "retrieval_fill"
            add_unique(out, seen, c)
            if len(out) >= args.top_k:
                break

    if len(out) < args.top_k:
        for cand in random_fill(all_pois, seen, args.top_k - len(out), rng):
            add_unique(out, seen, cand)
            if len(out) >= args.top_k:
                break

    out = out[: args.top_k]
    for i, cand in enumerate(out, start=1):
        cand["rank"] = i
    return out


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
    default_tag = "dst_time_layered12_cat4_farmismatch2_tail2_predcat_soft"
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
    parser.add_argument("--cat_rerank_k", type=int, default=4)
    parser.add_argument("--far_mismatch_k", type=int, default=2)
    parser.add_argument("--tail_k", type=int, default=2)
    parser.add_argument("--tail_min_rank", type=int, default=101)
    parser.add_argument("--far_min_km", type=float, default=10.0)
    parser.add_argument("--cat_rerank_pool_k", type=int, default=100)
    parser.add_argument("--category_reference_top_k", type=int, default=50)
    parser.add_argument("--category_reference_max_categories", type=int, default=3)
    parser.add_argument("--predicted_category_top_k", type=int, default=50)
    parser.add_argument("--predicted_category_max_categories", type=int, default=3)
    parser.add_argument("--category_lambda", type=float, default=0.7,
                        help="Weight for retriever score in soft category reranking; category prior gets 1-lambda.")
    parser.add_argument("--train_shuffle_strong", action="store_true",
                        help="Shuffle strong candidates during training. Default keeps retrieval top12 stable.")
    parser.add_argument("--random_insert_supplements", action="store_true",
                        help="Randomly insert supplementary candidates. Default appends after strong candidates for stable evaluation.")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--candidate_strategy", type=str, default="final20", choices=["final20", "legacy"])
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--max_graph_nodes", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--train_use_top12", action="store_true")
    parser.add_argument("--test_use_top12", action="store_true")
    parser.add_argument("--final20_base_k", type=int, default=14)
    parser.add_argument("--final20_cross_k", type=int, default=4)
    parser.add_argument("--final20_near_k", type=int, default=2)
    parser.add_argument("--final20_pool_k", type=int, default=100)
    parser.add_argument("--final20_near_min_km", type=float, default=0.5)
    parser.add_argument("--final20_near_max_km", type=float, default=5.0)
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
    args.train_output = args.train_output or (
        f"{data_dir}/train_{tag}.json"
    )
    args.test_output = args.test_output or (
        f"{data_dir}/test_{tag}.txt"
    )
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
    train_retrieval = retrieve_candidates(model, train_loader, idx_to_poi, device, args.retrieval_top_k)
    test_retrieval = retrieve_candidates(model, test_loader, idx_to_poi, device, args.retrieval_top_k)

    category_index = build_category_index(poi_info, all_pois)
    train_candidate_map = {
        raw_idx: build_layered(
            train_rows[raw_idx], raw_idx, cands, all_pois, poi_info, category_index, args, use_top12=args.train_use_top12
        )
        for raw_idx, cands in zip(train_raw_indices, train_retrieval)
    }
    test_candidate_map = {
        raw_idx: build_layered(
            test_rows[raw_idx], raw_idx, cands, all_pois, poi_info, category_index, args, use_top12=args.test_use_top12
        )
        for raw_idx, cands in zip(test_raw_indices, test_retrieval)
    }

    train_out, train_stats = build_augmented_rows(train_rows, train_candidate_map, poi_info)
    test_out, test_stats = build_augmented_rows(test_rows, test_candidate_map, poi_info)
    train_stats["retrieved_rows"] = len(train_candidate_map)
    train_stats["skipped_rows"] = len(train_skipped)
    test_stats["retrieved_rows"] = len(test_candidate_map)
    test_stats["skipped_rows"] = len(test_skipped)

    save_json(train_out, args.train_output)
    save_txt(test_out, args.test_output)
    stats = {
        "rule": (
            "candidate_strategy=final20 by default: top14 retrieved_strong + cross_pool4 + near_category2; "
            "cross_pool picks POIs inside retriever top pool whose category is in predicted top categories; "
            "near_category further selects predicted-category POIs with distance in [near_min_km, near_max_km]; "
            "all train/test use the same strategy (no oracle). "
            "Set --candidate_strategy legacy to fallback to the older layered rule."
        ),
        "model_path": model_path,
        "artifact_dir": str(artifact_dir),
        "random_seed": args.random_seed,
        "retrieval_top_k": args.retrieval_top_k,
        "cat_rerank_k": args.cat_rerank_k,
        "far_mismatch_k": args.far_mismatch_k,
        "tail_k": args.tail_k,
        "cat_rerank_pool_k": args.cat_rerank_pool_k,
        "category_lambda": args.category_lambda,
        "category_reference_top_k": args.category_reference_top_k,
        "category_reference_max_categories": args.category_reference_max_categories,
        "predicted_category_top_k": args.predicted_category_top_k,
        "predicted_category_max_categories": args.predicted_category_max_categories,
        "candidate_strategy": args.candidate_strategy,
        "final20_base_k": args.final20_base_k,
        "final20_cross_k": args.final20_cross_k,
        "final20_near_k": args.final20_near_k,
        "final20_pool_k": args.final20_pool_k,
        "final20_near_min_km": args.final20_near_min_km,
        "final20_near_max_km": args.final20_near_max_km,
        "train": train_stats,
        "test": test_stats,
        "outputs": {"train": args.train_output, "test": args.test_output},
    }
    save_json(stats, args.stats_output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
