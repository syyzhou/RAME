import argparse
import json
import os
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.dirname(__file__))
RAG_ROOT = os.path.abspath(os.path.join(ROOT, "rag"))
MULTIVIEW_ROOT = os.path.abspath(os.path.join(RAG_ROOT, "target_poi_multiview"))
if RAG_ROOT not in sys.path:
    sys.path.insert(0, RAG_ROOT)
if MULTIVIEW_ROOT not in sys.path:
    sys.path.insert(0, MULTIVIEW_ROOT)

import train as mv  # noqa: E402


ANSWER_RE = re.compile(r"POI\s+id\s+(\d+)", re.IGNORECASE)
VISIT_RE = re.compile(
    r"At\s+(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}),\s+"
    r"user\s+\d+\s+visited\s+POI\s+id\s+(\d+)\s+"
    r"which\s+is\s+a\s+(.+?)\s+with\s+Category\s+id\s+(\d+)"
)


def load_rows(path):
    text = Path(path).read_text(encoding="utf-8").strip()
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
        if "<answer>:" in line:
            q, a = line.split("<answer>:", 1)
            rows.append({"question": q.replace("<question>:", "").strip(), "answer": a.strip()})
        else:
            rows.append(json.loads(line))
    return rows


def parse_target_poi(answer):
    m = ANSWER_RE.search(answer or "")
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(description="Evaluate actual retriever top-k POI and category coverage.")
    parser.add_argument("--model_dir", required=True, help="Retriever artifact dir that contains model.pth and config.pth.")
    parser.add_argument("--dataset_name", default="nyc")
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--train_qa", default=None)
    parser.add_argument("--test_qa", default=None)
    parser.add_argument("--test_qa_for_dataset", default=None)
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--top_ks", default="5,10")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_seq_len", type=int, default=20)
    parser.add_argument("--max_graph_nodes", type=int, default=20)
    args = parser.parse_args()

    data_dir = os.path.join(ROOT, "datasets", args.dataset_name, "preprocessed")
    args.train_csv = args.train_csv or f"{data_dir}/train_sample.csv"
    args.test_csv = args.test_csv or f"{data_dir}/test_sample_with_traj.csv"
    args.train_qa = args.train_qa or f"{data_dir}/train_qa_pairs_kqt.json"
    args.test_qa = args.test_qa or f"{data_dir}/test_qa_pairs_kqt.txt"
    args.test_qa_for_dataset = args.test_qa_for_dataset or f"{data_dir}/test_qa_pairs_kqt.json"
    args.feature_cache = args.feature_cache or os.path.join(ROOT, "rag", "feature_cache", f"bert_{args.dataset_name}")

    model_dir = Path(args.model_dir)
    config_path = model_dir / "config.pth"
    model_path = model_dir / "model.pth"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.pth under {model_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pth under {model_dir}")

    cfg = torch.load(config_path, map_location="cpu")
    args.hidden_size = int(cfg.get("hidden_size", 128))
    args.dropout = float(cfg.get("dropout", 0.4))
    args.fusion_type = str(cfg.get("fusion_type", "gated"))
    args.fusion_heads = int(cfg.get("fusion_heads", 4))
    args.graph_alignment = str(cfg.get("graph_alignment", "none"))
    args.graph_align_heads = int(cfg.get("graph_align_heads", 4))
    args.graph_align_residual = bool(cfg.get("graph_align_residual", True))

    tm = mv.TimeBucketManager()
    dataset = mv.load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa_for_dataset)
    all_pois = sorted(int(x) for x in dataset.poi_dict.keys())
    p2i = {pid: i for i, pid in enumerate(all_pois)}
    idx_to_poi = {i: pid for pid, i in p2i.items()}
    category_ids = sorted(int(x) for x in dataset.category_id_to_name.keys())
    cat2i = {cat_id: idx for idx, cat_id in enumerate(category_ids)}
    poi_category_map = {int(pid): cat2i[int(info.category_id)] for pid, info in dataset.poi_dict.items() if int(info.category_id) in cat2i}
    sem, geo = mv.load_feature_cache(args.feature_cache, all_pois)
    transition_index = mv.build_transition_index(dataset, p2i, tm)
    graph_builder = mv.SubgraphBuilder(transition_index, max_nodes=args.max_graph_nodes)
    samples = mv.build_samples(args.test_qa, p2i, poi_category_map, tm, graph_builder, args.max_seq_len, 0)
    loader = DataLoader(mv.MultiViewDataset(samples), batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=mv.collate_fn)

    device = torch.device(args.device)
    model = mv.MultiViewPOIRetriever(
        sem,
        geo,
        args.hidden_size,
        args.dropout,
        num_categories=len(cat2i),
        fusion_type=args.fusion_type,
        fusion_heads=args.fusion_heads,
        graph_alignment=args.graph_alignment,
        graph_align_heads=args.graph_align_heads,
        graph_align_residual=args.graph_align_residual,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_embed = model.poi_encoder.all_embeddings()

    top_ks = sorted({int(x) for x in args.top_ks.split(",") if x.strip()})
    poi_hit = {k: 0 for k in top_ks}
    cat_hit = {k: 0 for k in top_ks}
    total = 0
    cat_total = 0

    with torch.no_grad():
        for batch in loader:
            batch = mv.move_batch(batch, device)
            out = model(batch)
            scores = out["fused"] @ all_embed.t()
            top = scores.topk(max(top_ks), dim=-1).indices.cpu()
            cat_top = out["category_logits"].topk(max(top_ks), dim=-1).indices.cpu()
            targets = batch["target"].cpu()
            target_categories = batch["target_category"].cpu()
            for i in range(top.size(0)):
                total += 1
                cat_total += 1
                tgt_poi = all_pois[int(targets[i].item())]
                pred_pois = [int(idx_to_poi[int(j)]) for j in top[i].tolist()]
                tgt_cat = int(target_categories[i].item())
                pred_cats = cat_top[i].tolist()
                for k in top_ks:
                    if int(tgt_poi) in pred_pois[:k]:
                        poi_hit[k] += 1
                    if tgt_cat in pred_cats[:k]:
                        cat_hit[k] += 1

    result = {
        "model_dir": str(model_dir),
        "test_qa": args.test_qa,
        "total_samples": total,
        "usable_samples": cat_total,
        "topk_poi_hit_rate": {f"top{k}": poi_hit[k] / max(cat_total, 1) for k in top_ks},
        "topk_category_coverage": {f"top{k}": cat_hit[k] / max(cat_total, 1) for k in top_ks},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
