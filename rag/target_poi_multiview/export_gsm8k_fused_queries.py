import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = os.path.abspath(os.path.dirname(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

import train_gsm8k as mv  # noqa: E402
import export_gsm8k_mixed_candidate_prompts as base  # noqa: E402


@torch.no_grad()
def export_queries(model, loader, device):
    model.eval()
    chunks = []
    for batch in tqdm(loader, desc="export_gsm8k_fused_queries"):
        batch = mv.move_batch(batch, device)
        chunks.append(model(batch)["fused"].detach().cpu())
    if not chunks:
        return torch.empty((0, model.hidden_size), dtype=torch.float32)
    return torch.cat(chunks, dim=0).to(torch.float32)


def build_aligned_tensor(num_rows, raw_indices, vecs):
    aligned = torch.zeros((num_rows, vecs.size(1)), dtype=vecs.dtype)
    valid_mask = torch.zeros(num_rows, dtype=torch.bool)
    for vec_idx, raw_idx in enumerate(raw_indices):
        aligned[int(raw_idx)] = vecs[vec_idx]
        valid_mask[int(raw_idx)] = True
    return aligned, valid_mask


def save_pt(path, embeddings, valid_mask, raw_indices, skipped_rows, meta):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "valid_mask": valid_mask,
            "raw_indices": list(map(int, raw_indices)),
            "skipped_rows": list(map(int, skipped_rows)),
            "num_samples": int(embeddings.size(0)),
            "dim": int(embeddings.size(1)),
            "meta": meta,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", default="nyc")
    parser.add_argument("--train_csv", default="./datasets/NYC/preprocessed/train_sample.csv")
    parser.add_argument("--test_csv", default="./datasets/NYC/preprocessed/test_sample_with_traj.csv")
    parser.add_argument("--train_qa", default="./datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k.json")
    parser.add_argument("--test_qa", default="./datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt")
    parser.add_argument("--feature_cache", default="./rag/feature_cache/bert_nyc")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--poi_info_csv", default="./datasets/nyc/data/poi_info.csv")
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--test_out", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_train_rows", type=int, default=0)
    parser.add_argument("--max_test_rows", type=int, default=0)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    cfg = torch.load(artifact_dir / "config.pth", map_location="cpu")
    max_seq_len = int(cfg.get("max_seq_len", 20))
    max_graph_nodes = int(cfg.get("max_graph_nodes", 20))
    hidden_size = int(cfg.get("hidden_size", 96))
    dropout = float(cfg.get("dropout", 0.55))
    fusion_type = str(cfg.get("fusion_type", "gated"))
    fusion_heads = int(cfg.get("fusion_heads", 4))
    graph_alignment = str(cfg.get("graph_alignment", "none"))
    graph_align_heads = int(cfg.get("graph_align_heads", 4))
    graph_align_residual = bool(cfg.get("graph_align_residual", True))
    active_views = mv.parse_active_views(cfg.get("active_views", "sem,str,traj"))

    train_rows = base.load_rows(args.train_qa)
    test_rows = base.load_rows(args.test_qa)
    if args.max_train_rows > 0:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_test_rows > 0:
        test_rows = test_rows[: args.max_test_rows]

    tm = mv.TimeBucketManager()
    dataset = mv.load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa)
    all_pois = sorted(int(x) for x in dataset.poi_dict.keys())
    p2i = {pid: idx for idx, pid in enumerate(all_pois)}
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

    train_samples, train_raw_indices, _train_ctx, train_skipped = base.build_export_samples(
        train_rows, p2i, poi_category_map, tm, graph_builder, max_seq_len
    )
    test_samples, test_raw_indices, _test_ctx, test_skipped = base.build_export_samples(
        test_rows, p2i, poi_category_map, tm, graph_builder, max_seq_len
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

    device = torch.device(args.device)
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
        graph_align_residual=graph_align_residual,
        active_views=active_views,
    ).to(device)
    model.load_state_dict(torch.load(artifact_dir / "model.pth", map_location=device))

    train_vecs = export_queries(model, train_loader, device)
    test_vecs = export_queries(model, test_loader, device)
    train_aligned, train_valid = build_aligned_tensor(len(train_rows), train_raw_indices, train_vecs)
    test_aligned, test_valid = build_aligned_tensor(len(test_rows), test_raw_indices, test_vecs)

    meta = {
        "artifact_dir": str(artifact_dir),
        "model_path": str(artifact_dir / "model.pth"),
        "config_path": str(artifact_dir / "config.pth"),
        "dataset_name": args.dataset_name,
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "train_qa": args.train_qa,
        "test_qa": args.test_qa,
        "feature_cache": args.feature_cache,
        "fusion_type": fusion_type,
        "active_views": list(active_views),
    }
    save_pt(args.train_out, train_aligned, train_valid, train_raw_indices, train_skipped, meta)
    save_pt(args.test_out, test_aligned, test_valid, test_raw_indices, test_skipped, meta)

    print(
        json.dumps(
            {
                "train_out": args.train_out,
                "test_out": args.test_out,
                "train": {
                    "rows": len(train_rows),
                    "valid": int(train_valid.sum().item()),
                    "skipped": len(train_skipped),
                    "dim": int(train_aligned.size(1)),
                },
                "test": {
                    "rows": len(test_rows),
                    "valid": int(test_valid.sum().item()),
                    "skipped": len(test_skipped),
                    "dim": int(test_aligned.size(1)),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
