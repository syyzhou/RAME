import argparse
import json
import os
import sys
from pathlib import Path

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
import export_layered_candidates_predcat as base  # noqa: E402


@torch.no_grad()
def export_queries(model, loader, device):
    model.eval()
    out = []
    for batch in tqdm(loader, desc="export_fused_queries"):
        batch = mv.move_batch(batch, device)
        fused = model(batch)["fused"].detach().cpu()
        out.append(fused)
    if len(out) == 0:
        return torch.empty((0, model.hidden_size), dtype=torch.float32)
    return torch.cat(out, dim=0).to(torch.float32)


def build_aligned_tensor(num_rows, raw_indices, vecs):
    aligned = torch.zeros((num_rows, vecs.size(1)), dtype=vecs.dtype)
    valid_mask = torch.zeros(num_rows, dtype=torch.bool)
    for i, raw_idx in enumerate(raw_indices):
        aligned[int(raw_idx)] = vecs[i]
        valid_mask[int(raw_idx)] = True
    return aligned, valid_mask


def save_pt(path, embeddings, valid_mask, raw_indices, skipped_rows, meta):
    obj = {
        "embeddings": embeddings,
        "valid_mask": valid_mask,
        "raw_indices": raw_indices,
        "skipped_rows": skipped_rows,
        "num_samples": int(embeddings.size(0)),
        "dim": int(embeddings.size(1)),
        "meta": meta,
    }
    torch.save(obj, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifact_dir", required=True)
    p.add_argument("--dataset_name", default="nyc")
    p.add_argument("--train_csv", default=None)
    p.add_argument("--test_csv", default=None)
    p.add_argument("--train_qa", default=None)
    p.add_argument("--test_qa", default=None)
    p.add_argument("--test_qa_for_dataset", default=None)
    p.add_argument("--feature_cache", default=None)
    p.add_argument("--max_seq_len", type=int, default=20)
    p.add_argument("--max_graph_nodes", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--train_out", required=True)
    p.add_argument("--test_out", required=True)
    p.add_argument("--max_train_rows", type=int, default=0)
    p.add_argument("--max_test_rows", type=int, default=0)
    args = p.parse_args()

    data_dir = f"./datasets/{args.dataset_name}/preprocessed"
    args.train_csv = args.train_csv or f"{data_dir}/train_sample.csv"
    args.test_csv = args.test_csv or f"{data_dir}/test_sample_with_traj.csv"
    args.train_qa = args.train_qa or f"{data_dir}/train_qa_pairs_kqt.json"
    args.test_qa = args.test_qa or f"{data_dir}/test_qa_pairs_kqt.json"
    args.test_qa_for_dataset = args.test_qa_for_dataset or f"{data_dir}/test_qa_pairs_kqt.json"
    args.feature_cache = args.feature_cache or f"./rag/feature_cache/bert_{args.dataset_name}"

    artifact_dir = Path(args.artifact_dir)
    config_path = artifact_dir / "config.pth"
    model_path = artifact_dir / "model.pth"
    cfg = torch.load(config_path, map_location="cpu")

    hidden_size = int(cfg.get("hidden_size", 128))
    dropout = float(cfg.get("dropout", 0.4))
    fusion_type = str(cfg.get("fusion_type", "gated"))
    fusion_heads = int(cfg.get("fusion_heads", 4))
    graph_alignment = str(cfg.get("graph_alignment", "none"))
    graph_align_heads = int(cfg.get("graph_align_heads", 4))
    graph_align_residual = bool(cfg.get("graph_align_residual", True))
    max_seq_len = int(cfg.get("max_seq_len", args.max_seq_len))
    max_graph_nodes = int(cfg.get("max_graph_nodes", args.max_graph_nodes))

    train_rows = base.load_rows(args.train_qa)
    test_rows = base.load_rows(args.test_qa)
    poi_info = base.load_poi_info(
        f"{data_dir}/poi_info.csv",
        f"{data_dir}/poi_desc_with_id.csv",
        train_rows,
        test_rows,
    )
    if args.max_train_rows > 0:
        train_rows = train_rows[:args.max_train_rows]
    if args.max_test_rows > 0:
        test_rows = test_rows[:args.max_test_rows]

    tm = mv.TimeBucketManager()
    ds = mv.load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa_for_dataset)
    all_pois = sorted(int(x) for x in ds.poi_dict.keys())
    p2i = {pid: i for i, pid in enumerate(all_pois)}
    sem, geo = mv.load_feature_cache(args.feature_cache, all_pois)
    transition_index = mv.build_transition_index(ds, p2i, tm)
    graph_builder = mv.SubgraphBuilder(transition_index, max_nodes=max_graph_nodes)

    train_samples, train_raw_indices, train_skipped = base.build_export_samples(
        train_rows, p2i, poi_info, tm, graph_builder, max_seq_len
    )
    test_samples, test_raw_indices, test_skipped = base.build_export_samples(
        test_rows, p2i, poi_info, tm, graph_builder, max_seq_len
    )

    device = torch.device(args.device)
    model = mv.MultiViewPOIRetriever(
        sem,
        geo,
        hidden_size,
        dropout,
        num_categories=max(1, int(getattr(ds, "num_categories", 1))),
        fusion_type=fusion_type,
        fusion_heads=fusion_heads,
        graph_alignment=graph_alignment,
        graph_align_heads=graph_align_heads,
        graph_align_residual=graph_align_residual,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

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

    train_vecs = export_queries(model, train_loader, device)
    test_vecs = export_queries(model, test_loader, device)
    train_aligned, train_valid = build_aligned_tensor(len(train_rows), train_raw_indices, train_vecs)
    test_aligned, test_valid = build_aligned_tensor(len(test_rows), test_raw_indices, test_vecs)

    meta = {
        "artifact_dir": str(artifact_dir),
        "model_path": str(model_path),
        "config_path": str(config_path),
        "dataset_name": args.dataset_name,
        "train_qa": args.train_qa,
        "test_qa": args.test_qa,
        "feature_cache": args.feature_cache,
        "fusion_type": fusion_type,
    }
    save_pt(args.train_out, train_aligned, train_valid, train_raw_indices, train_skipped, meta)
    save_pt(args.test_out, test_aligned, test_valid, test_raw_indices, test_skipped, meta)

    print(json.dumps({
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
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
