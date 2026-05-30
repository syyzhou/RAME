import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import train as mv


def build_loader(qa_path, p2i, poi_category_map, tm, graph_builder, max_seq_len, batch_size):
    samples = mv.build_samples(
        qa_path,
        p2i,
        poi_category_map,
        tm,
        graph_builder,
        max_seq_len=max_seq_len,
        max_samples=0,
    )
    loader = DataLoader(
        mv.MultiViewDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=mv.collate_fn,
    )
    return samples, loader


def main():
    default_dataset = "nyc"
    default_data_dir = f"./datasets/{default_dataset}/preprocessed"
    default_artifact_dir = (
        f"./rag/target_poi_multiview/artifacts_cross_attn_dropout04_wd3e3_{default_dataset}_ep200_cathead"
    )

    p = argparse.ArgumentParser(description="Evaluate direct POI retrieval Acc@K on train/test.")
    p.add_argument("--dataset_name", default=default_dataset)
    p.add_argument("--train_csv", default=f"{default_data_dir}/train_sample.csv")
    p.add_argument("--test_csv", default=f"{default_data_dir}/test_sample_with_traj.csv")
    p.add_argument("--train_qa", default=f"{default_data_dir}/train_qa_pairs_kqt.json")
    p.add_argument("--test_qa", default=f"{default_data_dir}/test_qa_pairs_kqt.json")
    p.add_argument("--feature_cache", default=f"./rag/feature_cache/bert_{default_dataset}")
    p.add_argument("--artifact_dir", default=default_artifact_dir)
    p.add_argument("--model_path", default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_seq_len", type=int, default=20)
    p.add_argument("--max_graph_nodes", type=int, default=20)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_json", default=None)
    p.add_argument("--history_path", default=None, help="Optional training history.json to export curve data.")
    args = p.parse_args()

    data_dir = f"./datasets/{args.dataset_name}/preprocessed"
    args.train_csv = args.train_csv or f"{data_dir}/train_sample.csv"
    args.test_csv = args.test_csv or f"{data_dir}/test_sample_with_traj.csv"
    args.train_qa = args.train_qa or f"{data_dir}/train_qa_pairs_kqt.json"
    args.test_qa = args.test_qa or f"{data_dir}/test_qa_pairs_kqt.json"
    args.feature_cache = args.feature_cache or f"./rag/feature_cache/bert_{args.dataset_name}"
    args.output_json = args.output_json or f"{data_dir}/target_poi_multiview_acc_topk_eval.json"
    args.history_path = args.history_path or str(Path(args.artifact_dir) / "history.json")

    artifact_dir = Path(args.artifact_dir)
    model_path = args.model_path or str(artifact_dir / "model.pth")
    config_path = artifact_dir / "config.pth"

    hidden_size = 128
    dropout = 0.4
    fusion_type = "gated"
    fusion_heads = 4
    graph_alignment = "none"
    graph_align_heads = 4
    graph_align_residual = True
    if config_path.exists():
        cfg = torch.load(config_path, map_location="cpu")
        hidden_size = int(cfg.get("hidden_size", hidden_size))
        dropout = float(cfg.get("dropout", dropout))
        fusion_type = str(cfg.get("fusion_type", fusion_type))
        fusion_heads = int(cfg.get("fusion_heads", fusion_heads))
        graph_alignment = str(cfg.get("graph_alignment", graph_alignment))
        graph_align_heads = int(cfg.get("graph_align_heads", graph_align_heads))
        graph_align_residual = bool(cfg.get("graph_align_residual", graph_align_residual))
        args.max_seq_len = int(cfg.get("max_seq_len", args.max_seq_len))
        args.max_graph_nodes = int(cfg.get("max_graph_nodes", args.max_graph_nodes))

    device = torch.device(args.device)
    tm = mv.TimeBucketManager()
    dataset = mv.load_full_dataset(args.train_csv, args.test_csv, args.train_qa, args.test_qa)
    all_pois = sorted(int(x) for x in dataset.poi_dict.keys())
    p2i = {pid: i for i, pid in enumerate(all_pois)}
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

    train_samples, train_loader = build_loader(
        args.train_qa, p2i, poi_category_map, tm, graph_builder, args.max_seq_len, args.batch_size
    )
    test_samples, test_loader = build_loader(
        args.test_qa, p2i, poi_category_map, tm, graph_builder, args.max_seq_len, args.batch_size
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
        graph_align_residual=graph_align_residual,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    train_metrics = mv.evaluate(model, train_loader, device=device, top_ks=(1, 5, 10, 20))
    test_metrics = mv.evaluate(model, test_loader, device=device, top_ks=(1, 5, 10, 20))

    result = {
        "model_path": model_path,
        "artifact_dir": str(artifact_dir),
        "dataset_name": args.dataset_name,
        "train_samples": len(train_samples),
        "test_samples": len(test_samples),
        "train": {
            "acc@1": train_metrics["recall@1"],
            "acc@5": train_metrics["recall@5"],
            "acc@10": train_metrics["recall@10"],
            "acc@20": train_metrics["recall@20"],
        },
        "test": {
            "acc@1": test_metrics["recall@1"],
            "acc@5": test_metrics["recall@5"],
            "acc@10": test_metrics["recall@10"],
            "acc@20": test_metrics["recall@20"],
        },
    }

    history_path = Path(args.history_path)
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        curve = {
            "epoch": [],
            "train_loss": [],
            "val_acc@1": [],
            "val_acc@5": [],
            "val_acc@10": [],
            "val_acc@20": [],
        }
        for row in history:
            curve["epoch"].append(int(row.get("epoch", len(curve["epoch"]) + 1)))
            curve["train_loss"].append(float(row.get("loss", 0.0)))
            curve["val_acc@1"].append(float(row.get("recall@1", 0.0)))
            curve["val_acc@5"].append(float(row.get("recall@5", 0.0)))
            curve["val_acc@10"].append(float(row.get("recall@10", 0.0)))
            curve["val_acc@20"].append(float(row.get("recall@20", 0.0)))
        result["curves"] = curve
        result["history_path"] = str(history_path)

    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
