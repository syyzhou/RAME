import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router_dump_path", required=True)
    parser.add_argument("--output_path", default="./router_analysis/router_soft_routing.png")
    parser.add_argument("--example_output_path", default=None)
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--stats_path", default=None)
    parser.add_argument("--title", default="Test-time soft routing")
    return parser.parse_args()


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stack_field(records, field):
    values = [r[field] for r in records if field in r and r[field] is not None]
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


def plot_expert_bar(ax, values, title, color):
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    x = np.arange(len(mean))
    ax.bar(x, mean, yerr=std, capsize=3, color=color, edgecolor="#222222", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel("LoRA expert")
    ax.set_ylabel("Average softmax weight")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in x])
    ax.set_ylim(0, max(0.35, float((mean + std).max()) * 1.2))
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)


def plot_gate_hist(ax, values):
    gate = values[:, 1]
    ax.hist(gate, bins=np.linspace(0.0, 1.0, 11), color="#7a5195", edgecolor="#222222", linewidth=0.6)
    ax.axvline(gate.mean(), color="#d62728", linestyle="--", linewidth=1.4, label=f"mean={gate.mean():.3f}")
    ax.set_title("Fusion gate distribution")
    ax.set_xlabel("Trajectory-router fusion weight")
    ax.set_ylabel("Number of samples")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="y", color="#dddddd", linewidth=0.8, alpha=0.8)
    ax.legend(frameon=False)


def select_example_records(records, num_examples):
    usable = [
        r for r in records
        if r.get("router1_weight") is not None
        and r.get("router2_weight") is not None
        and r.get("fusion_gate") is not None
    ]
    if not usable or num_examples <= 0:
        return []
    if len(usable) <= num_examples:
        return usable
    indices = np.linspace(0, len(usable) - 1, num_examples, dtype=int)
    return [usable[i] for i in indices]


def plot_examples(records, output_path, num_examples):
    examples = select_example_records(records, num_examples)
    if not examples:
        return

    router1 = np.asarray([r["router1_weight"] for r in examples], dtype=np.float64)
    router2 = np.asarray([r["router2_weight"] for r in examples], dtype=np.float64)
    fusion = np.asarray([r["fusion_gate"] for r in examples], dtype=np.float64)
    labels = [
        f"S{r.get('sample_id', i)} ({'T' if r.get('correct_at_1') else 'F'})"
        for i, r in enumerate(examples)
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(
        1, 3, figsize=(13, max(3.8, 0.45 * len(examples))), constrained_layout=True
    )

    im1 = axes[0].imshow(router1, aspect="auto", vmin=0.0, vmax=max(0.35, router1.max()), cmap="Blues")
    axes[0].set_title("Router1 examples")
    axes[0].set_xlabel("LoRA expert")
    axes[0].set_ylabel("Test sample")
    axes[0].set_xticks(np.arange(router1.shape[1]))
    axes[0].set_xticklabels([str(i + 1) for i in range(router1.shape[1])])
    axes[0].set_yticks(np.arange(len(labels)))
    axes[0].set_yticklabels(labels)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(router2, aspect="auto", vmin=0.0, vmax=max(0.35, router2.max()), cmap="Oranges")
    axes[1].set_title("Router2 examples")
    axes[1].set_xlabel("LoRA expert")
    axes[1].set_xticks(np.arange(router2.shape[1]))
    axes[1].set_xticklabels([str(i + 1) for i in range(router2.shape[1])])
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_yticklabels([])
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    y = np.arange(len(examples))
    axes[2].barh(y, fusion[:, 0], color="#4c78a8", edgecolor="#222222", linewidth=0.4, label="Router1")
    axes[2].barh(y, fusion[:, 1], left=fusion[:, 0], color="#f58518", edgecolor="#222222", linewidth=0.4, label="Router2")
    axes[2].set_title("Fusion gate examples")
    axes[2].set_xlabel("Fusion weight")
    axes[2].set_xlim(0.0, 1.0)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].invert_yaxis()
    axes[2].legend(frameon=False, loc="lower right")

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    records = load_jsonl(args.router_dump_path)
    if not records:
        raise ValueError(f"No records found in {args.router_dump_path}")

    router1 = stack_field(records, "router1_weight")
    router2 = stack_field(records, "router2_weight")
    fusion = stack_field(records, "fusion_gate")
    if router1 is None or router2 is None or fusion is None:
        raise ValueError("router_dump_path must contain router1_weight, router2_weight, and fusion_gate fields.")

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True)
    fig.suptitle(args.title, fontsize=13)
    plot_expert_bar(axes[0], router1, "Router1: token-context experts", "#4c78a8")
    plot_expert_bar(axes[1], router2, "Router2: trajectory experts", "#f58518")
    plot_gate_hist(axes[2], fusion)
    fig.savefig(args.output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if args.example_output_path is not None:
        plot_examples(records, args.example_output_path, args.num_examples)

    stats = {
        "num_samples": len(records),
        "router1_mean": router1.mean(axis=0).tolist(),
        "router2_mean": router2.mean(axis=0).tolist(),
        "fusion_gate_mean": fusion.mean(axis=0).tolist(),
        "correct_at_1": float(np.mean([bool(r.get("correct_at_1", False)) for r in records])),
        "correct_at_5": float(np.mean([bool(r.get("correct_at_5", False)) for r in records])),
        "correct_at_10": float(np.mean([bool(r.get("correct_at_10", False)) for r in records])),
    }
    stats_path = args.stats_path
    if stats_path is None:
        base, _ = os.path.splitext(args.output_path)
        stats_path = base + "_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Saved figure to {args.output_path}")
    if args.example_output_path is not None:
        print(f"Saved example figure to {args.example_output_path}")
    print(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()
