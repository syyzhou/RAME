#!/usr/bin/env python3
"""
Precompute trajectory embeddings for GSM8K-style LLM4POI QA files.

This script intentionally avoids answers. It can encode either the original
text from <history> through </current>, or a natural-language representation
for trajectory routing.

Compared with the original compact mode, this version changes compact mode
from symbolic tags like <HIST>/<CURR>/<TARGET_TIME>/<REP> to a natural-language
prompt that distinguishes long-term history and current session.

The compact mode in this script does NOT include target time, because the goal
is to represent the trajectory itself rather than a trajectory-query pair.
"""

import io
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from tqdm import tqdm


TXT_QA_RE = re.compile(
    r"<question>:\s*(.*?)\s*<answer>:\s*(.*?)\s*(?=(<question>:|$))",
    re.IGNORECASE | re.DOTALL,
)
HISTORY_CURRENT_RE = re.compile(
    r"(<history>.*?</current>)",
    re.IGNORECASE | re.DOTALL,
)
HISTORY_RE = re.compile(r"<history>(.*?)</history>", re.IGNORECASE | re.DOTALL)
CURRENT_RE = re.compile(r"<current>(.*?)</current>", re.IGNORECASE | re.DOTALL)
ENTRY_RE = re.compile(
    r"(\d{4}-\d{2}-\d{2})[ T](\d{2}):(\d{2})(?::\d{2})?\s*\|\s*(\d+)\s*\|\s*([^\n<]+)"
)


def jload(path, mode="r"):
    if not isinstance(path, io.IOBase):
        path = open(path, mode=mode, encoding="utf-8")
    data = json.load(path)
    path.close()
    return data


@dataclass
class Args:
    model_name_or_path: str = field(default="./Qwen2.5-3B")
    dataset_name: str = field(default_factory=lambda: os.getenv("DATASET_NAME", "NYC"))
    dataset: Optional[str] = field(default=None)
    output_path: Optional[str] = field(default=None)
    split: str = field(default="train", metadata={"help": "train / test / validate"})
    model_max_length: int = field(default=4096)
    batch_size: int = field(default=1)
    pooling: str = field(default="last", metadata={"help": "last / mean / weighted_mean"})
    cache_dir: Optional[str] = field(default=None)
    device: Optional[str] = field(default=None)
    strict: bool = field(default=True)
    extraction_mode: str = field(
        default="history_current",
        metadata={"help": "history_current / compact"},
    )


def load_qa(path):
    if path.endswith(".json"):
        data = jload(path)
        return [{"question": str(x["question"]), "answer": str(x["answer"])} for x in data]

    if path.endswith(".txt"):
        text = open(path, "r", encoding="utf-8").read().strip()
        rows = []
        for m in TXT_QA_RE.finditer(text):
            rows.append({"question": m.group(1).strip(), "answer": m.group(2).strip()})
        if not rows and text:
            raise ValueError(f"Cannot parse QA txt file: {path}")
        return rows

    raise ValueError(f"Unsupported dataset format: {path}")


def extract_history_current_text(question, strict=True):
    match = HISTORY_CURRENT_RE.search(question)
    if match:
        return match.group(1).strip(), True
    if strict:
        raise ValueError("Cannot find <history>...</current> segment")
    return question.strip(), False


def normalize_category(category):
    """
    Normalize category into a natural-language phrase.

    Original compact version used CAT_Restaurant.
    This version uses "Restaurant" / "Fast Food Restaurant",
    which is easier for a general LLM to understand.
    """
    category = category.strip()
    category = re.sub(r"[^0-9A-Za-z]+", " ", category)
    category = re.sub(r"\s+", " ", category).strip()
    return category or "Unknown"


def parse_entries(section):
    """
    Parse check-in entries from one section.

    Expected input line:
        2024-01-01 08:30 | 123 | Restaurant

    Output entry:
        At 2024-01-01 08:30, the user visited POI 123, category Restaurant.
    """
    entries = []
    for date, hour, minute, poi_id, category in ENTRY_RE.findall(section):
        entries.append(
            f"At {date} {hour}:{minute}, the user visited POI {int(poi_id)}, "
            f"category {normalize_category(category)}."
        )
    return entries


def format_natural_trajectory(history_entries, current_entries):
    """
    Build the natural-language prompt for compact mode.

    The final sentence is intentionally used as a semantic aggregation point
    for last-token pooling.
    """
    parts = [
        "This is a user's check-in trajectory, including long-term historical visits and the current session.",
        "",
        "Long-term history visits:",
    ]

    if history_entries:
        parts.extend([f"- {entry}" for entry in history_entries])
    else:
        parts.append("- None.")

    parts.extend([
        "",
        "Current session visits:",
    ])

    if current_entries:
        parts.extend([f"- {entry}" for entry in current_entries])
    else:
        parts.append("- None.")

    parts.extend([
        "",
        "This trajectory represents the user's mobility patterns.",
    ])

    return "\n".join(parts).strip()


def extract_compact_trajectory_text(question, strict=True):
    """
    Keep the original function name for compatibility with existing pipeline.

    Original compact mode:
        <HIST>
        2024-01-01T08:30 | POI_123 | CAT_Restaurant

        <CURR>
        ...

        <TARGET_TIME>
        ...

        <REP>

    New compact mode:
        This is a user's check-in trajectory, including long-term historical visits and the current session.

        Long-term history visits:
        - At 2024-01-01 08:30, the user visited POI 123, category Restaurant.

        Current session visits:
        - At 2024-01-05 18:00, the user visited POI 789, category Mall.

        This trajectory represents the user's mobility preferences, temporal habits, and location transition patterns:

    Note:
        Target time is intentionally not included.
    """
    history_match = HISTORY_RE.search(question)
    current_match = CURRENT_RE.search(question)

    if not history_match or not current_match:
        if strict:
            missing = []
            if not history_match:
                missing.append("<history>")
            if not current_match:
                missing.append("<current>")
            raise ValueError(f"Cannot build natural compact trajectory text, missing: {', '.join(missing)}")
        return question.strip(), False, {
            "history_entries": 0,
            "current_entries": 0,
            "has_target_time": False,
        }

    history_entries = parse_entries(history_match.group(1))
    current_entries = parse_entries(current_match.group(1))

    if strict and not current_entries:
        raise ValueError("Cannot build natural compact trajectory text: <current> has no trajectory entries")

    text = format_natural_trajectory(history_entries, current_entries)

    return text, True, {
        "history_entries": len(history_entries),
        "current_entries": len(current_entries),
        # Kept only for compatibility with the old statistics code.
        # This mode intentionally does not use target time.
        "has_target_time": False,
    }


def last_non_pad_indices(attention_mask):
    mask = attention_mask.to(torch.long)
    return mask.size(1) - 1 - torch.argmax(mask.flip(dims=[1]), dim=1)


def compute_embeddings(model, tokenizer, texts, args, device):
    all_embeddings = []
    model.eval()

    with torch.no_grad():
        desc = "Encoding natural compact trajectories" if args.extraction_mode == "compact" else "Encoding <history>...</current>"
        for start in tqdm(range(0, len(texts), args.batch_size), desc=desc):
            batch_texts = texts[start:start + args.batch_size]
            encodings = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.model_max_length,
            ).to(device)

            outputs = model(
                input_ids=encodings["input_ids"],
                attention_mask=encodings["attention_mask"],
                output_hidden_states=False,
            )
            hidden_states = outputs.last_hidden_state
            attention_mask = encodings["attention_mask"]

            if args.pooling == "last":
                seq_lengths = last_non_pad_indices(attention_mask)
                embeddings = hidden_states[
                    torch.arange(hidden_states.size(0), device=device),
                    seq_lengths,
                ]
            elif args.pooling == "mean":
                mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                embeddings = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            elif args.pooling == "weighted_mean":
                positions = torch.arange(hidden_states.size(1), device=device).float()
                weights = positions.unsqueeze(0) * attention_mask.float()
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1)
                embeddings = (hidden_states * weights.unsqueeze(-1)).sum(dim=1)
            else:
                raise ValueError(f"Unknown pooling: {args.pooling}")

            all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


def default_dataset_path(args):
    data_dir = f"./datasets/{args.dataset_name}/preprocessed"
    if args.split == "train":
        return f"{data_dir}/train_qa_pairs_llm4poi_gsm8k.json"
    if args.split == "test":
        return f"{data_dir}/test_qa_pairs_llm4poi_gsm8k.txt"
    if args.split == "validate":
        return f"{data_dir}/validate_qa_pairs_llm4poi_gsm8k.txt"
    raise ValueError(f"Unsupported split: {args.split}")


def default_output_path(args):
    data_dir = f"./datasets/{args.dataset_name}/preprocessed"
    if args.extraction_mode == "compact":
        if args.split == "train":
            return f"{data_dir}/trajectory_embeddings_gsm8k_compact.pt"
        if args.split == "test":
            return f"{data_dir}/test_embeddings_gsm8k_compact.pt"
        if args.split == "validate":
            return f"{data_dir}/validate_embeddings_gsm8k_compact.pt"
        raise ValueError(f"Unsupported split: {args.split}")

    if args.split == "train":
        return f"{data_dir}/trajectory_embeddings_gsm8k_history_current.pt"
    if args.split == "test":
        return f"{data_dir}/test_embeddings_gsm8k_history_current.pt"
    if args.split == "validate":
        return f"{data_dir}/validate_embeddings_gsm8k_history_current.pt"
    raise ValueError(f"Unsupported split: {args.split}")


def main():
    parser = transformers.HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]
    if args.extraction_mode not in {"history_current", "compact"}:
        raise ValueError(f"Unsupported extraction_mode: {args.extraction_mode}")
    args.dataset = args.dataset or default_dataset_path(args)
    args.output_path = args.output_path or default_output_path(args)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    print(f"Loading dataset from {args.dataset}...")
    data = load_qa(args.dataset)
    print(f"Original dataset size: {len(data)}")

    texts = []
    matched_count = 0
    unmatched_indices = []
    answer_literal_in_segment = 0
    answer_literal_after_segment = 0
    total_history_entries = 0
    total_current_entries = 0
    missing_target_time = 0

    for idx, item in enumerate(data):
        if args.extraction_mode == "compact":
            text, matched, compact_stats = extract_compact_trajectory_text(
                item["question"], strict=args.strict
            )
            total_history_entries += compact_stats["history_entries"]
            total_current_entries += compact_stats["current_entries"]
            missing_target_time += int(not compact_stats["has_target_time"])
        else:
            text, matched = extract_history_current_text(item["question"], strict=args.strict)
        texts.append(text)
        matched_count += int(matched)
        if not matched:
            unmatched_indices.append(idx)

        answer = str(item.get("answer", "")).strip()
        if answer and answer in text:
            answer_literal_in_segment += 1
        if matched:
            match = HISTORY_CURRENT_RE.search(item["question"])
            after_segment = item["question"][match.end():]
            if answer and answer in after_segment:
                answer_literal_after_segment += 1

    print("\n" + "=" * 60)
    print("Trajectory Extraction Statistics")
    print("=" * 60)
    print(f"  Extraction mode:              {args.extraction_mode}")
    print(f"  Original samples:              {len(data)}")
    print(f"  Successfully matched:          {matched_count}")
    print(f"  Unmatched:                     {len(unmatched_indices)}")
    print(f"  Match rate:                    {matched_count / max(len(data), 1) * 100:.2f}%")
    if args.extraction_mode == "compact":
        print(f"  History entries:               {total_history_entries}")
        print(f"  Current entries:               {total_current_entries}")
        print("  Target time used:              NO")
    print(f"  Answer literal in segment:     {answer_literal_in_segment}")
    print(f"  Answer literal after segment:  {answer_literal_after_segment}")
    print(f"  Output text count:             {len(texts)}")
    print("=" * 60 + "\n")

    print(f"Loading model from {args.model_name_or_path} on {device}...")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="left",
        use_fast=False,
        cache_dir=args.cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = transformers.AutoModel.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    ).to(device)
    model.eval()

    embeddings = compute_embeddings(model, tokenizer, texts, args, device)
    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    assert embeddings.shape[0] == len(data), (
        f"FATAL: embeddings count ({embeddings.shape[0]}) != sample count ({len(data)})"
    )

    save_data = {
        "embeddings": embeddings,
        "pooling": args.pooling,
        "model_name": args.model_name_or_path,
        "model_class": "AutoModel",
        "hidden_dim": embeddings.shape[1],
        "num_samples": embeddings.shape[0],
        "source_dataset": args.dataset,
        "extraction_mode": (
            "gsm8k_natural_history_current_no_target"
            if args.extraction_mode == "compact"
            else "gsm8k_history_current_only"
        ),
        "extraction_span": (
            "natural prompt: long-term history + current session, no target time"
            if args.extraction_mode == "compact"
            else "<history>...</current>"
        ),
        "extraction_stats": {
            "original_count": len(data),
            "matched_count": matched_count,
            "unmatched_count": len(unmatched_indices),
            "unmatched_indices": unmatched_indices,
            "match_rate": matched_count / max(len(data), 1) * 100,
            "answer_literal_in_segment": answer_literal_in_segment,
            "answer_literal_after_segment": answer_literal_after_segment,
            "total_history_entries": total_history_entries,
            "total_current_entries": total_current_entries,
            # Kept for compatibility. In this version, compact mode intentionally does not use target time.
            "missing_target_time": missing_target_time,
            "target_time_used": False if args.extraction_mode == "compact" else None,
        },
    }
    torch.save(save_data, args.output_path)
    print(f"Saved embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
