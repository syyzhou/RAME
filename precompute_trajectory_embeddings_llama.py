#!/usr/bin/env python3
"""Llama-specific trajectory embedding precompute script.

Differences from precompute_trajectory_embeddings.py:
- Does not convert Llama 3.x rope_scaling to fake linear RoPE.
- Uses Llama EOS as PAD instead of adding <|endoftext|>.
- Reports Llama special tokens: BOS/EOS/PAD and Llama-3 chat tokens.
- Supports base mode and instruct/chat-template mode.
"""

import io
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers
from tqdm import tqdm


LLAMA3_START_HEADER = "<|start_header_id|>"
LLAMA3_END_HEADER = "<|end_header_id|>"
LLAMA3_EOT = "<|eot_id|>"


def jload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class Args:
    model_name_or_path: str = field(default="./Llama-3.2-3B")
    dataset_name: str = field(default_factory=lambda: os.getenv("DATASET_NAME", "nyc"))
    dataset: Optional[str] = field(default=None)
    output_path: Optional[str] = field(default=None)
    split: str = field(default="train")
    model_max_length: int = field(default=4096)
    batch_size: int = field(default=1)
    pooling: str = field(default="last")
    cache_dir: Optional[str] = field(default=None)
    device: str = field(default="auto")
    prompt_mode: str = field(default="auto", metadata={"help": "auto/base/instruct"})
    padding_side: str = field(default="right")
    trust_remote_code: bool = field(default=False)


def load_llama_config_4096_compatible(model_name_or_path: str, model_max_length: int, **kwargs):
    """Load Llama config safely on old transformers.

    Llama 3.1/3.2 configs may contain:
      rope_scaling = {
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
      }

    Old transformers versions reject this. For 4096 or other lengths not beyond
    original_max_position_embeddings, this loader removes rope_scaling instead
    of converting it to {"type": "linear", "factor": ...}. Linear is not the
    same as Llama-3 RoPE scaling.
    """
    try:
        return transformers.AutoConfig.from_pretrained(model_name_or_path, **kwargs)
    except ValueError as exc:
        if "rope_scaling" not in str(exc):
            raise

    config_path = os.path.join(model_name_or_path, "config.json")
    if not os.path.isfile(config_path):
        raise

    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = json.load(f)

    rope_scaling = config_dict.get("rope_scaling")
    if not isinstance(rope_scaling, dict):
        raise ValueError("rope_scaling error was raised, but config has no rope_scaling dict")

    rope_type = rope_scaling.get("rope_type") or rope_scaling.get("type")
    original_max_pos = int(
        rope_scaling.get(
            "original_max_position_embeddings",
            config_dict.get("max_position_embeddings", 4096),
        )
    )

    if rope_type != "llama3":
        raise ValueError(f"Unsupported rope_scaling for this loader: {rope_scaling}")
    if model_max_length > original_max_pos:
        raise ValueError(
            f"model_max_length={model_max_length} exceeds original_max_position_embeddings="
            f"{original_max_pos}. Upgrade transformers instead of using a fake linear fallback."
        )

    config_dict.pop("rope_scaling", None)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_config = os.path.join(tmp_dir, "config.json")
        with open(tmp_config, "w", encoding="utf-8") as f:
            json.dump(config_dict, f)
        config = transformers.AutoConfig.from_pretrained(tmp_dir, **kwargs)
    config._name_or_path = model_name_or_path
    return config


def prepare_llama_tokenizer(tokenizer, model=None):
    """Llama token policy.

    Typical Llama 3.x:
      bos_token: <|begin_of_text|>
      eos_token: <|end_of_text|>
      instruct turn end: <|eot_id|>
      pad_token: None in many checkpoints, so set pad=eos.

    Do not add Qwen/GPT style <|endoftext|> to Llama.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
            model.generation_config.eos_token_id = tokenizer.eos_token_id


def token_report(tokenizer):
    return {
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "llama3_start_header_id": tokenizer.convert_tokens_to_ids(LLAMA3_START_HEADER),
        "llama3_end_header_id": tokenizer.convert_tokens_to_ids(LLAMA3_END_HEADER),
        "llama3_eot_id": tokenizer.convert_tokens_to_ids(LLAMA3_EOT),
    }


def resolve_prompt_mode(args, tokenizer):
    if args.prompt_mode in {"base", "instruct"}:
        return args.prompt_mode
    if args.prompt_mode != "auto":
        raise ValueError(f"Unsupported prompt_mode={args.prompt_mode}")
    name = args.model_name_or_path.lower()
    if "instruct" in name:
        return "instruct"
    return "instruct" if getattr(tokenizer, "chat_template", None) else "base"


def load_trajectory_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    pattern = r"<question>:\s*(.*?)\s*<answer>:\s*(.*?)\s*(?=(<question>:|$))"
    return [{"question": q.strip(), "answer": a.strip()} for q, a, _ in re.findall(pattern, content, re.DOTALL)]


def extract_trajectory_text(question):
    match = re.search(r"(The following data contains.*?)(?:Given the data,)", question, re.DOTALL)
    if match:
        return match.group(1).strip(), True
    return question.strip(), False


def format_for_llama(tokenizer, text, prompt_mode):
    text = text.strip()
    if prompt_mode == "base":
        return text
    if prompt_mode != "instruct":
        raise ValueError(f"Unsupported prompt_mode={prompt_mode}")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Llama instruct mode requires tokenizer.chat_template")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def last_non_pad_indices(attention_mask):
    mask = attention_mask.to(torch.long)
    return mask.size(1) - 1 - torch.argmax(mask.flip(dims=[1]), dim=1)


def compute_embeddings(model, tokenizer, texts, args, device):
    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), args.batch_size), desc="Encoding trajectories"):
            batch_texts = texts[i:i + args.batch_size]
            enc = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.model_max_length,
            ).to(device)
            out = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                output_hidden_states=True,
            )
            hidden = out.hidden_states[-1]
            mask = enc["attention_mask"]
            if args.pooling == "last":
                idx = last_non_pad_indices(mask)
                emb = hidden[torch.arange(hidden.size(0), device=device), idx]
            elif args.pooling == "mean":
                mask_f = mask.unsqueeze(-1).to(hidden.dtype)
                emb = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
            elif args.pooling == "weighted_mean":
                pos = torch.arange(hidden.size(1), device=device).float()
                weights = pos.unsqueeze(0) * mask.float()
                weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1)
                emb = (hidden * weights.unsqueeze(-1)).sum(dim=1)
            else:
                raise ValueError(f"Unknown pooling={args.pooling}")
            all_embeddings.append(emb.cpu())
    return torch.cat(all_embeddings, dim=0)


def main():
    parser = transformers.HfArgumentParser(Args)
    args = parser.parse_args_into_dataclasses()[0]
    if args.split not in {"train", "test"}:
        raise ValueError("split must be train or test")

    data_dir = f"./datasets/{args.dataset_name}/preprocessed"
    if args.dataset is None:
        args.dataset = f"{data_dir}/train_qa_pairs_kqt.json" if args.split == "train" else f"{data_dir}/test_qa_pairs_kqt.txt"
    if args.output_path is None:
        args.output_path = f"{data_dir}/trajectory_embeddings_llama.pt" if args.split == "train" else f"{data_dir}/test_embeddings_llama.pt"

    device_name = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side=args.padding_side,
        use_fast=False,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    prepare_llama_tokenizer(tokenizer)
    prompt_mode = resolve_prompt_mode(args, tokenizer)

    print(f"Loading model: {args.model_name_or_path}")
    print(f"prompt_mode: {prompt_mode}")
    print("Llama token config:")
    for k, v in token_report(tokenizer).items():
        print(f"  {k}: {v}")

    config = load_llama_config_4096_compatible(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    print("Llama model config:")
    print(f"  model_type: {getattr(config, 'model_type', None)}")
    print(f"  hidden_size: {getattr(config, 'hidden_size', None)}")
    print(f"  num_hidden_layers: {getattr(config, 'num_hidden_layers', None)}")
    print(f"  max_position_embeddings: {getattr(config, 'max_position_embeddings', None)}")
    print(f"  rope_theta: {getattr(config, 'rope_theta', None)}")
    print(f"  rope_scaling: {getattr(config, 'rope_scaling', None)}")

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    ).to(device)
    prepare_llama_tokenizer(tokenizer, model)

    print(f"Loading dataset: {args.dataset}")
    data = jload(args.dataset) if args.dataset.endswith(".json") else load_trajectory_txt(args.dataset)
    trajectory_texts = []
    matched = 0
    unmatched_indices = []
    for idx, item in enumerate(data):
        traj, ok = extract_trajectory_text(item["question"])
        matched += int(ok)
        if not ok:
            unmatched_indices.append(idx)
        trajectory_texts.append(format_for_llama(tokenizer, traj, prompt_mode))

    print("Trajectory extraction:")
    print(f"  samples: {len(data)}")
    print(f"  matched: {matched}")
    print(f"  unmatched: {len(data) - matched}")
    if unmatched_indices:
        print(f"  unmatched first20: {unmatched_indices[:20]}")

    embeddings = compute_embeddings(model, tokenizer, trajectory_texts, args, device)
    save_data = {
        "embeddings": embeddings,
        "pooling": args.pooling,
        "model_name": args.model_name_or_path,
        "hidden_dim": embeddings.shape[1],
        "num_samples": embeddings.shape[0],
        "prompt_mode": prompt_mode,
        "padding_side": args.padding_side,
        "token_config": token_report(tokenizer),
        "llama_config": {
            "model_type": getattr(config, "model_type", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "rope_theta": getattr(config, "rope_theta", None),
            "rope_scaling": getattr(config, "rope_scaling", None),
        },
        "extraction_stats": {
            "original_count": len(data),
            "matched_count": matched,
            "unmatched_count": len(data) - matched,
            "unmatched_indices": unmatched_indices,
            "match_rate": matched / max(len(data), 1) * 100,
        },
    }
    torch.save(save_data, args.output_path)
    print(f"Embeddings shape: {tuple(embeddings.shape)}")
    print(f"Saved embeddings to {args.output_path}")


if __name__ == "__main__":
    main()
