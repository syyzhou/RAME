#!/usr/bin/env python3
# Written by GitHub Copilot

import os
import math
import json
import re
import argparse
import random
from typing import Dict, List, Optional

import torch
import numpy as np
import transformers
import importlib
from tqdm import tqdm

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"


def parse_config():
    parser = argparse.ArgumentParser(description='Two-router GSM8K evaluation')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--base_model', type=str, default="./Qwen2-1.5B")
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--seq_len', type=int, default=4096)
    parser.add_argument('--context_size', type=int, default=4096)
    parser.add_argument('--peft_model', type=str, default=None)
    parser.add_argument('--flash_attn', type=bool, default=False)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--data_path', type=str, default="./test.bin")
    parser.add_argument('--output_dir', type=str, default="./outputmodels/finetune-36/")
    parser.add_argument('--test_file', type=str, default="test_qa_pairs_kqt_100.txt")
    parser.add_argument('--test_path', type=str, default=None,
                        help='Direct path to test data file. If set, overrides test_file.')
    parser.add_argument('--trajectory_embedding_path', type=str, default=None,
                        help='预编码的测试集轨迹向量路径 (.pt 或 .npy)')
    parser.add_argument('--dataset_name', type=str, default='nyc',
                        help='Dataset name for compatibility with run scripts')
    parser.add_argument('--use_retrieval_query_routing', action='store_true',
                        help='启用检索query路由（仅实验开关）')
    parser.add_argument('--retrieval_query_path', type=str, default=None,
                        help='检索query向量路径 (.pt 或 .npy)')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda device, e.g. cuda:0')
    parser.add_argument('--max_new_tokens', type=int, default=5)
    parser.add_argument('--num_beams', type=int, default=7)
    parser.add_argument('--num_return_sequences', type=int, default=7)
    parser.add_argument('--repetition_penalty', type=float, default=1.176)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--router_dump_path', type=str, default=None,
                        help='Optional jsonl path to save test-time predictions and aggregated router weights.')
    parser.add_argument('--save_raw_generations', action='store_true',
                        help='If set, save raw decoded generations for each sample into router_dump_path.')
    parser.add_argument('--router_model_file', type=str, default='model8',
                        help='路由模型文件名（不带.py），如 model8 或 model9_ret_cross_router')
    parser.add_argument('--trust_remote_code', action='store_true',
                        help='Allow trust_remote_code when loading model/config')
    return parser.parse_args()


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def load_trajectory_embeddings(path):
    if path is None or not os.path.exists(path):
        print(f"No trajectory embeddings found at {path}")
        return None

    print(f"Loading trajectory embeddings from {path}...")
    if path.endswith(".pt"):
        embed_data = torch.load(path, map_location="cpu")
        if isinstance(embed_data, dict):
            embeddings = embed_data.get("embeddings", None)
            if embeddings is None:
                embeddings = embed_data.get("embeds", None)
            if embeddings is None:
                embeddings = embed_data
        else:
            embeddings = embed_data
        if isinstance(embeddings, dict):
            raise ValueError(f"Unsupported .pt payload structure: {path}")
        print(f"  Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
        if isinstance(embed_data, dict):
            print(f"  Pooling: {embed_data.get('pooling', 'unknown')}")
            print(f"  Model: {embed_data.get('model_name', 'unknown')}")
    elif path.endswith(".npy"):
        embeddings = torch.from_numpy(np.load(path))
        print(f"  Loaded {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    else:
        print(f"  Unknown embedding format: {path}")
        return None

    return embeddings


def load_retrieval_queries(path):
    if path is None or not os.path.exists(path):
        print(f"No retrieval queries found at {path}")
        return None
    print(f"Loading retrieval queries from {path}...")
    if path.endswith(".pt"):
        qdata = torch.load(path, map_location="cpu")
        if isinstance(qdata, dict):
            queries = qdata.get("queries") or qdata.get("query_embeddings") or qdata.get("embeddings")
        else:
            queries = qdata
    elif path.endswith(".npy"):
        queries = torch.from_numpy(np.load(path))
    else:
        print(f"  Unknown query format: {path}")
        return None
    if queries is None:
        print(f"  Cannot find query tensor in {path}")
        return None
    print(f"  Loaded {queries.shape[0]} queries, dim={queries.shape[1]}")
    return queries


def _collect_traj_projectors(model):
    if not hasattr(model, "router_manager"):
        return []

    projectors = []
    seen = set()

    if hasattr(model, "traj_projector") and model.traj_projector is not None:
        seen.add(id(model.traj_projector))
        projectors.append(model.traj_projector)

    for router in model.router_manager.token_routers:
        if getattr(router, 'use_trajectory', False) and getattr(router, 'traj_projector', None) is not None:
            pid = id(router.traj_projector)
            if pid not in seen:
                seen.add(pid)
                projectors.append(router.traj_projector)

    return projectors


def set_trajectory_embedding_for_all_routers(model, traj_emb, device):
    if traj_emb is None:
        return 0

    projectors = _collect_traj_projectors(model)
    if len(projectors) == 0:
        return 0

    if traj_emb.dim() == 1:
        traj_emb = traj_emb.unsqueeze(0)
    traj_emb = traj_emb.to(dtype=torch.bfloat16, device=device)

    for proj in projectors:
        proj.set_trajectory_embedding(traj_emb)

    return len(projectors)


def clear_all_router_cache(model, clear_traj_embedding=False):
    if hasattr(model, "router_manager") and hasattr(model.router_manager, "clear"):
        model.router_manager.clear()

    if clear_traj_embedding:
        for proj in _collect_traj_projectors(model):
            proj.clear()


def verify_router_trajectory_setup(model):
    if not hasattr(model, "router_manager"):
        print("  [verify] No router_manager found")
        return

    print("\n  [verify] Router trajectory setup:")
    miss = 0
    for i, router in enumerate(model.router_manager.token_routers):
        needs_traj = getattr(router, 'use_trajectory', False)
        if not needs_traj:
            continue
        proj = getattr(router, 'traj_projector', None)
        cached = getattr(proj, '_cached_traj_embedding', None) if proj is not None else None
        has_embedding = cached is not None
        status = "OK" if has_embedding else "MISSING!"
        if not has_embedding:
            miss += 1
        print(f"    TokenRouter[{i}] tag={getattr(router, 'tag', None)}: "
              f"use_trajectory={needs_traj}, has_embedding={has_embedding} [{status}]")

    if miss == 0:
        print("    All trajectory-enabled TokenRouter are ready.")
    print()


def count_routers(model):
    if not hasattr(model, "router_manager"):
        return {}

    stats = {
        'token_routers_total': len(model.router_manager.token_routers),
        'token_routers_with_traj': 0,
        'fusion_gates_total': len(model.router_manager.routers),
        'traj_projectors_total': len(_collect_traj_projectors(model)),
    }

    for router in model.router_manager.token_routers:
        if getattr(router, 'use_trajectory', False):
            stats['token_routers_with_traj'] += 1

    return stats


def evaluate_prediction_accuracy(prediction: str, ground_truth: str) -> int:
    pred = extract_ground_truth_digits(prediction)
    gt = extract_ground_truth_digits(ground_truth)
    if pred is None or gt is None:
        return 0
    return int(pred == gt)


def extract_ground_truth_digits(text: str) -> Optional[str]:
    match = re.search(r"\d+", text)
    return match.group(0) if match else None


def _tensor_to_mean_list(tensor):
    if tensor is None:
        return None
    tensor = tensor.detach().float().cpu()
    if tensor.dim() == 3:
        tensor = tensor.mean(dim=(0, 1))
    elif tensor.dim() == 2:
        tensor = tensor.mean(dim=0)
    return tensor.tolist()


def collect_router_snapshot(model):
    if not hasattr(model, "router_manager"):
        return {}

    router1_weights = []
    router2_weights = []
    for router in model.router_manager.token_routers:
        weight = _tensor_to_mean_list(getattr(router, "routing_weight", None))
        if weight is None:
            continue
        tag = getattr(router, "tag", "") or ""
        if tag.endswith("_G2") or getattr(router, "use_trajectory", False):
            router2_weights.append(weight)
        else:
            router1_weights.append(weight)

    fusion_weights = []
    for gate in model.router_manager.routers:
        weight = _tensor_to_mean_list(getattr(gate, "routing_weight", None))
        if weight is not None:
            fusion_weights.append(weight)

    snapshot = {}
    if router1_weights:
        snapshot["router1_weight"] = np.asarray(router1_weights, dtype=np.float64).mean(axis=0).tolist()
    if router2_weights:
        snapshot["router2_weight"] = np.asarray(router2_weights, dtype=np.float64).mean(axis=0).tolist()
    if fusion_weights:
        snapshot["fusion_gate"] = np.asarray(fusion_weights, dtype=np.float64).mean(axis=0).tolist()
    return snapshot


def normalize_question_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def load_test_samples(path: str) -> List[Dict[str, str]]:
    lower = path.lower()
    if lower.endswith(".txt"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "<answer>:" not in line:
                    continue
                q, a = line.split("<answer>:", 1)
                q = q.replace("<question>:", "").strip()
                rows.append({"question": normalize_question_text(q), "answer": a.strip()})
        return rows
    if lower.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for obj in data:
            if "question" in obj and "answer" in obj:
                rows.append({"question": normalize_question_text(str(obj["question"])), "answer": str(obj["answer"])})
            elif "user_prompt" in obj and "assistant_prompt" in obj:
                rows.append({"question": normalize_question_text(str(obj["user_prompt"])), "answer": str(obj["assistant_prompt"])})
        return rows
    if lower.endswith(".jsonl"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if "question" in obj and "answer" in obj:
                    rows.append({"question": normalize_question_text(str(obj["question"])), "answer": str(obj["answer"])})
                elif "user_prompt" in obj and "assistant_prompt" in obj:
                    rows.append({"question": normalize_question_text(str(obj["user_prompt"])), "answer": str(obj["assistant_prompt"])})
        return rows
    raise ValueError(f"Unsupported test format: {path}")


def main(args):
    device = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.set_device(device)

    seed = 2
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model_path = args.model_path
    output_dir = args.output_dir
    if args.test_path is not None:
        test_path = args.test_path
    else:
        test_path = f'./{args.test_file}'

    print("data path", test_path)
    print("base model", model_path)
    print("peft model", output_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=4096,
        padding_side="right",
        use_fast=False,
        trust_remote_code=args.trust_remote_code,
    )

    model_config = transformers.AutoConfig.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)

    context_size = args.context_size if args.context_size > 0 else args.seq_len
    orig_ctx_len = getattr(model_config, "max_position_embeddings", None)
    if orig_ctx_len and context_size > orig_ctx_len:
        scaling_factor = float(math.ceil(context_size / orig_ctx_len))
        model_config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
    )

    if output_dir:
        router_module = importlib.import_module(args.router_model_file)
        MoraModelCls = getattr(router_module, "MoraModel")
        model = MoraModelCls.from_pretrained(model, output_dir)
        peft_weights = torch.load(output_dir + '/' + 'adapter_model.safetensors', map_location='cpu')
        model.load_state_dict(peft_weights, strict=False)

    model.eval()
    model.to(device)

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    traj_embeddings = load_trajectory_embeddings(args.trajectory_embedding_path)
    retrieval_queries = load_retrieval_queries(args.retrieval_query_path) if args.use_retrieval_query_routing else None
    use_traj_routing = traj_embeddings is not None and hasattr(model, "router_manager")
    use_ret_routing = retrieval_queries is not None and hasattr(model, "router_manager") \
        and hasattr(model.router_manager, "set_retrieval_query")

    if use_traj_routing:
        print(f"\nTrajectory routing ENABLED:")
        print(f"  Embeddings: {traj_embeddings.shape[0]} samples, dim={traj_embeddings.shape[1]}")

        router_stats = count_routers(model)
        print(f"  TokenRouters: {router_stats['token_routers_total']} total, "
              f"{router_stats['token_routers_with_traj']} use trajectory")
        print(f"  FusionGates:  {router_stats['fusion_gates_total']} total")
        print(f"  TrajProjectors: {router_stats['traj_projectors_total']}")

        print("\n  Verifying with first sample...")
        set_trajectory_embedding_for_all_routers(model, traj_embeddings[0], device)
        verify_router_trajectory_setup(model)
        clear_all_router_cache(model, clear_traj_embedding=True)
    else:
        print("\nTrajectory routing DISABLED")
        if traj_embeddings is None:
            print("  Reason: no trajectory embeddings loaded")
        if not hasattr(model, "router_manager"):
            print("  Reason: model has no router_manager")

    if use_ret_routing:
        print(f"\nRetrieval-query routing ENABLED:")
        print(f"  Queries: {retrieval_queries.shape[0]} samples, dim={retrieval_queries.shape[1]}")
    else:
        print("\nRetrieval-query routing DISABLED")
        if args.use_retrieval_query_routing and retrieval_queries is None:
            print("  Reason: no retrieval queries loaded")
        if args.use_retrieval_query_routing and hasattr(model, "router_manager") and not hasattr(model.router_manager, "set_retrieval_query"):
            print("  Reason: current model/router_manager has no set_retrieval_query interface")

    generation_config = transformers.GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=None,
        do_sample=False,
        num_beams=args.num_beams,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        num_return_sequences=args.num_return_sequences
    )

    rows = load_test_samples(test_path)
    print(f"test path {test_path}")
    print(f"num test samples {len(rows)}")
    if len(rows) == 0:
        raise ValueError("No valid test samples loaded.")

    if use_traj_routing:
        n_embed = len(traj_embeddings)
        n_test = len(rows)
        if n_embed != n_test:
            print(f"\n  WARNING: embedding count ({n_embed}) != test sample count ({n_test})")
            print(f"  Will process min({n_embed}, {n_test}) = {min(n_embed, n_test)} samples with trajectory routing")
            print(f"  Remaining {max(0, n_test - n_embed)} samples will use uniform routing fallback")
    if use_ret_routing:
        n_ret = len(retrieval_queries)
        n_test = len(rows)
        if n_ret != n_test:
            print(f"\n  WARNING: retrieval-query count ({n_ret}) != test sample count ({n_test})")
            print(f"  Will apply retrieval query on min({n_ret}, {n_test}) samples")

    correct_predictions_1 = 0
    correct_predictions_5 = 0
    correct_predictions_10 = 0
    model.eval()

    correct_list = []
    unique_prediction_counts = []
    samples_with_at_least_5_unique = 0
    samples_with_at_least_10_unique = 0
    skipped = 0
    traj_used_count = 0
    traj_fallback_count = 0
    ret_used_count = 0
    ret_fallback_count = 0
    router_dump_file = None
    if args.router_dump_path is not None:
        router_dump_dir = os.path.dirname(args.router_dump_path)
        if router_dump_dir:
            os.makedirs(router_dump_dir, exist_ok=True)
        router_dump_file = open(args.router_dump_path, "w", encoding="utf-8")
        print(f"Saving per-sample router dump to {args.router_dump_path}")

    for index, sample in tqdm(enumerate(rows), desc="Processing samples", total=len(rows)):
        prompt = sample["question"]
        gt = sample["answer"].replace('[', '').replace(']', '').strip()
        if not prompt.startswith("<question>:"):
            prompt = "<question>: " + prompt
        if not prompt.endswith("<answer>:"):
            prompt = prompt + "<answer>: "

        if len(tokenizer.tokenize(prompt)) >= 4096:
            skipped += 1
            unique_prediction_counts.append(0)
            continue

        if use_traj_routing and index < len(traj_embeddings):
            set_trajectory_embedding_for_all_routers(model, traj_embeddings[index], device)
            traj_used_count += 1
        else:
            clear_all_router_cache(model, clear_traj_embedding=True)
            traj_fallback_count += 1

        if use_ret_routing and index < len(retrieval_queries):
            rq = retrieval_queries[index]
            if rq.dim() == 1:
                rq = rq.unsqueeze(0)
            rq = rq.to(dtype=torch.bfloat16, device=device)
            model.router_manager.set_retrieval_query(rq)
            ret_used_count += 1
        else:
            ret_fallback_count += 1

        if hasattr(model, 'task_encoder') and model.task_encoder is not None:
            prefix_tensors = tokenizer(
                prompt, padding=True, return_tensors='pt',
                add_special_tokens=False).to(device)
            embedding = getattr(model.base_model, "embed_tokens")
            hidden_states = embedding(prefix_tensors["input_ids"])
            task_embed = model.task_encoder(hidden_states, prefix_tensors["attention_mask"])
            model.router_manager.set_task_weight(task_embed)

        prompt_tokens = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(**prompt_tokens,
                                     generation_config=generation_config)
            router_snapshot = collect_router_snapshot(model)
            clear_all_router_cache(model, clear_traj_embedding=True)

        seen_predictions = set()
        topk_predictions = []
        raw_generations = []
        unique_rank = 0
        for i in range(outputs.shape[0]):
            try:
                raw_prediction = tokenizer.decode(
                    outputs[:, prompt_tokens.input_ids.shape[1]:][i],
                    skip_special_tokens=True).strip()
                if args.save_raw_generations:
                    raw_generations.append(raw_prediction)

                match = re.search(r'\d+', raw_prediction)
                if not match:
                    continue
                prediction = match.group(0)
                if prediction in seen_predictions:
                    continue
                seen_predictions.add(prediction)
                topk_predictions.append(prediction)
                unique_rank += 1
                correct = evaluate_prediction_accuracy(prediction, gt)
                if correct:
                    if unique_rank == 1:
                        correct_list.append(index)
                        correct_predictions_1 += 1
                    if unique_rank <= 5:
                        correct_predictions_5 += 1
                    if unique_rank <= 10:
                        correct_predictions_10 += 1
                    break
                if unique_rank >= 10:
                    break
            except Exception:
                continue

        unique_prediction_counts.append(unique_rank)
        if unique_rank >= 5:
            samples_with_at_least_5_unique += 1
        if unique_rank >= 10:
            samples_with_at_least_10_unique += 1

        if router_dump_file is not None:
            record = {
                "sample_id": index,
                "ground_truth": extract_ground_truth_digits(gt),
                "topk_predictions": topk_predictions,
                "correct_at_1": bool(len(topk_predictions) >= 1 and topk_predictions[0] == extract_ground_truth_digits(gt)),
                "correct_at_5": bool(extract_ground_truth_digits(gt) in topk_predictions[:5]),
                "correct_at_10": bool(extract_ground_truth_digits(gt) in topk_predictions[:10]),
            }
            if args.save_raw_generations:
                record["raw_generations"] = raw_generations
            record.update(router_snapshot)
            router_dump_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(rows)
    print(f"\n{'='*60}")
    print('Results:')
    print(f"{'='*60}")
    print(f"  Total samples: {total}")
    print(f"  Skipped (too long): {skipped}")
    print(f"  Evaluated: {total - skipped}")
    print(f"  ACC@1:  {correct_predictions_1 / total:.4f} ({correct_predictions_1}/{total})")
    print(f"  ACC@5:  {correct_predictions_5 / total:.4f} ({correct_predictions_5}/{total})")
    print(f"  ACC@10: {correct_predictions_10 / total:.4f} ({correct_predictions_10}/{total})")
    print(f"  Unique predictions >=5:  {samples_with_at_least_5_unique / total:.4f} "
          f"({samples_with_at_least_5_unique}/{total})")
    print(f"  Unique predictions >=10: {samples_with_at_least_10_unique / total:.4f} "
          f"({samples_with_at_least_10_unique}/{total})")

    if use_traj_routing:
        print(f"\n  Trajectory routing stats:")
        print(f"    With trajectory:    {traj_used_count}")
        print(f"    Fallback (uniform): {traj_fallback_count}")
    if use_ret_routing:
        print(f"\n  Retrieval-query routing stats:")
        print(f"    With retrieval query: {ret_used_count}")
        print(f"    Fallback (uniform):   {ret_fallback_count}")

    print(f"\n  Correct@1 indices: {correct_list}")
    print(f"{'='*60}")

    if router_dump_file is not None:
        router_dump_file.close()


if __name__ == "__main__":
    args = parse_config()
    main(args)
