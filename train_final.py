# Written by Peibo Li
# Original code based on https://github.com/dvlab-research/LongLoRA?tab=readme-ov-file
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import copy
import json
import math
import logging
import tempfile
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorForLanguageModeling, BitsAndBytesConfig
# from llama_attn_replace_sft import replace_llama_attn
# from gptneox_attn_replace import replace_gpt_neox_attn
# from peft import LoraConfig, get_peft_model
from torch.distributed import barrier
from config_model9 import TARGET_MODULE_TYPE, UmRaConfig
import re
import seaborn as sns
import importlib



IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<|endoftext|>"
DEFAULT_EOS_TOKEN = "<|endoftext|>"
os.environ["WANDB_DISABLED"]="true"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def infer_dataset_name(dataset_path: str, explicit_name: Optional[str] = None) -> str:
    if explicit_name is not None:
        return explicit_name

    path = dataset_path.replace("\\", "/").lower()
    basename = os.path.basename(path)
    for name in ["nyc", "tky", "ca"]:
        if f"/{name}/" in path or path.endswith(f"/{name}") or f"{name}_" in basename:
            return name

    raise ValueError(
        f"Cannot infer dataset_name from path: {dataset_path}. "
        "Please pass --dataset_name nyc|tky|ca"
    )


def load_auto_config_compatible(model_name_or_path: str, **kwargs):
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
        if not isinstance(rope_scaling, dict) or "type" in rope_scaling:
            raise

        # Older transformers releases only accept {"type", "factor"} here,
        # while Llama 3.1/3.2 configs use rope_type plus extra frequency fields.
        config_dict["rope_scaling"] = {
            "type": "linear",
            "factor": float(rope_scaling.get("factor", 1.0)),
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_config_path = os.path.join(tmp_dir, "config.json")
            with open(tmp_config_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f)
            config = transformers.AutoConfig.from_pretrained(tmp_dir, **kwargs)
        config._name_or_path = model_name_or_path
        return config


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class DataArguments:
    dataset: str = field(default=None, metadata={"help": "Path to the training data."})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Dataset name: nyc/tky/ca"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    use_full_attn: bool = field(
        default=False,
        metadata={"help": "Whether to use plain, full-attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default=None,
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    remove_unused_columns: bool = field(default=False)
@dataclass
class ConfigArguments:
    dropout: float = field(default=0.0, metadata={"help": "Dropout概率"})
    # LoRA 参数
    lora_r: int = field(default=8, metadata={"help": "LoRA 低秩矩阵的秩"})
    lora_alpha: int = field(default=16, metadata={"help": "LoRA 缩放因子"})
    target_modules: str = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
                                      metadata={"help": "LoRA 目标模块"})
    target_modules_lora: Optional[str] = field(default=None, metadata={"help": "LoRA 特定目标模块"})

    # HMORA 路由策略 
    top_k_routing_strategy: bool = field(default=False, metadata={"help": "是否启用 top-k 路由策略"})
    top_k: int = field(default=2, metadata={"help": "路由时选择的 top-k 值"})
    trajectory_top_k_routing_strategy: bool = field(
        default=False, metadata={"help": "是否仅对轨迹路由启用 top-k 稀疏选择"}
    )
    trajectory_top_k: int = field(
        default=2, metadata={"help": "轨迹路由选择的 top-k 值"}
    )
 
    # HMORA 路由共享相关
    use_task_router: bool = field(default=False, metadata={"help": "是否使用任务路由器"})
    task_router_only: bool = field(default=False, metadata={"help": "是否仅使用任务路由器"})
    share_router_for_qkv: bool = field(default=False, metadata={"help": "是否共享 QKV 路由器"})
    share_router_for_w_i: bool = field(default=False, metadata={"help": "是否共享 W_i 路由器"})

    # HMORA 路由配置
    num_router_mlp_layers: int = field(default=1, metadata={"help": "路由器 MLP 层数"})
    router_hidden_dim: int = field(default=32, metadata={"help": "路由器隐藏层维度"})
    epsilon_alpha: float = field(default=2.0, metadata={"help": "epsilon alpha 超参数"})
    alpha_shift: float = field(default=0.0, metadata={"help": "alpha 偏移"})
    alpha_up_bound: float = field(default=0.8, metadata={"help": "alpha 上限"})
    alpha_low_bound: float = field(default=0.2, metadata={"help": "alpha 下限"})

    # HMORA 损失项
    use_load_balancing_loss: bool = field(default=False, metadata={"help": "是否使用负载均衡损失"})
    use_div_loss: bool = field(default=False, metadata={"help": "是否使用多样性损失"})
    gamma_div_certain_t: float = field(default=0.5, metadata={"help": "γ 多样性确定性（任务）"})
    gamma_div_balance_t: float = field(default=0.98, metadata={"help": "γ 多样性平衡性（任务）"})
    gamma_div_certain_s: float = field(default=0.5, metadata={"help": "γ 多样性确定性（样本）"})
    gamma_div_balance_s: float = field(default=0.98, metadata={"help": "γ 多样性平衡性（样本）"})
    lambda_auxiliary: float = field(default=0.005, metadata={"help": "辅助损失权重"})
    lambda_lm: float = field(default=1.0, metadata={"help": "语言建模损失权重"})

    # HMORA Experts
    eta_b: float = field(default=1.2, metadata={"help": "专家路由冗余率"})
    num_experts: int = field(default=4, metadata={"help": "专家数量"})
    use_hydra_lora: bool = field(default=True, metadata={"help": "是否启用 HydraLoRA"})
    router1_use_shared_expert: bool = field(default=False, metadata={"help": "是否为router1启用共享专家"})
    router1_shared_expert_weight: float = field(default=1.0, metadata={"help": "router1共享专家权重"})
    disable_router1: bool = field(default=False, metadata={"help": "是否禁用router1（平均池化路由组）"})
    disable_router2: bool = field(default=False, metadata={"help": "是否禁用router2（轨迹路由组），只使用router1和router1共享专家"})
    
    # 轨迹编码路由
    use_trajectory_routing: bool = field(
        default=False, metadata={"help": "是否使用预编码轨迹向量辅助路由"}
    )
    router2_input_source: str = field(
        default="trajectory",
        metadata={"help": "router2输入来源: trajectory, hidden_mean 或 hidden_token"}
    )
    trajectory_embedding_path: Optional[str] = field(
        default=None, metadata={"help": "预编码轨迹向量的路径 (.pt 或 .npy)"}
    )
    trajectory_fusion_mode: str = field(
        default="gate", metadata={"help": "轨迹向量融合方式: gate/add/concat_proj/cross_attention"}
    )
    share_traj_projector: bool = field(
        default=True, metadata={"help": "是否所有层共享轨迹投影器"}
    )
    share_ret_query_projector: bool = field(
        default=True, metadata={"help": "是否所有层共享检索 query 投影器"}
    )
    use_retrieval_query_routing: bool = field(
        default=False, metadata={"help": "是否启用检索query向量路由（仅实验开关）"}
    )
    router1_input_source: str = field(
        default="retquery",
        metadata={"help": "router1输入来源: retquery 或 hidden_token"}
    )
    retrieval_query_path: Optional[str] = field(
        default=None, metadata={"help": "检索query向量路径 (.pt/.npy)，长度需与训练样本一致"}
    )
    retrieval_query_dim: int = field(
        default=0, metadata={"help": "检索query维度。0表示自动从retrieval_query_path推断"}
    )
    router_model_file: str = field(
        default="model8", metadata={"help": "路由模型文件名（不带.py），如 model8 或 model9_ret_cross_router"}
    )
    





def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


# class SupervisedDataset(Dataset):
#     """Dataset for supervised fine-tuning."""

#     def __init__(self, dataset: str, tokenizer: transformers.PreTrainedTokenizer):
#         super(SupervisedDataset, self).__init__()
#         logging.warning("Loading data...")
#         list_data_dict = jload(dataset)

#         logging.warning("Formatting inputs...")

#         if '<question>:' not in list_data_dict[0]["question"]:
#             sources = ['<question>:' + example["question"] for example in list_data_dict]
#             targets = ['<answer>:' + f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]
#         else:
#             sources = [example["question"] for example in list_data_dict]
#             targets = [f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]
#         logging.warning("Tokenizing inputs... This may take some time...")
#         data_dict = preprocess(sources, targets, tokenizer)

#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["labels"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(input_ids=self.input_ids[i], labels=self.labels[i])

# 在 sft.py 中修改 SupervisedDataset 和 DataCollator

class SupervisedDatasetWithEmbeddings(Dataset):
    """带有预编码轨迹向量的监督微调数据集"""

    def __init__(self, dataset: str, tokenizer: transformers.PreTrainedTokenizer,
                 trajectory_embedding_path: Optional[str] = None,
                 retrieval_query_path: Optional[str] = None):
        super(SupervisedDatasetWithEmbeddings, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(dataset)

        logging.warning("Formatting inputs...")
        if '<question>:' not in list_data_dict[0]["question"]:
            sources = ['<question>:' + example["question"] for example in list_data_dict]
            targets = ['<answer>:' + f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]
        else:
            sources = [example["question"] for example in list_data_dict]
            targets = [f"{example['answer']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        # 加载预编码的轨迹向量
        self.trajectory_embeddings = None
        if trajectory_embedding_path is not None and os.path.exists(trajectory_embedding_path):
            logging.warning(f"Loading trajectory embeddings from {trajectory_embedding_path}...")
            if trajectory_embedding_path.endswith(".pt"):
                embed_data = torch.load(trajectory_embedding_path, map_location="cpu")
                self.trajectory_embeddings = embed_data["embeddings"]  # [N, hidden_dim]
            elif trajectory_embedding_path.endswith(".npy"):
                self.trajectory_embeddings = torch.from_numpy(
                    np.load(trajectory_embedding_path)
                )
            
            assert len(self.trajectory_embeddings) == len(self.input_ids), \
                f"Embedding数量({len(self.trajectory_embeddings)})与样本数量({len(self.input_ids)})不匹配"
            logging.warning(f"Loaded {len(self.trajectory_embeddings)} trajectory embeddings, "
                          f"dim={self.trajectory_embeddings.shape[1]}")
        else:
            logging.warning("No trajectory embeddings loaded, router will use token hidden states only.")

        self.retrieval_queries = None
        if retrieval_query_path is not None and os.path.exists(retrieval_query_path):
            logging.warning(f"Loading retrieval queries from {retrieval_query_path}...")
            if retrieval_query_path.endswith(".pt"):
                rdata = torch.load(retrieval_query_path, map_location="cpu")
                if isinstance(rdata, dict):
                    self.retrieval_queries = (
                        rdata.get("queries")
                        or rdata.get("query_embeddings")
                        or rdata.get("embeddings")
                    )
                else:
                    self.retrieval_queries = rdata
            elif retrieval_query_path.endswith(".npy"):
                self.retrieval_queries = torch.from_numpy(np.load(retrieval_query_path))
            if self.retrieval_queries is None:
                raise ValueError(f"Cannot find query tensor in {retrieval_query_path}")
            assert len(self.retrieval_queries) == len(self.input_ids), \
                f"检索query数量({len(self.retrieval_queries)})与样本数量({len(self.input_ids)})不匹配"
            logging.warning(f"Loaded {len(self.retrieval_queries)} retrieval queries, "
                            f"dim={self.retrieval_queries.shape[1]}")
        else:
            logging.warning("No retrieval queries loaded.")

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        result = dict(input_ids=self.input_ids[i], labels=self.labels[i])
        if self.trajectory_embeddings is not None:
            result["trajectory_embedding"] = self.trajectory_embeddings[i]
        if self.retrieval_queries is not None:
            result["retrieval_query"] = self.retrieval_queries[i]
        return result


@dataclass
class DataCollatorWithEmbeddings(object):
    """支持轨迹向量的DataCollator"""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        result = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # 如果有轨迹向量，也打包进去
        if "trajectory_embedding" in instances[0]:
            trajectory_embeddings = torch.stack(
                [inst["trajectory_embedding"] for inst in instances], dim=0
            )
            result["trajectory_embedding"] = trajectory_embeddings
        if "retrieval_query" in instances[0]:
            retrieval_queries = torch.stack(
                [inst["retrieval_query"] for inst in instances], dim=0
            )
            result["retrieval_query"] = retrieval_queries

        return result


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    trajectory_embedding_path: Optional[str] = None,
    retrieval_query_path: Optional[str] = None,
) -> Dict:
    train_dataset = SupervisedDatasetWithEmbeddings(
        tokenizer=tokenizer,
        dataset=data_args.dataset,
        trajectory_embedding_path=trajectory_embedding_path,
        retrieval_query_path=retrieval_query_path,
    )
    data_collator = DataCollatorWithEmbeddings(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

# @dataclass
# class DataCollatorForSupervisedDataset(object):
#     """Collate examples for supervised fine-tuning."""

#     tokenizer: transformers.PreTrainedTokenizer

#     def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
#         input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
#         input_ids = torch.nn.utils.rnn.pad_sequence(
#             input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
#         )
#         labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
#         return dict(
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
#         )


# def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
#     """Make dataset and collator for supervised fine-tuning."""
#     train_dataset = SupervisedDataset(tokenizer=tokenizer, dataset=data_args.dataset)
#     data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
#     return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

# class UmraTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # 获取标准的模型输出和loss
#         outputs = model(**inputs)
#         loss = outputs.loss

#         # 获取attention_mask，如果需要的话
#         attention_mask = inputs.get("attention_mask", None)

#         # 添加自定义的 auxiliary loss
#         if hasattr(model, "router_manager") and hasattr(model.router_manager, "get_auxiliary_loss"):
#             loss = model.router_manager.get_auxiliary_loss(loss, attention_mask)

#         return (loss, outputs) if return_outputs else loss

#     def training_step(self, model, inputs):
#         model.train()
#         if isinstance(model, torch.nn.DataParallel):
#             print("模型被 DataParallel 包裹了")
#             model = model.module  # 解除包裹
#         inputs = self._prepare_inputs(inputs)
        
#         # if hasattr(model, "task_encoder") and model.task_encoder is not None:
#         #     if "input_ids" in inputs:
#         #         embedding_fn = getattr(
#         #             model.base_model,
#         #             TARGET_MODULE_TYPE[model.config.model_type]['embed']
#         #         )
#         #         hidden_states = embedding_fn(inputs["input_ids"])

#         #         task_embed = model.task_encoder(hidden_states, inputs['attention_mask'])
#         #         task_embed = task_embed.to(dtype=torch.bfloat16)  
#         #         model.router_manager.set_task_weight(task_embed)
#         # 梯度缩放（如果使用 fp16）
#         if self.args.fp16 and self.use_amp:
#             with self.autocast_smart_context_manager():
#                 loss = self.compute_loss(model, inputs)
#         else:
#             loss = self.compute_loss(model, inputs)

#         # 反向传播
#         if self.args.gradient_accumulation_steps > 1:
#             loss = loss / self.args.gradient_accumulation_steps

#         self.accelerator.backward(loss)

#         if hasattr(model, "router_manager") and hasattr(model.router_manager, "clear"):
#             model.router_manager.clear()

#         return loss.detach()
    
# sft.py 中修改 UmraTrainer

# class UmraTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False):
#         # 从inputs中取出轨迹向量（如果有的话）
#         trajectory_embedding = inputs.pop("trajectory_embedding", None)
        
#         # 在forward之前，将轨迹向量分发到所有router
#         if trajectory_embedding is not None and hasattr(model, "router_manager"):
#             trajectory_embedding = trajectory_embedding.to(
#                 dtype=model.dtype if hasattr(model, 'dtype') else torch.bfloat16,
#                 device=next(model.parameters()).device
#             )
#             for router in model.router_manager.token_routers:
#                 if hasattr(router, 'set_trajectory_embedding'):
#                     router.set_trajectory_embedding(trajectory_embedding)
        
#         outputs = model(**inputs)
#         loss = outputs.loss

#         attention_mask = inputs.get("attention_mask", None)
#         if hasattr(model, "router_manager") and hasattr(model.router_manager, "get_auxiliary_loss"):
#             loss = model.router_manager.get_auxiliary_loss(loss, attention_mask)

#         return (loss, outputs) if return_outputs else loss
    
# class UmraTrainer(Trainer):
#     def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
#         # 从inputs中取出轨迹向量
#         trajectory_embedding = inputs.pop("trajectory_embedding", None)
        
#         # 分发轨迹向量到 use_trajectory=True 的router（即router2）
#         if trajectory_embedding is not None and hasattr(model, "router_manager"):
#             trajectory_embedding = trajectory_embedding.to(
#                 dtype=torch.bfloat16,
#                 device=next(model.parameters()).device
#             )
#             for router in model.router_manager.token_routers:
#                 if getattr(router, 'use_trajectory', False):
#                     router.set_trajectory_embedding(trajectory_embedding)
        
#         outputs = model(**inputs)
#         loss = outputs.loss

#         attention_mask = inputs.get("attention_mask", None)
#         if hasattr(model, "router_manager") and hasattr(model.router_manager, "get_auxiliary_loss"):
#             loss = model.router_manager.get_auxiliary_loss(loss, attention_mask)

#         # 清理router缓存
#         if hasattr(model, "router_manager") and hasattr(model.router_manager, "clear"):
#             model.router_manager.clear()

#         return (loss, outputs) if return_outputs else loss

class UmraTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_metrics = {
            "epoch_times": [],
            "total_training_time": 0.0,
            "peak_memory_gb": 0.0,
            "start_time": None,
            "epoch_start_time": None,
            "gradient_norms": [],
            "parameter_gradients": {}
        }
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Track gradient norms
        total_norm = 0.0
        num_params = 0
        gradients = {}

        # Forward pass
        trajectory_embedding = inputs.pop("trajectory_embedding", None)
        retrieval_query = inputs.pop("retrieval_query", None)

        if trajectory_embedding is not None:
            trajectory_embedding = trajectory_embedding.to(
                dtype=torch.bfloat16,
                device=next(model.parameters()).device
            )
            seen = set()
            if hasattr(model, "router_manager"):
                for router in model.router_manager.token_routers:
                    if getattr(router, "use_trajectory", False) and router.traj_projector is not None:
                        pid = id(router.traj_projector)
                        if pid not in seen:
                            router.traj_projector.set_trajectory_embedding(trajectory_embedding)
                            seen.add(pid)

        if retrieval_query is not None and hasattr(model, "router_manager"):
            retrieval_query = retrieval_query.to(
                dtype=torch.bfloat16,
                device=next(model.parameters()).device
            )
            if hasattr(model.router_manager, "set_retrieval_query"):
                model.router_manager.set_retrieval_query(retrieval_query)

        outputs = model(**inputs)
        loss = outputs.loss

        attention_mask = inputs.get("attention_mask", None)
        if hasattr(model, "router_manager") and hasattr(model.router_manager, "get_auxiliary_loss"):
            loss = model.router_manager.get_auxiliary_loss(loss, attention_mask)

        # Track gradients
        if loss is not None:
            loss.backward()

            # Calculate gradient norms
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    num_params += 1
                    gradients[name] = {
                        "norm": param_norm.item(),
                        "mean": param.grad.data.mean().item(),
                        "std": param.grad.data.std().item(),
                        "min": param.grad.data.min().item(),
                        "max": param.grad.data.max().item()
                    }

            total_norm = total_norm ** (1. / 2) if num_params > 0 else 0.0

        # clear only routing_weight, not traj_embedding
        if hasattr(model, "router_manager"):
            model.router_manager.clear()

        # Store gradient statistics
        self.training_metrics["gradient_norms"].append(total_norm)
        self.training_metrics["parameter_gradients"] = gradients

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        trajectory_embedding = inputs.pop("trajectory_embedding", None)
        retrieval_query = inputs.pop("retrieval_query", None)

        if trajectory_embedding is not None:
            trajectory_embedding = trajectory_embedding.to(
                dtype=torch.bfloat16,
                device=next(model.parameters()).device
            )
            seen = set()
            if hasattr(model, "router_manager"):
                for router in model.router_manager.token_routers:
                    if getattr(router, "use_trajectory", False) and router.traj_projector is not None:
                        pid = id(router.traj_projector)
                        if pid not in seen:
                            router.traj_projector.set_trajectory_embedding(trajectory_embedding)
                            seen.add(pid)

            # ── 只设置到 traj_projector 上，所有router2共享 ──────
            # 梯度检查点重新forward时，这里的值仍然存在
            # if hasattr(model, 'traj_projector') and model.traj_projector is not None:
            #     model.traj_projector.set_trajectory_embedding(trajectory_embedding)

        if retrieval_query is not None and hasattr(model, "router_manager"):
            retrieval_query = retrieval_query.to(
                dtype=torch.bfloat16,
                device=next(model.parameters()).device
            )
            if hasattr(model.router_manager, "set_retrieval_query"):
                model.router_manager.set_retrieval_query(retrieval_query)

        outputs = model(**inputs)
        loss = outputs.loss

        attention_mask = inputs.get("attention_mask", None)
        if hasattr(model, "router_manager") and hasattr(model.router_manager, "get_auxiliary_loss"):
            loss = model.router_manager.get_auxiliary_loss(loss, attention_mask)

        # clear 只清 routing_weight，不清 traj_embedding
        if hasattr(model, "router_manager"):
            model.router_manager.clear()
        # # traj_embedding 在 compute_loss 结束后清除
        # if hasattr(model, 'traj_projector') and model.traj_projector is not None:
        #     model.traj_projector.clear()

        return (loss, outputs) if return_outputs else loss

    def train(self, model=None, resume_from_checkpoint=None, **kwargs):
        import time
        from transformers.trainer_utils import has_length
        from torch.utils.data.dataloader import DataLoader

        self.training_metrics["start_time"] = time.time()
        self.model = self.model if model is None else model
        self.total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps

        # Calculate total training time
        num_epochs = int(self.args.num_train_epochs)
        training_epochs = [i for i in range(num_epochs)]
        steps_in_epoch = len(self.get_train_dataloader()) // self.args.gradient_accumulation_steps

        for epoch, _ in enumerate(training_epochs):
            self.training_metrics["epoch_start_time"] = time.time()
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

            # Log epoch start
            epoch_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{num_epochs} - Start Time: {epoch_start}")
            print(f"{'='*60}")

            super().train(resume_from_checkpoint=resume_from_checkpoint, **kwargs)

            # Calculate epoch time and memory
            epoch_time = time.time() - self.training_metrics["epoch_start_time"]
            self.training_metrics["epoch_times"].append(epoch_time)
            self.training_metrics["total_training_time"] += epoch_time

            # Record peak memory
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
                self.training_metrics["peak_memory_gb"] = max(self.training_metrics["peak_memory_gb"], peak_memory)
                print(f"Peak Memory This Epoch: {peak_memory:.2f} GB")

            # Log epoch summary
            print(f"Epoch {epoch + 1} completed in {epoch_time:.2f} seconds")

        # Print final summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print(f"Total training time: {self.training_metrics['total_training_time']:.2f} seconds ({self.training_metrics['total_training_time']/3600:.2f} hours)")
        print(f"Average epoch time: {sum(self.training_metrics['epoch_times'])/len(self.training_metrics['epoch_times']):.2f} seconds")
        print(f"Peak GPU memory: {self.training_metrics['peak_memory_gb']:.2f} GB")
        self._save_training_metrics()

    def _save_training_metrics(self):
        """Save training metrics to file"""
        import json
        import os
        from datetime import datetime

        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Calculate theoretical FLOPs for MoE model
        total_params, active_params, trainable_params = self.get_model_parameters()

        # Calculate layer-wise parameter distribution
        layer_params = self._calculate_layer_wise_parameters()

        # Rough FLOPs estimation: 2 * params * sequence_length * batch_size
        seq_len = getattr(self.args, 'model_max_length', 4096)
        batch_size = getattr(self.args, 'per_device_train_batch_size', 1)
        gradient_accumulation_steps = getattr(self.args, 'gradient_accumulation_steps', 1)
        effective_batch_size = batch_size * gradient_accumulation_steps

        # Rough FLOPs estimation (multiply-adds count as 2 operations)
        model_flops = 2 * total_params * seq_len * effective_batch_size

        metrics_path = os.path.join(output_dir, "training_metrics.json")
        metrics_data = {
            "dataset_name": getattr(self.args, 'dataset_name', 'unknown'),
            "total_training_time_seconds": self.training_metrics["total_training_time"],
            "total_training_time_hours": self.training_metrics["total_training_time"] / 3600,
            "epoch_times_seconds": self.training_metrics["epoch_times"],
            "average_epoch_time_seconds": sum(self.training_metrics["epoch_times"]) / len(self.training_metrics["epoch_times"]),
            "peak_memory_gb": self.training_metrics["peak_memory_gb"],
            "num_epochs": len(self.training_metrics["epoch_times"]),
            "model_parameters": {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "activated_parameters": active_params,
                "total_parameters_billions": total_params / 1e9,
                "trainable_parameters_billions": trainable_params / 1e9,
                "activated_parameters_billions": active_params / 1e9
            },
            "layer_wise_parameters": layer_params,
            "theoretical_flops": model_flops,
            "flops_per_second": model_flops / self.training_metrics["total_training_time"] if self.training_metrics["total_training_time"] > 0 else 0,
            "model_complexity_analysis": {
                "sequence_length": seq_len,
                "effective_batch_size": effective_batch_size,
                "parameters_per_token": total_params / seq_len,
                "flops_per_second_per_gpu": model_flops / self.training_metrics["total_training_time"] if self.training_metrics["total_training_time"] > 0 else 0
            },
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"Training metrics saved to: {metrics_path}")

        # Save gradient statistics
        if self.training_metrics["gradient_norms"]:
            gradient_path = os.path.join(output_dir, "gradient_statistics.json")
            gradient_data = {
                "average_gradient_norm": sum(self.training_metrics["gradient_norms"]) / len(self.training_metrics["gradient_norms"]),
                "max_gradient_norm": max(self.training_metrics["gradient_norms"]),
                "min_gradient_norm": min(self.training_metrics["gradient_norms"]),
                "gradient_norms_history": self.training_metrics["gradient_norms"],
                "parameter_gradients": self.training_metrics["parameter_gradients"],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Only keep gradients for analysis (reduce memory)
            sampled_gradients = {}
            for param_name, grad_info in self.training_metrics["parameter_gradients"].items():
                if "lora" in param_name.lower() or "router" in param_name.lower():
                    sampled_gradients[param_name] = grad_info

            gradient_data["parameter_gradients"] = sampled_gradients

            with open(gradient_path, 'w') as f:
                json.dump(gradient_data, f, indent=2)
            print(f"Gradient statistics saved to: {gradient_path}")

    def _calculate_layer_wise_parameters(self):
        """Calculate parameters per layer"""
        layer_params = []

        # Get transformer layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            layers = self.model.model.layers
            for i, layer in enumerate(layers):
                layer_param_count = sum(p.numel() for p in layer.parameters() if p.requires_grad)
                layer_params.append({
                    "layer_id": i,
                    "parameters": layer_param_count,
                    "parameters_millions": layer_param_count / 1e6
                })

        # Add other components
        if hasattr(self.model, 'model'):
            embed_params = sum(p.numel() for p in self.model.model.embed_tokens.parameters() if p.requires_grad)
            lm_head_params = sum(p.numel() for p in self.model.model.lm_head.parameters() if p.requires_grad)

            layer_params.append({
                "layer_id": "embed_tokens",
                "parameters": embed_params,
                "parameters_millions": embed_params / 1e6
            })

            layer_params.append({
                "layer_id": "lm_head",
                "parameters": lm_head_params,
                "parameters_millions": lm_head_params / 1e6
            })

        return layer_params

    def get_model_parameters(self):
        """Calculate model parameters with detailed breakdown"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        # Detailed parameter breakdown
        param_breakdown = {
            "base_model": 0,
            "lora_adapters": 0,
            "router_components": 0,
            "expert_networks": 0,
            "projectors": 0,
            "shared_experts": 0
        }

        # Analyze each parameter
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "lora_" in name.lower():
                    param_breakdown["lora_adapters"] += param.numel()
                elif any(x in name.lower() for x in ["router", "gate", "moe", "router_manager"]):
                    if "expert" in name.lower():
                        param_breakdown["expert_networks"] += param.numel()
                    elif "projector" in name.lower():
                        param_breakdown["projectors"] += param.numel()
                    elif "shared" in name.lower():
                        param_breakdown["shared_experts"] += param.numel()
                    else:
                        param_breakdown["router_components"] += param.numel()
                else:
                    param_breakdown["base_model"] += param.numel()

        # For MoE models, analyze expert distribution in detail
        moe_analysis = {}
        if hasattr(self.model, 'router_manager') and hasattr(self.model.router_manager, 'token_routers'):
            for i, router in enumerate(self.model.router_manager.token_routers):
                if hasattr(router, 'experts'):
                    expert_count = len(router.experts)
                    moe_analysis[f'router_{i}'] = {
                        "num_experts": expert_count,
                        "router_params": sum(p.numel() for n, p in router.named_parameters() if p.requires_grad),
                        "expert_params": []
                    }

                    # Analyze each expert
                    for j, expert in enumerate(router.experts):
                        expert_param_count = sum(p.numel() for n, p in expert.named_parameters() if p.requires_grad)
                        moe_analysis[f'router_{i}']['expert_params'].append({
                            "expert_id": j,
                            "params": expert_param_count
                        })

            # Analyze projectors and shared experts
            if hasattr(self.model, 'router_manager'):
                projectors = []
                shared_experts = []

                for router in self.model.router_manager.token_routers:
                    if hasattr(router, 'traj_projector') and router.traj_projector is not None:
                        projector_params = sum(p.numel() for p in router.traj_projector.parameters() if p.requires_grad)
                        projectors.append({
                            "type": "trajectory_projector",
                            "params": projector_params
                        })

                    if hasattr(router, 'shared_expert') and router.shared_expert is not None:
                        shared_params = sum(p.numel() for p in router.shared_expert.parameters() if p.requires_grad)
                        shared_experts.append({
                            "router_id": i,
                            "params": shared_params
                        })

                moe_analysis["projectors"] = projectors
                moe_analysis["shared_experts"] = shared_experts

        # Save detailed analysis
        import json
        from datetime import datetime

        detailed_analysis = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "frozen_parameters": frozen_params,
            "parameter_breakdown": param_breakdown,
            "parameter_percentages": {
                "base_model": param_breakdown["base_model"] / total_params * 100,
                "lora_adapters": param_breakdown["lora_adapters"] / total_params * 100,
                "router_components": param_breakdown["router_components"] / total_params * 100,
                "expert_networks": param_breakdown["expert_networks"] / total_params * 100,
                "projectors": param_breakdown["projectors"] / total_params * 100,
                "shared_experts": param_breakdown["shared_experts"] / total_params * 100
            },
            "moe_analysis": moe_analysis,
            "model_efficiency": {
                "trainable_ratio": trainable_params / total_params,
                "router_expansion_ratio": sum(param_breakdown[x] for x in ["router_components", "expert_networks", "projectors", "shared_experts"]) / param_breakdown["base_model"]
            }
        }

        # Save detailed analysis
        output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        analysis_path = os.path.join(output_dir, "detailed_parameter_analysis.json")
        with open(analysis_path, 'w') as f:
            json.dump(detailed_analysis, f, indent=2)
        print(f"Detailed parameter analysis saved to: {analysis_path}")

        # Print summary
        print(f"\n{'='*60}")
        print("DETAILED PARAMETER ANALYSIS")
        print("="*60)
        print(f"Base Model: {param_breakdown['base_model']:,} ({param_breakdown['base_model']/total_params*100:.1f}%)")
        print(f"LoRA Adapters: {param_breakdown['lora_adapters']:,} ({param_breakdown['lora_adapters']/total_params*100:.1f}%)")
        print(f"Router Components: {param_breakdown['router_components']:,} ({param_breakdown['router_components']/total_params*100:.1f}%)")
        print(f"Expert Networks: {param_breakdown['expert_networks']:,} ({param_breakdown['expert_networks']/total_params*100:.1f}%)")
        print(f"Projectors: {param_breakdown['projectors']:,} ({param_breakdown['projectors']/total_params*100:.1f}%)")
        print(f"Shared Experts: {param_breakdown['shared_experts']:,} ({param_breakdown['shared_experts']/total_params*100:.1f}%)")
        print(f"\nTrainable Ratio: {trainable_params/total_params*100:.1f}%")
        print(f"Router Expansion Ratio: {sum(param_breakdown[x] for x in ['router_components', 'expert_networks', 'projectors', 'shared_experts']) / param_breakdown['base_model']:.2f}x")
        print("="*60)

        return total_params, trainable_params, trainable_params

    # def training_step(self, model, inputs):
    #     model.train()
    #     if isinstance(model, torch.nn.DataParallel):
    #         model = model.module
    #     inputs = self._prepare_inputs(inputs)

    #     if self.args.fp16 and self.use_amp:
    #         with self.autocast_smart_context_manager():
    #             loss = self.compute_loss(model, inputs)
    #     else:
    #         loss = self.compute_loss(model, inputs)

    #     if self.args.gradient_accumulation_steps > 1:
    #         loss = loss / self.args.gradient_accumulation_steps

    #     self.accelerator.backward(loss)

    #     if hasattr(model, "router_manager") and hasattr(model.router_manager, "clear"):
    #         model.router_manager.clear()

    #     return loss.detach()

def visualize_all_router_stats(model, save_path='router_analysis'):
    """
    训练结束后生成完整的可视化报告
    """
    import os
    os.makedirs(save_path, exist_ok=True)
    
    # 收集所有统计数据
    token_router_stats = []
    gate_router_stats = []
    
    for router in model.router_manager.token_routers:
        token_router_stats.append(router.get_stats())
    
    for router in model.router_manager.routers:
        gate_router_stats.append(router.get_stats())
    
    # ========== 1. 专家使用频次热力图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 提取数据
    layers = sorted(list(set([s['layer_id'] for s in token_router_stats])))
    num_layers = len(layers)
    
    freq_g1 = np.zeros((num_layers, 4))
    freq_g2 = np.zeros((num_layers, 4))
    
    for stat in token_router_stats:
        layer_idx = layers.index(stat['layer_id'])
        if '_G1' in stat['tag']:
            freq_g1[layer_idx] = stat['freq_ratio']
        elif '_G2' in stat['tag']:
            freq_g2[layer_idx] = stat['freq_ratio']
    
    # 绘制G1热力图
    sns.heatmap(freq_g1, annot=True, fmt='.2f', cmap='YlOrRd', 
                xticklabels=[f'E{i}' for i in range(4)],
                yticklabels=[f'L{i}' for i in layers],
                vmin=0, vmax=0.5, ax=axes[0], cbar_kws={'label': 'Selection Frequency'})
    axes[0].set_title('Group 1 - Expert Selection Frequency', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Expert ID')
    axes[0].set_ylabel('Layer ID')
    
    # 绘制G2热力图
    sns.heatmap(freq_g2, annot=True, fmt='.2f', cmap='YlGnBu',
                xticklabels=[f'E{i}' for i in range(4)],
                yticklabels=[f'L{i}' for i in layers],
                vmin=0, vmax=0.5, ax=axes[1], cbar_kws={'label': 'Selection Frequency'})
    axes[1].set_title('Group 2 - Expert Selection Frequency', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Expert ID')
    axes[1].set_ylabel('Layer ID')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/expert_selection_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}/expert_selection_heatmap.png")
    plt.close()
    
    # ========== 2. 专家平均权重热力图 ==========
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    weight_g1 = np.zeros((num_layers, 4))
    weight_g2 = np.zeros((num_layers, 4))
    
    for stat in token_router_stats:
        layer_idx = layers.index(stat['layer_id'])
        if '_G1' in stat['tag']:
            weight_g1[layer_idx] = stat['avg_weight']
        elif '_G2' in stat['tag']:
            weight_g2[layer_idx] = stat['avg_weight']
    
    sns.heatmap(weight_g1, annot=True, fmt='.3f', cmap='Reds',
                xticklabels=[f'E{i}' for i in range(4)],
                yticklabels=[f'L{i}' for i in layers],
                ax=axes[0], cbar_kws={'label': 'Avg Weight'})
    axes[0].set_title('Group 1 - Expert Average Weight', fontsize=14, fontweight='bold')
    
    sns.heatmap(weight_g2, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=[f'E{i}' for i in range(4)],
                yticklabels=[f'L{i}' for i in layers],
                ax=axes[1], cbar_kws={'label': 'Avg Weight'})
    axes[1].set_title('Group 2 - Expert Average Weight', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/expert_weight_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}/expert_weight_heatmap.png")
    plt.close()
    
    # ========== 3. 组间Gate使用对比 ==========
    gate_weights = np.zeros((num_layers, 2))
    for stat in gate_router_stats:
        layer_idx = layers.index(stat['layer_id'])
        gate_weights[layer_idx] = stat['group_weights']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(num_layers)
    width = 0.35
    
    ax.bar(x - width/2, gate_weights[:, 0], width, label='Group 1', color='coral')
    ax.bar(x + width/2, gate_weights[:, 1], width, label='Group 2', color='skyblue')
    
    ax.set_xlabel('Layer ID', fontsize=12)
    ax.set_ylabel('Average Gate Weight', fontsize=12)
    ax.set_title('Group Gate Usage Across Layers', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i}' for i in layers])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/group_gate_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {save_path}/group_gate_comparison.png")
    plt.close()
    
    # ========== 4. 异常检测报告 ==========
    with open(f'{save_path}/analysis_report.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("ROUTER USAGE ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # 检测专家坍缩
        f.write("🔍 Expert Collapse Detection:\n")
        f.write("-" * 80 + "\n")
        for stat in token_router_stats:
            min_freq = np.min(stat['freq_ratio'])
            max_freq = np.max(stat['freq_ratio'])
            if max_freq > 0.7:
                f.write(f"⚠️  {stat['tag']}: Expert {np.argmax(stat['freq_ratio'])} dominates ({max_freq:.1%})\n")
            if min_freq < 0.05:
                f.write(f"⚠️  {stat['tag']}: Expert {np.argmin(stat['freq_ratio'])} underused ({min_freq:.1%})\n")
        f.write("\n")
        
        # 检测组间不平衡
        f.write("🔍 Group Imbalance Detection:\n")
        f.write("-" * 80 + "\n")
        for stat in gate_router_stats:
            g1, g2 = stat['group_weights']
            if abs(g1 - g2) > 0.3:
                f.write(f"⚠️  {stat['tag']}: Imbalanced [G1:{g1:.2f}, G2:{g2:.2f}]\n")
        f.write("\n")
        
        # 详细统计
        f.write("📊 Detailed Statistics:\n")
        f.write("-" * 80 + "\n")
        for stat in token_router_stats:
            f.write(f"\n{stat['tag']}:\n")
            f.write(f"  Selection Freq: {[f'{x:.3f}' for x in stat['freq_ratio']]}\n")
            f.write(f"  Avg Weight:     {[f'{x:.3f}' for x in stat['avg_weight']]}\n")
            f.write(f"  Total Selects:  {stat['select_count'].tolist()}\n")
    
    print(f"✅ Saved: {save_path}/analysis_report.txt")
    print("\n" + "=" * 80)
    print("📊 Visualization Complete! Check the following files:")
    print(f"  - {save_path}/expert_selection_heatmap.png")
    print(f"  - {save_path}/expert_weight_heatmap.png")
    print(f"  - {save_path}/group_gate_comparison.png")
    print(f"  - {save_path}/analysis_report.txt")
    print("=" * 80 + "\n")
    
from transformers import TrainerCallback
import os
import torch
import json

class LoRACheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        model = kwargs["model"]
        optimizer = kwargs["optimizer"]

        save_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        os.makedirs(save_dir, exist_ok=True)

        # 1️⃣ 只保存可训练参数（LoRA）
        trainable_params = {
            k: v.detach().cpu()
            for k, v in model.named_parameters()
            if v.requires_grad
        }

        torch.save(
            {
                "model": trainable_params,
                "optimizer": optimizer.state_dict(),
                "step": state.global_step,
                "peft_config": model.peft_config.export(),
            },
            os.path.join(save_dir, "adapter_ckpt.pt"),
        )

        return control

    
def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ConfigArguments))
    model_args, data_args, training_args, config_args = parser.parse_args_into_dataclasses()
    if config_args.router2_input_source not in {"trajectory", "hidden_mean", "hidden_token"}:
        raise ValueError(
            "--router2_input_source must be 'trajectory', 'hidden_mean', or 'hidden_token', "
            f"got {config_args.router2_input_source!r}"
        )
    if config_args.router1_input_source not in {"retquery", "hidden_token"}:
        raise ValueError(
            "--router1_input_source must be 'retquery' or 'hidden_token', "
            f"got {config_args.router1_input_source!r}"
        )
    use_retquery_for_router1 = (
        config_args.use_retrieval_query_routing
        and config_args.router1_input_source == "retquery"
    )

    # NOTE: May expand supported model types in the future
    # if model_args.model_type == "gpt-neox":
    #     replace_gpt_neox_attn(training_args.use_flash_attn, training_args.use_full_attn)
    # else:
    #     replace_llama_attn(training_args.use_flash_attn, training_args.use_full_attn)

    # Set RoPE scaling factor
    config = load_auto_config_compatible(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir
    )

    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        # ),
    )

    for param in model.parameters():
        param.requires_grad = False  # freeze the model - train adapters later
        if param.ndim == 1:
            # cast the small parameters (e.g. layernorm) to fp32 for stability
            param.data = param.data.to(torch.float32)
    # [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

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

    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # ===== 使用带轨迹向量的数据集 =====
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        trajectory_embedding_path=(
            config_args.trajectory_embedding_path
            if config_args.router2_input_source == "trajectory"
            else None
        ),
        retrieval_query_path=(
            config_args.retrieval_query_path if use_retquery_for_router1 else None
        ),
    )
    inferred_ret_query_dim = None
    if use_retquery_for_router1:
        train_ds = data_module["train_dataset"]
        if getattr(train_ds, "retrieval_queries", None) is not None:
            inferred_ret_query_dim = int(train_ds.retrieval_queries.shape[1])
    dataset_name = infer_dataset_name(data_args.dataset, data_args.dataset_name)
    poi_num = {'nyc': 4981, 'tky': 7833, 'ca': 9690}[dataset_name.strip().lower()]
    print(f"Using dataset_name={dataset_name}, poi_num={poi_num}")
    peft_config = UmRaConfig(
        target_modules=config_args.target_modules,
        target_modules_lora=config_args.target_modules_lora,
        dropout=config_args.dropout,
        # poi_num=poi_num,
        # routing strategy
        top_k_routing_strategy=config_args.top_k_routing_strategy,
        top_k=config_args.top_k,
        trajectory_top_k_routing_strategy=config_args.trajectory_top_k_routing_strategy,
        trajectory_top_k=config_args.trajectory_top_k,
        # router sharing
        use_task_router=config_args.use_task_router,
        task_router_only=config_args.task_router_only,
        share_router_for_qkv=config_args.share_router_for_qkv,
        share_router_for_w_i=config_args.share_router_for_w_i,
        # router
        num_router_mlp_layers=config_args.num_router_mlp_layers,
        router_hidden_dim=config_args.router_hidden_dim,
        epsilon_alpha=config_args.epsilon_alpha,
        alpha_shift=config_args.alpha_shift,
        alpha_up_bound=config_args.alpha_up_bound,
        alpha_low_bound=config_args.alpha_low_bound,
        # loss
        use_load_balancing_loss=config_args.use_load_balancing_loss,
        use_div_loss=config_args.use_div_loss,
        gamma_div_certain_t=config_args.gamma_div_certain_t,
        gamma_div_balance_t=config_args.gamma_div_balance_t,
        gamma_div_certain_s=config_args.gamma_div_certain_s,
        gamma_div_balance_s=config_args.gamma_div_balance_s,
        lambda_lm=config_args.lambda_lm,
        lambda_auxiliary=config_args.lambda_auxiliary,
        # experts
        num_experts=config_args.num_experts,
        use_hydra_lora=config_args.use_hydra_lora,
        router1_use_shared_expert=config_args.router1_use_shared_expert,
        router1_shared_expert_weight=config_args.router1_shared_expert_weight,
        disable_router1=config_args.disable_router1,
        disable_router2=config_args.disable_router2,
        lora_r=config_args.lora_r,
        lora_alpha=config_args.lora_alpha,
        use_trajectory_routing=config_args.use_trajectory_routing,
        router2_input_source=config_args.router2_input_source,
        # trajectory_dim=model.config.hidden_size,  # 用同一个LLM编码，维度一致
        trajectory_fusion_mode=config_args.trajectory_fusion_mode,
        share_traj_projector=config_args.share_traj_projector,
        share_ret_query_projector=config_args.share_ret_query_projector,
        use_retrieval_query_routing=use_retquery_for_router1,
        router1_input_source=config_args.router1_input_source,
        retrieval_query_dim=(
            int(config_args.retrieval_query_dim)
            if int(config_args.retrieval_query_dim) > 0
            else (inferred_ret_query_dim if inferred_ret_query_dim is not None else model.config.hidden_size)
        ),
        )
    peft_config.torch_dtype = torch.bfloat16
    peft_config.padding_side = tokenizer.padding_side
    
    router_module = importlib.import_module(config_args.router_model_file)
    get_peft_model_fn = getattr(router_module, "get_peft_model")
    model = get_peft_model_fn(model, peft_config)
    class CastOutputToFloat(nn.Sequential):
        def forward(self, x):
            return super().forward(x).to(torch.float32)

    model.lm_head = CastOutputToFloat(model.lm_head)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    model.config.use_cache = False         # required for gradient checkpointing
    model.enable_input_require_grads()     # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing

    # trainer = UmraTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
     # ===== 使用UmraTrainer =====
    trainer = UmraTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        callbacks=[LoRACheckpointCallback()]
    )
    # 不用额外的loss

    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module, callbacks=[LoRACheckpointCallback()])
    trainer.train(resume_from_checkpoint=False)

    # Calculate and save model parameters
    total_params, active_params, trainable_params = trainer.get_model_parameters()

    # Calculate layer-wise distribution
    layer_params = trainer._calculate_layer_wise_parameters()
    total_layer_params = sum(p["parameters"] for p in layer_params if isinstance(p["layer_id"], int))
    router_params = total_params - total_layer_params

    print(f"\n{'='*60}")
    print("MODEL PARAMETERS SUMMARY")
    print("="*60)
    print(f"Total Parameters: {total_params:,} ({total_params/1e9:.2f} B)")
    print(f"Trainable Parameters: {trainable_params:,} ({trainable_params/1e9:.2f} B)")
    print(f"Activated Parameters: {active_params:,} ({active_params/1e9:.2f} B)")
    print(f"Frozen Parameters: {total_params - trainable_params:,} ({(total_params - trainable_params)/1e9:.2f} B)")

    print(f"\n{'='*60}")
    print("PARAMETER DISTRIBUTION")
    print("="*60)
    print(f"Transformer Layers: {total_layer_params:,} ({total_layer_params/total_params*100:.1f}%)")
    print(f"Router Components: {router_params:,} ({router_params/total_params*100:.1f}%)")

    if hasattr(model, 'router_manager') and hasattr(model.router_manager, 'token_routers'):
        print(f"\nNote: MoE uses softmax-weighted average, ALL experts are activated")
        print(f"Router includes experts, projectors, gates, and shared components")

    print(f"\n{'='*60}")
    print("TRAINING EFFICIENCY")
    print("="*60)
    print(f"Trainable Ratio: {trainable_params/total_params*100:.1f}%")
    print(f"Parameters per Million Tokens: {total_params/1e6:.1f}M")
    print(f"Estimated FLOPs per Token: ~{2 * total_params:,}")
    print("="*60)

    # Save parameter metrics
    import json
    from datetime import datetime

    params_path = os.path.join(training_args.output_dir, "model_parameters.json")
    params_data = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "activated_parameters": active_params,
        "total_parameters_billions": total_params / 1e9,
        "trainable_parameters_billions": trainable_params / 1e9,
        "activated_parameters_billions": active_params / 1e9,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(params_path, 'w') as f:
        json.dump(params_data, f, indent=2)
    print(f"Model parameters saved to: {params_path}")

    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    # trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)
    # 训练完成后
    # visualize_all_router_stats(model, save_path='./router_analysis')


if __name__ == "__main__":
    train()
