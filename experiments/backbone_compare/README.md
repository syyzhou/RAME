# Backbone comparison experiments

This directory contains scripts for comparing different backbone LLMs on the
combined retrieval-query plus trajectory-routing model.

Default experiment variables:

- Dataset: `nyc`
- Train QA: `datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0.json`
- Test QA: `datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0.txt`
- Train trajectory embeddings: `datasets/NYC/preprocessed/trajectory_embeddings_gsm8k.pt`
- Test trajectory embeddings: `datasets/NYC/preprocessed/test_embeddings_gsm8k.pt`
- Train retrieval-query embeddings: `datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_train_queries.pt`
- Test retrieval-query embeddings: `datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_test_queries.pt`
- Router model: `model9_ret_cross_router`
- Training entry: `supervised-fine-tune-qlora-two-router-model9.py`
- Evaluation entry: `eval_two_router_gsm8k.py`
- Output root: `output/backbone_compare/retquery_traj`

Run one backbone on the default NYC setup:

```bash
bash experiments/backbone_compare/run_one_backbone.sh qwen25_15b
```

Run one backbone on a specific dataset:

```bash
bash experiments/backbone_compare/run_one_backbone.sh qwen25_15b ca
bash experiments/backbone_compare/run_one_backbone.sh qwen25_15b tky
```

Run the recommended comparison set on NYC, CA, and TKY:

```bash
bash experiments/backbone_compare/run_all_backbones.sh
```

Limit the run to selected datasets or backbones:

```bash
bash experiments/backbone_compare/run_all_backbones.sh --datasets ca,tky --backbones qwen25_15b,llama32_3b
```

Supported backbone keys:

- `qwen25_05b`
- `qwen25_15b`
- `qwen25_3b`
- `llama32_1b`
- `llama32_3b`

Override local model paths if your folder names differ:

```bash
QWEN25_15B_PATH=/path/to/Qwen2.5-1.5B \
bash experiments/backbone_compare/run_one_backbone.sh qwen25_15b
```

Useful switches:

```bash
# Print commands, but do not train or evaluate.
DRY_RUN=1 bash experiments/backbone_compare/run_one_backbone.sh qwen25_15b

# Print the full CA/TKY command matrix.
DRY_RUN=1 bash experiments/backbone_compare/run_all_backbones.sh --datasets ca,tky --backbones qwen25_15b

# Evaluate an existing output directory without training.
SKIP_TRAIN=1 bash experiments/backbone_compare/run_one_backbone.sh qwen25_15b

# Override files.
TRAIN_FILE=/path/train.json TEST_FILE=/path/test.txt \
TRAIN_TRAJ_EMB=/path/train_embeddings.pt TEST_TRAJ_EMB=/path/test_embeddings.pt \
TRAIN_RET_QUERY=/path/train_queries.pt TEST_RET_QUERY=/path/test_queries.pt \
bash experiments/backbone_compare/run_one_backbone.sh llama32_3b
```

Dataset defaults:

- `nyc`: uses the reranked GSM8K candidate QA files and
  `gsm8k_final_valartifact_fused_{train,test}_queries.pt`.
- `ca`: uses `datasets/CA/preprocessed/train_qa_pairs_llm4poi_gsm8k.json`,
  `test_qa_pairs_llm4poi_gsm8k.txt`, trajectory embeddings, and
  `gsm8k_final_valartifact_fused_{train,test}_queries.pt`.
- `tky`: uses the same GSM8K naming convention under `datasets/TKY/preprocessed`.

Trajectory embedding defaults:

- `TRAJ_ENCODER_TAG=auto` is the default.
- Qwen backbones use the existing family-shared Qwen trajectory files:
  `trajectory_embeddings_gsm8k.pt` and `test_embeddings_gsm8k.pt`.
- Llama backbones use family-shared Llama 3.2-3B trajectory files:
  `trajectory_embeddings_gsm8k_llama32_3b.pt` and
  `test_embeddings_gsm8k_llama32_3b.pt`.
- Override with `TRAIN_TRAJ_EMB` and `TEST_TRAJ_EMB` for custom files, or set
  `TRAJ_ENCODER_TAG=default` to force the original unsuffixed files.

The training script infers `retrieval_query_dim` from the loaded query tensor
when `RETRIEVAL_QUERY_DIM=0`, and infers `trajectory_dim` from the loaded
trajectory tensor when `TRAJECTORY_DIM=0`.
