#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="${SRC_BASE:-/data/ChenWei/zhousiyu/LLM4POI}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-connect.westc.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-45395}"
REMOTE_BASE="${REMOTE_BASE:-/root/autodl-tmp/zyy/Mrag}"

INCLUDE_OUTPUTS="${INCLUDE_OUTPUTS:-0}"

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_CMD="ssh -p ${REMOTE_PORT}"

FILES=(
  # Backbone comparison scripts for the retquery + trajectory experiment.
  "experiments/backbone_compare/README.md"
  "experiments/backbone_compare/run_one_backbone.sh"
  "experiments/backbone_compare/run_all_backbones.sh"

  # Training/evaluation entry points and router implementation for the combined model.
  "supervised-fine-tune-qlora-two-router-model9.py"
  "eval_two_router_gsm8k.py"
  "model9_ret_cross_router.py"
  "config_model9.py"
  "ds_configs/stage2.json"

  # NYC defaults used by experiments/backbone_compare/run_one_backbone.sh.
  "datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0.json"
  "datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0.txt"
  "datasets/NYC/preprocessed/trajectory_embeddings_gsm8k.pt"
  "datasets/NYC/preprocessed/test_embeddings_gsm8k.pt"
  "datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_train_queries.pt"
  "datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_test_queries.pt"

  # CA GSM8K QA and trajectory embeddings.
  "datasets/CA/preprocessed/train_qa_pairs_llm4poi_gsm8k.json"
  "datasets/CA/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt"
  "datasets/CA/preprocessed/trajectory_embeddings_gsm8k.pt"
  "datasets/CA/preprocessed/test_embeddings_gsm8k.pt"

  # TKY GSM8K QA. Trajectory/query tensors are listed as optional below because
  # they may not have been generated locally yet.
  "datasets/TKY/preprocessed/train_qa_pairs_llm4poi_gsm8k.json"
  "datasets/TKY/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt"
)

OPTIONAL_FILES=(
  # CA/TKY retrieval-query tensors expected by run_one_backbone.sh.
  # "datasets/CA/preprocessed/gsm8k_final_valartifact_fused_train_queries.pt"
  # "datasets/CA/preprocessed/gsm8k_final_valartifact_fused_test_queries.pt"
  # "datasets/TKY/preprocessed/gsm8k_final_valartifact_fused_train_queries.pt"
  # "datasets/TKY/preprocessed/gsm8k_final_valartifact_fused_test_queries.pt"

  # TKY trajectory embeddings expected by run_one_backbone.sh.
  "datasets/TKY/preprocessed/trajectory_embeddings_gsm8k.pt"
  "datasets/TKY/preprocessed/test_embeddings_gsm8k.pt"

  # Proxy-generation helpers and intermediate proxy files.
  "run_ca_tky_gsm8k_traj_proxy_and_embed.sh"
  "build_gsm8k_trajectory_proxy_dataset.py"
  "precompute_trajectory_embeddings.py"
  "datasets/CA/preprocessed/train_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json"
  "datasets/CA/preprocessed/test_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json"
  "datasets/TKY/preprocessed/train_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json"
  "datasets/TKY/preprocessed/test_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json"
)

if [[ "${INCLUDE_OUTPUTS}" == "1" ]]; then
  OPTIONAL_FILES+=(
    "output/backbone_compare"
    "logs/backbone_compare"
  )
fi

echo "[info] remote: ${SSH_TARGET}:${REMOTE_BASE} (port ${REMOTE_PORT})"
echo "[step] ensure remote directory exists"
${SSH_CMD} "${SSH_TARGET}" "mkdir -p '${REMOTE_BASE}'"

echo "[step] check local files"
for file in "${FILES[@]}"; do
  if [[ ! -e "${SRC_BASE}/${file}" ]]; then
    echo "[error] missing local file: ${SRC_BASE}/${file}" >&2
    exit 1
  fi
done

EXISTING_OPTIONAL_FILES=()
for file in "${OPTIONAL_FILES[@]}"; do
  if [[ -e "${SRC_BASE}/${file}" ]]; then
    EXISTING_OPTIONAL_FILES+=("${file}")
  else
    echo "[warn] optional file missing, skip: ${SRC_BASE}/${file}" >&2
  fi
done

echo "[step] rsync files"
(
  cd "${SRC_BASE}"
  rsync -avzP --relative -e "ssh -p ${REMOTE_PORT}" \
    "${FILES[@]}" "${EXISTING_OPTIONAL_FILES[@]}" \
    "${SSH_TARGET}:${REMOTE_BASE}/"
)

echo "[done] transfer finished"
