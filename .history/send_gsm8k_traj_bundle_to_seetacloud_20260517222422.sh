#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="${SRC_BASE:-/data/ChenWei/zhousiyu/LLM4POI}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-connect.westb.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-17478}"
REMOTE_BASE="${REMOTE_BASE:-/root/autodl-tmp/zyy/Mrag}"

INCLUDE_OUTPUTS="${INCLUDE_OUTPUTS:-0}"

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_CMD="ssh -p ${REMOTE_PORT}"

FILES=(
#   # Backbone comparison scripts added for the retquery + trajectory experiment.
#   "experiments/backbone_compare/README.md"
#   "experiments/backbone_compare/run_one_backbone.sh"
#   "experiments/backbone_compare/run_all_backbones.sh"

#   # Training/evaluation entry points and router implementation for the combined model.
#   "supervised-fine-tune-qlora-two-router-model9.py"
#   "eval_two_router_gsm8k.py"
#   "model9_ret_cross_router.py"
#   "config_model9.py"
#   "ds_configs/stage2.json"

  # Default NYC GSM8K candidate QA files used by experiments/backbone_compare.
  "datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_valartifact_mix10trans4geo3hist3_rerank_at1gridbest_val.json"
  "datasets/NYC/preprocessed/datasets/test_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_valartifact_mix10trans4geo3hist3_rerank_at1gridbest_val.txt"

#   # Shared trajectory and retrieval-query tensors.
#   "datasets/NYC/preprocessed/trajectory_embeddings_gsm8k.pt"
#   "datasets/NYC/preprocessed/test_embeddings_gsm8k.pt"
#   "datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_train_queries.pt"
#   "datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_test_queries.pt"

#   # Existing pure top-20 files retained from the previous transfer list.
#   "datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_valartifact_puretop20.json"
#   "datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_valartifact_puretop20.txt"
)

# if [[ "${INCLUDE_OUTPUTS}" == "1" ]]; then
#   FILES+=(
#     "${SRC_BASE}/datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json"
#     "${SRC_BASE}/datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json"
#     "${SRC_BASE}/datasets/NYC/preprocessed/trajectory_embeddings_gsm8k.pt"
#     "${SRC_BASE}/datasets/NYC/preprocessed/test_embeddings_gsm8k.pt"
#     "${SRC_BASE}/datasets/nyc/preprocessed/train_test_merged_for_gsm8k_proxy.csv"
#   )
# fi

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

echo "[step] rsync files"
(
  cd "${SRC_BASE}"
  rsync -avzP --relative -e "ssh -p ${REMOTE_PORT}" "${FILES[@]}" "${SSH_TARGET}:${REMOTE_BASE}/"
)

echo "[done] transfer finished"
