#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="${SRC_BASE:-/data/ChenWei/zhousiyu/LLM4POI}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-connect.westc.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-47675}"
REMOTE_BASE="${REMOTE_BASE:-/root/autodl-tmp/zyy/Mrag/datasets/CA/preprocessed/}"

INCLUDE_OUTPUTS="${INCLUDE_OUTPUTS:-0}"

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_CMD="ssh -p ${REMOTE_PORT}"

FILES=(
  "${SRC_BASE}/datasets/CA/preprocessed/trajectory_embeddings_gsm8k.pt"
  "${SRC_BASE}/datasets/CA/preprocessed/test_embeddings_gsm8k.pt"
  # "${SRC_BASE}/datasets/TKY"
  # "${SRC_BASE}/datasets/CA"
  # "${SRC_BASE}/build_gsm8k_trajectory_proxy_dataset.py"
  "${SRC_BASE}/datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_valartifact_puretop20.json"
  "${SRC_BASE}/datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_valartifact_puretop20.txt"
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

echo "[step] rsync files"
rsync -avzP -e "ssh -p ${REMOTE_PORT}" "${FILES[@]}" "${SSH_TARGET}:${REMOTE_BASE}/"

echo "[done] transfer finished"
