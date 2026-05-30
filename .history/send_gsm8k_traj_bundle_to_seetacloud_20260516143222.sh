#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="${SRC_BASE:-/data/ChenWei/zhousiyu/LLM4POI}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-connect.westd.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-41323}"
REMOTE_BASE="${REMOTE_BASE:-/root/autodl-tmp/zyy/Mrag/datasets/NYC/preprocessed/}"

INCLUDE_OUTPUTS="${INCLUDE_OUTPUTS:-0}"

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_CMD="ssh -p ${REMOTE_PORT}"

FILES=(
  # "${SRC_BASE}/datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json"
  # "${SRC_BASE}/datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json"
  "${SRC_BASE}/precompute_trajectory_embeddings.py"
  "${SRC_BASE}/run_ca_tky_gsm8k_traj_proxy_and_embed.sh"
  "${SRC_BASE}/build_gsm8k_trajectory_proxy_dataset.py"
  # "${SRC_BASE}/datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0_top10.json"
  # "${SRC_BASE}/datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0_top10.txt"
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
