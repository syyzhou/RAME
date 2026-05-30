#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="${SRC_BASE:-/data/ChenWei/zhousiyu/LLM4POI}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_HOST="${REMOTE_HOST:-connect.westb.seetacloud.com}"
REMOTE_PORT="${REMOTE_PORT:-17478}"
REMOTE_BASE="${REMOTE_BASE:-/root/autodl-tmp/zyy/Mrag/datasets/NYC/preprocessed/}"

INCLUDE_OUTPUTS="${INCLUDE_OUTPUTS:-0}"

SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
SSH_CMD="ssh -p ${REMOTE_PORT}"

FILES=(
  "${SRC_BASE}/datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0.json"
  "${SRC_BASE}/datasets/NYC/preprocessed/datasets/NYC/preprocessed/gsm8k_final_fused_test_queries.pt.txt"
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
