#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

BASE_ROOT="${BASE_ROOT:-/mnt/nfsData19/ChenWei/zhousiyu}"

env \
  BASE_ROOT="${BASE_ROOT}" \
  DATASET_NAME="NYC" \
  RUN_TAG="valartifact-full" \
  # OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_ROOT}/output/gsm8k_valartifact_full}" \
  OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_ROOT}/output/reqry-traj}" \
  LOG_DIR="${LOG_DIR:-./logs/gsm8k_valartifact_full}" \
  RUN_EVAL="${RUN_EVAL:-True}" \
  TRAIN_FILE="${TRAIN_FILE:-./datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k.json}" \
  TEST_FILE="${TEST_FILE:-./datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt}" \
  TRAIN_RET_QUERY="${TRAIN_RET_QUERY:-./datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_train_queries.pt}" \
  TEST_RET_QUERY="${TEST_RET_QUERY:-./datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_test_queries.pt}" \
  TRAIN_TRAJ_EMB="${TRAIN_TRAJ_EMB:-./datasets/NYC/preprocessed/trajectory_embeddings_gsm8k.pt}" \
  TEST_TRAJ_EMB="${TEST_TRAJ_EMB:-./datasets/NYC/preprocessed/test_embeddings_gsm8k.pt}" \
  ROUTER_MODEL_FILE="${ROUTER_MODEL_FILE:-model9_ret_cross_router}" \
  NUM_EXPERTS="${NUM_EXPERTS:-4}" \
  TOP_K_ROUTING_STRATEGY="${TOP_K_ROUTING_STRATEGY:-False}" \
  TOP_K="${TOP_K:-2}" \
  TRAJECTORY_TOP_K_ROUTING_STRATEGY="${TRAJECTORY_TOP_K_ROUTING_STRATEGY:-False}" \
  TRAJECTORY_TOP_K="${TRAJECTORY_TOP_K:-2}" \
  ROUTER1_USE_SHARED_EXPERT="${ROUTER1_USE_SHARED_EXPERT:-True}" \
  ROUTER1_SHARED_EXPERT_WEIGHT="${ROUTER1_SHARED_EXPERT_WEIGHT:-1.0}" \
  DISABLE_ROUTER1="${DISABLE_ROUTER1:-False}" \
  DISABLE_ROUTER2="${DISABLE_ROUTER2:-False}" \
  USE_RETRIEVAL_QUERY_ROUTING="${USE_RETRIEVAL_QUERY_ROUTING:-True}" \
  ROUTER1_INPUT_SOURCE="${ROUTER1_INPUT_SOURCE:-retquery}" \
  USE_TRAJECTORY_ROUTING="${USE_TRAJECTORY_ROUTING:-True}" \
  ROUTER2_INPUT_SOURCE="${ROUTER2_INPUT_SOURCE:-trajectory}" \
  bash run_two_router_gsm8k_retquery.sh
