#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Space-separated list, e.g. DATASETS="CA NYC TKY".
DATASETS="${DATASETS:-CA}"
BASE_ROOT="${BASE_ROOT:-/mnt/nfsData19/ChenWei/zhousiyu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_ROOT}/output/gsm8k_routing_ablation}"
LOG_ROOT="${LOG_ROOT:-./logs/gsm8k_routing_ablation}"
RUN_EVAL="${RUN_EVAL:-True}"

run_one() {
  local dataset="$1"
  local tag="$2"
  shift 2

  echo "============================================================"
  echo "Dataset: ${dataset}"
  echo "Routing ablation: ${tag}"
  echo "Output: ${OUTPUT_ROOT}/two-router-${dataset}-${tag}"
  echo "============================================================"

  env \
    DATASET_NAME="${dataset}" \
    RUN_TAG="${tag}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    LOG_DIR="${LOG_ROOT}/${dataset}" \
    RUN_EVAL="${RUN_EVAL}" \
    "$@" \
    bash run_two_router_gsm8k_retquery.sh
}

for dataset in ${DATASETS}; do
  # Full model: retrieval-query router1 + trajectory router2 + shared router1 expert.
  # run_one "${dataset}" "route-full-E4-softmax" \
  #   NUM_EXPERTS=4 TOP_K_ROUTING_STRATEGY=False TOP_K=2 \
  #   ROUTER1_USE_SHARED_EXPERT=True DISABLE_ROUTER1=False DISABLE_ROUTER2=False \
  #   USE_RETRIEVAL_QUERY_ROUTING=True ROUTER1_INPUT_SOURCE=retquery \
  #   USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=trajectory

  # Remove or alter router groups.
  run_one "${dataset}" "route-no-router1-traj-only" \
    NUM_EXPERTS=4 TOP_K_ROUTING_STRATEGY=False TOP_K=2 \
    ROUTER1_USE_SHARED_EXPERT=False DISABLE_ROUTER1=True DISABLE_ROUTER2=False \
    USE_RETRIEVAL_QUERY_ROUTING=False ROUTER1_INPUT_SOURCE=hidden_token \
    USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=trajectory

  run_one "${dataset}" "route-no-router2-retquery-only" \
    NUM_EXPERTS=4 TOP_K_ROUTING_STRATEGY=False TOP_K=2 \
    ROUTER1_USE_SHARED_EXPERT=True DISABLE_ROUTER1=False DISABLE_ROUTER2=True \
    USE_RETRIEVAL_QUERY_ROUTING=True ROUTER1_INPUT_SOURCE=retquery \
    USE_TRAJECTORY_ROUTING=False ROUTER2_INPUT_SOURCE=trajectory

  run_one "${dataset}" "route-no-router1-shared" \
    NUM_EXPERTS=4 TOP_K_ROUTING_STRATEGY=False TOP_K=2 \
    ROUTER1_USE_SHARED_EXPERT=False DISABLE_ROUTER1=False DISABLE_ROUTER2=False \
    USE_RETRIEVAL_QUERY_ROUTING=True ROUTER1_INPUT_SOURCE=retquery \
    USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=trajectory

  # # Router input-source ablations.
  run_one "${dataset}" "route-router1-hidden-mean" \
    NUM_EXPERTS=4 TOP_K_ROUTING_STRATEGY=False TOP_K=2 \
    ROUTER1_USE_SHARED_EXPERT=True DISABLE_ROUTER1=False DISABLE_ROUTER2=False \
    USE_RETRIEVAL_QUERY_ROUTING=False ROUTER1_INPUT_SOURCE=hidden_mean \
    USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=trajectory

  # run_one "${dataset}" "route-router1-hidden-token" \
  #   NUM_EXPERTS=4 TOP_K_ROUTING_STRATEGY=False TOP_K=2 \
  #   ROUTER1_USE_SHARED_EXPERT=True DISABLE_ROUTER1=False DISABLE_ROUTER2=False \
  #   USE_RETRIEVAL_QUERY_ROUTING=False ROUTER1_INPUT_SOURCE=hidden_token \
  #   USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=trajectory

  # run_one "${dataset}" "route-router2-hidden-mean" \
  #   NUM_EXPERTS=4 TOP_K_ROUTING_STRATEGY=False TOP_K=2 \
  #   ROUTER1_USE_SHARED_EXPERT=True DISABLE_ROUTER1=False DISABLE_ROUTER2=False \
  #   USE_RETRIEVAL_QUERY_ROUTING=True ROUTER1_INPUT_SOURCE=retquery \
  #   USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=hidden_mean

  # run_one "${dataset}" "route-router2-hidden-token" \
  #   NUM_EXPERTS=4 TOP_K_ROUTING_STRATEGY=False TOP_K=2 \
  #   ROUTER1_USE_SHARED_EXPERT=True DISABLE_ROUTER1=False DISABLE_ROUTER2=False \
  #   USE_RETRIEVAL_QUERY_ROUTING=True ROUTER1_INPUT_SOURCE=retquery \
  #   USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=hidden_token

  # # Expert-count sensitivity under dense soft routing.
  # run_one "${dataset}" "route-E2-softmax" \
  #   NUM_EXPERTS=2 TOP_K_ROUTING_STRATEGY=False TOP_K=2 \
  #   ROUTER1_USE_SHARED_EXPERT=True DISABLE_ROUTER1=False DISABLE_ROUTER2=False \
  #   USE_RETRIEVAL_QUERY_ROUTING=True ROUTER1_INPUT_SOURCE=retquery \
  #   USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=trajectory

  # run_one "${dataset}" "route-E8-softmax" \
  #   NUM_EXPERTS=8 TOP_K_ROUTING_STRATEGY=False TOP_K=2 \
  #   ROUTER1_USE_SHARED_EXPERT=True DISABLE_ROUTER1=False DISABLE_ROUTER2=False \
  #   USE_RETRIEVAL_QUERY_ROUTING=True ROUTER1_INPUT_SOURCE=retquery \
  #   USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=trajectory

  # # Sparse routing sensitivity with E fixed to 4.
  # run_one "${dataset}" "route-E4-top1" \
  #   NUM_EXPERTS=4 TOP_K_ROUTING_STRATEGY=True TOP_K=1 \
  #   ROUTER1_USE_SHARED_EXPERT=True DISABLE_ROUTER1=False DISABLE_ROUTER2=False \
  #   USE_RETRIEVAL_QUERY_ROUTING=True ROUTER1_INPUT_SOURCE=retquery \
  #   USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=trajectory

  # run_one "${dataset}" "route-E4-top2" \
  #   NUM_EXPERTS=4 TOP_K_ROUTING_STRATEGY=True TOP_K=2 \
  #   ROUTER1_USE_SHARED_EXPERT=True DISABLE_ROUTER1=False DISABLE_ROUTER2=False \
  #   USE_RETRIEVAL_QUERY_ROUTING=True ROUTER1_INPUT_SOURCE=retquery \
  #   USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=trajectory

  # # Router2-only sparse routing.
  # run_one "${dataset}" "route-router2-top1" \
  #   NUM_EXPERTS=4 TOP_K_ROUTING_STRATEGY=False TOP_K=2 \
  #   TRAJECTORY_TOP_K_ROUTING_STRATEGY=True TRAJECTORY_TOP_K=1 \
  #   ROUTER1_USE_SHARED_EXPERT=True DISABLE_ROUTER1=False DISABLE_ROUTER2=False \
  #   USE_RETRIEVAL_QUERY_ROUTING=True ROUTER1_INPUT_SOURCE=retquery \
  #   USE_TRAJECTORY_ROUTING=True ROUTER2_INPUT_SOURCE=trajectory
done

echo "All routing ablation jobs finished."
