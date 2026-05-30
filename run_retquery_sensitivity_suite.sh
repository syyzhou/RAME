#!/usr/bin/env bash
set -euo pipefail

# Train the core retquery + trajectory sensitivity configurations.
# Outputs are grouped under one parent directory for later batch evaluation.

OUTPUT_ROOT="${OUTPUT_ROOT:-./output/retquery_sensitivity}"
LOG_ROOT="${LOG_ROOT:-./logs/retquery_sensitivity}"
RUN_EVAL="${RUN_EVAL:-False}"

COMMON_ENV=(
  OUTPUT_ROOT="${OUTPUT_ROOT}"
  LOG_DIR="${LOG_ROOT}"
  RUN_EVAL="${RUN_EVAL}"
  ROUTER2_INPUT_SOURCE="${ROUTER2_INPUT_SOURCE:-trajectory}"
)

run_one() {
  local tag="$1"
  local experts="$2"
  local topk_strategy="$3"
  local topk="$4"

  echo "============================================================"
  echo "Running ${tag}: E=${experts}, top_k_strategy=${topk_strategy}, top_k=${topk}"
  echo "Output: ${OUTPUT_ROOT}/two-router-${DATASET_NAME:-nyc}-${tag}"
  echo "============================================================"

  env "${COMMON_ENV[@]}" \
    RUN_TAG="${tag}" \
    NUM_EXPERTS="${experts}" \
    TOP_K_ROUTING_STRATEGY="${topk_strategy}" \
    TOP_K="${topk}" \
    bash run_two_router_gsm8k_retquery.sh
}

# Expert-number sensitivity under full soft routing.
run_one "retquery-traj-E2-softmax" 2 False 2
run_one "retquery-traj-E4-softmax" 4 False 2
run_one "retquery-traj-E8-softmax" 8 False 2

# Routing sparsity sensitivity with E fixed to 4.
run_one "retquery-traj-E4-top1" 4 True 1
run_one "retquery-traj-E4-top2" 4 True 2

echo "All sensitivity training jobs finished."
echo "Models are under: ${OUTPUT_ROOT}"
