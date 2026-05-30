#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Space-separated list, e.g. DATASETS="CA NYC TKY".
DATASETS="${DATASETS:-CA}"
VIEWS="${VIEWS:-full no_sem no_str no_traj sem_only str_only traj_only}"
BASE_ROOT="${BASE_ROOT:-/mnt/nfsData19/ChenWei/zhousiyu}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${BASE_ROOT}/output/gsm8k_view_ablation}"
LOG_ROOT="${LOG_ROOT:-./logs/gsm8k_view_ablation}"
RUN_EVAL="${RUN_EVAL:-True}"
RUN_QUERY_EXPORT="${RUN_QUERY_EXPORT:-True}"

query_dir_for_dataset() {
  local dataset="$1"
  local upper
  upper="$(printf '%s' "${dataset}" | tr '[:lower:]' '[:upper:]')"
  printf './datasets/%s/preprocessed/gsm8k_ablation_fused_queries' "${upper}"
}

run_view_queries() {
  local dataset="$1"
  if [[ "${RUN_QUERY_EXPORT}" =~ ^(True|true|1)$ ]]; then
    echo "============================================================"
    echo "Dataset: ${dataset}"
    echo "Building/exporting multiview fused-query ablations"
    echo "============================================================"
    env DATASET_NAME="${dataset}" LOG_DIR="${LOG_ROOT}/${dataset}/query_export" \
      bash run_gsm8k_multiview_ablation_fused_queries.sh
  fi
}

run_view_train() {
  local dataset="$1"
  local tag="$2"
  local query_dir
  query_dir="$(query_dir_for_dataset "${dataset}")"

  echo "============================================================"
  echo "Dataset: ${dataset}"
  echo "View ablation: ${tag}"
  echo "Train query: ${query_dir}/${tag}_train_fused_queries.pt"
  echo "Test query: ${query_dir}/${tag}_test_fused_queries.pt"
  echo "Output: ${OUTPUT_ROOT}/two-router-${dataset}-view-${tag}"
  echo "============================================================"

  env \
    DATASET_NAME="${dataset}" \
    RUN_TAG="view-${tag}" \
    OUTPUT_ROOT="${OUTPUT_ROOT}" \
    LOG_DIR="${LOG_ROOT}/${dataset}/router_train" \
    RUN_EVAL="${RUN_EVAL}" \
    ROUTER1_USE_SHARED_EXPERT=True \
    NUM_EXPERTS=4 \
    TOP_K_ROUTING_STRATEGY=False \
    TOP_K=2 \
    USE_RETRIEVAL_QUERY_ROUTING=True \
    ROUTER1_INPUT_SOURCE=retquery \
    USE_TRAJECTORY_ROUTING=True \
    ROUTER2_INPUT_SOURCE=trajectory \
    TRAIN_RET_QUERY="${query_dir}/${tag}_train_fused_queries.pt" \
    TEST_RET_QUERY="${query_dir}/${tag}_test_fused_queries.pt" \
    bash run_two_router_gsm8k_retquery.sh
}

for dataset in ${DATASETS}; do
  run_view_queries "${dataset}"
  for view_tag in ${VIEWS}; do
    run_view_train "${dataset}" "${view_tag}"
  done
done

echo "All view ablation jobs finished."
