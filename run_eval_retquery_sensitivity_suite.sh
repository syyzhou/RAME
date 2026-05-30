#!/usr/bin/env bash
set -euo pipefail

# Evaluate all retquery + trajectory sensitivity outputs produced by
# run_retquery_sensitivity_suite.sh.

MODEL_PATH="${MODEL_PATH:-./Qwen2.5-3B}"
SEQ_LEN="${SEQ_LEN:-4096}"
DATASET_NAME="${DATASET_NAME:-nyc}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/retquery_sensitivity}"
LOG_DIR="${LOG_DIR:-./logs/retquery_sensitivity_eval}"
TEST_FILE="${TEST_FILE:-./datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt}"
TEST_TRAJ_EMB="${TEST_TRAJ_EMB:-./datasets/NYC/preprocessed/test_embeddings_gsm8k.pt}"
TEST_RET_QUERY="${TEST_RET_QUERY:-./datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_test_queries.pt}"
ROUTER_MODEL_FILE="${ROUTER_MODEL_FILE:-model9_ret_cross_router}"
THRESHOLD_MIB="${THRESHOLD_MIB:-10000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-5}"
EVAL_NUM_BEAMS="${EVAL_NUM_BEAMS:-7}"
EVAL_NUM_RETURN_SEQUENCES="${EVAL_NUM_RETURN_SEQUENCES:-7}"

mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
STATUS_LOG="${LOG_DIR}/eval_status_${RUN_TS}.log"
SUMMARY_FILE="${SUMMARY_FILE:-${LOG_DIR}/eval_summary_${RUN_TS}.tsv}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

find_gpu_over_threshold() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[$(timestamp)] nvidia-smi not found; cannot detect GPU availability." | tee -a "${STATUS_LOG}" >&2
    return 1
  fi

  nvidia-smi --query-gpu=index,memory.total,memory.free,name --format=csv,noheader,nounits \
  | awk -F', ' -v threshold="${THRESHOLD_MIB}" '$3 >= threshold {print $1","$2","$3","$4}' \
  | sort -t',' -k3,3nr \
  | head -n 1
}

wait_for_gpu() {
  echo "[$(timestamp)] Waiting for GPU with free memory >= ${THRESHOLD_MIB} MiB" | tee -a "${STATUS_LOG}" >&2
  while true; do
    local gpu_info
    if ! gpu_info="$(find_gpu_over_threshold 2>/dev/null)"; then
      echo "[$(timestamp)] GPU detection failed; please check NVIDIA driver/CUDA visibility." | tee -a "${STATUS_LOG}" >&2
      return 1
    fi
    if [[ -n "${gpu_info}" ]]; then
      echo "${gpu_info}"
      return 0
    fi
    echo "[$(timestamp)] No suitable GPU; sleeping ${POLL_SECONDS}s" | tee -a "${STATUS_LOG}" >&2
    sleep "${POLL_SECONDS}"
  done
}

TAGS=(
  retquery-traj-E2-softmax
  retquery-traj-E4-softmax
  retquery-traj-E8-softmax
  retquery-traj-E4-top1
  retquery-traj-E4-top2
)

extract_result_value() {
  local log_file="$1"
  local label="$2"
  awk -v label="${label}" '
    index($0, label) {
      value = $0
      sub(".*" label "[[:space:]]*", "", value)
      sub("[[:space:]]*\\(.*", "", value)
      print value
    }
  ' "${log_file}" | tail -n 1
}

extract_result_count() {
  local log_file="$1"
  local label="$2"
  awk -v label="${label}" '
    index($0, label) {
      value = $0
      sub(".*" label "[[:space:]]*", "", value)
      print value
    }
  ' "${log_file}" | tail -n 1
}

printf "tag\tstatus\tacc1\tacc5\tacc10\ttotal\tskipped\tevaluated\tuniq_ge5\tuniq_ge10\ttraj_with\ttraj_fallback\tret_with\tret_fallback\toutput_dir\tlog_file\n" > "${SUMMARY_FILE}"

GPU_INFO="$(wait_for_gpu)"
GPU_ID="$(echo "$GPU_INFO" | cut -d',' -f1)"
GPU_NAME="$(echo "$GPU_INFO" | cut -d',' -f4-)"
echo "[$(timestamp)] Using GPU ${GPU_ID} (${GPU_NAME}) for evaluation" | tee -a "${STATUS_LOG}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

for tag in "${TAGS[@]}"; do
  output_dir="${OUTPUT_ROOT}/two-router-${DATASET_NAME}-${tag}"
  if [[ ! -f "${output_dir}/adapter_model.safetensors" ]]; then
    echo "[$(timestamp)] Skipping ${tag}: missing ${output_dir}/adapter_model.safetensors" | tee -a "${STATUS_LOG}"
    printf "%s\t%s\t\t\t\t\t\t\t\t\t\t\t\t\t%s\t\n" \
      "${tag}" "missing_adapter" "${output_dir}" >> "${SUMMARY_FILE}"
    continue
  fi

  echo "[$(timestamp)] Evaluating ${tag}" | tee -a "${STATUS_LOG}"
  log_file="${LOG_DIR}/${tag}_${RUN_TS}.log"
  eval_cmd=(
    python eval_two_router_gsm8k.py
    --model_path "${MODEL_PATH}"
    --output_dir "${output_dir}"
    --test_file "${TEST_FILE}"
    --dataset_name "${DATASET_NAME}"
    --context_size "${SEQ_LEN}"
    --seq_len "${SEQ_LEN}"
    --batc