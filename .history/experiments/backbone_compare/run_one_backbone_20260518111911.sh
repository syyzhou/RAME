#!/usr/bin/env bash
set -euo pipefail

BACKBONE_KEY="${1:-qwen25_15b}"
DATASET_KEY="${2:-${DATASET_NAME:-nyc}}"
DATASET_KEY="$(printf '%s' "${DATASET_KEY}" | tr '[:upper:]' '[:lower:]')"

case "${DATASET_KEY}" in
  nyc)
    DATASET_NAME="nyc"
    DATASET_DIR="NYC"
    DEFAULT_TRAIN_FILE="./datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k.json"
    DEFAULT_TEST_FILE="./datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt"
    DEFAULT_TRAIN_RET_QUERY="./datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_train_queries.pt"
    DEFAULT_TEST_RET_QUERY="./datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_test_queries.pt"
    ;;
  ca)
    DATASET_NAME="ca"
    DATASET_DIR="CA"
    DEFAULT_TRAIN_FILE="./datasets/CA/preprocessed/train_qa_pairs_llm4poi_gsm8k.json"
    DEFAULT_TEST_FILE="./datasets/CA/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt"
    DEFAULT_TRAIN_RET_QUERY="./datasets/CA/preprocessed/gsm8k_final_fused_train_queries.pt"
    DEFAULT_TEST_RET_QUERY="./datasets/CA/preprocessed/gsm8k_final_fused_test_queries.pt"
    ;;
  tky)
    DATASET_NAME="tky"
    DATASET_DIR="TKY"
    DEFAULT_TRAIN_FILE="./datasets/TKY/preprocessed/train_qa_pairs_llm4poi_gsm8k.json"
    DEFAULT_TEST_FILE="./datasets/TKY/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt"
    DEFAULT_TRAIN_RET_QUERY="./datasets/TKY/preprocessed/gsm8k_final_fused_train_queries.pt"
    DEFAULT_TEST_RET_QUERY="./datasets/TKY/preprocessed/gsm8k_final_fused_test_queries.pt"
    ;;
  *)
    echo "Unknown dataset key: ${DATASET_KEY}" >&2
    echo "Supported datasets: nyc ca tky" >&2
    exit 2
    ;;
esac

SEQ_LEN="${SEQ_LEN:-4096}"
EXP_TAG="${EXP_TAG:-retquery_traj}"
TRAIN_FILE="${TRAIN_FILE:-${DEFAULT_TRAIN_FILE}}"
TEST_FILE="${TEST_FILE:-${DEFAULT_TEST_FILE}}"
TRAIN_TRAJ_EMB="${TRAIN_TRAJ_EMB:-./datasets/${DATASET_DIR}/preprocessed/trajectory_embeddings_gsm8k.pt}"
TEST_TRAJ_EMB="${TEST_TRAJ_EMB:-./datasets/${DATASET_DIR}/preprocessed/test_embeddings_gsm8k.pt}"
TRAIN_RET_QUERY="${TRAIN_RET_QUERY:-${DEFAULT_TRAIN_RET_QUERY}}"
TEST_RET_QUERY="${TEST_RET_QUERY:-${DEFAULT_TEST_RET_QUERY}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./output/backbone_compare/${EXP_TAG}/${DATASET_KEY}}"
LOG_ROOT="${LOG_ROOT:-./logs/backbone_compare/${EXP_TAG}/${DATASET_KEY}}"
THRESHOLD_MIB="${THRESHOLD_MIB:-20000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
TF32_ENABLED="${TF32_ENABLED:-False}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
DRY_RUN="${DRY_RUN:-0}"
ROUTER_MODEL_FILE="${ROUTER_MODEL_FILE:-model9_ret_cross_router}"
RETRIEVAL_QUERY_DIM="${RETRIEVAL_QUERY_DIM:-0}"

case "${BACKBONE_KEY}" in
  qwen25_05b)
    MODEL_PATH="${MODEL_PATH:-${QWEN25_05B_PATH:-./Qwen2.5-0.5B}}"
    BACKBONE_NAME="qwen2.5-0.5b"
    ;;
  qwen25_15b)
    MODEL_PATH="${MODEL_PATH:-${QWEN25_15B_PATH:-./Qwen2.5-1.5B}}"
    BACKBONE_NAME="qwen2.5-1.5b"
    ;;
  qwen25_3b)
    MODEL_PATH="${MODEL_PATH:-${QWEN25_3B_PATH:-./Qwen2.5-3B}}"
    BACKBONE_NAME="qwen2.5-3b"
    ;;
  llama32_1b)
    MODEL_PATH="${MODEL_PATH:-${LLAMA32_1B_PATH:-./Llama-3.2-1B}}"
    BACKBONE_NAME="llama3.2-1b"
    ;;
  llama32_3b)
    MODEL_PATH="${MODEL_PATH:-${LLAMA32_3B_PATH:-./Llama-3.2-3B}}"
    BACKBONE_NAME="llama3.2-3b"
    ;;
  *)
    echo "Unknown backbone key: ${BACKBONE_KEY}" >&2
    echo "Supported: qwen25_05b qwen25_15b qwen25_3b llama32_1b llama32_3b" >&2
    exit 2
    ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${BACKBONE_KEY}}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/${BACKBONE_KEY}}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
STATUS_LOG="${LOG_DIR}/run_status_${RUN_TS}.log"
TRAIN_LOG="${LOG_DIR}/train_${RUN_TS}.log"
EVAL_LOG="${LOG_DIR}/eval_${RUN_TS}.log"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

log() {
  echo "[$(timestamp)] $*" | tee -a "${STATUS_LOG}"
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
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
  local stage="$1"
  echo "[$(timestamp)] Waiting for GPU for ${stage} with free memory >= ${THRESHOLD_MIB} MiB" | tee -a "${STATUS_LOG}" >&2
  while true; do
    local gpu_info
    if ! gpu_info="$(find_gpu_over_threshold 2>/dev/null)"; then
      echo "[$(timestamp)] GPU detection failed for ${stage}; please check NVIDIA driver/CUDA visibility." | tee -a "${STATUS_LOG}" >&2
      return 1
    fi
    if [[ -n "${gpu_info}" ]]; then
      echo "${gpu_info}"
      return 0
    fi
    echo "[$(timestamp)] No suitable GPU for ${stage}; sleeping ${POLL_SECONDS}s" | tee -a "${STATUS_LOG}" >&2
    sleep "${POLL_SECONDS}"
  done
}

if [[ ! -e "${MODEL_PATH}" ]]; then
  log "Model path does not exist locally: ${MODEL_PATH}"
  log "If this is a Hugging Face id or cached model path, transformers may still load it; otherwise set MODEL_PATH or the matching *_PATH variable."
fi

log "Backbone key: ${BACKBONE_KEY}"
log "Backbone name: ${BACKBONE_NAME}"
log "Dataset key: ${DATASET_KEY}"
log "Dataset name: ${DATASET_NAME}"
log "Model path: ${MODEL_PATH}"
log "Train file: ${TRAIN_FILE}"
log "Test file: ${TEST_FILE}"
log "Train trajectory embeddings: ${TRAIN_TRAJ_EMB}"
log "Test trajectory embeddings: ${TEST_TRAJ_EMB}"
log "Train retrieval-query embeddings: ${TRAIN_RET_QUERY}"
log "Test retrieval-query embeddings: ${TEST_RET_QUERY}"
log "Router model file: ${ROUTER_MODEL_FILE}"
log "Output dir: ${OUTPUT_DIR}"

TRAIN_CMD=(
  python supervised-fine-tune-qlora-two-router-model9.py
  --model_name_or_path "${MODEL_PATH}"
  --bf16 True
  --output_dir "${OUTPUT_DIR}"
  --use_flash_attn False
  --dataset "${TRAIN_FILE}"
  --dataset_name "${DATASET_NAME}"
  --low_rank_training True
  --num_train_epochs 1
  --per_device_train_batch_size 1
  --per_device_eval_batch_size 2
  --evaluation_strategy no
  --gradient_accumulation_steps 8
  --save_strategy no
  --save_steps 100
  --learning_rate 2e-5
  --weight_decay 0.0
  --warmup_steps 20
  --lr_scheduler_type constant_with_warmup
  --logging_steps 1
  --logging_dir "${LOG_DIR}"
  --deepspeed ds_configs/stage2.json
  --model_max_length "${SEQ_LEN}"
  --tf32 "${TF32_ENABLED}"
  --gradient_checkpointing True
  --router_model_file "${ROUTER_MODEL_FILE}"
  --use_trajectory_routing True
  --trajectory_embedding_path "${TRAIN_TRAJ_EMB}"
  --trajectory_fusion_mode gate
  --share_traj_projector False
  --router1_use_shared_expert True
  --router1_shared_expert_weight 1.0
  --use_retrieval_query_routing True
  --retrieval_query_path "${TRAIN_RET_QUERY}"
  --retrieval_query_dim "${RETRIEVAL_QUERY_DIM}"
  --share_ret_query_projector False
)

EVAL_CMD=(
  python eval_two_router_gsm8k.py
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --test_file "${TEST_FILE}"
  --dataset_name "${DATASET_NAME}"
  --context_size "${SEQ_LEN}"
  --seq_len "${SEQ_LEN}"
  --batch_size 1
  --max_new_tokens 5
  --num_beams 7
  --num_return_sequences 7
  --repetition_penalty 1.176
  --router_model_file "${ROUTER_MODEL_FILE}"
  --trajectory_embedding_path "${TEST_TRAJ_EMB}"
  --use_retrieval_query_routing
  --retrieval_query_path "${TEST_RET_QUERY}"
)

if [[ "${EVAL_DEVICE}" != "auto" ]]; then
  EVAL_CMD+=(--device "${EVAL_DEVICE}")
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY_RUN=1; commands are:"
  printf 'Training command: %q ' "${TRAIN_CMD[@]}" | tee -a "${STATUS_LOG}"
  printf '\n' | tee -a "${STATUS_LOG}"
  printf 'Evaluation command: %q ' "${EVAL_CMD[@]}" | tee -a "${STATUS_LOG}"
  printf '\n' | tee -a "${STATUS_LOG}"
  exit 0
fi

require_file "${TRAIN_FILE}" "training QA file"
require_file "${TEST_FILE}" "test QA file"
require_file "${TRAIN_TRAJ_EMB}" "training trajectory embedding file"
require_file "${TEST_TRAJ_EMB}" "test trajectory embedding file"
require_file "${TRAIN_RET_QUERY}" "training retrieval-query file"
require_file "${TEST_RET_QUERY}" "test retrieval-query file"

if [[ "${TRAIN_RET_QUERY}" == "${TEST_RET_QUERY}" ]]; then
  echo "TRAIN_RET_QUERY and TEST_RET_QUERY must be different files." >&2
  exit 1
fi

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  GPU_INFO="$(wait_for_gpu training)"
  GPU_ID="$(echo "${GPU_INFO}" | cut -d',' -f1)"
  GPU_TOTAL="$(echo "${GPU_INFO}" | cut -d',' -f2)"
  GPU_FREE="$(echo "${GPU_INFO}" | cut -d',' -f3)"
  GPU_NAME="$(echo "${GPU_INFO}" | cut -d',' -f4-)"
  log "Using GPU ${GPU_ID} (${GPU_NAME}) for training (total=${GPU_TOTAL} MiB, free=${GPU_FREE} MiB)"
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"

  log "Starting training"
  "${TRAIN_CMD[@]}" 2>&1 | tee "${TRAIN_LOG}"
  log "Training finished"
else
  log "SKIP_TRAIN=1; training skipped"
fi

if [[ ! -f "${OUTPUT_DIR}/adapter_model.safetensors" ]]; then
  echo "Missing adapter after training: ${OUTPUT_DIR}/adapter_model.safetensors" >&2
  exit 1
fi

GPU_INFO_AFTER="$(wait_for_gpu evaluation)"
GPU_ID_AFTER="$(echo "${GPU_INFO_AFTER}" | cut -d',' -f1)"
GPU_TOTAL_AFTER="$(echo "${GPU_INFO_AFTER}" | cut -d',' -f2)"
GPU_FREE_AFTER="$(echo "${GPU_INFO_AFTER}" | cut -d',' -f3)"
GPU_NAME_AFTER="$(echo "${GPU_INFO_AFTER}" | cut -d',' -f4-)"
log "Using GPU ${GPU_ID_AFTER} (${GPU_NAME_AFTER}) for evaluation (total=${GPU_TOTAL_AFTER} MiB, free=${GPU_FREE_AFTER} MiB)"
export CUDA_VISIBLE_DEVICES="${GPU_ID_AFTER}"

# log "Starting evaluation"
# "${EVAL_CMD[@]}" 2>&1 | tee "${EVAL_LOG}"
# log "Evaluation finished"
