#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="./Qwen2.5-3B"
SEQ_LEN=4096
DATASET_NAME="${DATASET_NAME:-nyc}"
RUN_TAG="${RUN_TAG:-retquery}"

# 默认使用原始 QA；如需换数据可通过环境变量覆盖 TRAIN_FILE/TEST_FILE。
TRAIN_FILE="${TRAIN_FILE:-./datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_valartifact_mix10trans4geo3hist3_rerank_at1gridbest_val.json}"
TEST_FILE="${TEST_FILE:-./datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_valartifact_mix10trans4geo3hist3_rerank_at1gridbest_val.txt}"

OUTPUT_ROOT="${OUTPUT_ROOT:-./output}"
OUTPUT_DIR="${OUTPUT_ROOT}/two-router-${DATASET_NAME}-${RUN_TAG}"
THRESHOLD_MIB="${THRESHOLD_MIB:-10000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-./logs/run_two_router_retquery}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
STATUS_LOG="${LOG_DIR}/run_status_${RUN_TS}.log"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
TF32_ENABLED="${TF32_ENABLED:-False}"
RUN_EVAL="${RUN_EVAL:-True}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-5}"
EVAL_NUM_BEAMS="${EVAL_NUM_BEAMS:-7}"
EVAL_NUM_RETURN_SEQUENCES="${EVAL_NUM_RETURN_SEQUENCES:-7}"

# 路由模型文件（对应你新建的 model9）
ROUTER_MODEL_FILE="${ROUTER_MODEL_FILE:-model9_ret_cross_router}"
RETRIEVAL_QUERY_DIM="${RETRIEVAL_QUERY_DIM:-0}"
ROUTER1_INPUT_SOURCE="${ROUTER1_INPUT_SOURCE:-retquery}"
NUM_EXPERTS="${NUM_EXPERTS:-4}"
TOP_K_ROUTING_STRATEGY="${TOP_K_ROUTING_STRATEGY:-False}"
TOP_K="${TOP_K:-2}"
ROUTER1_USE_SHARED_EXPERT="${ROUTER1_USE_SHARED_EXPERT:-False}"

# 轨迹向量
TRAIN_TRAJ_EMB="${TRAIN_TRAJ_EMB:-./datasets/NYC/preprocessed/trajectory_embeddings_gsm8k.pt}"
TEST_TRAJ_EMB="${TEST_TRAJ_EMB:-./datasets/NYC/preprocessed/test_embeddings_gsm8k.pt}"
ROUTER2_INPUT_SOURCE="${ROUTER2_INPUT_SOURCE:-trajectory}"

# 检索 query 向量（由 export_fused_queries.py 导出）
TRAIN_RET_QUERY="${TRAIN_RET_QUERY:-./datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_train_queries.pt}"
TEST_RET_QUERY="${TEST_RET_QUERY:-./datasets/NYC/preprocessed/gsm8k_final_valartifact_fused_test_queries.pt}"

mkdir -p "${LOG_DIR}"

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

echo "[$(timestamp)] Model path: ${MODEL_PATH}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Train file: ${TRAIN_FILE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Test file: ${TEST_FILE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Output dir: ${OUTPUT_DIR}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Run eval: ${RUN_EVAL}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Router model file: ${ROUTER_MODEL_FILE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Router1 input source: ${ROUTER1_INPUT_SOURCE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Router2 input source: ${ROUTER2_INPUT_SOURCE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Num experts: ${NUM_EXPERTS}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Top-k routing strategy: ${TOP_K_ROUTING_STRATEGY}, top_k=${TOP_K}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Router1 shared expert: ${ROUTER1_USE_SHARED_EXPERT}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Train retrieval query: ${TRAIN_RET_QUERY}" | tee -a "${STATUS_LOG}"
echo "[$(timest