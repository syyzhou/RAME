#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="./Qwen2.5-3B"
SEQ_LEN=4096
DATASET_NAME="${DATASET_NAME:-nyc}"
RUN_TAG="${RUN_TAG:-retquery_gsm8k}"

# 默认使用原始 QA；如需换数据可通过环境变量覆盖 TRAIN_FILE/TEST_FILE。
TRAIN_FILE="${TRAIN_FILE:-./datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k.json}"
TEST_FILE="${TEST_FILE:-./datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt}"

OUTPUT_DIR="./output/two-router-${DATASET_NAME}-${RUN_TAG}"
THRESHOLD_MIB="${THRESHOLD_MIB:-20000}"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_DIR="${LOG_DIR:-./logs/run_two_router_hsm8kretquery}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
STATUS_LOG="${LOG_DIR}/run_status_${RUN_TS}.log"
EVAL_DEVICE="${EVAL_DEVICE:-auto}"
TF32_ENABLED="${TF32_ENABLED:-False}"

# 路由模型文件（对应你新建的 model9）
ROUTER_MODEL_FILE="${ROUTER_MODEL_FILE:-model9_ret_cross_router}"
RETRIEVAL_QUERY_DIM="${RETRIEVAL_QUERY_DIM:-0}"

# 轨迹向量
TRAIN_TRAJ_EMB="${TRAIN_TRAJ_EMB:-./datasets/${DATASET_NAME}/preprocessed/trajectory_embeddings_gsm8k.pt}"
TEST_TRAJ_EMB="${TEST_TRAJ_EMB:-./datasets/${DATASET_NAME}/preprocessed/test_embeddings.pt}"

# 检索 query 向量（由 export_fused_queries.py 导出）
TRAIN_RET_QUERY="${TRAIN_RET_QUERY:-./datasets/${DATASET_NAME}/preprocessed/train_retrieval_queries_cross_attn_dropout05_wd1e2_${DATASET_NAME}_ep60_regv1.pt}"
TEST_RET_QUERY="${TEST_RET_QUERY:-./datasets/${DATASET_NAME}/preprocessed/test_retrieval_queries_cross_attn_dropout05_wd1e2_${DATASET_NAME}_ep60_regv1.pt}"

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
echo "[$(timestamp)] Router model file: ${ROUTER_MODEL_FILE}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Train retrieval query: ${TRAIN_RET_QUERY}" | tee -a "${STATUS_LOG}"
echo "[$(timestamp)] Test retrieval query: ${TEST_RET_QUERY}" | tee -a "${STATUS_LOG}"

if [[ "${TRAIN_RET_QUERY}" == "${TEST_RET_QUERY}" ]]; then
  echo "[$(timestamp)] ERROR: TRAIN_RET_QUERY and TEST_RET_QUERY are identical. Please use different files." | tee -a "${STATUS_LOG}" >&2
  exit 1
fi

GPU_INFO="$(wait_for_gpu training)"
GPU_ID="$(echo "$GPU_INFO" | cut -d',' -f1)"
GPU_TOTAL="$(echo "$GPU_INFO" | cut -d',' -f2)"
GPU_FREE="$(echo "$GPU_INFO" | cut -d',' -f3)"
GPU_NAME="$(echo "$GPU_INFO" | cut -d',' -f4-)"

echo "[$(timestamp)] Using GPU ${GPU_ID} (${GPU_NAME}) for training (total=${GPU_TOTAL} MiB, free=${GPU_FREE} MiB)" | tee -a "${STATUS_LOG}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

TRAIN_CMD=(
  python supervised-fine-tune-qlora-two-router.py
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
  --learning_rate 2e-5
  --weight_decay 0.0
  --warmup_steps 20
  --lr_scheduler_type constant_with_warmup
  --logging_steps 1
  --logging_dir ./logs
  --deepspeed ds_configs/stage2.json
  --model_max_length "${SEQ_LEN}"
  --tf32 "${TF32_ENABLED}"
  --gradient_checkpointing True
  --router_model_file "${ROUTER_MODEL_FILE}"
  --router1_use_shared_expert True
  --router1_shared_expert_weight 1.0
  --use_retrieval_query_routing True
  --retrieval_query_dim "${RETRIEVAL_QUERY_DIM}"
)

TRAIN_CMD+=(--use_trajectory_routing True --trajectory_fusion_mode gate --share_traj_projector False)
if [[ -n "${TRAIN_TRAJ_EMB}" ]]; then
  TRAIN_CMD+=(--trajectory_embedding_path "${TRAIN_TRAJ_EMB}")
fi
if [[ -n "${TRAIN_RET_QUERY}" ]]; then
  TRAIN_CMD+=(--retrieval_query_path "${TRAIN_RET_QUERY}")
fi

echo "[$(timestamp)] Starting training..." | tee -a "${STATUS_LOG}"
"${TRAIN_CMD[@]}"
echo "[$(timestamp)] Training finished." | tee -a "${STATUS_LOG}"

GPU_INFO_AFTER="$(wait_for_gpu evaluation)"
GPU_ID_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f1)"
GPU_TOTAL_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f2)"
GPU_FREE_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f3)"
GPU_NAME_AFTER="$(echo "$GPU_INFO_AFTER" | cut -d',' -f4-)"

echo "[$(timestamp)] Starting evaluation on GPU ${GPU_ID_AFTER} (${GPU_NAME_AFTER}) (total=${GPU_TOTAL_AFTER} MiB, free=${GPU_FREE_AFTER} MiB)" | tee -a "${STATUS_LOG}"
export CUDA_VISIBLE_DEVICES="${GPU_ID_AFTER}"

EVAL_CMD=(
  python eval_two_router.py
  --model_path "${MODEL_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --test_file "${TEST_FILE}"
  --dataset_name "${DATASET_NAME}"
  --context_size "${SEQ_LEN}"
  --seq_len "${SEQ_LEN}"
  --batch_size 1
  --router_model_file "${ROUTER_MODEL_FILE}"
  --use_retrieval_query_routing
)

if [[ "${EVAL_DEVICE}" != "auto" ]]; then
  EVAL_CMD+=(--device "${EVAL_DEVICE}")
fi
if [[ -n "${TEST_TRAJ_EMB}" ]]; then
  EVAL_CMD+=(--trajectory_embedding_path "${TEST_TRAJ_EMB}")
fi
if [[ -n "${TEST_RET_QUERY}" ]]; then
  EVAL_CMD+=(--retrieval_query_path "${TEST_RET_QUERY}")
fi

"${EVAL_CMD[@]}"
echo "[$(timestamp)] Evaluation finished." | tee -a "${STATUS_LOG}"
