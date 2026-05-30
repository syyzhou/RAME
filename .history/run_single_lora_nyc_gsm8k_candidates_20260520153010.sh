#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/Qwen2.5-3B}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/datasets/NYC/preprocessed}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0.json}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/test_qa_pairs_llm4poi_gsm8k_candidates_gsm8k_final_mix10trans4geo3hist3_rerank_ret075_trans075_hist1_geo0.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/single-lora-nyc-gsm8k-candidates-rerank-ret075}"
SEQ_LEN="${SEQ_LEN:-4096}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_EVAL="${RUN_EVAL:-1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
echo "[info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[info] model      : ${MODEL_PATH}"
echo "[info] train file : ${TRAIN_FILE}"
echo "[info] test file  : ${TEST_FILE}"
echo "[info] output dir : ${OUTPUT_DIR}"
echo "[info] seq len    : ${SEQ_LEN}"

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[error] TRAIN_FILE not found: ${TRAIN_FILE}" >&2
  exit 1
fi

if [[ ! -f "${TEST_FILE}" ]]; then
  echo "[error] TEST_FILE not found: ${TEST_FILE}" >&2
  exit 1
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  echo "[info] start training with candidate-augmented data..."
  python "${SCRIPT_DIR}/supervised-fine-tune-qlora-old.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --bf16 True \
    --output_dir "${OUTPUT_DIR}" \
    --use_flash_attn False \
    --dataset "${TRAIN_FILE}" \
    --low_rank_training True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --evaluation_strategy no \
    --gradient_accumulation_steps 8 \
    --save_strategy no \
    --save_steps 100 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_steps 20 \
    --lr_scheduler_type constant_with_warmup \
    --logging_steps 1 \
    --logging_dir "${SCRIPT_DIR}/logs" \
    --deepspeed "${SCRIPT_DIR}/ds_configs/stage2.json" \
    --model_max_length "${SEQ_LEN}" \
    --tf32 False \
    --gradient_checkpointing True
else
  echo "[info] skip training because RUN_TRAIN=${RUN_TRAIN}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  echo "[info] start eval with candidate-augmented test data..."
  if [[ ! -f "${OUTPUT_DIR}/adapter_model.safetensors" && ! -f "${OUTPUT_DIR}/adapter_model.bin" ]]; then
    echo "[warn] No LoRA adapter found in OUTPUT_DIR=${OUTPUT_DIR}"
    echo "[warn] Eval will run with base model only, ACC may be very low."
  fi

  python "${SCRIPT_DIR}/eval_next_poi.py" \
    --model_path "${MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --test_path "${TEST_FILE}" \
    --context_size "${SEQ_LEN}" \
    --seq_len "${SEQ_LEN}" \
    --batch_size 1 \
    --device cuda:0 \
    --save_raw_predictions
else
  echo "[info] skip eval because RUN_EVAL=${RUN_EVAL}"
fi
