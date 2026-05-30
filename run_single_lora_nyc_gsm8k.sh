#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${MODEL_PATH:-${SCRIPT_DIR}/Qwen2.5-3B}"
DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/datasets/NYC}"
TRAIN_PARQUET="${TRAIN_PARQUET:-${DATA_DIR}/nyc_gsm8k_train_llm4poi.parquet}"
TEST_PARQUET="${TEST_PARQUET:-${DATA_DIR}/nyc_gsm8k_test_llm4poi.parquet}"
TRAIN_JSONL_FALLBACK="${TRAIN_JSONL_FALLBACK:-${DATA_DIR}/nyc_llm4poi_train_samples.jsonl}"
TEST_JSONL_FALLBACK="${TEST_JSONL_FALLBACK:-${DATA_DIR}/nyc_llm4poi_test_samples.jsonl}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/preprocessed/train_qa_pairs_llm4poi_gsm8k.json}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output/single-lora-nyc-gsm8k}"
SEQ_LEN="${SEQ_LEN:-4096}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
echo "[info] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [[ -f "${TRAIN_PARQUET}" && -f "${TEST_PARQUET}" ]]; then
  TRAIN_INPUT="${TRAIN_PARQUET}"
  TEST_INPUT="${TEST_PARQUET}"
else
  TRAIN_INPUT="${TRAIN_JSONL_FALLBACK}"
  TEST_INPUT="${TEST_JSONL_FALLBACK}"
fi

echo "[info] prepare data from:"
echo "  train: ${TRAIN_INPUT}"
echo "  test : ${TEST_INPUT}"
python "${SCRIPT_DIR}/prepare_gsm8k_for_llm4poi.py" \
  --train_input "${TRAIN_INPUT}" \
  --test_input "${TEST_INPUT}" \
  --train_output_json "${TRAIN_FILE}" \
  --test_output_txt "${TEST_FILE}"

echo "[info] start training..."
# python supervised-fine-tune-qlora-old.py \
#   --model_name_or_path "${MODEL_PATH}" \
#   --bf16 True \
#   --output_dir "${OUTPUT_DIR}" \
#   --use_flash_attn False \
#   --dataset "${TRAIN_FILE}" \
#   --low_rank_training True \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 1 \
#   --per_device_eval_batch_size 2 \
#   --evaluation_strategy no \
#   --gradient_accumulation_steps 8 \
#   --save_strategy no \
#   --save_steps 100 \
#   --learning_rate 2e-5 \
#   --weight_decay 0.0 \
#   --warmup_steps 20 \
#   --lr_scheduler_type constant_with_warmup \
#   --logging_steps 1 \
#   --logging_dir ./logs \
#   --deepspeed ds_configs/stage2.json \
#   --model_max_length "${SEQ_LEN}" \
#   --tf32 False \
#   --gradient_checkpointing True

echo "[info] start eval..."
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
