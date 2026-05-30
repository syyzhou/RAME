#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONDA_SH="${CONDA_SH:-/data/ChenWei/miniconda3/etc/profile.d/conda.sh}"
ENV_PATH="${ENV_PATH:-/data/ChenWei/zhousiyu/llm4poi}"

TRAIN_QA="${TRAIN_QA:-datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k.json}"
TEST_QA="${TEST_QA:-datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt}"

TRAIN_RAW="${TRAIN_RAW:-datasets/nyc/preprocessed/train_sample.csv}"
TEST_RAW="${TEST_RAW:-datasets/nyc/preprocessed/test_sample_with_traj.csv}"
MERGED_RAW="${MERGED_RAW:-datasets/nyc/preprocessed/train_test_merged_for_gsm8k_proxy.csv}"

TRAIN_PROXY="${TRAIN_PROXY:-datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json}"
TEST_PROXY="${TEST_PROXY:-datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json}"

TRAIN_EMB="${TRAIN_EMB:-datasets/NYC/preprocessed/trajectory_embeddings_gsm8k.pt}"
TEST_EMB="${TEST_EMB:-datasets/NYC/preprocessed/test_embeddings_gsm8k.pt}"

MODEL_PATH="${MODEL_PATH:-./Qwen2.5-3B}"
POI_UPPER="${POI_UPPER:-4981}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${ENV_PATH}"
else
  echo "[warn] conda init script not found at ${CONDA_SH}, continue without conda activate"
fi

echo "[info] root dir: ${ROOT_DIR}"
echo "[info] env path: ${ENV_PATH}"
echo "[info] model path: ${MODEL_PATH}"

echo "[step] torch cuda availability"
python - <<'PY'
import torch
print(f"[info] torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"[info] torch.cuda.device_count() = {torch.cuda.device_count()}")
PY

echo "[step] build train proxy dataset"
python build_gsm8k_trajectory_proxy_dataset.py \
  --qa_input "${TRAIN_QA}" \
  --raw_csv "${TRAIN_RAW}" \
  --output_json "${TRAIN_PROXY}" \
  --poi_upper "${POI_UPPER}"

echo "[step] merge train+test raw csv for test proxy"
python - <<PY
import pandas as pd
train = "${TRAIN_RAW}"
test = "${TEST_RAW}"
out = "${MERGED_RAW}"
df = pd.concat([pd.read_csv(train), pd.read_csv(test)], ignore_index=True)
df.to_csv(out, index=False)
print(f"[ok] merged csv saved: {out}, rows={len(df)}")
PY

echo "[step] build test proxy dataset"
python build_gsm8k_trajectory_proxy_dataset.py \
  --qa_input "${TEST_QA}" \
  --raw_csv "${MERGED_RAW}" \
  --output_json "${TEST_PROXY}" \
  --poi_upper "${POI_UPPER}"

echo "[step] precompute train trajectory embeddings"
python precompute_trajectory_embeddings.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset "${TRAIN_PROXY}" \
  --output_path "${TRAIN_EMB}" \
  --split train

echo "[step] precompute test trajectory embeddings"
python precompute_trajectory_embeddings.py \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset "${TEST_PROXY}" \
  --output_path "${TEST_EMB}" \
  --split test

echo "[done] all steps finished"
echo "[done] train embedding: ${TRAIN_EMB}"
echo "[done] test embedding : ${TEST_EMB}"
