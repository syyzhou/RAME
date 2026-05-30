#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

CONDA_SH="${CONDA_SH:-/data/ChenWei/miniconda3/etc/profile.d/conda.sh}"
ENV_PATH="${ENV_PATH:-/data/ChenWei/zhousiyu/llm4poi}"
MODEL_PATH="${MODEL_PATH:-./Qwen2.5-3B}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -f "${CONDA_SH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_SH}"
  conda activate "${ENV_PATH}"
fi

echo "[info] root dir: ${ROOT_DIR}"
echo "[info] model path: ${MODEL_PATH}"
echo "[info] python: $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"

echo "[step] torch cuda availability"
"${PYTHON_BIN}" - <<'PY'
import torch
print(f"[info] torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"[info] torch.cuda.device_count() = {torch.cuda.device_count()}")
PY

run_dataset() {
  local upper_name="$1"
  local lower_name="$2"
  local poi_upper="$3"

  local train_qa="datasets/${upper_name}/preprocessed/train_qa_pairs_llm4poi_gsm8k.json"
  local test_qa="datasets/${upper_name}/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt"
  local train_raw="datasets/${lower_name}/preprocessed/train_sample.csv"
  local test_raw="datasets/${lower_name}/preprocessed/test_sample_with_traj.csv"
  local merged_raw="datasets/${lower_name}/preprocessed/train_test_merged_for_gsm8k_proxy.csv"
  local train_proxy="datasets/${upper_name}/preprocessed/train_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json"
  local test_proxy="datasets/${upper_name}/preprocessed/test_qa_pairs_llm4poi_gsm8k_proxy_for_traj.json"
  local train_emb="datasets/${upper_name}/preprocessed/trajectory_embeddings_gsm8k.pt"
  local test_emb="datasets/${upper_name}/preprocessed/test_embeddings_gsm8k.pt"

  echo "[step][${upper_name}] merge train+test raw csv for test proxy"
  "${PYTHON_BIN}" - <<PY
import pandas as pd
train = "${train_raw}"
test = "${test_raw}"
out = "${merged_raw}"
df = pd.concat([pd.read_csv(train), pd.read_csv(test)], ignore_index=True)
df.to_csv(out, index=False)
print(f"[ok] merged csv saved: {out}, rows={len(df)}")
PY

  echo "[step][${upper_name}] build train proxy dataset"
  "${PYTHON_BIN}" build_gsm8k_trajectory_proxy_dataset.py \
    --qa_input "${train_qa}" \
    --raw_csv "${train_raw}" \
    --output_json "${train_proxy}" \
    --poi_upper "${poi_upper}"

  echo "[step][${upper_name}] build test proxy dataset"
  "${PYTHON_BIN}" build_gsm8k_trajectory_proxy_dataset.py \
    --qa_input "${test_qa}" \
    --raw_csv "${merged_raw}" \
    --output_json "${test_proxy}" \
    --poi_upper "${poi_upper}"

  echo "[step][${upper_name}] precompute train trajectory embeddings"
  "${PYTHON_BIN}" precompute_trajectory_embeddings.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${train_proxy}" \
    --output_path "${train_emb}" \
    --split train

  echo "[step][${upper_name}] precompute test trajectory embeddings"
  "${PYTHON_BIN}" precompute_trajectory_embeddings.py \
    --model_name_or_path "${MODEL_PATH}" \
    --dataset "${test_proxy}" \
    --output_path "${test_emb}" \
    --split test

  echo "[done][${upper_name}] train embedding: ${train_emb}"
  echo "[done][${upper_name}] test embedding : ${test_emb}"
}

run_dataset "CA" "ca" "9690"
run_dataset "TKY" "tky" "7833"

echo "[done] CA and TKY GSM8K trajectory proxy + embedding pipeline finished"
