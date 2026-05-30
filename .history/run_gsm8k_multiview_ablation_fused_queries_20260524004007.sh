#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/data/ChenWei/miniconda3/envs/graphgpt/bin/python}"
DATASET_NAME="${DATASET_NAME:-nyc}"

TRAIN_CSV="${TRAIN_CSV:-./datasets/NYC/preprocessed/train_sample.csv}"
TEST_CSV="${TEST_CSV:-./datasets/NYC/preprocessed/test_sample_with_traj.csv}"
VAL_CSV="${VAL_CSV:-./datasets/NYC/preprocessed/validate_sample_with_traj.csv}"
TRAIN_QA="${TRAIN_QA:-./datasets/NYC/preprocessed/train_qa_pairs_llm4poi_gsm8k.json}"
VAL_QA="${VAL_QA:-./datasets/NYC/preprocessed/validate_qa_pairs_llm4poi_gsm8k.txt}"
TEST_QA="${TEST_QA:-./datasets/NYC/preprocessed/test_qa_pairs_llm4poi_gsm8k.txt}"
FEATURE_CACHE="${FEATURE_CACHE:-./rag/feature_cache/bert_${DATASET_NAME}}"

ARTIFACT_ROOT="${ARTIFACT_ROOT:-./rag/target_poi_multiview/gsm8k_ablation}"
QUERY_OUT_DIR="${QUERY_OUT_DIR:-./datasets/NYC/preprocessed/gsm8k_ablation_fused_queries}"
LOG_DIR="${LOG_DIR:-./logs/gsm8k_multiview_ablation}"
SUMMARY_JSON="${SUMMARY_JSON:-${ARTIFACT_ROOT}/ablation_summary.json}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-256}"
EPOCHS="${EPOCHS:-120}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-10}"

# Keep these defaults aligned with the existing gsm8k_final/gsm8k_final_val artifacts.
HIDDEN_SIZE="${HIDDEN_SIZE:-64}"
DROPOUT="${DROPOUT:-0.45}"
LR="${LR:-0.00012}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.005}"
FUSION_TYPE="${FUSION_TYPE:-gated}"
GRAPH_ALIGNMENT="${GRAPH_ALIGNMENT:-context_cross_attn}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-20}"
MAX_GRAPH_NODES="${MAX_GRAPH_NODES:-20}"
TRAJ_WEIGHT="${TRAJ_WEIGHT:-0.08}"
SEM_WEIGHT="${SEM_WEIGHT:-0.06}"
STR_WEIGHT="${STR_WEIGHT:-0.06}"
ALIGN_WEIGHT="${ALIGN_WEIGHT:-0.0}"
RANK_WEIGHT="${RANK_WEIGHT:-0.0}"
CATEGORY_WEIGHT="${CATEGORY_WEIGHT:-0.0}"

# Set RERUN_EXISTING=True to retrain even when model.pth already exists.
RERUN_EXISTING="${RERUN_EXISTING:-False}"
# Set REEXPORT_EXISTING=True to overwrite existing fused-query .pt files.
REEXPORT_EXISTING="${REEXPORT_EXISTING:-False}"

mkdir -p "${ARTIFACT_ROOT}" "${QUERY_OUT_DIR}" "${LOG_DIR}"

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

run_one() {
  local tag="$1"
  local active_views="$2"
  local artifact_dir="${ARTIFACT_ROOT}/${tag}"
  local train_query="${QUERY_OUT_DIR}/${tag}_train_fused_queries.pt"
  local test_query="${QUERY_OUT_DIR}/${tag}_test_fused_queries.pt"
  local log_file="${LOG_DIR}/${tag}.log"

  echo "============================================================" | tee -a "${log_file}"
  echo "[$(timestamp)] variant=${tag} active_views=${active_views}" | tee -a "${log_file}"
  echo "[$(timestamp)] artifact_dir=${artifact_dir}" | tee -a "${log_file}"
  echo "[$(timestamp)] train_query=${train_query}" | tee -a "${log_file}"
  echo "[$(timestamp)] test_query=${test_query}" | tee -a "${log_file}"

  mkdir -p "${artifact_dir}"

  if [[ "${RERUN_EXISTING}" == "True" || "${RERUN_EXISTING}" == "true" || ! -f "${artifact_dir}/model.pth" ]]; then
    echo "[$(timestamp)] training ${tag}" | tee -a "${log_file}"
    "${PYTHON_BIN}" rag/target_poi_multiview/train_gsm8k.py \
      --dataset_name "${DATASET_NAME}" \
      --train_csv "${TRAIN_CSV}" \
      --test_csv "${VAL_CSV}" \
      --train_qa "${TRAIN_QA}" \
      --test_qa "${VAL_QA}" \
      --feature_cache "${FEATURE_CACHE}" \
      --save_dir "${artifact_dir}" \
      --device "${DEVICE}" \
      --hidden_size "${HIDDEN_SIZE}" \
      --dropout "${DROPOUT}" \
      --fusion_type "${FUSION_TYPE}" \
      --graph_alignment "${GRAPH_ALIGNMENT}" \
      --max_seq_len "${MAX_SEQ_LEN}" \
      --max_graph_nodes "${MAX_GRAPH_NODES}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --eval_batch_size "${EVAL_BATCH_SIZE}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --traj_weight "${TRAJ_WEIGHT}" \
      --sem_weight "${SEM_WEIGHT}" \
      --str_weight "${STR_WEIGHT}" \
      --align_weight "${ALIGN_WEIGHT}" \
      --rank_weight "${RANK_WEIGHT}" \
      --category_weight "${CATEGORY_WEIGHT}" \
      --active_views "${active_views}" \
      --early_stop_patience "${EARLY_STOP_PATIENCE}" \
      2>&1 | tee -a "${log_file}"
  else
    echo "[$(timestamp)] skip training because ${artifact_dir}/model.pth exists" | tee -a "${log_file}"
  fi

  if [[ "${REEXPORT_EXISTING}" == "True" || "${REEXPORT_EXISTING}" == "true" || ! -f "${train_query}" || ! -f "${test_query}" ]]; then
    echo "[$(timestamp)] exporting fused queries for ${tag}" | tee -a "${log_file}"
    "${PYTHON_BIN}" rag/target_poi_multiview/export_gsm8k_fused_queries.py \
      --dataset_name "${DATASET_NAME}" \
      --train_csv "${TRAIN_CSV}" \
      --test_csv "${TEST_CSV}" \
      --train_qa "${TRAIN_QA}" \
      --test_qa "${TEST_QA}" \
      --feature_cache "${FEATURE_CACHE}" \
      --artifact_dir "${artifact_dir}" \
      --train_out "${train_query}" \
      --test_out "${test_query}" \
      --batch_size "${EXPORT_BATCH_SIZE}" \
      --device "${DEVICE}" \
      2>&1 | tee -a "${log_file}"
  else
    echo "[$(timestamp)] skip export because fused-query files already exist" | tee -a "${log_file}"
  fi

  "${PYTHON_BIN}" - "${tag}" "${active_views}" "${artifact_dir}" "${train_query}" "${test_query}" "${SUMMARY_JSON}" <<'PY'
import json
import sys
from pathlib import Path

tag, active_views, artifact_dir, train_query, test_query, summary_path = sys.argv[1:]
history_path = Path(artifact_dir) / "history.json"
rows = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
best = max(rows, key=lambda r: r.get("recall@20", -1.0)) if rows else {}
entry = {
    "tag": tag,
    "active_views": active_views.split(","),
    "artifact_dir": artifact_dir,
    "train_fused_query": train_query,
    "test_fused_query": test_query,
    "best_epoch": best.get("epoch"),
    "best_recall@1": best.get("recall@1"),
    "best_recall@5": best.get("recall@5"),
    "best_recall@10": best.get("recall@10"),
    "best_recall@20": best.get("recall@20"),
    "best_train_recall@20": best.get("train_recall@20"),
    "best_gate_sem": best.get("gate_sem"),
    "best_gate_str": best.get("gate_str"),
    "best_gate_traj": best.get("gate_traj"),
}
summary_file = Path(summary_path)
summary_file.parent.mkdir(parents=True, exist_ok=True)
if summary_file.exists():
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
else:
    summary = []
summary = [x for x in summary if x.get("tag") != tag]
summary.append(entry)
order = ["full", "no_sem", "no_str", "no_traj", "sem_only", "str_only", "traj_only"]
summary.sort(key=lambda x: order.index(x["tag"]) if x.get("tag") in order else 999)
summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(entry, ensure_ascii=False, indent=2))
PY

  echo "[$(timestamp)] finished ${tag}" | tee -a "${log_file}"
}

# run_one "full" "sem,str,traj"
# run_one "no_sem" "str,traj"
# run_one "no_str" "sem,traj"
run_one "no_traj" "sem,str"
run_one "sem_only" "sem"
run_one "str_only" "str"
run_one "traj_only" "traj"

echo "============================================================"
echo "[$(timestamp)] all ablation jobs finished"
echo "summary: ${SUMMARY_JSON}"
echo "fused query dir: ${QUERY_OUT_DIR}"
