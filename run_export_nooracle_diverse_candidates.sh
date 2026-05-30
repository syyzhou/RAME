#!/usr/bin/env bash
set -euo pipefail

DATASET_NAME="${DATASET_NAME:-nyc}"
CANDIDATE_TAG="${CANDIDATE_TAG:-dst_time_top12_easy3_contrast3_tail2_nooracle}"
ARTIFACT_DIR="${ARTIFACT_DIR:-./rag/target_poi_multiview/artifacts_cross_attn_dropout04_wd3e3_${DATASET_NAME}_ep200}"
FEATURE_CACHE="${FEATURE_CACHE:-./rag/feature_cache/bert_${DATASET_NAME}}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"

DATA_DIR="./datasets/${DATASET_NAME}/preprocessed"

"${PYTHON}" rag/target_poi_multiview/export_nooracle_diverse_candidates.py \
  --dataset_name "${DATASET_NAME}" \
  --candidate_tag "${CANDIDATE_TAG}" \
  --feature_cache "${FEATURE_CACHE}" \
  --artifact_dir "${ARTIFACT_DIR}" \
  --train_output "${DATA_DIR}/train_${CANDIDATE_TAG}.json" \
  --test_output "${DATA_DIR}/test_${CANDIDATE_TAG}.txt" \
  --stats_output "${DATA_DIR}/target_poi_multiview_${CANDIDATE_TAG}_stats.json" \
  --test_use_top12 \
  --device "${DEVICE}"
