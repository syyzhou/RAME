#!/usr/bin/env bash
set -euo pipefail

BACKBONES=()
DATASETS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --datasets)
      shift
      if [[ "$#" -eq 0 ]]; then
        echo "Missing value after --datasets" >&2
        exit 2
      fi
      IFS=',' read -r -a DATASETS <<< "$1"
      ;;
    --backbones)
      shift
      if [[ "$#" -eq 0 ]]; then
        echo "Missing value after --backbones" >&2
        exit 2
      fi
      IFS=',' read -r -a BACKBONES <<< "$1"
      ;;
    *)
      BACKBONES+=("$1")
      ;;
  esac
  shift
done

if [[ "${#BACKBONES[@]}" -eq 0 ]]; then
  BACKBONES=(qwen25_05b qwen25_15b qwen25_3b llama32_3b)
fi

if [[ "${#DATASETS[@]}" -eq 0 ]]; then
  DATASETS=(nyc ca tky)
fi

for dataset in "${DATASETS[@]}"; do
  for backbone in "${BACKBONES[@]}"; do
    echo "===== Running dataset: ${dataset}; backbone: ${backbone} ====="
    bash experiments/backbone_compare/run_one_backbone.sh "${backbone}" "${dataset}"
  done
done
