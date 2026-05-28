#!/usr/bin/env bash
set -euo pipefail
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
cd /workspace/PDM
python "$EXP/scripts/scan_latent_dataset.py" \
  --config "$EXP/configs/b3_meanflow_realdata_full.yaml" \
  --output-dir "$EXP/results/partial_fullcache_dataset_scan" \
  --label-preview-count 1024 \
  2>&1 | tee "$EXP/logs/020_partial_fullcache_dataset_scan.log"
