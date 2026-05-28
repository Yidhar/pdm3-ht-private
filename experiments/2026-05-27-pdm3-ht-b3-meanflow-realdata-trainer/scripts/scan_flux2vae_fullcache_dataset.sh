#!/usr/bin/env bash
set -euo pipefail
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
cd /workspace/PDM
mkdir -p "$EXP/logs"
python "$EXP/scripts/scan_latent_dataset.py" \
  --config "$EXP/configs/b3_meanflow_flux2vae_full.yaml" \
  --output-dir "$EXP/results/flux2vae_fullcache_dataset_scan" \
  --label-preview-count 2048 \
  2>&1 | tee "$EXP/logs/031_flux2vae_fullcache_dataset_scan.log"
