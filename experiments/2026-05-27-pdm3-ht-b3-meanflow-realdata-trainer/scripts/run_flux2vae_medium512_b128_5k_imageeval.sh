#!/usr/bin/env bash
set -euo pipefail
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
cd /workspace/PDM
mkdir -p "$EXP/logs"
LOG="$EXP/logs/053_flux2vae_medium512_b128_5k_imageeval_$(date -u +%Y%m%dT%H%M%SZ).log"
python "$EXP/scripts/train_b3_meanflow_realdata.py" \
  --config "$EXP/configs/b3_meanflow_flux2vae_fullcache_medium512_b128_5k_imageeval.yaml" \
  2>&1 | tee "$LOG"
