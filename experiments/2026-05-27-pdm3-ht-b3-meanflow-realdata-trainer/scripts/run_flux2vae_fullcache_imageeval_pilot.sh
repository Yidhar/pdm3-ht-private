#!/usr/bin/env bash
set -euo pipefail
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
cd /workspace/PDM
mkdir -p "$EXP/logs"
LOG="$EXP/logs/045_flux2vae_fullcache_imageeval_pilot_$(date -u +%Y%m%dT%H%M%SZ).log"
python "$EXP/scripts/train_b3_meanflow_realdata.py" \
  --config "$EXP/configs/b3_meanflow_flux2vae_fullcache_imageeval_pilot.yaml" \
  2>&1 | tee "$LOG"
