#!/usr/bin/env bash
set -euo pipefail
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
cd /workspace/PDM
mkdir -p "$EXP/logs"
LOG="$EXP/logs/044_flux2vae_imagespace_fid_mmd_smoke_$(date -u +%Y%m%dT%H%M%SZ).log"
python "$EXP/scripts/train_b3_meanflow_realdata.py" \
  --config "$EXP/configs/b3_meanflow_flux2vae_imagespace_fid_mmd_smoke.yaml" \
  2>&1 | tee "$LOG"

