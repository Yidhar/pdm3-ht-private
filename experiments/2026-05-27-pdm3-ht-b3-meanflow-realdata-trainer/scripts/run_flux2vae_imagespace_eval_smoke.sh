#!/usr/bin/env bash
set -euo pipefail
cd /workspace/PDM
CFG="experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_flux2vae_imagespace_eval_smoke.yaml"
LOG="experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/032_flux2vae_imagespace_eval_smoke.log"
mkdir -p "$(dirname "$LOG")"
python experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py --config "$CFG" 2>&1 | tee "$LOG"
