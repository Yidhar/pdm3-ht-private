#!/usr/bin/env bash
set -euo pipefail
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
cd /workspace/PDM
python "$EXP/scripts/train_b3_meanflow_realdata.py" \
  --config "$EXP/configs/b3_meanflow_realdata_smoke4096.yaml" \
  2>&1 | tee "$EXP/logs/010_smoke4096.log"
