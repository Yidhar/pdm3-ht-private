#!/usr/bin/env bash
set -euo pipefail
source /workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/configs/full_cache.env
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results" "$IMAGENET1K_PARQUET_DIR"
python "$EXP_DIR/scripts/download_imagenet1k_hf_parquet_snapshot.py" \
  --repo-id "$DATASET" \
  --split "$SPLIT" \
  --local-dir "$IMAGENET1K_PARQUET_DIR" \
  --max-workers "$DOWNLOAD_WORKERS" \
  --manifest "$EXP_DIR/results/imagenet1k_${SPLIT}_parquet_snapshot_manifest.json"
