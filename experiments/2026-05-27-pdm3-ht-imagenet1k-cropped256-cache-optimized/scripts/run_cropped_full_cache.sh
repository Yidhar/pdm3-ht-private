#!/usr/bin/env bash
set -euo pipefail
source /workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized/configs/optimized_cache.env
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results" "$CROP_OUT"
python "$EXP_DIR/scripts/build_cropped_uint8_cache.py" \
  --data-files-glob "$SOURCE_PARQUET_GLOB" \
  --output-dir "$CROP_OUT" \
  --image-size "$IMAGE_SIZE" \
  --shard-size "$CROP_SHARD_SIZE" \
  --chunk-size "$CROP_CHUNK_SIZE" \
  --read-batch-rows "$CROP_READ_BATCH_ROWS" \
  --num-workers "$CROP_WORKERS" \
  --prefetch-chunks "$CROP_PREFETCH_CHUNKS" \
  --max-samples "$EXPECTED_TRAIN_SIZE" \
  --resume \
  --log-every-chunks 8
cp "$CROP_OUT/build_summary.json" "$EXP_DIR/results/cropped_full_summary.json"
