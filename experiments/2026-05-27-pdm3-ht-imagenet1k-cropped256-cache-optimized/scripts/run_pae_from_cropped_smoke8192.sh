#!/usr/bin/env bash
set -euo pipefail
source /workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized/configs/optimized_cache.env
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results" "$LATENT_SMOKE_FAST_OUT"
python "$EXP_DIR/scripts/build_pae_latents_from_cropped_cache.py" \
  --crop-cache-dir "$CROP_SMOKE_OUT" \
  --output-dir "$LATENT_SMOKE_FAST_OUT" \
  --pae-config "$PAE_CONFIG" \
  --pae-ckpt "$PAE_CKPT" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$PAE_BATCH_SIZE" \
  --latent-shard-size "$LATENT_SHARD_SIZE" \
  --max-samples 8192 \
  --device "$DEVICE" \
  --model-dtype "$MODEL_DTYPE" \
  --save-dtype "$SAVE_DTYPE" \
  --overwrite
cp "$LATENT_SMOKE_FAST_OUT/build_summary.json" "$EXP_DIR/results/pae_from_cropped_smoke8192_summary.json"
