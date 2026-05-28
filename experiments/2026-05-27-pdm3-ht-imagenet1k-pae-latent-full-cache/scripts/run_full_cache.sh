#!/usr/bin/env bash
set -euo pipefail
source /workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/configs/full_cache.env
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results" "$FULL_OUT"
python "$EXP_DIR/scripts/build_pae_latents_full_cache.py" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --streaming \
  --pae-config "$PAE_CONFIG" \
  --pae-ckpt "$PAE_CKPT" \
  --output-dir "$FULL_OUT" \
  --image-size "$IMAGE_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --shard-size "$SHARD_SIZE" \
  --max-samples "$EXPECTED_TRAIN_SIZE" \
  --expected-total "$EXPECTED_TRAIN_SIZE" \
  --device "$DEVICE" \
  --model-dtype "$MODEL_DTYPE" \
  --save-dtype "$SAVE_DTYPE" \
  --preprocess-workers "$PREPROCESS_WORKERS" \
  --prefetch-batches "$PREFETCH_BATCHES" \
  --resume \
  --sha256 \
  --hard-exit
python "$EXP_DIR/scripts/verify_pae_latent_cache.py" "$FULL_OUT" \
  --expected-total "$EXPECTED_TRAIN_SIZE" \
  --no-latent-norm \
  --output-json "$EXP_DIR/results/full_cache_verify.json"
cp "$FULL_OUT/build_summary.json" "$EXP_DIR/results/full_cache_build_summary.json"
cp "$FULL_OUT/progress.json" "$EXP_DIR/results/full_cache_progress_final.json"
