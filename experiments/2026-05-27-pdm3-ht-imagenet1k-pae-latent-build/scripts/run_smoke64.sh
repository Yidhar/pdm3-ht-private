#!/usr/bin/env bash
set -euo pipefail
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-build
source "$EXP/configs/imagenet1k_pae_dinov2l_d32_smoke64.env"
python "$EXP/scripts/build_pae_latents_from_hf_imagenet.py" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --streaming \
  --hf-token-env "$HF_TOKEN_ENV" \
  --pae-config "$PAE_CONFIG" \
  --pae-ckpt "$PAE_CKPT" \
  --output-dir "$OUTPUT_DIR" \
  --image-size "$IMAGE_SIZE" \
  --max-samples "$MAX_SAMPLES" \
  --batch-size "$BATCH_SIZE" \
  --shard-size "$SHARD_SIZE" \
  --model-dtype "$MODEL_DTYPE" \
  --save-dtype "$SAVE_DTYPE" \
  --random-self-test \
  --hard-exit
