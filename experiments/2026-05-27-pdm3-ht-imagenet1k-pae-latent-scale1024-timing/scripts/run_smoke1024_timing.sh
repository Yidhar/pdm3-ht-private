#!/usr/bin/env bash
set -euo pipefail
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-scale1024-timing
SRC_EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-build
source "$EXP/configs/imagenet1k_pae_dinov2l_d32_smoke1024_b32.env"
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"
echo "### Real HF ImageNet-1k train smoke1024 -> PAE_DINOv2L_d32 latents timing"
date -u
START_NS=$(date +%s%N)
set +e
python "$SRC_EXP/scripts/build_pae_latents_from_hf_imagenet.py" \
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
STATUS=$?
set -e
END_NS=$(date +%s%N)
SHELL_TOTAL_SECONDS=$(python - <<PY
start=int("$START_NS")
end=int("$END_NS")
print((end-start)/1e9)
PY
)
echo "SHELL_TOTAL_SECONDS=$SHELL_TOTAL_SECONDS"
echo "PYTHON_EXIT_STATUS=$STATUS"
exit "$STATUS"
