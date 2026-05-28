#!/usr/bin/env bash
set -euo pipefail
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-batch-saturation
source "$EXP/configs/batch_saturation_sweep.env"
rm -rf "$OUTPUT_BASE"
mkdir -p "$OUTPUT_BASE"
echo "### PAE ImageNet latent batch saturation sweep"
date -u
START_NS=$(date +%s%N)
set +e
python "$EXP/scripts/batch_saturation_sweep.py" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --streaming \
  --hf-token-env "$HF_TOKEN_ENV" \
  --pae-config "$PAE_CONFIG" \
  --pae-ckpt "$PAE_CKPT" \
  --output-base "$OUTPUT_BASE" \
  --image-size "$IMAGE_SIZE" \
  --batch-sizes "$BATCH_SIZES" \
  --samples-per-batch-size "$SAMPLES_PER_BATCH_SIZE" \
  --initial-skip "$INITIAL_SKIP" \
  --model-dtype "$MODEL_DTYPE" \
  --save-dtype "$SAVE_DTYPE" \
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
echo "WRAPPER_SHELL_TOTAL_SECONDS=$SHELL_TOTAL_SECONDS"
echo "PYTHON_EXIT_STATUS=$STATUS"
exit "$STATUS"
