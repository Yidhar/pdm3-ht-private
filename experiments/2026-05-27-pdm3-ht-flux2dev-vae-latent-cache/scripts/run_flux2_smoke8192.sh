#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$EXP_DIR/configs/flux2dev_vae_cache.env"
LOG="$EXP_DIR/logs/flux2_smoke8192_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "$LOG" > "$EXP_DIR/results/flux2_smoke8192.logpath"
python "$SCRIPT_DIR/build_diffusers_vae_latents_from_cropped_cache.py" \
  --crop-cache-dir "$CROP_CACHE_DIR" \
  --output-dir "$FLUX2_SMOKE8192_OUT" \
  --repo-id "$FLUX2_REPO_ID" \
  --subfolder "$FLUX2_SUBFOLDER" \
  --vae-class "$FLUX2_VAE_CLASS" \
  --backend-name AutoencoderKLFlux2_d32 \
  --image-size "$IMAGE_SIZE" \
  --batch-size "${BATCH_SIZE:-64}" \
  --latent-shard-size "$LATENT_SHARD_SIZE" \
  --max-samples 8192 \
  --device "$DEVICE" \
  --model-dtype "$MODEL_DTYPE" \
  --save-dtype "$SAVE_DTYPE" \
  --latent-mode "$LATENT_MODE" \
  --expected-latent-channels 32 \
  --expected-latent-hw 32x32 \
  --overwrite \
  2>&1 | tee "$LOG"
