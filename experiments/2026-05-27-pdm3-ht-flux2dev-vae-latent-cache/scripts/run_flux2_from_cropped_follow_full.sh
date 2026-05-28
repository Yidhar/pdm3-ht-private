#!/usr/bin/env bash
set -euo pipefail

# Follow-mode full FLUX.2 latent cache builder.
# It repeatedly encodes whatever cropped ImageNet-256 shards are durable, then
# resumes later as the crop cache grows, until EXPECTED_TRAIN_SIZE is reached.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# The batch sweep showed B=256 is the best safe point on the 98GB Blackwell GPU.
export BATCH_SIZE="${BATCH_SIZE:-256}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
source "$EXP_DIR/configs/flux2dev_vae_cache.env"

mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results" "$FLUX2_FULL_OUT"
: "${FLUX2_FOLLOW_MIN_GAP:=131072}"
: "${FLUX2_FOLLOW_POLL_SEC:=60}"
: "${FLUX2_FOLLOW_MAX_PASSES:=30}"

count_samples() {
  local dir="$1" pattern="$2" key="$3"
  python - "$dir" "$pattern" "$key" <<'PY'
import sys
from pathlib import Path
from safetensors import safe_open
p=Path(sys.argv[1]); pattern=sys.argv[2]; key=sys.argv[3]
total=0; nfiles=0; last=None
if p.exists():
    for f in sorted(p.glob(pattern)):
        with safe_open(str(f), framework='pt', device='cpu') as sf:
            total += int(sf.get_tensor(key).numel())
        nfiles += 1; last=f.name
print(f"{total} {nfiles} {last or ''}")
PY
}

json_status() {
  local event="$1" crop_samples="$2" crop_files="$3" crop_last="$4" latent_samples="$5" latent_files="$6" latent_last="$7" pass_idx="$8"
  python - "$event" "$crop_samples" "$crop_files" "$crop_last" "$latent_samples" "$latent_files" "$latent_last" "$pass_idx" "$EXPECTED_TRAIN_SIZE" "$FLUX2_FOLLOW_MIN_GAP" "$BATCH_SIZE" "$LATENT_SHARD_SIZE" "$FLUX2_FULL_OUT" <<'PY'
import json, sys, datetime
keys=['event','crop_samples','crop_files','crop_last','latent_samples','latent_files','latent_last','pass_idx','expected','min_gap','batch_size','latent_shard_size','output_dir']
vals=dict(zip(keys, sys.argv[1:]))
for k in ['crop_samples','crop_files','latent_samples','latent_files','pass_idx','expected','min_gap','batch_size','latent_shard_size']:
    vals[k]=int(vals[k])
vals['time_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
vals['remaining_latents']=max(0, vals['expected']-vals['latent_samples'])
vals['available_gap']=vals['crop_samples']-vals['latent_samples']
print('FLUX2_FOLLOW_STATUS '+json.dumps(vals, sort_keys=True), flush=True)
PY
}

pass_idx=0
while (( pass_idx < FLUX2_FOLLOW_MAX_PASSES )); do
  read -r crop_samples crop_files crop_last < <(count_samples "$CROP_CACHE_DIR" 'images_uint8_shard*.safetensors' labels)
  read -r latent_samples latent_files latent_last < <(count_samples "$FLUX2_FULL_OUT" 'latents_rank00_shard*.safetensors' labels)
  json_status poll "$crop_samples" "$crop_files" "${crop_last:-}" "$latent_samples" "$latent_files" "${latent_last:-}" "$pass_idx"

  if (( latent_samples >= EXPECTED_TRAIN_SIZE )); then
    echo "FLUX2_FOLLOW_DONE latent_samples=$latent_samples expected=$EXPECTED_TRAIN_SIZE" >&2
    break
  fi

  gap=$(( crop_samples - latent_samples ))
  if (( crop_samples >= EXPECTED_TRAIN_SIZE )); then
    should_run=1
  elif (( gap >= FLUX2_FOLLOW_MIN_GAP )); then
    should_run=1
  else
    should_run=0
  fi

  if (( should_run == 0 )); then
    sleep "$FLUX2_FOLLOW_POLL_SEC"
    continue
  fi

  pass_idx=$((pass_idx + 1))
  echo "FLUX2_FOLLOW_RUN_PASS pass=$pass_idx crop_samples=$crop_samples latent_samples=$latent_samples gap=$gap batch_size=$BATCH_SIZE" >&2
  python "$SCRIPT_DIR/build_diffusers_vae_latents_from_cropped_cache.py" \
    --crop-cache-dir "$CROP_CACHE_DIR" \
    --output-dir "$FLUX2_FULL_OUT" \
    --repo-id "$FLUX2_REPO_ID" \
    --subfolder "$FLUX2_SUBFOLDER" \
    --vae-class "$FLUX2_VAE_CLASS" \
    --backend-name AutoencoderKLFlux2_d32 \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --latent-shard-size "$LATENT_SHARD_SIZE" \
    --max-samples "$EXPECTED_TRAIN_SIZE" \
    --device "$DEVICE" \
    --model-dtype "$MODEL_DTYPE" \
    --save-dtype "$SAVE_DTYPE" \
    --latent-mode "$LATENT_MODE" \
    --expected-latent-channels 32 \
    --expected-latent-hw 32x32 \
    --resume

  cp "$FLUX2_FULL_OUT/build_summary.json" "$EXP_DIR/results/flux2_from_cropped_full_summary.json" || true
  cp "$FLUX2_FULL_OUT/build_summary.json" "$EXP_DIR/results/flux2_from_cropped_full_summary_pass${pass_idx}.json" || true

  read -r latent_samples latent_files latent_last < <(count_samples "$FLUX2_FULL_OUT" 'latents_rank00_shard*.safetensors' labels)
  if (( latent_samples >= EXPECTED_TRAIN_SIZE )); then
    echo "FLUX2_FOLLOW_DONE latent_samples=$latent_samples expected=$EXPECTED_TRAIN_SIZE" >&2
    break
  fi
done

read -r crop_samples crop_files crop_last < <(count_samples "$CROP_CACHE_DIR" 'images_uint8_shard*.safetensors' labels)
read -r latent_samples latent_files latent_last < <(count_samples "$FLUX2_FULL_OUT" 'latents_rank00_shard*.safetensors' labels)
json_status final "$crop_samples" "$crop_files" "${crop_last:-}" "$latent_samples" "$latent_files" "${latent_last:-}" "$pass_idx"
if (( latent_samples < EXPECTED_TRAIN_SIZE )); then
  echo "FLUX2_FOLLOW_INCOMPLETE latent_samples=$latent_samples expected=$EXPECTED_TRAIN_SIZE" >&2
  exit 2
fi
