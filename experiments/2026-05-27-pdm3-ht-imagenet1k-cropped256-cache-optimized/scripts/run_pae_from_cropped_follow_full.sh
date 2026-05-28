#!/usr/bin/env bash
set -euo pipefail
source /workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized/configs/optimized_cache.env
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results" "$LATENT_FULL_OUT"
: "${PAE_FOLLOW_MIN_GAP:=131072}"
: "${PAE_FOLLOW_POLL_SEC:=60}"
: "${PAE_FOLLOW_MAX_PASSES:=20}"

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
  python - "$event" "$crop_samples" "$crop_files" "$crop_last" "$latent_samples" "$latent_files" "$latent_last" "$pass_idx" "$EXPECTED_TRAIN_SIZE" "$PAE_FOLLOW_MIN_GAP" <<'PY'
import json, sys, datetime
keys=['event','crop_samples','crop_files','crop_last','latent_samples','latent_files','latent_last','pass_idx','expected','min_gap']
vals=dict(zip(keys, sys.argv[1:]))
for k in ['crop_samples','crop_files','latent_samples','latent_files','pass_idx','expected','min_gap']:
    vals[k]=int(vals[k])
vals['time_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
vals['remaining_latents']=max(0, vals['expected']-vals['latent_samples'])
vals['available_gap']=vals['crop_samples']-vals['latent_samples']
print('PAE_FOLLOW_STATUS '+json.dumps(vals, sort_keys=True), flush=True)
PY
}

pass_idx=0
while (( pass_idx < PAE_FOLLOW_MAX_PASSES )); do
  read -r crop_samples crop_files crop_last < <(count_samples "$CROP_OUT" 'images_uint8_shard*.safetensors' labels)
  read -r latent_samples latent_files latent_last < <(count_samples "$LATENT_FULL_OUT" 'latents_rank00_shard*.safetensors' labels)
  json_status poll "$crop_samples" "$crop_files" "${crop_last:-}" "$latent_samples" "$latent_files" "${latent_last:-}" "$pass_idx"

  if (( latent_samples >= EXPECTED_TRAIN_SIZE )); then
    echo "PAE_FOLLOW_DONE latent_samples=$latent_samples expected=$EXPECTED_TRAIN_SIZE" >&2
    break
  fi

  gap=$(( crop_samples - latent_samples ))
  if (( crop_samples >= EXPECTED_TRAIN_SIZE )); then
    should_run=1
  elif (( gap >= PAE_FOLLOW_MIN_GAP )); then
    should_run=1
  else
    should_run=0
  fi

  if (( should_run == 0 )); then
    sleep "$PAE_FOLLOW_POLL_SEC"
    continue
  fi

  pass_idx=$((pass_idx + 1))
  echo "PAE_FOLLOW_RUN_PASS pass=$pass_idx crop_samples=$crop_samples latent_samples=$latent_samples gap=$gap" >&2
  python "$EXP_DIR/scripts/build_pae_latents_from_cropped_cache.py" \
    --crop-cache-dir "$CROP_OUT" \
    --output-dir "$LATENT_FULL_OUT" \
    --pae-config "$PAE_CONFIG" \
    --pae-ckpt "$PAE_CKPT" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$PAE_BATCH_SIZE" \
    --latent-shard-size "$LATENT_SHARD_SIZE" \
    --max-samples "$EXPECTED_TRAIN_SIZE" \
    --device "$DEVICE" \
    --model-dtype "$MODEL_DTYPE" \
    --save-dtype "$SAVE_DTYPE" \
    --resume
  cp "$LATENT_FULL_OUT/build_summary.json" "$EXP_DIR/results/pae_from_cropped_full_summary.json" || true
  read -r latent_samples latent_files latent_last < <(count_samples "$LATENT_FULL_OUT" 'latents_rank00_shard*.safetensors' labels)
  if (( latent_samples >= EXPECTED_TRAIN_SIZE )); then
    echo "PAE_FOLLOW_DONE latent_samples=$latent_samples expected=$EXPECTED_TRAIN_SIZE" >&2
    break
  fi
done

read -r crop_samples crop_files crop_last < <(count_samples "$CROP_OUT" 'images_uint8_shard*.safetensors' labels)
read -r latent_samples latent_files latent_last < <(count_samples "$LATENT_FULL_OUT" 'latents_rank00_shard*.safetensors' labels)
json_status final "$crop_samples" "$crop_files" "${crop_last:-}" "$latent_samples" "$latent_files" "${latent_last:-}" "$pass_idx"
if (( latent_samples < EXPECTED_TRAIN_SIZE )); then
  echo "PAE_FOLLOW_INCOMPLETE latent_samples=$latent_samples expected=$EXPECTED_TRAIN_SIZE" >&2
  exit 2
fi
