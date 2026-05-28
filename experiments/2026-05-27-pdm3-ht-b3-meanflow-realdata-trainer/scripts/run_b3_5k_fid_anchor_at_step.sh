#!/usr/bin/env bash
# One-shot controller for a 5k true-Inception FID anchor at a checkpoint boundary.
#
# Default behavior:
#   * wait for STEP checkpoint and the trainer's small eval sample to exist;
#   * stop the live trainer process group to free the single H100;
#   * generate NUM_SAMPLES EMA latents on CUDA from the requested checkpoint;
#   * immediately resume training from latest.pt using the on-disk long-run config;
#   * launch CPU PAE decode + torchvision InceptionV3 metrics in the background.
#
# This script is intended for the active B3 b96 PAE+LightningDiT+MeanFlow route.
# It avoids uploading raw real ImageNet images/decoded NPZ payloads by passing
# --max-save-images 0 --grid-count 0 --no-save-npz to the decoder/eval helper.
set -Eeuo pipefail

EXP=${EXP:-/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer}
CFG=${CFG:-$EXP/configs/b3_meanflow_realdata_full_fast_h100_b96.yaml}
RES=${RES:-$EXP/results/fullcache_realdata_singleproc_template}

STEP=${STEP:-60000}
NUM_SAMPLES=${NUM_SAMPLES:-5000}
SAMPLE_BATCH_SIZE=${SAMPLE_BATCH_SIZE:-64}
SAMPLE_STEPS=${SAMPLE_STEPS:-32}
PRECISION_MODE=${PRECISION_MODE:-bf16_autocast}
SEED=${SEED:-2026052850}
REAL_SEED=${REAL_SEED:-20260528}

CHECK_INTERVAL=${CHECK_INTERVAL:-30}
WAIT_FOR_TRAINER_EVAL=${WAIT_FOR_TRAINER_EVAL:-1}
STOP_TRAINER=${STOP_TRAINER:-1}
RESUME_TRAINER=${RESUME_TRAINER:-1}
START_CPU_EVAL=${START_CPU_EVAL:-1}

DECODE_BATCH_SIZE=${DECODE_BATCH_SIZE:-8}
INCEPTION_BATCH_SIZE=${INCEPTION_BATCH_SIZE:-64}
TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-64}

TRAIN_SCRIPT=${TRAIN_SCRIPT:-$EXP/scripts/train_b3_meanflow_realdata.py}
SAMPLE_SCRIPT=${SAMPLE_SCRIPT:-$EXP/scripts/sample_b3_meanflow_eval_latents.py}
DECODE_SCRIPT=${DECODE_SCRIPT:-$EXP/scripts/decode_and_inception_eval_step.py}

printf -v STEP_PAD "%08d" "$STEP"
CKPT=${CKPT:-$RES/checkpoints/step_${STEP_PAD}.pt}
EVAL=${EVAL:-$RES/eval/step_${STEP_PAD}}
LATENTS=${LATENTS:-$EVAL/sample_latents_${NUM_SAMPLES}.safetensors}
METADATA_JSON=${METADATA_JSON:-${LATENTS%.safetensors}.json}
INCEPTION_DIR=${INCEPTION_DIR:-$EVAL/inception_eval_5k}
METRICS_JSONL=${METRICS_JSONL:-$RES/metrics.jsonl}

mkdir -p "$EVAL" "$EXP/logs"
LOCK=$RES/eval/.step_${STEP_PAD}_5k_anchor.lock
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] another 5k anchor controller already holds $LOCK"
  exit 0
fi

TRAIN_STOPPED=0
TRAIN_RESUMED=0

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

write_status() {
  local status=$1
  local detail=${2:-}
  python3 - "$EVAL/5k_anchor_control.json" "$status" "$detail" "$STEP" "$NUM_SAMPLES" "$LATENTS" "$INCEPTION_DIR" <<'PY'
import json, sys
from datetime import datetime, timezone
path, status, detail, step, n, latents, inception_dir = sys.argv[1:]
payload = {
    "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": status,
    "detail": detail,
    "step": int(step),
    "num_samples": int(n),
    "sample_latents": latents,
    "inception_eval_dir": inception_dir,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, ensure_ascii=False)
    f.write("\n")
PY
}

trainer_pgids() {
  # Use /proc argv matching instead of `ps | awk` substring matching.
  # The old awk command line contained the literal trainer path and could match
  # its own process group, causing the controller to TERM itself.
  python3 - "$TRAIN_SCRIPT" "$CFG" <<'PY'
import os, sys
train_script, cfg = sys.argv[1:]
pgids = set()
for pid_s in os.listdir('/proc'):
    if not pid_s.isdigit():
        continue
    try:
        raw = open(f'/proc/{pid_s}/cmdline', 'rb').read().split(b'\0')
        cmd = [x.decode('utf-8', 'ignore') for x in raw if x]
    except Exception:
        continue
    if len(cmd) < 4 or not os.path.basename(cmd[0]).startswith('python'):
        continue
    if cmd[1] != train_script:
        continue
    if not any(a == '--config' and i + 1 < len(cmd) and cmd[i + 1] == cfg for i, a in enumerate(cmd)):
        continue
    try:
        pgids.add(os.getpgid(int(pid_s)))
    except Exception:
        pass
for pgid in sorted(pgids):
    print(pgid)
PY
}

trainer_active() {
  [[ -n "$(trainer_pgids)" ]]
}

metric_eval_done() {
  python3 - "$METRICS_JSONL" "$STEP" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
step = int(sys.argv[2])
try:
    data = p.read_bytes()
except FileNotFoundError:
    raise SystemExit(1)
if len(data) > 32 * 1024 * 1024:
    data = data[-32 * 1024 * 1024:]
for raw in data.splitlines():
    try:
        obj = json.loads(raw)
    except Exception:
        continue
    if obj.get("type") == "eval" and int(obj.get("step", -1)) == step and obj.get("sample_status") == "ok":
        raise SystemExit(0)
raise SystemExit(1)
PY
}

wait_for_checkpoint() {
  log "waiting_for_checkpoint step=$STEP path=$CKPT"
  write_status waiting_for_checkpoint "$CKPT"
  while true; do
    if [[ -s "$CKPT" && ! -e "$RES/checkpoints/.step_${STEP_PAD}.pt.tmp" ]]; then
      local s1 s2
      s1=$(stat -c '%s' "$CKPT")
      sleep 10
      s2=$(stat -c '%s' "$CKPT")
      if [[ "$s1" == "$s2" && "$s1" -gt 1000000000 ]]; then
        log "checkpoint_ready step=$STEP size=$s2"
        return 0
      fi
      log "checkpoint_size_not_stable_yet step=$STEP size1=$s1 size2=$s2"
    fi
    sleep "$CHECK_INTERVAL"
  done
}

wait_for_small_eval() {
  if [[ "$WAIT_FOR_TRAINER_EVAL" != "1" ]]; then
    log "wait_for_trainer_eval_disabled"
    return 0
  fi
  log "waiting_for_trainer_small_eval step=$STEP"
  write_status waiting_for_trainer_small_eval "$EVAL/sample_latents.safetensors"
  while true; do
    if metric_eval_done || [[ -s "$EVAL/sample_latents.safetensors" ]]; then
      log "trainer_small_eval_ready step=$STEP"
      return 0
    fi
    sleep 10
  done
}

stop_trainer_for_gpu() {
  if [[ "$STOP_TRAINER" != "1" ]]; then
    log "stop_trainer_disabled"
    return 0
  fi
  mapfile -t pgids < <(trainer_pgids)
  if [[ "${#pgids[@]}" -eq 0 ]]; then
    log "trainer_not_active_before_sampling"
    return 0
  fi
  TRAIN_STOPPED=1
  write_status stopping_trainer "pgids=${pgids[*]}"
  log "stopping_trainer_for_5k_sampling pgids=${pgids[*]}"
  for pgid in "${pgids[@]}"; do
    kill -TERM -- "-$pgid" 2>/dev/null || true
  done

  local deadline=$((SECONDS + 180))
  while trainer_active; do
    if (( SECONDS >= deadline )); then
      log "trainer_still_active_after_TERM_sending_KILL"
      mapfile -t pgids < <(trainer_pgids)
      for pgid in "${pgids[@]}"; do
        kill -KILL -- "-$pgid" 2>/dev/null || true
      done
      break
    fi
    sleep 5
  done
  log "trainer_stopped_or_absent"
  sleep 5
}

start_trainer_after_sampling() {
  if [[ "$RESUME_TRAINER" != "1" ]]; then
    log "resume_trainer_disabled"
    return 0
  fi
  if trainer_active; then
    TRAIN_RESUMED=1
    log "trainer_already_active_skip_resume"
    return 0
  fi
  local train_log
  train_log="$EXP/logs/b3_train_resume_after_5k_step_${STEP_PAD}_$(date -u +%Y%m%dT%H%M%SZ).log"
  log "resuming_trainer config=$CFG log=$train_log"
  nohup python "$TRAIN_SCRIPT" --config "$CFG" > "$train_log" 2>&1 &
  local pid=$!
  echo "$pid" > "$EXP/fullcache_realdata.pid"
  TRAIN_RESUMED=1
  log "trainer_resume_started pid=$pid"
}

on_exit() {
  local st=$?
  trap - EXIT
  if [[ "$st" -ne 0 ]]; then
    log "anchor_controller_exit_status=$st"
    write_status controller_failed "exit_status=$st"
  fi
  if [[ "$TRAIN_STOPPED" == "1" && "$TRAIN_RESUMED" != "1" && "$RESUME_TRAINER" == "1" ]]; then
    log "exit_trap_resuming_trainer_after_failure"
    start_trainer_after_sampling || true
  fi
  exit "$st"
}
trap on_exit EXIT

log "5k_anchor_controller_start step=$STEP num_samples=$NUM_SAMPLES ckpt=$CKPT latents=$LATENTS"
write_status controller_start "step=$STEP"
wait_for_checkpoint
wait_for_small_eval
stop_trainer_for_gpu

if [[ -s "$LATENTS" && -s "$METADATA_JSON" ]]; then
  log "large_latents_already_exist skip_sampling latents=$LATENTS metadata=$METADATA_JSON"
else
  log "large_sampling_start latents=$LATENTS"
  write_status sampling_5k "$LATENTS"
  python "$SAMPLE_SCRIPT" \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --num-samples "$NUM_SAMPLES" \
    --sample-batch-size "$SAMPLE_BATCH_SIZE" \
    --sample-steps "$SAMPLE_STEPS" \
    --precision-mode "$PRECISION_MODE" \
    --seed "$SEED" \
    --output-latents "$LATENTS" \
    --metadata-json "$METADATA_JSON"
  log "large_sampling_done latents=$LATENTS metadata=$METADATA_JSON"
fi

start_trainer_after_sampling

if [[ "$START_CPU_EVAL" == "1" ]]; then
  mkdir -p "$INCEPTION_DIR"
  if [[ -s "$INCEPTION_DIR/inception_metrics.json" ]]; then
    log "inception_metrics_already_exist skip_cpu_eval metrics=$INCEPTION_DIR/inception_metrics.json"
  else
    log "starting_cpu_decode_inception_eval output_dir=$INCEPTION_DIR"
    write_status cpu_inception_started "$INCEPTION_DIR"
    nohup python "$DECODE_SCRIPT" \
      --sample-latents "$LATENTS" \
      --output-dir "$INCEPTION_DIR" \
      --num-real "$NUM_SAMPLES" \
      --real-seed "$REAL_SEED" \
      --decode-device cpu \
      --decode-batch-size "$DECODE_BATCH_SIZE" \
      --inception-device cpu \
      --inception-batch-size "$INCEPTION_BATCH_SIZE" \
      --torch-num-threads "$TORCH_NUM_THREADS" \
      --max-save-images 0 \
      --grid-count 0 \
      --no-save-npz \
      > "$EVAL/inception_eval_5k.log" 2>&1 &
    echo "$!" > "$EVAL/inception_eval_5k.pid"
    log "cpu_decode_inception_eval_started pid=$(cat "$EVAL/inception_eval_5k.pid") log=$EVAL/inception_eval_5k.log"
  fi
fi

write_status launched_cpu_eval_and_resumed_trainer "$LATENTS"
log "5k_anchor_controller_done step=$STEP"
