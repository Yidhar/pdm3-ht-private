#!/usr/bin/env bash
set -euo pipefail
cd /workspace/PDM

EXP="experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer"
RUN="${RUN:-fitprobe_100m_subset64k_b384_lr5e4}"
CFG="${CFG:-$EXP/configs/b3_meanflow_realdata_100m_subset64k_b384_lr5e4_fitprobe.yaml}"
OUT="${OUT:-$EXP/results/$RUN}"
PIDFILE="${PIDFILE:-$EXP/$RUN.pid}"
TARGET_STEP="${TARGET_STEP:?set TARGET_STEP, e.g. 6000}"
NUM_SAMPLES="${NUM_SAMPLES:-2000}"
NFE="${NFE:-32}"
REAL_MAX="${REAL_MAX:-65536}"
MODES="${MODES:-ema raw}"
MAX_STEPS="${MAX_STEPS:-20000}"
REAL_SEED="${REAL_SEED:-20260530}"
SAMPLE_BATCH_SIZE="${SAMPLE_BATCH_SIZE:-128}"
DECODE_BATCH_SIZE="${DECODE_BATCH_SIZE:-32}"
INCEPTION_BATCH_SIZE="${INCEPTION_BATCH_SIZE:-128}"
GPU_FREE_LIMIT_MB="${GPU_FREE_LIMIT_MB:-15000}"
POLL_SEC="${POLL_SEC:-60}"
RESTART_AFTER="${RESTART_AFTER:-auto}"   # auto|1|0

STEP_PADDED="$(printf '%08d' "$TARGET_STEP")"
RUNLOG="$EXP/logs/${RUN}_step${TARGET_STEP}_${NUM_SAMPLES}_nfe${NFE}_${MODES// /-}_anchor_$(date -u +%Y%m%dT%H%M%SZ).log"
CKPT="$OUT/checkpoints/step_${STEP_PADDED}.pt"
EVAL_DIR="$OUT/eval/step_${STEP_PADDED}"
LOCKDIR="$EXP/${RUN}_anchor_eval.lock"
TRAIN_STOPPED=0
TRAIN_RESTARTED=0

log(){ printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$RUNLOG"; }
gpu_mem_used(){ nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo 0; }

should_restart(){
  case "$RESTART_AFTER" in
    1|true|yes) return 0 ;;
    0|false|no) return 1 ;;
    auto)
      if (( TARGET_STEP < MAX_STEPS )); then return 0; else return 1; fi
      ;;
    *) log "unknown RESTART_AFTER=$RESTART_AFTER; using auto"; if (( TARGET_STEP < MAX_STEPS )); then return 0; else return 1; fi ;;
  esac
}

acquire_lock(){
  for i in $(seq 1 1440); do
    if mkdir "$LOCKDIR" 2>/dev/null; then
      echo $$ > "$LOCKDIR/pid"
      echo "$TARGET_STEP" > "$LOCKDIR/target_step"
      log "acquired eval lock $LOCKDIR"
      return 0
    fi
    local holder="unknown"
    [[ -f "$LOCKDIR/pid" ]] && holder="$(cat "$LOCKDIR/pid" 2>/dev/null || echo unknown)"
    log "eval lock busy holder=$holder; waiting"
    sleep 10
  done
  log "failed to acquire eval lock after long wait"
  return 1
}

release_lock(){
  if [[ -d "$LOCKDIR" ]] && [[ "$(cat "$LOCKDIR/pid" 2>/dev/null || true)" == "$$" ]]; then
    rm -rf "$LOCKDIR" || true
    log "released eval lock"
  fi
}

terminate_train(){
  if [[ ! -f "$PIDFILE" ]]; then log "pidfile missing; no train to terminate"; return 0; fi
  local pid; pid="$(cat "$PIDFILE" || true)"
  if [[ -z "${pid:-}" ]]; then log "empty pidfile"; return 0; fi
  if ! ps -p "$pid" >/dev/null 2>&1; then log "train parent pid $pid not alive"; return 0; fi
  log "terminating training process group pid=$pid after checkpoint $CKPT"
  kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for i in $(seq 1 90); do
    if ! ps -p "$pid" >/dev/null 2>&1 && ! pgrep -P "$pid" >/dev/null 2>&1; then
      log "training terminated cleanly"
      TRAIN_STOPPED=1
      return 0
    fi
    sleep 1
  done
  log "training still alive after TERM; sending KILL"
  kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  sleep 3
  TRAIN_STOPPED=1
}

wait_gpu_free(){
  local limit_mb="${1:-15000}"
  for i in $(seq 1 180); do
    local mem; mem="$(gpu_mem_used)"
    log "gpu memory used ${mem} MB"
    if [[ "$mem" =~ ^[0-9]+$ ]] && (( mem < limit_mb )); then return 0; fi
    sleep 2
  done
  log "warning: GPU memory did not drop below ${limit_mb}MB; continuing anyway"
}

restart_train(){
  if ! should_restart; then log "not restarting training after target_step=$TARGET_STEP max_steps=$MAX_STEPS restart_after=$RESTART_AFTER"; return 0; fi
  if [[ "$TRAIN_RESTARTED" == "1" ]]; then return 0; fi
  local newlog="$EXP/logs/${RUN}_resume_after_step${TARGET_STEP}_anchor_$(date -u +%Y%m%dT%H%M%SZ).log"
  log "restarting training from auto resume; log=$newlog"
  setsid bash -lc "cd /workspace/PDM && python3 '$EXP/scripts/train_b3_meanflow_realdata.py' --config '$CFG' > '$newlog' 2>&1" &
  local newpid=$!
  echo "$newpid" > "$PIDFILE"
  TRAIN_RESTARTED=1
  log "restarted train parent pid=$newpid"
}

on_exit(){
  local code=$?
  if [[ "$TRAIN_STOPPED" == "1" && "$TRAIN_RESTARTED" != "1" ]]; then
    if should_restart; then
      log "controller exiting code=$code with train stopped; attempting restart"
      restart_train || true
    fi
  fi
  release_lock || true
  log "controller exit code=$code"
}
trap on_exit EXIT

run_anchor_mode(){
  local mode="$1"
  local ema_arg=()
  case "$mode" in
    ema) ema_arg=(--use-ema) ;;
    raw) ema_arg=(--no-use-ema) ;;
    *) log "unsupported mode=$mode; expected ema or raw"; return 2 ;;
  esac
  local latents="$EVAL_DIR/sample_latents_${NUM_SAMPLES}_nfe${NFE}_subsetlabels_${mode}.safetensors"
  local metad="$EVAL_DIR/sample_latents_${NUM_SAMPLES}_nfe${NFE}_subsetlabels_${mode}.json"
  local fid_dir="$EVAL_DIR/inception_${NUM_SAMPLES}_subsetreal_nfe${NFE}_${mode}"
  log "sampling ${NUM_SAMPLES} latents mode=${mode} step=${TARGET_STEP} NFE=${NFE}"
  python3 "$EXP/scripts/sample_b3_meanflow_eval_latents.py" \
    --config "$CFG" \
    --checkpoint "$CKPT" \
    --output-latents "$latents" \
    --metadata-json "$metad" \
    --num-samples "$NUM_SAMPLES" \
    --sample-batch-size "$SAMPLE_BATCH_SIZE" \
    --sample-steps "$NFE" \
    --precision-mode bf16_autocast \
    --device cuda \
    "${ema_arg[@]}" \
    --label-source latent-subset \
    --label-subset-max-samples "$REAL_MAX" \
    --overwrite 2>&1 | tee -a "$RUNLOG"

  log "decode+FID mode=${mode} generated=${NUM_SAMPLES}, real first-${REAL_MAX}, NFE=${NFE}"
  python3 "$EXP/scripts/decode_and_inception_eval_step.py" \
    --sample-latents "$latents" \
    --output-dir "$fid_dir" \
    --real-image-cache /workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors \
    --real-max-samples "$REAL_MAX" \
    --num-real "$NUM_SAMPLES" \
    --real-seed "$REAL_SEED" \
    --decode-device cuda \
    --decode-dtype bf16 \
    --decode-batch-size "$DECODE_BATCH_SIZE" \
    --inception-device cuda \
    --inception-batch-size "$INCEPTION_BATCH_SIZE" \
    --max-save-images 64 \
    --grid-count 64 \
    --skip-compact 2>&1 | tee -a "$RUNLOG"

  log "anchor metrics summary mode=${mode}"
  python3 - <<PY 2>&1 | tee -a "$RUNLOG"
import json
from pathlib import Path
p=Path('$fid_dir/inception_metrics.json')
rec=json.loads(p.read_text())
metrics=rec.get('metrics') or {}
print(json.dumps({
  'target_step': int('$TARGET_STEP'),
  'mode': '$mode',
  'status': rec.get('status'),
  'fid': metrics.get('fid'),
  'kid': metrics.get('inception_kid_poly3'),
  'mmd': metrics.get('inception_mmd_rbf'),
  'num_generated': metrics.get('num_generated'),
  'num_real': metrics.get('num_real'),
  'real_max_samples': rec.get('real_max_samples'),
  'elapsed_sec': rec.get('elapsed_sec'),
}, indent=2, sort_keys=True))
PY
}

mkdir -p "$EVAL_DIR" "$EXP/logs"
log "controller start run=$RUN target_step=$TARGET_STEP num_samples=$NUM_SAMPLES nfe=$NFE real_max=$REAL_MAX modes=$MODES restart_after=$RESTART_AFTER max_steps=$MAX_STEPS"
python3 -m py_compile "$EXP/scripts/sample_b3_meanflow_eval_latents.py" "$EXP/scripts/decode_and_inception_eval_step.py" 2>&1 | tee -a "$RUNLOG"
log "waiting for checkpoint $CKPT"
while [[ ! -s "$CKPT" ]]; do
  python3 - <<PY 2>/dev/null | tee -a "$RUNLOG" || true
import json, statistics
from pathlib import Path
p=Path('/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/$RUN/metrics.jsonl')
rows=[]
if p.exists():
    for line in p.open():
        try:o=json.loads(line)
        except Exception: continue
        if o.get('type')=='train_step': rows.append(o)
if rows:
    last=rows[-1]
    tail=rows[-min(200,len(rows)):]
    losses=[float(r['loss']) for r in tail if isinstance(r.get('loss'),(int,float))]
    secs=[float(r.get('elapsed_sec',0)) for r in tail if isinstance(r.get('elapsed_sec'),(int,float))]
    avg=sum(secs)/len(secs) if secs else 0
    step=int(last.get('step',0) or 0)
    print(f"PROGRESS target=$TARGET_STEP step={step} loss={float(last.get('loss', float('nan'))):.6g} lr={float(last.get('optimizer_lr', float('nan'))):.6g} mean200={sum(losses)/len(losses):.6g} eta_min={max(0,int('$TARGET_STEP')-step)*avg/60:.2f}")
PY
  sleep "$POLL_SEC"
done
sleep 3
s1=$(stat -c%s "$CKPT")
sleep 5
s2=$(stat -c%s "$CKPT")
if [[ "$s1" != "$s2" ]]; then
  log "checkpoint size changed ${s1}->${s2}; waiting extra"
  sleep 10
fi
log "checkpoint ready: $CKPT size=$(stat -c%s "$CKPT")"
acquire_lock
terminate_train
wait_gpu_free "$GPU_FREE_LIMIT_MB"
for mode in $MODES; do
  run_anchor_mode "$mode"
done
restart_train
log "done; eval_dir=$EVAL_DIR"
