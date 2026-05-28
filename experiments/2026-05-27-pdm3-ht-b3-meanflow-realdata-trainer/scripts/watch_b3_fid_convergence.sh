#!/usr/bin/env bash
# Watch B3 MeanFlow trainer eval samples and run CPU-only PAE decode + Inception FID/MMD/KID.
# Intent: keep FD-loss deferred; measure how far the current PAE+LightningDiT+B3 MeanFlow path converges.
set -euo pipefail
EXP=${EXP:-/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer}
RES=${RES:-$EXP/results/fullcache_realdata_singleproc_template}
REAL_CACHE=${REAL_CACHE:-/workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors}
INTERVAL=${B3_FID_WATCH_INTERVAL:-180}
START_STEP=${B3_FID_WATCH_START_STEP:-30000}
STOP_STEP=${B3_FID_WATCH_STOP_STEP:-1070000}
REAL_SEED=${B3_FID_REAL_SEED:-20260528}
LOG_DIR="$EXP/logs"
mkdir -p "$LOG_DIR" "$RES/eval"
LOG="$LOG_DIR/b3_fid_convergence_watch_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "$LOG" > "$EXP/results/b3_fid_convergence_watch_latest.logpath"
exec >> "$LOG" 2>&1

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher_start exp=$EXP start_step=$START_STEP stop_step=$STOP_STEP interval=$INTERVAL real_cache=$REAL_CACHE"

summarize_trend() {
  python3 - <<'PY'
import json, re
from pathlib import Path
EXP=Path('/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer')
RES=EXP/'results/fullcache_realdata_singleproc_template'
rows=[]
for p in sorted((RES/'eval').glob('step_*/**/inception_metrics.json')):
    m=re.search(r'step_(\d+)', str(p))
    if not m: continue
    try:
        o=json.loads(p.read_text())
    except Exception:
        continue
    metrics=o.get('metrics') if isinstance(o.get('metrics'), dict) else o
    gen_img=o.get('generated_image_summary') if isinstance(o.get('generated_image_summary'), dict) else {}
    real_img=o.get('real_image_summary') if isinstance(o.get('real_image_summary'), dict) else {}
    rows.append({
        'step': int(m.group(1)),
        'fid': metrics.get('fid'),
        'inception_mmd_rbf': metrics.get('inception_mmd_rbf'),
        'inception_kid_poly3': metrics.get('inception_kid_poly3'),
        'num_generated': metrics.get('num_generated', o.get('num_generated_total')),
        'num_real': metrics.get('num_real', o.get('num_real_total')),
        'generated_image_std': o.get('generated_image_std', gen_img.get('std')),
        'real_image_std': o.get('real_image_std', real_img.get('std')),
        'metrics_path': str(p),
    })
# Prefer canonical full-64 inception_eval dirs over smoke dirs if duplicated.
best={}
for r in rows:
    old=best.get(r['step'])
    if old is None:
        best[r['step']]=r
    elif ('smoke' in old['metrics_path'] and 'smoke' not in r['metrics_path']):
        best[r['step']]=r
    elif ('/inception_eval/' in r['metrics_path'] and '/inception_eval/' not in old['metrics_path']):
        best[r['step']]=r
rows=[best[k] for k in sorted(best)]
out=RES/'eval'/'fid_convergence_summary.json'
out.write_text(json.dumps({'rows': rows}, indent=2, ensure_ascii=False)+"\n")
md=RES/'eval'/'fid_convergence_summary.md'
lines=['# B3 MeanFlow FID convergence summary\n', '\n', 'FD-loss is deferred; these are image-space Inception diagnostics for the current PAE+LightningDiT+B3 MeanFlow path. Current runs use small 64 generated / matched real samples unless noted, so treat as convergence diagnostics, not official 50k FID.\n', '\n', '| step | FID ↓ | MMD ↓ | KID ↓ | n_gen | n_real | gen_std | real_std |\n', '|---:|---:|---:|---:|---:|---:|---:|---:|\n']
for r in rows:
    def fmt(x):
        return '' if x is None else (f'{x:.6f}' if isinstance(x,(int,float)) else str(x))
    lines.append(f"| {r['step']} | {fmt(r['fid'])} | {fmt(r['inception_mmd_rbf'])} | {fmt(r['inception_kid_poly3'])} | {r.get('num_generated','')} | {r.get('num_real','')} | {fmt(r.get('generated_image_std'))} | {fmt(r.get('real_image_std'))} |\n")
md.write_text(''.join(lines))
print(json.dumps({'summary_json': str(out), 'summary_md': str(md), 'num_rows': len(rows), 'latest': rows[-1] if rows else None}, ensure_ascii=False))
PY
}

while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  latest_step=$(python3 - <<'PY'
import json
from pathlib import Path
p=Path('/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/metrics.jsonl')
last=0
try:
    for line in p.read_text().splitlines()[-500:]:
        try:o=json.loads(line)
        except Exception: continue
        if o.get('type')=='train_step': last=max(last,int(o.get('step',0)))
except Exception:
    pass
print(last)
PY
)
  echo "[$now] poll latest_step=$latest_step"

  for sample in "$RES"/eval/step_*/sample_latents.safetensors; do
    [[ -e "$sample" ]] || continue
    step_name=$(basename "$(dirname "$sample")")
    step=${step_name#step_}
    step_int=$((10#$step))
    if (( step_int < START_STEP || step_int > STOP_STEP )); then
      continue
    fi
    outdir="$(dirname "$sample")/inception_eval"
    metrics="$outdir/inception_metrics.json"
    running_marker="$outdir/.running"
    done_marker="$outdir/.done"
    if [[ -s "$metrics" || -e "$done_marker" || -e "$running_marker" ]]; then
      continue
    fi
    mkdir -p "$outdir"
    touch "$running_marker"
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eval_start step=$step sample=$sample outdir=$outdir"
    set +e
    python3 "$EXP/scripts/decode_and_inception_eval_step.py" \
      --sample-latents "$sample" \
      --output-dir "$outdir" \
      --real-image-cache "$REAL_CACHE" \
      --num-real 0 \
      --real-seed "$REAL_SEED" \
      --decode-device cpu \
      --decode-batch-size 2 \
      --inception-device cpu \
      --inception-batch-size 16 \
      --torch-num-threads 8 \
      --max-save-images 256 \
      --grid-count 64 \
      --no-save-npz
    rc=$?
    set -e
    rm -f "$running_marker"
    if [[ $rc -eq 0 && -s "$metrics" ]]; then
      touch "$done_marker"
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eval_done step=$step metrics=$metrics"
      summarize_trend || true
    else
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] eval_failed step=$step rc=$rc"
      rm -f "$done_marker"
    fi
  done

  train_pid=$(cat "$EXP/fullcache_realdata.pid" 2>/dev/null || true)
  if [[ -n "$train_pid" ]] && ! ps -p "$train_pid" >/dev/null 2>&1; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] train_pid_not_alive pid=$train_pid exiting"
    summarize_trend || true
    exit 0
  fi
  if [[ "$latest_step" =~ ^[0-9]+$ ]] && (( latest_step >= STOP_STEP )); then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] reached_stop_step latest_step=$latest_step exiting"
    summarize_trend || true
    exit 0
  fi
  sleep "$INTERVAL"
done
