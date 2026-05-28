#!/usr/bin/env bash
set -euo pipefail
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
INTERVAL=${B3_MONITOR_INTERVAL:-300}
OUT="$EXP/logs/b3_fullcache_monitor_$(date -u +%Y%m%dT%H%M%SZ).jsonl"
echo "$OUT" > "$EXP/results/b3_fullcache_monitor_latest.logpath"
while true; do
  TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  PID=$(cat "$EXP/fullcache_realdata.pid" 2>/dev/null || echo '')
  MET="$EXP/results/fullcache_realdata_singleproc_template/metrics.jsonl"
  CKPT_DIR="$EXP/results/fullcache_realdata_singleproc_template/checkpoints"
  PROC=""
  if [[ -n "$PID" ]]; then
    PROC=$(ps -p "$PID" -o pid=,stat=,etime=,%cpu=,%mem=,rss= 2>/dev/null | tr -s ' ' | sed 's/^ //') || true
  fi
  GPU=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | head -1 || true)
  LAST=$(grep '"type": "train_step"' "$MET" 2>/dev/null | tail -1 || true)
  FD=$(grep '"type": "fd_audit"' "$MET" 2>/dev/null | tail -1 || true)
  CKPTS=$(find "$CKPT_DIR" -maxdepth 1 \( -type f -o -type l \) -printf '%f:%s ' 2>/dev/null | sed 's/ $//' || true)
  python3 - <<PY >> "$OUT"
import json
print(json.dumps({
  'time_utc': '$TS',
  'pid': '$PID',
  'process': '''$PROC''',
  'gpu': '''$GPU''',
  'last_train_raw': '''$LAST''',
  'last_fd_raw': '''$FD''',
  'checkpoints': '''$CKPTS''',
}, ensure_ascii=False))
PY
  if [[ -z "$PID" ]] || ! ps -p "$PID" >/dev/null 2>&1; then
    exit 0
  fi
  sleep "$INTERVAL"
done
