#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$EXP_DIR/logs" "$EXP_DIR/results"
LOG="$EXP_DIR/logs/flux2_from_cropped_follow_full_setsid_$(date -u +%Y%m%dT%H%M%SZ).outer.log"
echo "$LOG" > "$EXP_DIR/results/flux2_from_cropped_follow_full_outer.logpath"
setsid env \
  PYTHONFAULTHANDLER=1 \
  MALLOC_ARENA_MAX=2 \
  PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
  BATCH_SIZE="${BATCH_SIZE:-256}" \
  FLUX2_FOLLOW_MIN_GAP="${FLUX2_FOLLOW_MIN_GAP:-131072}" \
  FLUX2_FOLLOW_POLL_SEC="${FLUX2_FOLLOW_POLL_SEC:-60}" \
  FLUX2_FOLLOW_MAX_PASSES="${FLUX2_FOLLOW_MAX_PASSES:-30}" \
  bash "$SCRIPT_DIR/run_flux2_from_cropped_follow_full.sh" > "$LOG" 2>&1 < /dev/null &
echo $! > "$EXP_DIR/flux2_from_cropped_follow_full.pid"
echo "launched pid=$(cat "$EXP_DIR/flux2_from_cropped_follow_full.pid") outer_log=$LOG"
