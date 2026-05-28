#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG="$EXP_DIR/logs/flux2_full_cache_setsid_$(date -u +%Y%m%dT%H%M%SZ).outer.log"
echo "$LOG" > "$EXP_DIR/results/flux2_full_cache_outer.logpath"
setsid env PYTHONFAULTHANDLER=1 MALLOC_ARENA_MAX=2 bash "$SCRIPT_DIR/run_flux2_full_cache.sh" > "$LOG" 2>&1 < /dev/null &
echo $! > "$EXP_DIR/flux2_full_cache.pid"
echo "launched pid=$(cat "$EXP_DIR/flux2_full_cache.pid") outer_log=$LOG"
