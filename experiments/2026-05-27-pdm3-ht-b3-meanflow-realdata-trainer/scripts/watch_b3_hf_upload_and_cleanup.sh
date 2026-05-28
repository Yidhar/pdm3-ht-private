#!/usr/bin/env bash
# Background launcher for B3 checkpoint/eval HF upload + local checkpoint cleanup.
#
# Defaults:
#   repo:            LAXMAYDAY/pdm3-ht-model-artifacts
#   checkpoint keep: archive every 10k steps to HF, keep current latest locally
#   cleanup:         delete non-archive old .pt files and uploaded old archive .pt files
#   eval upload:     generated/eval metadata only; raw real ImageNet images/NPZ are ignored
set -euo pipefail

EXP=${EXP:-/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer}
RES=${RES:-$EXP/results/fullcache_realdata_singleproc_template}
REPO_ID=${B3_HF_REPO_ID:-LAXMAYDAY/pdm3-ht-model-artifacts}
REMOTE_PREFIX=${B3_HF_REMOTE_PREFIX:-b3_meanflow_realdata/fullcache_b96}
ARCHIVE_EVERY=${B3_HF_ARCHIVE_EVERY:-10000}
CHECKPOINT_START_STEP=${B3_HF_CHECKPOINT_START_STEP:-20000}
EVAL_START_STEP=${B3_HF_EVAL_START_STEP:-30000}
STOP_STEP=${B3_HF_STOP_STEP:-1070000}
INTERVAL=${B3_HF_WATCH_INTERVAL:-300}

LOG_DIR="$EXP/logs"
mkdir -p "$LOG_DIR" "$RES"
LOG="$LOG_DIR/b3_hf_upload_cleanup_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "$LOG" > "$EXP/results/b3_hf_upload_cleanup_latest.logpath"

export HF_HUB_DISABLE_PROGRESS_BARS=${HF_HUB_DISABLE_PROGRESS_BARS:-1}
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}

exec >> "$LOG" 2>&1
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launcher_start exp=$EXP res=$RES repo=$REPO_ID remote_prefix=$REMOTE_PREFIX archive_every=$ARCHIVE_EVERY checkpoint_start=$CHECKPOINT_START_STEP eval_start=$EVAL_START_STEP stop_step=$STOP_STEP interval=$INTERVAL"

python3 "$EXP/scripts/watch_b3_hf_upload_and_cleanup.py" \
  --res "$RES" \
  --pid-file "$EXP/fullcache_realdata.pid" \
  --repo-id "$REPO_ID" \
  --repo-type model \
  --remote-prefix "$REMOTE_PREFIX" \
  --archive-every "$ARCHIVE_EVERY" \
  --checkpoint-start-step "$CHECKPOINT_START_STEP" \
  --eval-start-step "$EVAL_START_STEP" \
  --stop-step "$STOP_STEP" \
  --interval "$INTERVAL"

