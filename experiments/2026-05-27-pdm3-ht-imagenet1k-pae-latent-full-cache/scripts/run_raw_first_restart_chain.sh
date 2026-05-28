#!/usr/bin/env bash
set -euo pipefail
source /workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/configs/full_cache.env
CHAIN_LOG_MARKER="[raw-first-chain]"
ORDER_AUDIT_JSON="$EXP_DIR/results/local_parquet_order_audit_before_resume.json"
RECORD="$EXP_DIR/notes/EXPERIMENT_RECORD.md"

echo "$CHAIN_LOG_MARKER start $(date -u --iso-8601=seconds) pid=$$"
echo "$CHAIN_LOG_MARKER exp=$EXP_DIR"
echo "$CHAIN_LOG_MARKER raw_dir=$IMAGENET1K_PARQUET_DIR"
echo "$CHAIN_LOG_MARKER raw_glob=$IMAGENET1K_TRAIN_PARQUET_GLOB"
echo "$CHAIN_LOG_MARKER latent_out=$FULL_OUT"

echo "$CHAIN_LOG_MARKER durable shard snapshot before raw-first restart"
python - <<'PY'
from pathlib import Path
from safetensors import safe_open
import json, time
out=Path('/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full')
files=sorted(out.glob('latents_rank00_shard*.safetensors'))
total=0
for p in files:
    with safe_open(str(p), framework='pt', device='cpu') as f:
        total += int(f.get_tensor('labels').numel())
payload={'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'num_shards': len(files), 'durable_samples': total, 'first': files[0].name if files else None, 'last': files[-1].name if files else None}
print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
Path('/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/results/durable_before_raw_first_restart.json').write_text(json.dumps(payload, indent=2, sort_keys=True))
PY

echo "$CHAIN_LOG_MARKER phase1 download local HF parquet $(date -u --iso-8601=seconds)"
"$EXP_DIR/scripts/run_download_train_parquet.sh"

echo "$CHAIN_LOG_MARKER phase2 verify local parquet order against existing cache $(date -u --iso-8601=seconds)"
python "$EXP_DIR/scripts/verify_local_parquet_order_against_cache.py" \
  --latent-dir "$FULL_OUT" \
  --data-files-glob "$IMAGENET1K_TRAIN_PARQUET_GLOB" \
  --split "$SPLIT" \
  --output-json "$ORDER_AUDIT_JSON"

echo "$CHAIN_LOG_MARKER phase3 resume PAE latent cache from local parquet $(date -u --iso-8601=seconds)"
"$EXP_DIR/scripts/run_full_cache_from_local_parquet.sh"

echo "$CHAIN_LOG_MARKER final summary $(date -u --iso-8601=seconds)"
python - <<'PY'
import json, pathlib, time
EXP=pathlib.Path('/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache')
OUT=pathlib.Path('/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full')
summary_path=OUT/'build_summary.json'
verify_path=EXP/'results/full_cache_verify.json'
progress_path=OUT/'progress.json'
brief={'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'output_dir': str(OUT)}
for name,path in [('summary',summary_path),('verify',verify_path),('progress',progress_path)]:
    if path.exists():
        try: obj=json.loads(path.read_text())
        except Exception as e: obj={'read_error':repr(e)}
        brief[name]={k:obj.get(k) for k in ['ok','status','encoded_total','encoded_scanned','total_samples','expected_total','target_total','num_shards_total','num_files','total_bytes','elapsed_sec','elapsed_seconds','avg_new_samples_per_sec','new_samples_per_sec','eta_hms','event','source_loader','data_files_count']}
    else:
        brief[name]={'missing': True}
brief['safetensor_files']=len(list(OUT.glob('*.safetensors')))
brief['dir_bytes']=sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
path=EXP/'results/raw_first_restart_final_brief.json'
path.write_text(json.dumps(brief, indent=2, sort_keys=True))
print(json.dumps(brief, indent=2, sort_keys=True), flush=True)
PY

{
  echo
  echo "## $(date -u --iso-8601=seconds) — raw-first full cache restart chain finished"
  echo
  echo "- raw parquet dir: \`$IMAGENET1K_PARQUET_DIR\`"
  echo "- local parquet glob: \`$IMAGENET1K_TRAIN_PARQUET_GLOB\`"
  echo "- order audit: \`$ORDER_AUDIT_JSON\`"
  echo "- final brief: \`$EXP_DIR/results/raw_first_restart_final_brief.json\`"
  echo
  echo '```json'
  cat "$EXP_DIR/results/raw_first_restart_final_brief.json"
  echo '```'
} >> "$RECORD"

find "$EXP_DIR" /workspace/PDM/external/PAE/pae_with_generator -type d -name '__pycache__' -prune -print -exec rm -rf {} + 2>/dev/null || true

echo "$CHAIN_LOG_MARKER done $(date -u --iso-8601=seconds)"
