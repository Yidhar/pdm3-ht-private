#!/usr/bin/env bash
set -euo pipefail
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache
OUT=/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full
SUMMARY="$OUT/build_summary.json"
VERIFY="$EXP/results/full_cache_verify.json"
RECORD="$EXP/notes/EXPERIMENT_RECORD.md"

echo "[finalize] $(date -Is) waiting for full cache run pid if present"
if [[ -f "$EXP/run_full_cache.pid" ]]; then
  PID=$(cat "$EXP/run_full_cache.pid")
  while kill -0 "$PID" 2>/dev/null; do
    sleep 300
  done
fi

echo "[finalize] $(date -Is) run process ended; collecting status"
python - <<'PY' > "$EXP/results/full_cache_final_brief.json"
import json, pathlib, glob, os, time
EXP=pathlib.Path('/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache')
OUT=pathlib.Path('/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full')
summary_path=OUT/'build_summary.json'
verify_path=EXP/'results/full_cache_verify.json'
progress_path=OUT/'progress.json'
brief={'time': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'output_dir': str(OUT)}
for name,path in [('summary',summary_path),('verify',verify_path),('progress',progress_path)]:
    if path.exists():
        try:
            obj=json.loads(path.read_text())
        except Exception as e:
            obj={'read_error':repr(e)}
        brief[name]={k:obj.get(k) for k in ['ok','status','encoded_total','encoded_scanned','total_samples','expected_total','target_total','num_shards_total','num_files','total_bytes','elapsed_seconds','new_samples_per_sec','eta_hms','event']}
    else:
        brief[name]={'missing': True}
brief['safetensor_files']=len(list(OUT.glob('*.safetensors')))
brief['dir_bytes']=sum(p.stat().st_size for p in OUT.glob('*') if p.is_file()) if OUT.exists() else 0
print(json.dumps(brief, indent=2, sort_keys=True))
PY

BRIEF=$(cat "$EXP/results/full_cache_final_brief.json")
{
  echo
  echo "## $(date -Iseconds) — full cache finalizer"
  echo
  echo '```json'
  echo "$BRIEF"
  echo '```'
} >> "$RECORD"

find "$EXP" /workspace/PDM/external/PAE/pae_with_generator -type d -name '__pycache__' -prune -print -exec rm -rf {} + 2>/dev/null || true

echo "[finalize] done; brief=$EXP/results/full_cache_final_brief.json"
