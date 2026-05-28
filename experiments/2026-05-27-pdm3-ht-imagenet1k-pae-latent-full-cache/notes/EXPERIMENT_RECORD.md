# Experiment Record — Full ImageNet-1k PAE latent cache

## 2026-05-27 — setup/audit

- User directive: “把这个做好 完成full cache”.
- Switchyard bridge requested by AGENTS instructions but unavailable in container: `switchyard: command not found`; proceeding locally.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB VRAM, idle at start.
- Disk: `/workspace` has ~111T free; final latent cache estimate ~42GB, safe.
- Existing validated outputs retained:
  - `imagenet256_train_smoke64`
  - `imagenet256_train_smoke1024`
  - `batch_saturation_2026-05-27`

## Implementation notes

Production builder requirements:

- resume by scanning completed safetensors shards;
- source skip = completed samples;
- atomic temp save → verify → rename;
- JSONL manifest + JSON progress;
- `bf16` PAE backbone and `bf16` latent save;
- preprocess only one ADM center crop per image, generate flip at batch/GPU level;
- optional threaded preprocessing/prefetch;
- `--hard-exit` available for HF streaming finalizer issues.


## 2026-05-27 — production builder smoke

Commands/logs:

- `logs/001_access_check.log` — HF streaming access check PASS.
- `logs/002_pae_self_test.log` — PAE random uint8 encode self-test PASS after patching CUDA memory-stat helper to set current device before reset/sync.
- `logs/010_smoke4096_builder.log` — optimized production builder smoke on 4096 ImageNet train samples.

Smoke output:

- Directory: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full_smoke4096_opt`
- Shards: 1 × `latents_rank00_shard000000.safetensors`
- Samples: 4096
- Schema: `latents` BF16 `[4096,32,16,16]`, `latents_flip` BF16 `[4096,32,16,16]`, `labels` I64 `[4096]`
- SHA256 recorded in manifest/build summary.
- End-to-end builder throughput including PAE load/HF startup: ~36.38 samples/s.
- CUDA peak allocated: ~12.38 GiB.
- Fast verifier PASS: first/middle/last `ImgLatentDataset` reads work with `latent_norm=False`.

Compared with the earlier serial builder (~8-10 samples/s), the new one-crop + GPU flip + threaded prefetch builder is substantially faster on smoke. Full-run ETA from conservative smoke average: ~9.8 h; sustained post-warmup intervals suggest lower is possible if HF streaming remains smooth.

## 2026-05-27 — full cache launched

Launch method:

```bash
setsid bash -lc "cd /workspace/PDM && exec bash /workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/scripts/run_full_cache.sh > /workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/logs/020_full_cache_nohup.log 2>&1" </dev/null >/dev/null 2>&1 &
```

Runtime files:

- Full output: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full`
- PID file: `run_full_cache.pid`
- Main log: `logs/020_full_cache_nohup.log`
- Progress: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full/progress.json`
- Manifest: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full/manifest.jsonl`
- Finalizer watcher: `finalize_full_cache.pid`, log `logs/030_full_cache_finalizer.log`

Checkpoint around 2026-05-27T16:40:00Z:

- Encoded: 31,744 / 1,281,167
- Published shards: 7
- Directory size: ~899M
- Average throughput so far: ~36.08 samples/s
- Recent interval throughput: ~41.76 samples/s
- ETA from progress JSON: ~09:37:08 remaining
- CUDA peak allocated: ~12.38 GiB

The run is resume-safe; if interrupted, rerun `scripts/run_full_cache.sh` and it will skip completed shard samples.

## 2026-05-27 UTC — Cache I/O audit after user concern

Question raised: current cache script may be repeatedly loading / network-bound; clarify whether it streams HF data directly into PAE and whether raw staging is needed.

Findings:

- The current full-cache process was started from `scripts/run_full_cache.sh`; it uses `datasets.load_dataset('ILSVRC/imagenet-1k', streaming=True)` and encodes directly into PAE latents. It does **not** first stage raw ImageNet files locally.
- Token logging in both the old smoke64 builder and the current full-cache builder was misleading: `HF_TOKEN` env is unset, but `huggingface-cli login` cache is valid. Patched scripts now resolve token as `env -> hf_cache -> none` and log `token_source=hf_cache token_set=True` without printing the token.
- Current throughput snapshot showed clear input-pipeline bottleneck:
  - end-to-end: ~37.4 samples/s
  - PAE encode-only: ~151.0 samples/s
  - GPU encode fraction of wall time: ~24.8%
  - conclusion: network/decode/preprocess supply is slower than PAE encoding.
- HF repo plan for raw-first staging:
  - train parquet shards: 294 files
  - train parquet bytes: 146,477,630,673 bytes (~146.5 GB decimal)
  - expanded uint8 256-crop cache would be ~251.9 GB, so local HF parquet mirror is a better raw stage than expanded tensor raw.
  - bf16 latent+flip output estimate is ~42.0 GB.

Actions completed:

- Added raw-first mirror script:
  - `scripts/download_imagenet1k_hf_parquet_snapshot.py`
  - `scripts/run_download_train_parquet.sh`
  - `scripts/run_download_train_parquet_dryrun.sh`
- Added local-parquet encode runner:
  - `scripts/run_full_cache_from_local_parquet.sh`
- Patched `scripts/build_pae_latents_full_cache.py`:
  - `--data-files-glob` loads local parquet via `load_dataset('parquet', data_files=..., streaming=True)`.
  - source info and token source are written to metadata/progress for future runs.
- Patched old smoke script token reporting:
  - `experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-build/scripts/build_pae_latents_from_hf_imagenet.py`
- Wrote audit JSON:
  - `results/cache_io_audit.json`
- Wrote dry-run plan:
  - `results/imagenet1k_train_parquet_snapshot_plan.json`
- Logs:
  - `logs/041_raw_parquet_dryrun.log`
  - `logs/043_access_check_token_fallback_hard_exit.log`
  - `logs/044_cache_io_audit.log`

Important operational note:

- The current running full-cache Python process was launched before these patches, so the patches affect future resumes/runs, not the already-running process.
- I did not start the 146.5 GB raw-parquet mirror concurrently with the active streaming cache, because that would compete for the same network bottleneck. If switching strategy, stop/pause the streaming job first or let it finish.

## 2026-05-27T17:23:47+00:00 — raw-first restart launched

Stopped previous HF streaming run and stale finalizer. Launched raw-first chain.

- chain pid: `1595701`
- chain log: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/logs/046_raw_first_restart_chain_20260527T172346Z.log`
- launch JSON: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/results/raw_first_restart_launch.json`
- stages: download local parquet -> order audit -> local parquet PAE cache resume -> verify

## 2026-05-27T18:08:22Z — raw-first status checkpoint

```json
{
  "avg_new_samples_per_sec": 38.05348333200933,
  "completed_at_start": 122880,
  "cuda_peak_allocated_mb": 12375.349609375,
  "data_files_count": 294,
  "download_elapsed_sec": 1446.5228960514069,
  "download_finished_utc": "2026-05-27T17:47:58+00:00",
  "download_ok": true,
  "durable_samples": 163840,
  "durable_shards": 40,
  "encoded_total_progress": 166912,
  "eta_hms": "08:08:01",
  "interval_samples_per_sec": 75.06507459218206,
  "last_batch_samples_per_sec": 150.61392724089106,
  "new_encoded_this_run": 44032,
  "order_audit_ok": true,
  "order_audit_time_utc": "2026-05-27T17:48:04+00:00",
  "pae_cache_running": true,
  "raw_bytes": 146477630673,
  "raw_files": 294,
  "remaining_to_target": 1114255,
  "source_loader": "local_parquet",
  "target_total": 1281167,
  "time_utc": "2026-05-27T18:08:22Z"
}
```


## 2026-05-27T18:15Z live bottleneck diagnosis
- Current full-cache job remains running; no interruption performed.
- Diagnosis: input_preprocess_bound_not_gpu_bound.
- Throughput: end-to-end avg 38.63 img/s, last GPU encode batch 149.26 img/s.
- Wall split: GPU encode 25.6%, H2D+D2H 0.4%, non-encode/input/save 74.0%.
- Recent shard save avg: 1.298s per 4096 samples.
- Primary bottleneck: local parquet/HF iterator + JPEG decode + PIL crop/resize + CPU batch assembly; GPU utilization jumps because GPU receives a batch, encodes quickly, then waits for next CPU-preprocessed batch.
- Artifact: results/bottleneck_diagnosis_live_20260527T1815Z.json
