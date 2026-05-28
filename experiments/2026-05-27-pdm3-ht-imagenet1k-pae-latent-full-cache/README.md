# Full ImageNet-1k PAE Latent Cache Experiment

This experiment builds the full ImageNet-1k train latent cache for `PAE_DINOv2L_d32`.

## Main files

- `scripts/build_pae_latents_full_cache.py` — resumable production cache builder.
- `scripts/verify_pae_latent_cache.py` — fast full-cache verifier / summary.
- `scripts/run_smoke_full_cache_builder.sh` — production-script smoke run.
- `scripts/run_full_cache.sh` — full ImageNet train cache run.
- `configs/full_cache.env` — common paths and hyperparameters.
- `notes/EXPERIMENT_RECORD.md` — chronological command/result log.
- `logs/` — command logs.
- `results/` — copied summaries / benchmark summaries.

## Output directories

- Smoke: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full_smoke4096_opt`
- Full: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full`

## Monitor full run

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache
OUT=/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full
cat "$EXP/run_full_cache.pid"
tail -f "$EXP/logs/020_full_cache_nohup.log"
cat "$OUT/progress.json" | python -m json.tool
nvidia-smi
```

## Raw-first / offline parquet path

The original full-cache runner streams HF ImageNet examples directly into PAE. This is correct but network/decode/preprocess-bound on the current machine. A raw-first path was added:

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache

# Plan only: lists expected train parquet shards and bytes.
$EXP/scripts/run_download_train_parquet_dryrun.sh 2>&1 | tee $EXP/logs/raw_parquet_dryrun.log

# Full raw mirror: downloads 294 train parquet files (~146.5 GB) with parallel HF workers.
$EXP/scripts/run_download_train_parquet.sh 2>&1 | tee $EXP/logs/raw_parquet_download.log

# Offline/local PAE encoding from mirrored parquet.
$EXP/scripts/run_full_cache_from_local_parquet.sh 2>&1 | tee $EXP/logs/full_cache_from_local_parquet.log
```

Local parquet glob used by the offline runner:

```text
/workspace/PDM/data/raw/imagenet1k_hf_parquet/data/train-*.parquet
```

Prefer mirroring HF parquet over saving expanded uint8 crops: train parquet is ~146.5 GB, while expanded `[N,3,256,256] uint8` crops would be ~251.9 GB.
