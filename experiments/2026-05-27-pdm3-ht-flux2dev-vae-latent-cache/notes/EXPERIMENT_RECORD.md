# Experiment record — VAE-agnostic latent cache

## 2026-05-27 — Framing revision accepted

- PAE is no longer framed as an innovation component.
- PDM-3-HT novelty should be HT + MeanFlow per-patch integration / two-level decoupling.
- VAE backend should be swappable: PAE, FLUX.2 VAE, Qwen Image VAE.
- FID outcomes must be reported honestly:
  - A: all work, PAE best due ImageNet tuning;
  - B: all work, close numbers;
  - C: Qwen/FLUX fail or degrade, weakening VAE-agnostic claim.

## 2026-05-27 — FLUX.2 CPU shape probe

`AutoencoderKLFlux2` from `diffusers/FLUX.2-dev-bnb-4bit`, subfolder `vae`, was loadable in this environment.

Observed with one 256x256 input on CPU:

```json
{
  "latent_dist_type": "DiagonalGaussianDistribution",
  "mode_shape": [1, 32, 32, 32],
  "mode_dtype": "torch.float32",
  "mode_mean": -0.09721769392490387,
  "mode_std": 1.3858556747436523,
  "sample_shape": [1, 32, 32, 32]
}
```

Decision: cache deterministic `posterior.mode()` unless a downstream FLUX pipeline convention requires sampled latents.

## 2026-05-27 — Generic diffusers VAE cache builder CPU one-sample validation

Implemented:

```text
scripts/build_diffusers_vae_latents_from_cropped_cache.py
```

Validation command encoded 1 cropped ImageNet-256 sample with `AutoencoderKLFlux2` on CPU to avoid disturbing the running PAE GPU cache.

Result:

```json
{
  "ok": true,
  "encoded_total": 1,
  "first_latent_shape": [1, 32, 32, 32],
  "latent_mode": "mode",
  "model_dtype": "fp32",
  "save_dtype": "bf16",
  "encode_time_total_sec": 10.906434059143066,
  "output_dir": "/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_cpu_one_sample"
}
```

CPU speed is not representative. GPU smoke is still needed once it will not interfere with the active PAE pass.

## 2026-05-27 — VAE-agnostic trade-off / outcome taxonomy locked

User-level reporting rule recorded as a hard constraint: if PAE reaches e.g. FID 1.0 while Qwen-VAE/FLUX.2-VAE reach e.g. FID 1.5 because they are not ImageNet-256 tuned, we must report that honestly. This does **not** invalidate the experiment, but it changes the strength of the VAE-agnostic claim.

Outcome taxonomy for paper framing:

- **A / best**: PAE, Qwen-VAE, and FLUX.2-VAE all work; PAE is best because it was ImageNet-256 tuned. Use this to sell generality while acknowledging backend-specific tuning.
- **B / medium**: all three work with close numbers. This is the cleanest ablation.
- **C / worst**: Qwen/FLUX fail or degrade substantially. Then the honest conclusion is that PDM-3-HT is not yet truly VAE-agnostic and currently works reliably only on PAE.

Implementation implication: B3/MeanFlow loader and evaluation must keep `vae_backend`, latent shape, decoder, FID, and MMD separated in logs/results so backend-specific degradation cannot be accidentally hidden.

## 2026-05-27 — FLUX.2 GPU smoke8192 throughput, batch64 effective

Ran `run_flux2_smoke8192.sh`; due current config sourcing, attempted external `BATCH_SIZE=256` was overridden and the actual effective batch size was 64. This exposed an engineering issue: run scripts must preserve explicit env overrides for batch tuning.

Result summary:

```json
{
  "ok": true,
  "status": "complete",
  "encoded_total": 8192,
  "first_latent_shape": [
    64,
    32,
    32,
    32
  ],
  "avg_new_samples_per_sec": 47.50006533797097,
  "elapsed_sec": 172.4629180431366,
  "encode_time_total_sec": 152.17204332351685,
  "h2d_time_total_sec": 7.7229390144348145,
  "save_time_total_sec": 4.055973052978516,
  "cuda_peak_allocated_mb": 10665.30126953125,
  "num_new_shards": 2
}
```

Extrapolation at 47.500 samples/s (each sample writes original + flipped latent):

- 1,000,000 images: 5.85 h.
- ImageNet train 1,281,167 images: 7.49 h.
- FLUX.2 full latent+flip cache estimated size: 167.9 GB.

Next action: patch run scripts so explicit batch-size overrides work, then test larger batch sizes (128/256) before launching full cache.

## 2026-05-27 — Config override bug fixed for batch tuning

Observed during FLUX.2 smoke8192 that `BATCH_SIZE=256 scripts/run_flux2_smoke8192.sh` was silently reset to 64 because `configs/flux2dev_vae_cache.env` used unconditional `export BATCH_SIZE=64`. Patched the config to use shell-default form, e.g. `export BATCH_SIZE="${BATCH_SIZE:-64}"`, and did the same for output paths/device/dtype settings. This allows clean batch-size and output-dir sweeps without editing scripts.

## 2026-05-27 — FLUX.2 batch sweep completed


Measured batch-size points for `AutoencoderKLFlux2` latent cache from cropped ImageNet-256. Samples/sec counts images while encoding both original and horizontal-flip latents.

| point | target | avg samples/s | peak MB | est 1M h | est 1.281M h |
|---|---:|---:|---:|---:|---:|
| bs64_8192 | 8192 | 47.500 | 10665.3 | 5.85 | 7.49 |
| bs128_8192 | 8192 | 50.208 | 21001.3 | 5.53 | 7.09 |
| bs256_8192 | 8192 | 50.743 | 41673.3 | 5.47 | 7.01 |
| bs512_4096 | 4096 | 46.608 | 83016.7 | 5.96 | 7.64 |

Decision: **use batch size 256** for a dedicated FLUX.2 full cache run. Batch 128 is nearly as good and uses half the memory; batch 512 almost fills the GPU (~83GB) but is slower, so it is not worth the risk.

Machine-readable summary: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/results/flux2_batch_sweep_summary.json`

## 2026-05-27 — FLUX.2 full cache follow run launched

Launched FLUX.2 VAE full-cache in follow/resume mode while the cropped ImageNet-256 cache is still being produced.

- Script: `experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/scripts/run_flux2_from_cropped_follow_full.sh`
- Launcher: `experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/scripts/launch_flux2_from_cropped_follow_full_setsid.sh`
- PID: `1660571`
- Log: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/logs/flux2_from_cropped_follow_full_setsid_20260527T194322Z.outer.log`
- Output: `/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full`
- Batch size: `256`
- Expected latent: `[32, 32, 32]`, saved as bf16; deterministic posterior `mode()`.
- Reason: avoid waiting for crop cache completion and avoid a partial one-shot run. The follow loop resumes as new crop shards appear until all `1,281,167` train samples are encoded.
- GPU scheduling: PAE follow watcher remains SIGSTOPed so this FLUX.2 pass gets the GPU exclusively.


## HF FLUX.2 VAE latent dataset upload launched (2026-05-28T02:00:09+00:00)

- repo: `LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents` (https://huggingface.co/datasets/LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents)
- repo_type: `dataset`, private: `true`
- supervisor PID: `1688293`
- log: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/logs/hf_upload_flux2_vae_latents_20260528T015927Z.log`
- behavior: waits for full cache validation, then uploads only `latents_rank00_shard*.safetensors` plus `README.md`, `run_config.json`, `build_summary.json`, `progress.json`, `manifest.jsonl`, and `progress.jsonl`.
- explicit excludes: raw ImageNet parquet, cropped uint8 shards, PAE latents, checkpoints, logs, temp files.

## 2026-05-28 handoff document uploaded to private HF code repo

Created and uploaded a compact handoff bundle for running PAE-backed work on a second machine.

Local files:

```text
/workspace/PDM/HANDOFF_2026-05-28_PDM3_HT_PRIVATE_HF_AND_PAE_PARALLEL.md
/workspace/PDM/.hf_publish/handoff_bundle/README.md
/workspace/PDM/.hf_publish/handoff_bundle/state/handoff_state_20260528T022421Z.json
/workspace/PDM/.hf_publish/handoff_upload_result.json
```

Private HF code repo paths:

```text
https://huggingface.co/LAXMAYDAY/pdm3-ht-20260528-code/blob/main/HANDOFF_2026-05-28_PDM3_HT_PRIVATE_HF_AND_PAE_PARALLEL.md
https://huggingface.co/LAXMAYDAY/pdm3-ht-20260528-code/tree/main/handoff
```

Upload policy: handoff README, compact log snippets, and state JSON only. No raw ImageNet, no cropped uint8 cache, no PAE/FLUX latent shards, no checkpoints/model weights, no `/data`, no `/external`.

<!-- FLUX2_ROUTE_RECORD_COMPLETION_20260528 -->

## 2026-05-28 — FLUX.2 route record completion / missing entries backfilled

Audit note: the previous record stopped after the FLUX.2 latent upload supervisor was launched and the HF handoff bundle was created. The later engineering results existed locally but were not written into this experiment record. This section backfills the missing FLUX.2 VAE cache, public dataset upload, real ImageNet-256 reference stats, image-space eval wiring, and B3/MeanFlow short-run results.

Source files for this backfill are local result summaries under:

```text
experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/results/
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/
```

Important artifact policy is unchanged: keep code/config/docs/lightweight summaries in Git; keep large model/checkpoint/eval products in the HF artifact repo only when explicitly selected; do not commit raw ImageNet, cropped uint8 cache, latent shards, logs, full decoded PNG directories, checkpoints, or temporary files.

## 2026-05-28 — FLUX.2 full ImageNet-256 VAE latent cache completed

Source summary:

```text
experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/results/flux2_from_cropped_full_summary.json
```

Final status:

```json
{
  "ok": true,
  "status": "complete",
  "vae_backend": "AutoencoderKLFlux2_d32",
  "repo_id": "diffusers/FLUX.2-dev-bnb-4bit",
  "diffusers_class": "AutoencoderKLFlux2",
  "latent_mode": "mode",
  "encoded_total": 1281167,
  "target_total": 1281167,
  "num_shards_total": 313,
  "first_latent_shape": [256, 32, 32, 32],
  "finished_utc": "2026-05-28T02:55:44+00:00",
  "output_dir": "/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full"
}
```

Details:

- Input cache: `/workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors`.
- Output cache: `/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full`.
- Backend: `AutoencoderKLFlux2`, `diffusers/FLUX.2-dev-bnb-4bit`, subfolder `vae`.
- Posterior choice: deterministic `mode()`.
- Model compute dtype during encode: fp32; saved latent dtype: bf16.
- Stored tensors per shard: `labels`, `latents`, `latents_flip`.
- Latent shape per image: `[32, 32, 32]`, compared with PAE `[32, 16, 16]`; FLUX.2 route therefore needs backend-specific model/eval config and cannot reuse PAE shape assumptions.
- Full train set: `1,281,167` samples, `313` shards.
- Final shard: `latents_rank00_shard000312.safetensors`, `3215` samples, `[3215,32,32,32]`, bf16, validation ok.
- Upload-validation size later measured the dataset at `167,935,991,352` bytes (`167.936 GB`, decimal), as expected for original + flipped FLUX.2 latents.

Final builder/resume segment timing from the summary:

| metric | value |
|---|---:|
| new samples in final resume segment | 511,119 |
| elapsed sec in final segment | 10,514.968 |
| avg new samples/sec | 48.609 |
| encode sec total | 9,462.528 |
| H2D sec total | 253.846 |
| save sec total | 453.767 |
| CUDA peak allocated MB | 41,673.3 |

Interpretation: the full FLUX.2 latent cache is done and locally validated. The earlier batch sweep estimate of ~7 h for full ImageNet was realistic; the final summary reports the last resumed builder segment rather than a clean from-zero benchmark.

## 2026-05-28 — FLUX.2 VAE latent cache uploaded as public HF dataset

Source summary:

```text
experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/results/hf_flux2_vae_latents_upload_result_20260528.json
.hf_publish/flux2_vae_latents_upload_result.json
```

Final upload status supersedes the earlier "private upload launched" note:

```json
{
  "status": "complete",
  "repo_id": "LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents-public",
  "repo_type": "dataset",
  "private": false,
  "url": "https://huggingface.co/datasets/LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents-public",
  "started_utc": "2026-05-28T03:18:02+00:00",
  "finished_utc": "2026-05-28T03:23:32+00:00"
}
```

Validated contents:

| item | value |
|---|---:|
| samples | 1,281,167 |
| shards | 313 |
| bytes | 167,935,991,352 |
| GB decimal | 167.936 |
| first shard | `latents_rank00_shard000000.safetensors` |
| last shard | `latents_rank00_shard000312.safetensors` |
| bad files | 0 |
| temp files | 0 |

Allowed upload patterns were intentionally narrow:

```text
README.md
run_config.json
build_summary.json
progress.json
manifest.jsonl
progress.jsonl
latents_rank00_shard*.safetensors
```

Explicitly excluded from this dataset repo: raw ImageNet parquet, cropped uint8 cache, PAE latents, checkpoints, logs, temp files, and unrelated experiment outputs.

## 2026-05-28 — FLUX.2 full-cache trainer scan validated

Source:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_fullcache_dataset_scan/scan_summary.json
```

Trainer-facing dataset scan result:

```json
{
  "num_shards": 313,
  "total_samples": 1281167,
  "latent_shape": [32, 32, 32],
  "flip_prob": 0.5,
  "latent_norm": false,
  "latent_multiplier": 1.0,
  "skipped_files_count": 0
}
```

This confirms that the B3 real-data loader can see the complete FLUX.2 full cache, can use original/flip latents, and does not skip unreadable shards.

## 2026-05-28 — Real ImageNet-256 reference stats and FLUX image-space eval wiring completed

Reference-stat source summaries:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/imagenet256_refstats_full_summary.json
data/reference_stats/imagenet256_adm_train_inception2048/refstats_summary.json
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/real_imagenet256_fid_mmd_wiring_summary.json
```

Real ImageNet-256 reference stats:

| metric | value |
|---|---:|
| reference images | 1,281,167 |
| cropped shards | 313 |
| Inception feature dim | 2048 |
| TF32 | off |
| batch size | 1024 |
| elapsed sec | 2,592.585 |
| samples/sec | 494.166 |
| CUDA peak MB | 43,027.3 |
| covariance trace | 180.5874 |
| MMD/KID reference features | 8,192 x 2,048 |

Reference files:

```text
/workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_stats.npz
/workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_mmd8192_features.npz
/workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/refstats_summary.json
```

Eval wiring status:

- `eval_flux2vae_imagespace.py` is connected to a real FID/MMD/KID command path rather than a dummy placeholder.
- `decode_flux2vae_latents.py` decodes sampled FLUX.2 latents to PNG using the FLUX.2 VAE decoder.
- Decode manifest policy was changed to `first_last` so eval does not write enormous full image manifests by default.
- Reference stats remain local data artifacts and are not to be committed to Git.

4-image wiring smoke source:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_imagespace_fid_mmd_smoke_summary.json
```

Smoke result:

```json
{
  "practical_gate_pass": true,
  "final_step": 3,
  "fd_rel_err_full_jvp": 0.00010939302601559381,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0,
  "image_eval_status": "decoded",
  "fid_status": "ok",
  "num_images": 4,
  "fid": 456.339409075125,
  "mmd2": 0.45975885396356464,
  "kid_x1000": 459.75885396356466
}
```

This 4-image value is integration-only and is not a quality metric.

## 2026-05-28 — FLUX.2 B3/MeanFlow calibration completed

Source:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_b3medium_calibration_summary.json
```

Purpose: choose a workable B3-medium config on FLUX.2 `[32,32,32]` latents and verify the Phase-3 precision recipe on the real FLUX latent cache.

| run | hidden/depth/heads | batch | params M | final loss | mean r=t | peak MB | max FD rel | last FD rel | samples/s | live JVP | degen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| `h384_d8_b16_50s` | 384/8/6 | 16 | 22.661 | 2.8722 | 0.7275 | 1,440.3 | 2.02e-4 | 1.43e-4 | 58.276 | yes | 0 |
| `h384_d8_b32_20s` | 384/8/6 | 32 | 22.661 | 4.2200 | 0.7563 | 2,439.2 | 7.87e-5 | 7.87e-5 | 81.699 | yes | 0 |
| `h512_d12_b16_20s` | 512/12/8 | 16 | 58.819 | 3.8392 | 0.7594 | 3,022.5 | 9.89e-5 | 5.64e-5 | 34.830 | yes | 0 |
| `h512_d12_b64_20s` | 512/12/8 | 64 | 58.819 | 3.5659 | 0.7594 | 8,913.5 | 1.14e-4 | 1.14e-4 | 41.929 | yes | 0 |
| `h512_d12_b128_20s` | 512/12/8 | 128 | 58.819 | 3.5324 | 0.7660 | 16,783.7 | 8.00e-5 | 8.00e-5 | 59.898 | yes | 0 |

Calibration conclusion:

- `bf16_backbone + fp32_jvp` works on FLUX.2 real latents.
- JVP target uses live parameters; EMA is not used for target JVP.
- FD/JVP audits are in the expected `~1e-4` range during calibration.
- `r=t` degeneration check is exact (`target-v=0`).
- `h512_d12_b128` is the selected B3-medium short-run point: it is larger than the tiny smoke model, still fits comfortably, and has acceptable throughput.

## 2026-05-28 — FLUX.2 full-cache tiny 1k image-space route validation

Source:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_fullcache_imageeval_pilot_100s_1024img_summary.json
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_fullcache_1k_imageeval_summary.json
```

Tiny-model pilot (`~0.92M` params, 100 optimizer steps, batch 16):

```json
{
  "final_step": 100,
  "final_loss": 3.4786744117736816,
  "mean_realized_r_eq_t_fraction": 0.744375,
  "loop_peak_memory_mb": 168.74365234375,
  "fd_rel_err_full_jvp": 9.486990313151122e-05,
  "degenerate_target_minus_v_max_abs": 0.0,
  "fid": 382.1860103089217,
  "mmd2": 0.4709882374694505,
  "kid_x1000": 470.9882374694505,
  "num_images": 1024
}
```

Tiny-model 1k run (`~0.92M` params, batch 16):

| step | FID | MMD2/KID | images | note |
|---:|---:|---:|---:|---|
| 500 | 381.9068 | 0.470545 | 1024 | decoded, real ImageNet-256 ref |
| 1000 | 380.8357 | 0.468050 | 1024 | decoded, real ImageNet-256 ref |

Training/gate summary for tiny 1k:

```json
{
  "final_step": 1000,
  "global_batch_size": 16,
  "final_loss": 2.915365695953369,
  "mean_realized_r_eq_t_fraction": 0.746625,
  "last_fd_rel_err_full_jvp": 0.00032816254595992897,
  "max_fd_rel_err_full_jvp": 0.0009662954806282971,
  "degenerate_target_minus_v_max_abs": 0.0,
  "practical_gate_pass": true
}
```

Interpretation: the full data loader, live-param fp32 JVP target, bf16 backbone, sampler, checkpointing, FLUX decode, and real image-space FID/MMD/KID wiring are all functional. Quality numbers from the tiny model are not publishable; they only validate the route and a weak early trend.

## 2026-05-28 — FLUX.2 B3-medium h512/d12/b128 1k short run

Source:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_b3medium_h512_b128_1k_imageeval_summary.json
```

Model:

```json
{
  "hidden_size": 512,
  "depth": 12,
  "num_heads": 8,
  "patch_size": 2,
  "params_m": 58.819192
}
```

Training/gate result:

```json
{
  "final_step": 1000,
  "global_batch_size": 128,
  "final_loss": 2.2662835121154785,
  "loss_all_finite": true,
  "grad_all_finite": true,
  "mixed_modes_ok": true,
  "jvp_param_source_all_live": true,
  "ema_used_for_target_any_step": false,
  "target_detached_all_steps": true,
  "class_cond_all_steps": true,
  "mean_realized_r_eq_t_fraction": 0.75159375,
  "degenerate_target_minus_v_max_abs": 0.0,
  "loop_peak_memory_mb": 16783.7314453125,
  "effective_train_samples_per_sec": 64.8740279785637,
  "practical_gate_pass": true
}
```

FD/JVP audit summary: `11` audits, last FD rel `1.752e-4`, max FD rel `1.197e-3`, all finite.

Image-space eval with real ImageNet-256 reference stats:

| step | FID | MMD2/KID | KID x1000 | images | ref features | status |
|---:|---:|---:|---:|---:|---:|---|
| 500 | 346.9461 | 0.402373 | 402.373 | 1024 | 8192 | ok |
| 1000 | 348.3859 | 0.406783 | 406.783 | 1024 | 8192 | ok |

Interpretation: B3-medium has better early numbers than the tiny full-cache route, but this remains a 1k/1024-image short-run diagnostic, not a publishable FID.

## 2026-05-28 — FLUX.2 B3-medium h512/d12/b128 5k short run

Source run directory:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/shorttrain_flux2vae_b3medium_h512_d12_b128_5k_1024img
```

Training result parsed from `metrics.jsonl`:

```json
{
  "final_step": 5000,
  "final_loss": 1.9944214820861816,
  "batch_size": 128,
  "peak_memory_mb": 16783.626953125,
  "jvp_param_source": "live",
  "ema_used_for_target_jvp": false,
  "target_detached": true,
  "last_fd_rel_err_full_jvp": 0.0005316118372723249,
  "fd_no_nan_or_inf": true,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0
}
```

Image-space eval table, using 1024 generated images and 8192 reference features:

| step | FID | MMD2/KID | KID x1000 | images | status |
|---:|---:|---:|---:|---:|---|
| 1000 | 350.3428 | 0.411324 | 411.324 | 1024 | ok |
| 2000 | 359.3986 | 0.433757 | 433.757 | 1024 | ok |
| 3000 | 371.9172 | 0.464737 | 464.737 | 1024 | ok |
| 4000 | 391.8534 | 0.509279 | 509.279 | 1024 | ok |
| 5000 | 420.7126 | 0.574157 | 574.157 | 1024 | ok |

Local checkpoints present:

```text
latest.pt
step_00003000.pt
step_00004000.pt
step_00005000.pt
```

Interpretation / paper-framing constraint:

- The FLUX.2 VAE route engineering smoke is complete: full cache, loader, class conditioning, mixed precision, live-param fp32 JVP target, `r=t` sampler, checkpointing, FLUX decode, and real ImageNet-256 image-space eval all run end-to-end.
- The current ImageNet-256 class-conditional B3-medium shorttrain does **not** improve image-space metrics under this config; after step 1000 the 1024-image FID/MMD worsens through step 5000.
- This is **not** a conclusive verdict that FLUX.2 VAE is worse or that the VAE-agnostic story fails. FLUX.2 VAE was not specifically tuned for ImageNet-256 class-conditional ADM-style latent modeling, and this run is a short conditional-generation/domain-transfer stress test.
- Keep FLUX/Qwen as modern VAE and future T2I/general-image baselines. Report backend-dependent results honestly: PAE may remain the strongest ImageNet-256 backend if it is ImageNet-tuned, while FLUX/Qwen may be stronger baselines for later text-to-image or broader image-distribution work.
- Before drawing quality conclusions, run a VAE reconstruction-ceiling diagnostic: `real image -> FLUX.2 VAE encode -> decode`, then compute recon FID/MMD/KID plus optional LPIPS/PSNR/SSIM. This separates VAE reconstruction/domain mismatch from MeanFlow/HT modeling quality.

Practical next diagnostics for the FLUX route:

1. VAE reconstruction baseline on ImageNet-256 for FLUX.2 and PAE side by side.
2. Latent statistics and scaling/normalization check for FLUX.2 latents (`mean/std`, channel stats, possible latent multiplier) before longer B3 runs.
3. If continuing class-conditional ImageNet on FLUX.2, retune LR, model size, sampler steps, latent normalization, and possibly patchification; do not reuse PAE hyperparameters blindly.
4. For paper story, phrase this as VAE-backend stress/ablation, not as proof that FLUX.2 VAE is weak.

## 2026-05-28 — FLUX.2 VAE reconstruction-ceiling diagnostic on real ImageNet-256 crops

Purpose: diagnose whether the bad FLUX B3-medium ImageNet-256 short-run FID (`~350 -> 421` over 1k..5k) is caused primarily by the FLUX.2 VAE reconstruction/domain ceiling, or by the MeanFlow/HT/B3 training configuration in FLUX latent space.

Diagnostic definition:

```text
real ADM-cropped ImageNet-256 uint8 image
  -> AutoencoderKLFlux2 encode, posterior.mode(), raw latent convention
  -> AutoencoderKLFlux2 decode
  -> reconstructed PNG
  -> Inception FID/MMD/KID against real ImageNet-256 reference stats
```

New script:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/diagnose_flux2vae_reconstruction.py
```

Main run:

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
python3 "$EXP/scripts/diagnose_flux2vae_reconstruction.py" \
  --crop-cache-dir /workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors \
  --output-dir "$EXP/results/flux2vae_reconstruction_diagnostic_1024_fp32_strict" \
  --repo-id diffusers/FLUX.2-dev-bnb-4bit \
  --subfolder vae \
  --vae-class AutoencoderKLFlux2 \
  --local-files-only \
  --max-images 1024 \
  --batch-size 32 \
  --device cuda:0 \
  --model-dtype fp32 \
  --ref-stats /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_stats.npz \
  --ref-features /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_mmd8192_features.npz \
  --fid-batch-size 64 \
  --fid-num-workers 2 \
  --mmd-max-ref 8192
```

Local-only outputs:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_reconstruction_diagnostic_1024_fp32_strict/
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/054_flux2vae_reconstruction_diag_1024_fp32_strict_20260528T115802Z.log
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/055_imagenet_real1024_baseline_fid_mmd_20260528T120341Z.log
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/056_flux2vae_recon_vs_input1024_paired_feature_diag_20260528T120424Z.log
```

These are ignored runtime artifacts and should not be committed/uploaded by default.

### Results

All metrics below use `1024` first ImageNet train crops unless otherwise noted. The real full-reference stats are the existing `1,281,167`-image ImageNet-256 ADM-crop Inception-2048 stats with `8192` real feature-bank samples for KID/MMD.

| comparison | FID ↓ | MMD2/KID ↓ | KID x1000 ↓ | images | note |
|---|---:|---:|---:|---:|---|
| FLUX.2 recon vs full real ref | `44.3251` | `1.83545e-05` | `0.01835` | 1024 | encode/decode reconstruction ceiling |
| Real first-1024 inputs vs full real ref | `44.4002` | `-8.34277e-06` | `-0.00834` | 1024 | finite-sample baseline |
| FLUX.2 recon vs same first-1024 inputs | `2.7447` | `-6.12531e-04` | `-0.61253` | 1024 | paired distribution drift |

Pixel reconstruction metrics:

| metric | value |
|---|---:|
| MSE `[0,1]` | `8.338014e-04` |
| RMSE `[0,1]` | `0.0288756` |
| MAE `[0,1]` | `0.0167331` |
| global PSNR | `30.7894 dB` |
| mean per-image PSNR | `33.0736 dB` |
| median per-image PSNR | `32.9017 dB` |
| max abs uint8 error | `227` |

Latent / decode stats:

| item | value |
|---|---:|
| latent shape | `[B, 32, 32, 32]` |
| latent mean | `-0.00627987` |
| latent std | `1.72098` |
| latent min / max | `-13.3578 / 12.2725` |
| latent RMS | `1.72099` |
| decoded float min / max before uint8 | `-2.1255 / 1.7374` |
| encode+decode throughput | `25.77 images/sec` |
| total wall time | `289.64 sec` |
| CUDA peak | `12.44 GB` |

Additional paired feature drift:

```json
{
  "paired_feature_cosine_mean": 0.991840691139065,
  "paired_feature_l2_mean": 2.167182747933199
}
```

### Interpretation

This diagnostic **does not support** the hypothesis that FLUX.2 VAE reconstruction quality is the main cause of the current B3-medium FLUX ImageNet-256 FID `~350-421` short-run behavior.

Reason: FLUX.2 reconstruction FID against the full real reference (`44.3251`) is essentially equal to the real first-1024 finite-sample baseline (`44.4002`). Against the same first-1024 inputs, reconstruction drift is much smaller (`FID 2.7447`, feature cosine mean `0.99184`). Therefore, at least for deterministic `posterior.mode()` reconstruction of ADM-cropped ImageNet-256 images, FLUX.2 VAE can represent the images well enough that the immediate bottleneck is more likely one of:

1. FLUX latent scale/normalization mismatch relative to PAE hyperparameters (`std ~= 1.72`, `[32,32,32]` spatial grid).
2. B3/MeanFlow training configuration not retuned for FLUX latents.
3. Model capacity / patchification / LR / sampler-step mismatch for `[32,32,32]` latents.
4. Short-run class-conditional ImageNet prior learning difficulty, not VAE reconstruction ceiling.

Guardrail: this is a `1024`-sample diagnostic, not official 50k FID. It clears the immediate reconstruction-ceiling concern but does not prove the FLUX route is fully tuned. FLUX/Qwen should remain modern VAE/T2I baselines; do not overstate VAE-agnostic claims until backend-specific training is retuned and compared honestly.

Practical next steps for FLUX route:

1. Add latent normalization/standardization experiment for FLUX B3 instead of reusing PAE-scale assumptions.
2. Run a short B3 smoke with normalized FLUX latents and compare loss/FID trend to the existing unnormalized 1k/5k runs.
3. If needed, run the same reconstruction-ceiling diagnostic for PAE and later Qwen VAE for side-by-side backend reporting.
4. Treat FLUX ImageNet-256 as a backend stress/ablation; keep stronger FLUX/Qwen relevance for later T2I/general-image experiments.

---

## 2026-05-28 — FLUX.2 latent_std diagnostic

Confirmed FLUX.2 raw latent scale before further FLUX B3 tuning.

Diagnostic script:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/compute_latent_cache_stats.py
```

Local outputs:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_latent_cache_stats/flux2_latent_stats_subset16shards_65536_eachkey.json
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_latent_cache_stats/flux2_latent_stats_linspace32shards_130191_latents.json
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_latent_cache_stats/pae_latent_stats_linspace32shards_131072_latents.json
```

Core numbers:

| route / scan | samples | mean | std | RMS | absmax |
|---|---:|---:|---:|---:|---:|
| FLUX first 16 shards, `latents+latents_flip` | `131,072` | `-0.00923639` | `1.71402432` | `1.71404920` | `17.375` |
| FLUX linspace 32 shards, `latents` | `130,191` | `-0.00905418` | `1.71404304` | `1.71406695` | `17.375` |
| PAE linspace 32 shards, local partial cache | `131,072` | `0.21101030` | `0.97748311` | `0.99999928` | `4.8125` |

Conclusion:

- FLUX raw `latent_std ~= 1.714` is a real cache property, not a one-off reconstruction diagnostic artifact.
- This does **not** mean FLUX.2 VAE reconstruction is bad; the separate reconstruction diagnostic remains healthy.
- The immediate FLUX B3 issue is a scale-policy mismatch: raw FLUX latent endpoint std `~1.714`, but MeanFlow noise/sampling endpoint is unit Gaussian and current image eval decodes with identity scale.
- Recommended first validation: scalar scale-only smoke with `latent_multiplier=0.5834159221481118` and decode `pre_decode_scale=1.7140430386576422`, then move to per-channel latent normalization if that improves the trend.

Detailed handoff:

```text
handoff/FLUX2_LATENT_STD_DIAGNOSTIC_2026-05-28.md
```

<!-- FLUX2_SCALEAWARE_B3_SMOKE_20260528 -->

## FLUX normalized / scale-aware B3 smoke completed

更新时间：`2026-05-28T13:10Z`

Detailed handoff:

```text
/workspace/PDM/handoff/FLUX2_SCALEAWARE_B3_SMOKE_2026-05-28.md
```

What was run:

- Full ImageNet-256 FLUX.2 VAE latent cache: `1,281,167` samples, `[32,32,32]` latents.
- B3-medium h512/d12, batch 128, 1000 optimizer steps.
- Scalar training scale: `latent_multiplier=0.5834159221481118 = 1 / 1.7140430386576422`.
- Official image eval inverse scale: `pre_decode_scale=1.7140430386576422`.
- Mixed precision: `bf16` backbone + `fp32` JVP target.

Mechanical result: **PASS**.

| check | result |
|---|---:|
| final step | `1000` |
| final loss | `1.240724` |
| practical gate | `true` |
| live-param JVP target | `true` |
| EMA used for JVP target | `false` |
| realized r=t fraction | `0.7491875` |
| r=t degenerate target-v max | `0.0` |
| final FD rel err | `3.96541e-4` |
| loop peak memory | `16.76 GB` |

Official image-space result with inverse decode scale:

| step | sample std, trainer space | raw-equivalent std after inverse scale | FID ↓ | MMD2/KID ↓ |
|---:|---:|---:|---:|---:|
| 500 | `1.583183` | `2.713643` | `389.4661` | `0.508815` |
| 1000 | `1.562140` | `2.677575` | `391.8537` | `0.514579` |

Comparison / diagnosis:

- Raw FLUX 1k baseline step 1000: `FID 348.3859`, `MMD2/KID 0.406783`.
- Same scale-aware step-1000 samples decoded with no inverse scale, eval-only ablation: `FID 345.6951`, `MMD2/KID 0.399690`.
- Therefore the scalar-normalized training/JVP route works mechanically, but the early sampler outputs over-dispersed normalized latents. Multiplying by `1.714` before decode produces raw-equivalent std `~2.68`, above true FLUX raw latent std `~1.714`, hurting image metrics.

Conclusion:

- Do not conclude “FLUX VAE bad.” Reconstruction diagnostic is still healthy.
- Do not yet conclude scalar normalization is bad either; this is a sample/decode scale-calibration issue at 1k.
- Before longer FLUX training, add a scale-calibrated eval diagnostic: log raw-decode-space stats and compare identity decode / official inverse decode / adaptive ref-std decode / per-channel inverse normalization.

---

## 2026-05-28 — FLUX scale-calibrated eval diagnostic

Detailed handoff:

```text
handoff/FLUX2_SCALE_CALIBRATED_EVAL_2026-05-28.md
```

Purpose: before launching a longer scalar-normalized FLUX B3 run, check whether the poor official inverse-scale metric is mainly caused by decode amplitude overshoot.

Code updated:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/decode_flux2vae_latents.py
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/eval_flux2vae_imagespace.py
```

New eval/decode records now include sample-space and decode-space latent stats, plus reference raw FLUX stats.

Adaptive eval run:

```text
base samples: experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/smoke_flux2vae_b3medium_h512_d12_b128_1k_scaleaware_1024img/eval/step_00001000/sample_latents.safetensors
output:       .../eval/step_00001000_refstd_decode_ablate/
log:          experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/058_flux2vae_scaleaware_step1000_refstd_decode_ablate_20260528T131922Z.log
```

Scale derivation:

| item | value |
|---|---:|
| generated sample-space std | `1.5621399879455566` |
| target real FLUX raw std | `1.7140430386576422` |
| adaptive pre-decode scale | `1.0972403573842702` |
| resulting decode-space std | `1.714043038657642` |

Same step-1000 samples, 1024-image real ImageNet-256 FID/MMD:

| decode variant | pre-decode scale | decode-space std | FID ↓ | MMD2/KID ↓ | KID x1000 ↓ |
|---|---:|---:|---:|---:|---:|
| official inverse scale | `1.714043` | `2.677575` | `391.8537` | `0.514579` | `514.579` |
| identity ablation | `1.0` | `1.562140` | `345.6951` | `0.399690` | `399.690` |
| adaptive ref-std match | `1.097240` | `1.714043` | `354.0362` | `0.421643` | `421.643` |

Conclusion: adaptive std matching confirms the official inverse scale was too large for early generated samples and recovers a large part of the FID loss, but scalar calibration alone is not sufficient. Next FLUX route should test per-channel normalization/inverse decode or explicit sampler/noise scale policy before any long training.

<!-- FLUX2_CHANNELNORM_B3_SMOKE_20260528 -->

## Update — FLUX per-channel latent_norm B3 smoke completed

更新时间：`2026-05-28T17:15Z`

Detailed handoff:

```text
/workspace/PDM/handoff/FLUX2_CHANNELNORM_B3_SMOKE_2026-05-28.md
```

What changed:

- Added decoder support for trainer stats based per-channel inverse decode:
  `z_raw = (z_sample * pre_decode_scale + pre_decode_shift) * channel_std + channel_mean`.
- Added `eval_flux2vae_imagespace.py --pre-decode-stats-path` forwarding.
- Added B3-medium h512/d12 b128 1k config using:
  `data.latent_norm=true`, `latent_multiplier=1.0`, and the FLUX channel stats file from `130,191` ImageNet-256 train latents.

Mechanical result: **PASS**.

| check | result |
|---|---:|
| final step | `1000` |
| final loss | `1.240308` |
| practical gate | `true` |
| live-param fp32 JVP target | `true` |
| EMA used for JVP target | `false` |
| mean realized r=t fraction | `0.7491875` |
| r=t degenerate target-v max | `0.0` |
| final FD rel err | `4.07025e-4` |
| loop peak memory | `16.76 GB` |

Image-space result with official per-channel inverse decode:

| step | normalized sample std | raw decode-space std | decode/ref std ratio | FID ↓ | MMD2/KID ↓ |
|---:|---:|---:|---:|---:|---:|
| 500 | `1.583158` | `2.716459` | `1.584825` | `370.2665` | `0.464136` |
| 1000 | `1.562104` | `2.680442` | `1.563813` | `372.7879` | `0.469827` |

Five-way 1k comparison:

| variant | decode-space std | FID ↓ | MMD2/KID ↓ |
|---|---:|---:|---:|
| raw FLUX baseline | n/a | `348.3859` | `0.406783` |
| scalar normalized + official inverse | `2.677575` | `391.8537` | `0.514579` |
| scalar normalized + identity eval ablation | `1.562140` | `345.6951` | `0.399690` |
| scalar normalized + adaptive ref-std eval | `1.714043` | `354.0362` | `0.421643` |
| per-channel latent_norm + official inverse | `2.680442` | `372.7879` | `0.469827` |

Interpretation: per-channel latent_norm is mechanically correct and slightly better than scalar official inverse, but it does not fix the FLUX smoke quality issue. The generated normalized latent std is still `~1.56`; after the correct inverse it becomes raw std `~2.68`, above the real FLUX raw std `~1.714`. This points to sampler/output-scale/training-retune issues rather than VAE reconstruction failure.

---

## 2026-05-28 — Same-100 VAE rFID ceiling: PAE vs FLUX.2

Detailed handoff:

```text
handoff/VAE_RFID_CEILING_100_PAE_VS_FLUX2_2026-05-28.md
```

Purpose: answer the requested binary reconstruction-ceiling question before spending more time on the FLUX ImageNet-256 route:

```text
same 100 ImageNet-256 ADM crops -> PAE encode/decode    -> rFID(originals, PAE recon)
same 100 ImageNet-256 ADM crops -> FLUX.2 encode/decode -> rFID(originals, FLUX recon)
```

Script:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_vae_rfid_ceiling_100.py
```

Run artifacts, local only:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/vae_rfid_ceiling_100_pae_vs_flux2/
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/060_vae_rfid_ceiling_100_pae_vs_flux2_20260528T173728Z.log
```

Settings:

- source: first 100 images from the local ImageNet-256 ADM cropped safetensor cache;
- source indices: `0..99`;
- image shape: `[100, 3, 256, 256]`;
- deterministic VAE reconstructions: PAE `encode -> decode`, FLUX posterior `mode() -> decode`;
- dtype: fp32 for both VAEs;
- TF32: off;
- Inception dim: 2048.

Primary result:

| VAE | rFID vs same 100 originals ↓ | MMD2/KID ↓ | KID x1000 ↓ | PSNR ↑ | MAE ↓ |
|---|---:|---:|---:|---:|---:|
| PAE DINOv2-L d32 | `15.851984` | `-0.00627811` | `-6.27811` | `24.264884 dB` | `0.0341667` |
| FLUX.2 VAE | `4.866488` | `-0.00666353` | `-6.66353` | `30.733994 dB` | `0.0168896` |

Decision numbers:

```text
FLUX/PAE rFID ratio = 0.306996
FLUX - PAE rFID     = -10.985496
automatic label     = flux_rfid_not_much_worse_than_pae
```

Conclusion:

- The stop condition `FLUX rFID >> PAE rFID` is not met.
- On this same-100 reconstruction ceiling check, FLUX.2 is better than PAE in rFID and pixel metrics.
- Therefore the current FLUX B3 ImageNet-256 poor sample FID is more likely a training/sampler/output-scale issue than a FLUX VAE reconstruction-ceiling issue.
- Action: do not spend more time retuning FLUX right now; create a deferred ticket and return to PAE/mainline conditional generation.

---

## 2026-05-28 — 50K protocol VAE rFID ceiling: PAE vs FLUX.2

Detailed handoff:

```text
handoff/VAE_RFID_PROTOCOL_50K_PAE_VS_FLUX2_2026-05-28.md
```

Purpose: rerun the VAE reconstruction-ceiling comparison with the paper-standard 50K protocol after the same-100 diagnostic.

Protocol:

```text
ImageNet validation 50,000 ADM-center-crop 256 originals
same 50K -> PAE DINOv2-L d32 deterministic encode/decode -> rFID(originals, recon)
same 50K -> FLUX.2 VAE deterministic mode/decode        -> rFID(originals, recon)
```

Script:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_vae_rfid_ceiling_50k_direct.py
```

Run artifacts, local only:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/vae_rfid_protocol_val50k_pae_vs_flux2_direct_20260528T182712Z/
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/065_vae_rfid_protocol_val50k_pae_vs_flux2_direct_20260528T182712Z.log
```

Settings:

- source: `/workspace/PDM/data/cropped_uint8/imagenet1k_validation_256_adm_safetensors`;
- sample count: `50,000`;
- image shape: `[N, 3, 256, 256]` uint8 RGB;
- dtype: fp32 for both VAEs;
- TF32: off;
- Inception dim: 2048;
- MMD/KID: skipped; rFID is the primary metric;
- feature implementation: direct uint8 tensor to Inception, validated against PNG path on N=128.

Primary result:

| VAE | rFID vs same 50K originals ↓ | PSNR ↑ | MAE ↓ | latent std |
|---|---:|---:|---:|---:|
| PAE DINOv2-L d32 | `0.2648149351` | `24.238722 dB` | `0.0346135` | `0.9771646` |
| FLUX.2 VAE | `0.1562118224` | `30.544935 dB` | `0.0173406` | `1.7200218` |

Runtime:

| phase | elapsed | throughput |
|---|---:|---:|
| original features | `77.25 sec` | `647.25 img/s` |
| PAE reconstruction/features | `1653.95 sec` | `30.23 img/s` |
| FLUX reconstruction/features | `1810.51 sec` | `27.62 img/s` |
| total | `3568.51 sec` | `~59.5 min` |

Decision numbers:

```text
FLUX/PAE rFID ratio = 0.5898905298
FLUX - PAE rFID     = -0.1086031128
automatic label     = flux_rfid_not_much_worse_than_pae
```

Conclusion:

- PAE rFID `0.2648` matches the expected paper-level `~0.26`, validating the 50K protocol path.
- FLUX.2 rFID `0.1562` is better than PAE under the same protocol.
- The FLUX route should not be stopped due to VAE reconstruction ceiling.
- Current FLUX B3 generated-sample issues should be treated as latent-prior / MeanFlow training / sampler / output-scale problems.

<!-- FLUX2_DECODE_SCALE_SWEEP_20260528 -->

## 2026-05-28 — FLUX.2 B3 decode/output-scale sweep

Detailed handoff:

```text
handoff/FLUX2_DECODE_SCALE_SWEEP_2026-05-28.md
```

Script:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/sweep_flux2vae_decode_scale.py
```

Artifacts, local only:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_decode_scale_sweep_channelnorm_step1000_20260528T201155Z/
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_decode_scale_sweep_channelnorm_step1000_lowrange_20260528T202427Z/
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/066_flux2vae_decode_scale_sweep_channelnorm_step1000_20260528T201155Z.log
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/067_flux2vae_decode_scale_sweep_channelnorm_step1000_lowrange_20260528T202427Z.log
```

Combined result, 1024 images from existing channelnorm step-1000 sample latents:

| scale | decode std | decode/ref std | FID ↓ | MMD2/KID ↓ |
|---:|---:|---:|---:|---:|
| `0.20` | `0.539710` | `0.314875` | `447.8535` | `0.567919` |
| `0.30` | `0.806882` | `0.470748` | `329.5382` | `0.337264` |
| `0.35` | `0.940596` | `0.548758` | **`322.3539`** | **`0.324473`** |
| `0.40` | `1.074353` | `0.626794` | `322.4108` | `0.328168` |
| `0.45` | `1.208138` | `0.704847` | `323.1042` | `0.336568` |
| `0.50` | `1.341944` | `0.782911` | `328.7377` | `0.355028` |
| `0.55` | `1.475764` | `0.860984` | `334.8103` | `0.372971` |
| `0.64` | `1.716666` | `1.001530` | `344.6390` | `0.400106` |
| `0.70` | `1.877281` | `1.095236` | `351.2454` | `0.417133` |
| `0.80` | `2.144988` | `1.251420` | `361.2771` | `0.442044` |
| `1.00` | `2.680442` | `1.563813` | `372.7879` | `0.469827` |

Conclusion: output scale compression helps but does not solve FLUX B3. The std-matched scale `~0.64` is not best; best FID occurs with decode/ref std `~0.55`. Treat scale `0.35` as diagnostic only. Next FLUX step is checkpoint resampling/sample-step sweep before lower-LR retraining.
