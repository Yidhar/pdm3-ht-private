# B3 MeanFlow Real-Data Trainer

This experiment adds a real-data training loop for Phase-0/B3 MeanFlow over PAE latent safetensors.

## Main script

```bash
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py
```

It imports the already validated MeanFlow model and transport from:

```text
/workspace/PDM/external/PAE/pae_with_generator
```

## Smoke run

Uses the fixed 4096-sample real PAE latent cache:

```bash
cd /workspace/PDM
python experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py \
  --config experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_smoke4096.yaml \
  2>&1 | tee experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/010_smoke4096.log
```

Expected outputs:

```text
results/smoke_realdata_4096/
  log.txt
  config_resolved.yaml
  dataset_snapshot.json
  metrics.jsonl
  train_summary.json
  train_summary.md
  checkpoints/latest.pt
  checkpoints/step_00000003.pt
  eval/step_00000003/sample_latents.safetensors
```

## Full-cache config

`configs/b3_meanflow_realdata_full.yaml` points at the full ImageNet-1k PAE latent cache:

```text
/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full
```

It is intended for use after the full cache is complete. The dataset scanner snapshots the file list at startup and records it in `dataset_snapshot.json`.

## Resume

Set in config:

```yaml
train:
  resume_from: auto   # or an explicit checkpoint path
```

`auto` loads `checkpoints/latest.pt` from the configured output/exp directory if present.

## Eval/FID hook

The smoke config samples latent tensors and skips FID. For a real FID pipeline, set:

```yaml
eval:
  enabled: true
  every_steps: 10000
  fid_command_template: "python path/to/fid.py --samples {sample_dir} --out {output_dir} --step {step}"
```

The trainer writes stdout/stderr and attempts to parse a numeric `FID` from command output.

## Current smoke status — 2026-05-27

Smoke `smoke_realdata_4096` completed successfully.

- Gate: `practical_gate_pass=true`
- Dataset: 4096 real PAE latents, 1 shard, shape `[32,16,16]`, labels min/max `0/999`
- Training: 6/6 optimizer steps, all finite
- Class-conditioning: labels injected every step
- MeanFlow recipe: bf16 backbone + fp32 live-param JVP target
- r=t sampler: observed mean r=t fraction `0.7917` for tiny batch smoke; configured probability `0.75`
- FD audit last rel err: `6.78e-05`
- r=t degenerate target-v max abs: `0.0`
- CUDA loop peak memory: `63.06 MiB` for Tiny smoke, while full-cache builder stayed active
- Checkpoints: `step_00000003.pt`, `step_00000006.pt`, `latest.pt`
- Eval: latent samples saved at steps 3 and 6; FID skipped because no FID command configured

Partial full-cache scan at `2026-05-27T17:06:13Z` saw 22 completed shards / 90,112 samples; expected full total is 1,281,167, so the full train config is not ready for full run yet.

## Convenience wrappers

```bash
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_smoke4096.sh
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/scan_fullcache_dataset.sh
```

## FLUX.2 VAE backend route — 2026-05-28

FLUX.2 VAE latent cache is complete and can be used as a second VAE backend:

```text
/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full
```

Full-cache scan result:

- shards: `313`
- samples: `1,281,167`
- latent shape: `[32,32,32]`
- last shard: `3215` samples
- `latents_flip`: present
- unreadable shards: `0`

Design choice for the DiT route: FLUX.2 VAE has spatial latent size `32x32`, while PAE uses `16x16`. For comparable transformer token count, FLUX configs use `downsample_ratio=8` and `patch_size=2`, giving a `16x16` token grid. This keeps the B3/MeanFlow mechanism backend-agnostic while making the VAE swap explicit in config/logs.

Created configs/scripts:

```bash
# one-shard smoke, Tiny-equivalent Custom patch_size=2
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_smoke4096.sh

# scan all 313 FLUX latent shards
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/scan_flux2vae_fullcache_dataset.sh

# short engineering run over all 1.281M FLUX latents, Custom patch_size=2
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_fullcache_short20.sh

# full template, XL/2, not launched by smoke scripts
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_full_template.sh
```

Smoke results:

| run | dataset | model | steps | gate | final loss | mean r=t | last FD rel err | degen max | peak MB | eval/FID |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| `smoke_flux2vae_4096` | 4,096 / 1 shard | Custom Tiny-like `/2` | 6 | pass | 5.0054 | 0.8333 | 5.7467e-05 | 0.0 | 64.95 | latent sample ok; FID skipped |
| `shorttrain_flux2vae_fullcache_20` | 1,281,167 / 313 shards | Custom Tiny-like `/2` | 20 | pass | 4.9092 | 0.8000 | 6.4823e-05 | 0.0 | 64.95 | latent sample ok; FID skipped |

Both runs verify:

- class labels loaded/injected every step;
- bf16 backbone + fp32 live-parameter JVP target;
- JVP target is not EMA;
- r=t sampler path works (`equal_prob=0.75`);
- FD audit passes comfortably below `1e-2`;
- checkpoint/eval latent sampling works for `[N,32,32,32]` FLUX latents.

Remaining FLUX-route work: wire FLUX VAE decoder + FID/MMD evaluation hook. Current smoke eval saves latent samples and intentionally leaves `fid_command_template` empty.


## FLUX.2 image-space eval hook — 2026-05-28

The FLUX backend now has an image-space decode/eval hook wired through the existing trainer `eval.fid_command_template` path.

Added scripts/configs:

```text
scripts/decode_flux2vae_latents.py
scripts/eval_flux2vae_imagespace.py
scripts/run_flux2vae_imagespace_eval_smoke.sh
configs/b3_meanflow_flux2vae_imagespace_eval_smoke.yaml
```

Decoder convention:

- latent key: `samples` for trainer eval output; also supports cache keys `latents` / `latents_flip`.
- latent shape: `[N, 32, 32, 32]`.
- VAE: `diffusers/FLUX.2-dev-bnb-4bit`, subfolder `vae`, class `AutoencoderKLFlux2`.
- decode transform: raw `vae.decode(z).sample`, because the FLUX cache was built from raw `vae.encode(x).latent_dist.mode()` without an external scaling/shift factor.
- output: PNGs plus `decode_summary.json`; optional `preview_grid.png`.

Smoke command:

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_imagespace_eval_smoke.sh
```

Smoke result:

- output: `results/smoke_flux2vae_imagespace_eval`
- training gate: `practical_gate_pass=true`
- final step: `3`
- final loss: `5.1767`
- live-param fp32 JVP target + bf16 backbone: pass
- r=t sampler: pass
- FD audit last rel err: `1.0939e-04`
- r=t degenerate target-v max abs: `0.0`
- eval latent sample: `[4,32,32,32]`
- image-space decode: `4` PNGs at `eval/step_00000003/decoded_flux2vae_png`
- decode CUDA peak: about `1296.86 MiB`
- FID: not faked; current hook reports `fid=null` / `decoded_no_fid` unless an inner FID command is provided.

To add real FID later, keep the trainer hook pointed at `eval_flux2vae_imagespace.py` and pass an inner command via `--fid-command-template`, e.g. one that compares `{images_dir}` against precomputed ImageNet-256 reference stats. The wrapper will parse a numeric `FID`/`fid` from that inner command's output and forward it to the trainer.

## Real ImageNet-256 FID/MMD reference stats + FLUX.2 image-space metric wiring — 2026-05-28

The FLUX.2 image-space eval hook is now connected to a real ImageNet-256 reference-stat pipeline. No fake metric is emitted: the trainer calls `eval_flux2vae_imagespace.py`, which decodes FLUX latents to PNG, then calls an inner real metric command over the decoded image directory.

Added/updated metric scripts:

```text
scripts/inception_metrics_lib.py
scripts/build_imagenet256_reference_stats.py
scripts/eval_imagespace_fid_mmd.py
scripts/eval_flux2vae_imagespace.py
scripts/train_b3_meanflow_realdata.py
configs/b3_meanflow_flux2vae_imagespace_fid_mmd_smoke.yaml
scripts/run_flux2vae_imagespace_fid_mmd_smoke.sh
```

Reference stats were built from the full local ADM-style cropped ImageNet-256 uint8 cache:

```text
cache: /workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors
stats: /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_stats.npz
mmd feature bank: /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_mmd8192_features.npz
summary: /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/refstats_summary.json
```

Full reference build summary:

```json
{
  "status": "ok",
  "count": 1281167,
  "dims": 2048,
  "num_shards": 313,
  "allow_tf32": false,
  "batch_size": 1024,
  "elapsed_sec": 2592.5845,
  "samples_per_sec": 494.1659,
  "cuda_peak_memory_mb": 43027.2896,
  "sigma_trace": 180.5874165407635
}
```

Shape validation:

```text
stats keys: ['mu', 'sigma', 'count', 'dims', 'image_size', 'created_unix']
mu:     [2048] float64
sigma:  [2048, 2048] float64
count:  1281167
feature bank: [8192, 2048] float32; global index 0..1281166
```

The trainer config now carries a nested command like:

```bash
python .../eval_flux2vae_imagespace.py \
  --sample-dir {sample_dir} \
  --output-dir {output_dir} \
  --step {step} \
  --device cuda:0 \
  --batch-size 4 \
  --max-images 4 \
  --images-subdir decoded_flux2vae_png_realmetrics \
  --save-grid \
  --fid-command-template 'python .../eval_imagespace_fid_mmd.py --images-dir {images_dir} --ref-stats /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_stats.npz --ref-features /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_mmd8192_features.npz --output-json {sample_dir}/fid_mmd_metrics_real_imagenet256.json --batch-size 4 --num-workers 0 --device cuda:0 --sqrt-device cuda:0 --mmd-device cuda:0 --mmd-max-ref 8192'
```

Run command:

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_imagespace_fid_mmd_smoke.sh
```

Integrated smoke result:

```json
{
  "practical_gate_pass": true,
  "final_step": 3,
  "final_loss": 5.176722526550293,
  "fd_rel_err_full_jvp": 0.00010939302601559381,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0,
  "image_eval_status": "decoded",
  "fid_status": "ok",
  "num_images": 4,
  "fid": 456.339409075125,
  "mmd2": 0.45975885396356464,
  "kid_x1000": 459.75885396356466,
  "ref_count": 1281167,
  "ref_features_used": 8192
}
```

Artifacts:

```text
results/imagenet256_refstats_full_summary.json
results/flux2vae_imagespace_fid_mmd_smoke_summary.json
results/real_imagenet256_fid_mmd_wiring_summary.json
results/smoke_flux2vae_imagespace_fid_mmd/train_summary.json
results/smoke_flux2vae_imagespace_fid_mmd/eval/step_00000003/fid_mmd_metrics_real_imagenet256.json
results/smoke_flux2vae_imagespace_fid_mmd/eval/step_00000003/flux2vae_imagespace_eval_record.json
logs/043_imagenet256_refstats_full.log
logs/044_flux2vae_imagespace_fid_mmd_smoke_*.log
```

Caveats:

- The reference stats are real/full ImageNet-256 stats; the smoke generated only 4 images, so the numeric FID/KID above is **integration-only** and is not a reportable quality number.
- `preview_grid.png`, `contact_sheet.*`, and `preview_*` files are intentionally excluded from metric image enumeration.
- The `.npz` reference files are data artifacts under `/workspace/PDM/data`; do not upload them to the lightweight/private code repository unless explicitly requested.

## 2026-05-28 — FLUX.2 full-cache 100-step + 1024-image real FID/MMD pilot

Purpose: move beyond the 4-image wiring smoke and validate the full local FLUX.2 VAE route on the complete ImageNet-1k latent cache with a short B3 MeanFlow train, chunked eval sampling, image-space decode, and real ImageNet-256 FID/MMD/KID.

Run command:

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_fullcache_imageeval_pilot.sh
```

Config:

```text
configs/b3_meanflow_flux2vae_fullcache_imageeval_pilot.yaml
```

Key implementation addition:

- `train_b3_meanflow_realdata.py` eval sampling now supports `eval.sample_batch_size`; this pilot used `num_samples=1024`, `sample_batch_size=16`, i.e. 64 sampling chunks, avoiding a single huge eval batch.

Pilot result:

```json
{
  "status": "ok",
  "practical_gate_pass": true,
  "final_step": 100,
  "final_loss": 3.4786744117736816,
  "fd_rel_err_full_jvp": 0.00009486990313151122,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0,
  "mean_realized_r_eq_t_fraction": 0.744375,
  "loop_peak_memory_mb": 168.74365234375,
  "sample_shape": [1024, 32, 32, 32],
  "sample_batch_size": 16,
  "num_sample_batches": 64,
  "sample_elapsed_sec": 12.468012206023559,
  "decode_elapsed_sec": 109.26908582891338,
  "decode_effective_images_per_sec": 9.371360547514001,
  "decoded_metric_png_count": 1024,
  "fid_status": "ok",
  "fid": 382.1860103089217,
  "mmd2": 0.4709882374694505,
  "kid_x1000": 470.9882374694505,
  "ref_count": 1281167,
  "ref_features_used": 8192,
  "metrics_elapsed_sec": 9.787689667893574,
  "metrics_effective_images_per_sec": 104.62121652252762,
  "metrics_cuda_peak_memory_mb": 1332.16064453125
}
```

Artifacts:

```text
results/flux2vae_fullcache_imageeval_pilot_100s_1024img_summary.json
results/pilot_flux2vae_fullcache_imageeval_100s_1024img/train_summary.json
results/pilot_flux2vae_fullcache_imageeval_100s_1024img/eval/step_00000100/flux2vae_imagespace_eval_record.json
results/pilot_flux2vae_fullcache_imageeval_100s_1024img/eval/step_00000100/fid_mmd_metrics_real_imagenet256_1024.json
logs/045_flux2vae_fullcache_imageeval_pilot_*.log
```

Caveats:

- This is an engineering/pilot signal, not a publishable quality metric: the model is tiny (`0.92M` params), trained for only 100 optimizer steps, and evaluated with only 1024 generated images.
- The eval uses EMA model weights for sampling, while the MeanFlow JVP target during training remains live-param fp32; this is intentional.
- Local decoded PNGs, sample latents, checkpoints, full logs, latent shards, and reference `.npz` stats are data/heavy artifacts and should not be uploaded to the lightweight/private code repo by default.

## 2026-05-28 — FLUX.2 full-cache 1k-step short run + 1024-image FID/MMD trend check

Purpose: after the 100-step pilot, run a longer but still lightweight FLUX.2 B3 MeanFlow short train and verify that the image-space metric route remains stable at repeated eval checkpoints.

Implementation changes before the run:

- `decode_flux2vae_latents.py` now supports `--manifest-mode {full,first_last,none}`.
- `eval_flux2vae_imagespace.py` now forwards `--decode-manifest-mode`, defaulting to `first_last`.
- This prevents `decode_summary.json` from containing a huge per-image `images_manifest` during larger evals while retaining `first_image`, `last_image`, image count, and decode stats.

Run command:

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_fullcache_1k_imageeval.sh
```

Config:

```text
configs/b3_meanflow_flux2vae_fullcache_1k_imageeval.yaml
```

Result:

```json
{
  "status": "ok",
  "practical_gate_pass": true,
  "final_step": 1000,
  "final_loss": 2.915365695953369,
  "min_loss": 2.1960299015045166,
  "last_fd_rel_err_full_jvp": 0.00032816254595992897,
  "max_fd_rel_err_full_jvp": 0.0009662954806282971,
  "degenerate_target_minus_v_max_abs": 0.0,
  "mean_realized_r_eq_t_fraction": 0.746625,
  "loop_peak_memory_mb": 168.74365234375,
  "total_elapsed_sec": 485.92449224786833
}
```

Eval trend, each point uses 1024 decoded FLUX.2 images against the full ImageNet-256 reference stats:

| step | FID | MMD2/KID | KID x1000 | sample sec | decode sec | metric sec | manifest |
|---:|---:|---:|---:|---:|---:|---:|---|
| 500 | 381.9068369846536 | 0.4705451225255155 | 470.5451225255155 | 12.354675956070423 | 113.98510063579306 | 10.714315253077075 | first/last only |
| 1000 | 380.83573517219475 | 0.46804980426009113 | 468.04980426009115 | 12.050162501865998 | 104.19804051890969 | 11.387822730932385 | first/last only |

Compared with the earlier 100-step pilot:

```json
{
  "pilot100_fid": 382.1860103089217,
  "step500_fid": 381.9068369846536,
  "step1000_fid": 380.83573517219475,
  "fid_delta_1000_minus_100": -1.3502751367269639,
  "pilot100_kid_x1000": 470.9882374694505,
  "step500_kid_x1000": 470.5451225255155,
  "step1000_kid_x1000": 468.04980426009115,
  "kid_x1000_delta_1000_minus_100": -2.9384332093593457
}
```

Artifacts:

```text
results/flux2vae_fullcache_1k_imageeval_summary.json
results/shorttrain_flux2vae_fullcache_1k_1024img/train_summary.json
results/shorttrain_flux2vae_fullcache_1k_1024img/eval/step_00000500/fid_mmd_metrics_real_imagenet256_1024.json
results/shorttrain_flux2vae_fullcache_1k_1024img/eval/step_00001000/fid_mmd_metrics_real_imagenet256_1024.json
logs/046_flux2vae_fullcache_1k_imageeval_*.log
```

Caveat: still not a publishable quality run. The model remains tiny (`0.92M` params), training is only 1000 steps, and each metric point uses 1024 generated images.

## 2026-05-28 — FLUX.2 B3-medium calibration + 1k image-space smoke

Purpose: move beyond the `0.92M` tiny route and verify that a larger B3 MeanFlow model can use the full FLUX.2 latent cache with the same production precision recipe, class conditioning, live-param fp32 JVP target, r=t sampler, FD audit, checkpointing, and image-space FID/MMD eval.

Engineering patch before the medium run:

- `sample_meanflow_latents()` in `scripts/train_b3_meanflow_realdata.py` now wraps eval sampling in `torch.no_grad()`.
- This keeps EMA sampling as inference-only and avoids accidental graph retention when scaling `eval.sample_batch_size`.
- The training JVP target path is unchanged: **live parameters**, fp32 JVP/FD, detached target, no EMA target.

Calibration configs/scripts added:

```text
configs/b3_meanflow_flux2vae_fullcache_medium_calib_b16_50s.yaml
configs/b3_meanflow_flux2vae_fullcache_medium_calib_b32_20s.yaml
configs/b3_meanflow_flux2vae_fullcache_medium512_calib_b16_20s.yaml
configs/b3_meanflow_flux2vae_fullcache_medium512_calib_b64_20s.yaml
configs/b3_meanflow_flux2vae_fullcache_medium512_calib_b128_20s.yaml
scripts/run_flux2vae_medium_calib_b16_50s.sh
scripts/run_flux2vae_medium_calib_b32_20s.sh
scripts/run_flux2vae_medium512_calib_b16_20s.sh
scripts/run_flux2vae_medium512_calib_b64_20s.sh
scripts/run_flux2vae_medium512_calib_b128_20s.sh
```

Calibration summary:

| model | params | batch | steps | final loss | max FD rel | r=t mean | loop peak MB | eff samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h384/d8/heads6 | 22.6606M | 16 | 50 | 2.8721611499786377 | 0.00020186840536935322 | 0.7275 | 1440.2607421875 | 58.275970818800054 |
| h384/d8/heads6 | 22.6606M | 32 | 20 | 4.220029830932617 | 0.00007870426877958057 | 0.75625 | 2439.2255859375 | 81.69916392329296 |
| h512/d12/heads8 | 58.8192M | 16 | 20 | 3.8391733169555664 | 0.0000989313113513627 | 0.759375 | 3022.45556640625 | 34.830103093051086 |
| h512/d12/heads8 | 58.8192M | 64 | 20 | 3.5658743381500244 | 0.00011434621593090143 | 0.759375 | 8913.51513671875 | 41.92922497839281 |
| h512/d12/heads8 | 58.8192M | 128 | 20 | 3.5324220657348633 | 0.0000800312485497428 | 0.766015625 | 16783.7314453125 | 59.89824844510914 |

All calibration runs had finite loss/grad, live-param JVP target, bf16 backbone + fp32 JVP target, class conditioning, and exact r=t degenerate target handling (`target-v max abs = 0.0`).

Medium 1k run command:

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_medium512_b128_1k_imageeval.sh
```

Config:

```text
configs/b3_meanflow_flux2vae_fullcache_medium512_b128_1k_imageeval.yaml
```

Run summary:

```json
{
  "status": "ok",
  "practical_gate_pass": true,
  "model": "h512/d12/heads8",
  "params_m": 58.819192,
  "global_batch_size": 128,
  "final_step": 1000,
  "final_loss": 2.2662835121154785,
  "min_loss": 1.9191009998321533,
  "max_loss": 5.642007350921631,
  "loss_all_finite": true,
  "grad_all_finite": true,
  "mixed_modes_ok": true,
  "jvp_param_source_all_live": true,
  "ema_used_for_target_any_step": false,
  "target_detached_all_steps": true,
  "class_cond_all_steps": true,
  "mean_realized_r_eq_t_fraction": 0.75159375,
  "degenerate_target_minus_v_max_abs": 0.0,
  "last_fd_rel_err_full_jvp": 0.00017521927982342016,
  "max_fd_rel_err_full_jvp": 0.0011974215362778894,
  "loop_peak_memory_mb": 16783.7314453125,
  "sample_batch_size": 32,
  "total_elapsed_sec": 1973.0546104258392,
  "effective_train_samples_per_sec": 64.8740279785637
}
```

Eval table, each point uses 1024 decoded FLUX.2 images against the full ImageNet-256 reference stats:

| step | FID | MMD2/KID | KID x1000 | sample sec | decode sec | metric sec | sample peak MB | ref features |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 346.9461242148833 | 0.40237275729309285 | 402.37275729309283 | 11.524835258023813 | 115.33161097788252 | 9.750014807796106 | 1506.23828125 | 8192 |
| 1000 | 348.38588136862217 | 0.40678265356824905 | 406.78265356824903 | 11.21557610318996 | 133.1801801288966 | 18.63674869108945 | 1504.4267578125 | 8192 |

Interpretation:

- The scaled model path is now validated: `58.8M` params, real full-cache FLUX.2 latents, batch `128`, checkpoints at steps 500/1000, and real image-space metrics all work.
- Compared with the tiny 1k run (`FID≈380.84`, `KIDx1000≈468.05` at step 1000), the medium 1k run improves to `FID≈348.39`, `KIDx1000≈406.78` at the same 1024-image eval scale.
- Step-500 FID was slightly better than step-1000 (`346.95` vs `348.39`), so this is **not** a convergence claim. Treat it as scale-route validation plus a substantially better short-run quality signal than the tiny model.
- FD audits remained well below the practical gate threshold `1e-2`; two audits were around `1e-3` (`step 200 ≈ 0.001001`, `step 900 ≈ 0.001197`), while the final audit returned to `1.75e-4`.
- The `r=t` sampler realized fraction was `0.7516`, matching the configured 75%; the degenerate path stayed exact (`0.0`).

Artifacts:

```text
results/flux2vae_b3medium_calibration_summary.json
results/flux2vae_b3medium_h512_b128_1k_imageeval_summary.json
results/shorttrain_flux2vae_b3medium_h512_d12_b128_1k_1024img/train_summary.json
results/shorttrain_flux2vae_b3medium_h512_d12_b128_1k_1024img/eval/step_00000500/fid_mmd_metrics_real_imagenet256_1024.json
results/shorttrain_flux2vae_b3medium_h512_d12_b128_1k_1024img/eval/step_00001000/fid_mmd_metrics_real_imagenet256_1024.json
logs/047_flux2vae_medium_calib_b16_50s_*.log
logs/048_flux2vae_medium_calib_b32_20s_*.log
logs/049_flux2vae_medium512_calib_b16_20s_*.log
logs/050_flux2vae_medium512_calib_b64_20s_*.log
logs/051_flux2vae_medium512_calib_b128_20s_*.log
logs/052_flux2vae_medium512_b128_1k_imageeval_*.log
```

Heavy local artifacts not for lightweight upload by default:

```text
results/shorttrain_flux2vae_b3medium_h512_d12_b128_1k_1024img/checkpoints/
results/shorttrain_flux2vae_b3medium_h512_d12_b128_1k_1024img/eval/step_00000500/sample_latents.safetensors
results/shorttrain_flux2vae_b3medium_h512_d12_b128_1k_1024img/eval/step_00001000/sample_latents.safetensors
results/shorttrain_flux2vae_b3medium_h512_d12_b128_1k_1024img/eval/step_00000500/decoded_flux2vae_png_realmetrics_1024/
results/shorttrain_flux2vae_b3medium_h512_d12_b128_1k_1024img/eval/step_00001000/decoded_flux2vae_png_realmetrics_1024/
```
