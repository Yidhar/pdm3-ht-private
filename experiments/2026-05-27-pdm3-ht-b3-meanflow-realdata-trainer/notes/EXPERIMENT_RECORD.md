# Experiment Record — B3 Real-Data Training Loop

## 2026-05-27 start

User request: while full ImageNet-1k PAE latent cache is running, write the real-data B3 training loop engineering layer:

- DataLoader over 1.28M latents
- label/class conditional injection
- eval loop with sampling + FID hook
- checkpoint save/resume

Initial full-cache status observed before starting this experiment:

- full-cache process active under experiment `2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache`
- progress around 70k/1,281,167 samples encoded
- throughput around 37 samples/s end-to-end
- smoke4096 latent cache already verified

Switchyard note: `switchyard` command was not present in this environment, so no peer HYARD delegation could be launched; proceeded locally.

## Commands

Commands and results will be appended below.

## 2026-05-27 implementation

Created:

- `scripts/train_b3_meanflow_realdata.py`
- `configs/b3_meanflow_realdata_smoke4096.yaml`
- `configs/b3_meanflow_realdata_full.yaml`
- result directories under `results/`

Implemented features:

- `ShardIndexedLatentDataset`: snapshots safetensor shard list and uses cumulative shard offsets instead of a 1.28M-entry Python map.
- DataLoader knobs: workers, prefetch, persistent workers, shuffle/drop_last, random `latents` vs `latents_flip`.
- class-conditional injection: batch labels are passed as `model_kwargs['y']`; `force_drop_ids` is sampled once per train batch and reused through JVP target and bf16 train forward.
- MeanFlow precision: `bf16_backbone_fp32_jvp`, live-param JVP target, target detached, no EMA target.
- r=t sampler: `equal_prob=0.75` from transport.
- metrics JSONL, FD audit, CUDA peak memory logging.
- checkpoint save/resume state: model, EMA, optimizer, RNG, config, dataset snapshot; `latest.pt` symlink points to latest step checkpoint.
- eval hook: latent Euler sampler saves `sample_latents.safetensors`; optional external `fid_command_template` can be configured. Smoke leaves it empty and records `skipped_no_command`.

## 2026-05-27 smoke4096 command

```bash
cd /workspace/PDM
python experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py \
  --config experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_smoke4096.yaml \
  2>&1 | tee experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/010_smoke4096.log
```

## 2026-05-27 smoke4096 result

Output directory:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/smoke_realdata_4096
```

Key result:

```json
{
  "practical_gate_pass": true,
  "dataset_total_samples": 4096,
  "dataset_num_shards": 1,
  "latent_shape": [32, 16, 16],
  "final_step": 6,
  "optimizer_steps_this_run": 6,
  "final_loss": 3.4510116577148438,
  "mean_realized_r_eq_t_fraction": 0.7916666666666666,
  "fd_rel_err_full_jvp_last": 6.783736791015577e-05,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0,
  "loop_peak_memory_mb": 63.06005859375,
  "checkpoint_ok": true,
  "eval_sample_ok": true,
  "fid_status": "skipped_no_command"
}
```

Artifacts verified:

- `metrics.jsonl` written.
- `dataset_snapshot.json` written.
- checkpoints: `step_00000003.pt`, `step_00000006.pt`, `latest.pt` symlink.
- checkpoint keys verified: `model`, `ema`, `optimizer`, `rng_state`, `config`, `dataset_summary`, `step`.
- eval latent samples saved at steps 3 and 6.
- class conditional labels verified per step (`class_cond_injected=True`).
- FD audits at steps 1, 3, 6 all passed `<1e-2`; last rel err `6.78e-05`.

## 2026-05-27 partial full-cache dataset scan

Command imported the experiment dataset and scanned active full-cache directory without training.

Result at `2026-05-27T17:06:13Z`:

```json
{
  "num_shards": 22,
  "total_samples": 90112,
  "expected_total_from_config": 1281167,
  "ready_for_full_train": false,
  "latent_shape": [32, 16, 16],
  "skipped_files_count": 0,
  "scan_elapsed_sec": 0.14347953093238175
}
```

Interpretation: full-cache writer is still running; current scanner correctly snapshots only completed `.safetensors` shards and does not include the pending unsaved shard samples.

## 2026-05-28 — FLUX.2 VAE backend route started

Context: FLUX.2 VAE ImageNet-1k latent cache finished, so the backend-swap route can start locally without waiting for PAE full cache.

FLUX full-cache validation:

```json
{
  "data_dir": "/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full",
  "num_shards": 313,
  "total_samples": 1281167,
  "latent_shape": [32, 32, 32],
  "last_shard_samples": 3215,
  "has_latents_flip": true,
  "skipped_files_count": 0,
  "ready_for_full_train": true
}
```

Added FLUX-specific configs/scripts:

- `configs/b3_meanflow_flux2vae_smoke4096.yaml`
- `configs/b3_meanflow_flux2vae_fullcache_short20.yaml`
- `configs/b3_meanflow_flux2vae_full.yaml`
- `scripts/run_flux2vae_smoke4096.sh`
- `scripts/run_flux2vae_fullcache_short20.sh`
- `scripts/scan_flux2vae_fullcache_dataset.sh`
- `scripts/run_flux2vae_full_template.sh`

Important shape decision: FLUX.2 VAE latents are `[32,32,32]`, not PAE `[32,16,16]`. The FLUX route uses `downsample_ratio=8` and `patch_size=2`, so transformer token grid remains `16x16`. This is the correct backend swap engineering point; do not reuse PAE `/1` configs blindly.

### FLUX one-shard smoke command

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_smoke4096.sh
```

Result:

```json
{
  "practical_gate_pass": true,
  "dataset_total_samples": 4096,
  "dataset_num_shards": 1,
  "latent_shape": [32, 32, 32],
  "model_type": "B3MeanFlowLightningDiT-Custom",
  "patch_size": 2,
  "final_step": 6,
  "optimizer_steps_this_run": 6,
  "final_loss": 5.00540018081665,
  "mean_realized_r_eq_t_fraction": 0.8333333333333334,
  "fd_rel_err_full_jvp_last": 5.746658066108667e-05,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0,
  "loop_peak_memory_mb": 64.95361328125,
  "eval_sample_shape_last": [8, 32, 32, 32],
  "fid_status": "skipped_no_command"
}
```

### FLUX all-shard shorttrain command

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_fullcache_short20.sh
```

Result:

```json
{
  "practical_gate_pass": true,
  "dataset_total_samples": 1281167,
  "dataset_num_shards": 313,
  "latent_shape": [32, 32, 32],
  "model_type": "B3MeanFlowLightningDiT-Custom",
  "patch_size": 2,
  "final_step": 20,
  "optimizer_steps_this_run": 20,
  "final_loss": 4.909223556518555,
  "mean_realized_r_eq_t_fraction": 0.8,
  "fd_rel_err_full_jvp_last": 6.482259910222848e-05,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0,
  "loop_peak_memory_mb": 64.95361328125,
  "eval_sample_shape_last": [8, 32, 32, 32],
  "fid_status": "skipped_no_command"
}
```

Conclusion: the B3 MeanFlow training loop is now VAE-backend agnostic at the latent-loader/training/JVP/checkpoint/eval-latent level for both PAE and FLUX.2. The remaining missing part for publishable FLUX numbers is the decoder + FID/MMD image-space evaluation hook.


## 2026-05-28 — FLUX.2 image-space eval hook implemented and smoke-tested

Implemented the FLUX image-space eval path requested after the backend-swap route.

Added:

- `scripts/decode_flux2vae_latents.py`
  - decodes trainer `sample_latents.safetensors` (`samples` key) or cache shards (`latents`/`latents_flip`) to PNG.
  - loads `diffusers/FLUX.2-dev-bnb-4bit`, subfolder `vae`, class `AutoencoderKLFlux2`.
  - default decode is raw `vae.decode(z).sample`; no SD-style latent scale/shift is applied because cache builder stored raw posterior `mode()` latents.
  - writes `decode_summary.json` and optional `preview_grid.png`.
- `scripts/eval_flux2vae_imagespace.py`
  - trainer hook called through `eval.fid_command_template`.
  - decodes `{sample_dir}/sample_latents.safetensors` to `{sample_dir}/decoded_flux2vae_png`.
  - optionally runs an inner FID command over `{images_dir}`; if none is supplied, it reports `fid=null` and `image_eval_status=decoded_no_fid` without faking a metric.
- `configs/b3_meanflow_flux2vae_imagespace_eval_smoke.yaml`
- `scripts/run_flux2vae_imagespace_eval_smoke.sh`

Direct decoder smoke on previous `smoke_flux2vae_4096` sample:

```json
{
  "status": "ok",
  "num_images": 4,
  "decoded_shape": [2, 3, 256, 256],
  "cuda_peak_memory_mb": 816.36376953125,
  "output_dir": "results/smoke_flux2vae_4096/eval/step_00000006/decoded_flux2vae_png"
}
```

Eval-hook direct smoke on previous `shorttrain_flux2vae_fullcache_20` sample:

```json
{
  "image_eval_status": "decoded_no_fid",
  "num_images": 4,
  "decode_cuda_peak_memory_mb": 1296.86376953125,
  "fid_status": "decoded_no_fid",
  "fid": null
}
```

Trainer-integrated smoke command:

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_imagespace_eval_smoke.sh
```

Result summary:

```json
{
  "practical_gate_pass": true,
  "final_step": 3,
  "final_loss": 5.176722526550293,
  "fd_rel_err_full_jvp_last": 0.00010939302601559381,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0,
  "eval_sample_shape_last": [4, 32, 32, 32],
  "trainer_fid_status": "ok",
  "trainer_fid": null,
  "image_eval_status": "decoded_no_fid",
  "decoded_png_count": 4,
  "decode_cuda_peak_memory_mb": 1296.86376953125
}
```

Artifacts:

```text
results/smoke_flux2vae_imagespace_eval/train_summary.json
results/smoke_flux2vae_imagespace_eval/eval/step_00000003/sample_latents.safetensors
results/smoke_flux2vae_imagespace_eval/eval/step_00000003/flux2vae_imagespace_eval_record.json
results/smoke_flux2vae_imagespace_eval/eval/step_00000003/decoded_flux2vae_png/{000000.png..000003.png,decode_summary.json,preview_grid.png}
logs/032_flux2vae_imagespace_eval_smoke.log
```

Interpretation: FLUX route now covers full-cache dataloading, class-conditional training, live-param fp32 JVP target, bf16 backbone, r=t sampler/degenerate path, checkpointing, latent sampling, and image-space decode eval. The only remaining metric work is to provide a real ImageNet-256 reference FID/MMD command; no fake FID number is emitted.

## 2026-05-28 — Real ImageNet-256 reference stats and FID/MMD wired into FLUX image-space eval

Task: build real ImageNet-256 reference stats and connect `eval_flux2vae_imagespace.py --fid-command-template ...` to a real FID/MMD command.

Implemented:

- `scripts/inception_metrics_lib.py`
  - Inception feature extraction compatible with the LightningDiT/pytorch-fid Inception path.
  - Frechet/FID computed without SciPy; small generated-smoke cases use a low-rank exact path.
  - Polynomial degree-3 unbiased MMD/KID implementation.
  - Metric image enumeration excludes `preview_grid.png`, `contact_sheet.*`, and `preview_*` sidecars.
- `scripts/build_imagenet256_reference_stats.py`
  - Reads local cropped ImageNet-256 safetensors cache directly: CHW uint8 images + labels/source indices.
  - Outputs full reference `mu`/`sigma` and deterministic real-feature bank for MMD/KID.
- `scripts/eval_imagespace_fid_mmd.py`
  - Computes FID plus MMD/KID for a decoded PNG/JPEG directory.
  - Emits JSON fields `fid`, `FID`, `mmd2`, `kid`, `kid_x1000`, `num_images`, `ref_count`, and `ref_features_used`.
- `scripts/eval_flux2vae_imagespace.py`
  - Parses inner JSON metrics and forwards FID/MMD/KID fields.
- `scripts/train_b3_meanflow_realdata.py`
  - Trainer eval record now stores inner metric JSON and propagates `fid`, `mmd2`, `kid`, `kid_x1000`, `num_images`, and `ref_features_used`.
- `configs/b3_meanflow_flux2vae_imagespace_fid_mmd_smoke.yaml`
- `scripts/run_flux2vae_imagespace_fid_mmd_smoke.sh`

Reference stats build command:

```bash
python experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/build_imagenet256_reference_stats.py \
  --cache-dir /workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors \
  --output-stats /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_stats.npz \
  --output-summary /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/refstats_summary.json \
  --output-mmd-features /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_mmd8192_features.npz \
  --mmd-ref-samples 8192 \
  --batch-size 1024 \
  --device cuda:0 \
  --progress-every-batches 20
```

Full reference stats result:

```json
{
  "status": "ok",
  "count": 1281167,
  "dims": 2048,
  "num_shards": 313,
  "allow_tf32": false,
  "batch_size": 1024,
  "elapsed_sec": 2592.584533799207,
  "samples_per_sec": 494.16594991991946,
  "cuda_peak_memory_mb": 43027.28955078125,
  "sigma_trace": 180.5874165407635
}
```

Validation:

```text
stats .npz: mu [2048] float64, sigma [2048,2048] float64, count 1281167, dims 2048
MMD feature bank: features [8192,2048] float32, global_indices [8192] int64, first/last 0/1281166
```

Trainer-integrated smoke command:

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_imagespace_fid_mmd_smoke.sh
```

Trainer-integrated smoke result:

```json
{
  "gate": true,
  "final_step": 3,
  "final_loss": 5.176722526550293,
  "fd_rel": 0.00010939302601559381,
  "degen": 0.0,
  "image_eval_status": "decoded",
  "fid_status": "ok",
  "num_images": 4,
  "fid": 456.339409075125,
  "mmd2": 0.45975885396356464,
  "kid": 0.45975885396356464,
  "kid_x1000": 459.75885396356466,
  "ref_count": 1281167,
  "ref_features_used": 8192,
  "decode_cuda_peak_memory_mb": 1296.86376953125,
  "metrics_cuda_peak_memory_mb": 476.03271484375,
  "metrics_elapsed_sec": 2.181860954966396
}
```

Interpretation:

- The requested wiring is complete: `train_b3_meanflow_realdata.py -> eval_flux2vae_imagespace.py -> eval_imagespace_fid_mmd.py -> real ImageNet-256 stats`.
- The full real reference stats cover all `1,281,167` training images in the local ADM-style cropped cache.
- The 4-image smoke verifies command plumbing, JSON propagation, PNG sidecar exclusion, and full reference-stat loading. The smoke FID/KID number is not a publishable quality result.
- Reference `.npz` files remain local data artifacts and should not be included in a lightweight code repo upload by default.

## 2026-05-28 — FLUX.2 full-cache short-train + 1024-image real ImageNet-256 FID/MMD pilot

Task: run the FLUX.2 route on the complete ImageNet-1k latent cache, then evaluate 1024 EMA samples through FLUX.2 VAE decode and real ImageNet-256 FID/MMD/KID.

Command:

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_fullcache_imageeval_pilot.sh
```

Config/script:

```text
configs/b3_meanflow_flux2vae_fullcache_imageeval_pilot.yaml
scripts/run_flux2vae_fullcache_imageeval_pilot.sh
```

Result summary:

```json
{
  "gate": true,
  "final_step": 100,
  "final_loss": 3.4786744117736816,
  "fd_rel": 0.00009486990313151122,
  "degen": 0.0,
  "mean_realized_r_eq_t_fraction": 0.744375,
  "sample_status": "ok",
  "sample_shape": [1024, 32, 32, 32],
  "sample_batch_size": 16,
  "num_sample_batches": 64,
  "sample_elapsed_sec": 12.468012206023559,
  "image_eval_status": "decoded",
  "decoded_metric_png_count": 1024,
  "decode_elapsed_sec": 109.26908582891338,
  "decode_effective_images_per_sec": 9.371360547514001,
  "fid_status": "ok",
  "fid": 382.1860103089217,
  "mmd2": 0.4709882374694505,
  "kid_x1000": 470.9882374694505,
  "ref_count": 1281167,
  "ref_features_used": 8192,
  "metrics_elapsed_sec": 9.787689667893574,
  "metrics_cuda_peak_memory_mb": 1332.16064453125
}
```

Interpretation:

- Full FLUX.2 cache DataLoader path is usable: `313` shards, `1,281,167` samples, latent shape `[32,32,32]`, labels injected class-conditionally.
- Mixed precision route remains correct: bf16 backbone + fp32 live-param JVP target, no EMA target JVP; FD audit at step 100 is `9.49e-05`.
- r=t sampler is active and close to configured 75% (`0.744375` realized); degenerate path max error is `0.0`.
- Trainer-integrated image-space eval works end-to-end with chunked sampling, FLUX.2 decode, and real FID/MMD/KID command.
- FID/KID numbers are pilot-only because this is a tiny 100-step model and only 1024 generated images.

Local artifacts:

```text
results/flux2vae_fullcache_imageeval_pilot_100s_1024img_summary.json
results/pilot_flux2vae_fullcache_imageeval_100s_1024img/train_summary.json
results/pilot_flux2vae_fullcache_imageeval_100s_1024img/train_summary.md
results/pilot_flux2vae_fullcache_imageeval_100s_1024img/eval/step_00000100/fid_mmd_metrics_real_imagenet256_1024.json
results/pilot_flux2vae_fullcache_imageeval_100s_1024img/eval/step_00000100/flux2vae_imagespace_eval_record.json
results/pilot_flux2vae_fullcache_imageeval_100s_1024img/eval/step_00000100/decoded_flux2vae_png_realmetrics_1024/  # heavy, do not upload
results/pilot_flux2vae_fullcache_imageeval_100s_1024img/checkpoints/  # heavy, do not upload
```

## 2026-05-28 — FLUX.2 full-cache 1k-step short run with repeated 1024-image real FID/MMD eval

Task: continue the FLUX.2 route after the 100-step pilot by patching large-eval decode summaries and running a 1k-step full-cache short train with 1024-image real ImageNet-256 FID/MMD/KID at steps 500 and 1000.

Implementation patch:

```text
scripts/decode_flux2vae_latents.py
  + --manifest-mode {full,first_last,none}

scripts/eval_flux2vae_imagespace.py
  + --decode-manifest-mode {full,first_last,none}
  default: first_last
```

Validation: both step-500 and step-1000 decode summaries used `images_manifest_mode=first_last`, `images_manifest_omitted=true`, `images_manifest_count=0`, and did not contain a full `images_manifest` field. Metric PNG count remained 1024 per eval.

Command:

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_fullcache_1k_imageeval.sh
```

Result summary:

```json
{
  "gate": true,
  "final_step": 1000,
  "final_loss": 2.915365695953369,
  "min_loss": 2.1960299015045166,
  "last_fd_rel": 0.00032816254595992897,
  "max_fd_rel": 0.0009662954806282971,
  "degen": 0.0,
  "mean_realized_r_eq_t_fraction": 0.746625,
  "loop_peak_memory_mb": 168.74365234375,
  "total_elapsed_sec": 485.92449224786833
}
```

Eval table:

| step | FID | MMD2/KID | KID x1000 | sample sec | decode sec | metric sec | manifest |
|---:|---:|---:|---:|---:|---:|---:|---|
| 500 | 381.9068369846536 | 0.4705451225255155 | 470.5451225255155 | 12.354675956070423 | 113.98510063579306 | 10.714315253077075 | first_last/no full manifest |
| 1000 | 380.83573517219475 | 0.46804980426009113 | 468.04980426009115 | 12.050162501865998 | 104.19804051890969 | 11.387822730932385 | first_last/no full manifest |

Trend vs earlier 100-step pilot:

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

Interpretation:

- Full-cache train/eval route remains stable over 1k steps.
- FD audit remains comfortably below `1e-2`; highest observed FD rel in this run was `9.66e-04`.
- FID/KID moved slightly downward from 100-step to 1k-step, but this is only a weak pilot trend because the model/eval scale is small.
- The full images manifest has been successfully removed from decode summaries for this route, preparing for larger 5k/10k/50k image evals.

Local artifacts:

```text
results/flux2vae_fullcache_1k_imageeval_summary.json
results/shorttrain_flux2vae_fullcache_1k_1024img/train_summary.json
results/shorttrain_flux2vae_fullcache_1k_1024img/eval/step_00000500/fid_mmd_metrics_real_imagenet256_1024.json
results/shorttrain_flux2vae_fullcache_1k_1024img/eval/step_00001000/fid_mmd_metrics_real_imagenet256_1024.json
results/shorttrain_flux2vae_fullcache_1k_1024img/eval/step_00000500/decoded_flux2vae_png_realmetrics_1024/  # heavy, do not upload
results/shorttrain_flux2vae_fullcache_1k_1024img/eval/step_00001000/decoded_flux2vae_png_realmetrics_1024/  # heavy, do not upload
results/shorttrain_flux2vae_fullcache_1k_1024img/checkpoints/  # heavy, do not upload
```

## 2026-05-28 — FLUX.2 B3-medium memory/batch calibration and 1k smoke

Task: execute the next scale step after the FLUX.2 tiny route: calibrate B3-medium memory/batch settings, then run a 1k-step real full-cache FLUX.2 short train with 1024-image real ImageNet-256 FID/MMD/KID eval.

Implementation patch:

```text
scripts/train_b3_meanflow_realdata.py
  sample_meanflow_latents(): add torch.no_grad() around EMA/inference sampling
```

Reason: eval sampling should not retain autograd graphs, especially after increasing `eval.sample_batch_size` from 16 to 32 for the medium model. This does not change the training target path; target JVP remains live-param fp32 and detached.

Calibration runs:

```bash
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_medium_calib_b16_50s.sh
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_medium_calib_b32_20s.sh
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_medium512_calib_b16_20s.sh
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_medium512_calib_b64_20s.sh
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_medium512_calib_b128_20s.sh
```

Calibration results:

| model | params | batch | steps | final loss | max FD rel | r=t mean | loop peak MB | eff samples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h384/d8/heads6 | 22.6606M | 16 | 50 | 2.8721611499786377 | 0.00020186840536935322 | 0.7275 | 1440.2607421875 | 58.275970818800054 |
| h384/d8/heads6 | 22.6606M | 32 | 20 | 4.220029830932617 | 0.00007870426877958057 | 0.75625 | 2439.2255859375 | 81.69916392329296 |
| h512/d12/heads8 | 58.8192M | 16 | 20 | 3.8391733169555664 | 0.0000989313113513627 | 0.759375 | 3022.45556640625 | 34.830103093051086 |
| h512/d12/heads8 | 58.8192M | 64 | 20 | 3.5658743381500244 | 0.00011434621593090143 | 0.759375 | 8913.51513671875 | 41.92922497839281 |
| h512/d12/heads8 | 58.8192M | 128 | 20 | 3.5324220657348633 | 0.0000800312485497428 | 0.766015625 | 16783.7314453125 | 59.89824844510914 |

Selected setting for 1k smoke: `h512/d12/heads8`, `global_batch_size=128`. It leaves large 96GB GPU headroom while giving better useful scale than the tiny model.

1k command:

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_medium512_b128_1k_imageeval.sh
```

1k summary:

```json
{
  "gate": true,
  "model_params_m": 58.819192,
  "global_batch_size": 128,
  "final_step": 1000,
  "final_loss": 2.2662835121154785,
  "min_loss": 1.9191009998321533,
  "last_fd_rel": 0.00017521927982342016,
  "max_fd_rel": 0.0011974215362778894,
  "degen": 0.0,
  "mean_realized_r_eq_t_fraction": 0.75159375,
  "loop_peak_memory_mb": 16783.7314453125,
  "total_elapsed_sec": 1973.0546104258392,
  "effective_train_samples_per_sec": 64.8740279785637
}
```

Eval table:

| step | FID | MMD2/KID | KID x1000 | sample sec | decode sec | metric sec | sample peak MB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | 346.9461242148833 | 0.40237275729309285 | 402.37275729309283 | 11.524835258023813 | 115.33161097788252 | 9.750014807796106 | 1506.23828125 |
| 1000 | 348.38588136862217 | 0.40678265356824905 | 406.78265356824903 | 11.21557610318996 | 133.1801801288966 | 18.63674869108945 | 1504.4267578125 |

Interpretation:

- Medium full-cache route passes practical gate with checkpoints, real image-space eval, and all key MeanFlow diagnostics.
- Quality is substantially better than the tiny 1k route (`FID≈380.84` -> `348.39`; `KIDx1000≈468.05` -> `406.78` at 1024-image eval), but this is still a short-run smoke, not a publishable FID.
- Step-500 metric is slightly better than step-1000; do not claim monotonic convergence.
- Highest FD rel observed was `1.197e-3`, below the current practical threshold `1e-2` but worth monitoring in longer runs.

Local lightweight summaries:

```text
results/flux2vae_b3medium_calibration_summary.json
results/flux2vae_b3medium_h512_b128_1k_imageeval_summary.json
```

Heavy local artifacts, do not upload by default:

```text
results/shorttrain_flux2vae_b3medium_h512_d12_b128_1k_1024img/checkpoints/
results/shorttrain_flux2vae_b3medium_h512_d12_b128_1k_1024img/eval/*/sample_latents.safetensors
results/shorttrain_flux2vae_b3medium_h512_d12_b128_1k_1024img/eval/*/decoded_flux2vae_png_realmetrics_1024/
```

## 2026-05-28 — STARTED: FLUX.2 B3-medium h512/d12 b128 5k image-space short train

Task: start the next post-1k FLUX.2 route validation run: `h512/depth12/heads8`, `global_batch_size=128`, full FLUX.2 ImageNet-256 latent cache, live-param fp32 JVP target, bf16 backbone, r=t sampler, and real image-space ImageNet-256 FID/MMD/KID eval.

Config/script:

```text
configs/b3_meanflow_flux2vae_fullcache_medium512_b128_5k_imageeval.yaml
scripts/run_flux2vae_medium512_b128_5k_imageeval.sh
```

Planned settings:

```text
max_steps=5000
global_batch_size=128
fd_audit_every=250
checkpoint_every=1000 / keep_last_checkpoints=3
eval.every_steps=1000
eval.num_samples=1024
eval.sample_batch_size=32
eval.use_ema=true
fid/mmd reference=/workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048
```

Expected purpose:

- Check whether the non-PAE FLUX.2 latent route improves beyond the 1k smoke (`FID≈348.39`, `KIDx1000≈406.78`).
- Monitor FD rel; previous max was `1.197e-3`, threshold remains `1e-2`.
- Keep upload hygiene: checkpoints, sample latents, decoded PNGs, and raw stats remain local-only unless explicitly requested.

Status: launched in background; final metrics to be appended after completion.

Launch note: initial `nohup` launch exited before training steps in this execution environment (only startup log/config written, no checkpoints or metrics). The empty startup output directory was removed and the same run was relaunched inside a detached `tmux` session for persistence.

