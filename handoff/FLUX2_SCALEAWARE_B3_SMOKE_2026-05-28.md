# FLUX normalized / scale-aware B3 smoke — 2026-05-28

## TL;DR

The requested **FLUX normalized / scale-aware B3 smoke** has been completed on the full ImageNet-256 FLUX.2 VAE latent cache.

Mechanical pipeline result: **PASS**.

- Full-cache FLUX latent DataLoader works: `1,281,167` samples, `[32,32,32]` latents.
- Scalar latent scaling was applied during training:
  - `latent_multiplier = 0.5834159221481118 = 1 / 1.7140430386576422`.
- Official scale-aware image eval applied inverse scalar decode:
  - `pre_decode_scale = 1.7140430386576422`, `pre_decode_shift = 0`.
- Mixed precision recipe works:
  - `bf16` backbone + `fp32` JVP target.
- Live-param JVP target verified:
  - `jvp_param_source_all_live = true`.
  - `ema_used_for_target_any_step = false`.
- `r=t` sampler behavior is correct:
  - realized `r=t` fraction: `0.7491875`.
  - degenerate target check: `all_r_eq_t_degenerate_target_minus_v_max_abs = 0.0`.
- FD/JVP audit is healthy:
  - final `fd_rel_err_full_jvp = 3.9654e-4`, all audits `< 4e-4` except initial still `6.64e-5`.
- Memory logging works:
  - training loop peak `16.76 GB`.

Quality result: **the official inverse-scale decode is worse than raw FLUX baseline at 1k**.

- Official scale-aware inverse decode at step 1000: `FID 391.8537`, `MMD2/KID 0.514579`.
- Raw FLUX 1k baseline at step 1000: `FID 348.3859`, `MMD2/KID 0.406783`.
- An eval-only ablation on the same scale-aware step-1000 samples with **no inverse decode scale** gives `FID 345.6951`, `MMD2/KID 0.399690`.

Interpretation: scalar normalization did not break training/JVP, but the early 1k sampler in normalized latent space is over-dispersed (`sample_std=1.562` in normalized space). Multiplying by `1.714` before decode produces raw-equivalent std `~2.678`, above the real FLUX raw latent std `~1.714`, which likely explains the worse official inverse-scale image FID. The no-inverse ablation is not the semantically correct inverse transform, but it confirms that the immediate failure mode is **decode/sample scale calibration**, not the training loop or JVP chain.

---

## Experiment definition

Config added:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_flux2vae_fullcache_medium512_b128_1k_scaleaware_imageeval.yaml
```

Run script added:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_medium512_b128_1k_scaleaware_imageeval.sh
```

Eval wrapper change:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/eval_flux2vae_imagespace.py
```

New eval args:

```bash
--pre-decode-scale
--pre-decode-shift
```

These are forwarded to:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/decode_flux2vae_latents.py
```

Latent stats script added earlier for the scale estimate:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/compute_latent_cache_stats.py
```

Scale source:

```text
FLUX full-cache linspace 32-shard scan, latents only:
mean = -0.009054178792269978
std  = 1.7140430386576422
RMS  = 1.7140669521708671
```

Training/eval scale policy:

```yaml
data:
  latent_norm: false
  latent_multiplier: 0.5834159221481118  # 1 / 1.7140430386576422

eval fid command:
  --pre-decode-scale 1.7140430386576422
  --pre-decode-shift 0.0
```

Important wording: this is **scalar scale-aware**, not yet per-channel `latent_norm: true` with inverse per-channel decode.

---

## Commands

Official scale-aware smoke:

```bash
cd /workspace/PDM
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_medium512_b128_1k_scaleaware_imageeval.sh
```

Log:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/053_flux2vae_medium512_b128_1k_scaleaware_imageeval_20260528T124057Z.log
```

Output directory, local/ignored:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/smoke_flux2vae_b3medium_h512_d12_b128_1k_scaleaware_1024img/
```

Additional eval-only ablation on the same step-1000 samples, without inverse decode scaling:

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
ROOT=$EXP/results/smoke_flux2vae_b3medium_h512_d12_b128_1k_scaleaware_1024img
AB=$ROOT/eval/step_00001000_scale1decode_ablate
mkdir -p "$AB"
ln -sf ../step_00001000/sample_latents.safetensors "$AB/sample_latents.safetensors"
python "$EXP/scripts/eval_flux2vae_imagespace.py" \
  --sample-dir "$AB" \
  --output-dir "$ROOT" \
  --step 1000 \
  --device cuda:0 \
  --batch-size 8 \
  --max-images 1024 \
  --images-subdir decoded_flux2vae_png_realmetrics_1024_scale1decode_ablate \
  --pre-decode-scale 1.0 \
  --pre-decode-shift 0.0 \
  --save-grid \
  --decode-manifest-mode first_last \
  --fid-timeout-sec 7200 \
  --fid-command-template "python $EXP/scripts/eval_imagespace_fid_mmd.py --images-dir {images_dir} --ref-stats /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_stats.npz --ref-features /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_mmd8192_features.npz --output-json $AB/fid_mmd_metrics_real_imagenet256_1024_scale1decode.json --batch-size 64 --num-workers 4 --device cuda:0 --sqrt-device cuda:0 --mmd-device cuda:0 --mmd-max-ref 8192"
```

Ablation log:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/057_flux2vae_scaleaware_step1000_scale1decode_ablate_20260528T130450Z.log
```

---

## Training health

Run summary:

```json
{
  "final_step": 1000,
  "final_loss": 1.2407240867614746,
  "min_loss": 1.1974533796310425,
  "max_loss": 3.7270028591156006,
  "optimizer_steps_this_run": 1000,
  "loss_all_finite": true,
  "grad_all_finite": true,
  "u_du_target_all_finite": true,
  "target_detached_all_steps": true,
  "jvp_param_source_all_live": true,
  "ema_used_for_target_any_step": false,
  "mixed_modes_ok": true,
  "class_cond_all_steps": true,
  "mean_realized_r_eq_t_fraction": 0.7491875,
  "max_actual_rt_samples_target_minus_v_abs": 0.0,
  "loop_peak_memory_mb": 16757.45458984375,
  "total_elapsed_sec": 1109.9494186439551
}
```

Gate summary:

```json
{
  "train_ok": true,
  "class_cond_ok": true,
  "live_param_jvp_target": true,
  "ema_not_used_for_jvp_target": true,
  "mixed_precision_recipe_ok": true,
  "fd_ok_lt_1e_2": true,
  "fd_rel_err_full_jvp": 0.0003965412291613331,
  "degenerate_ok_lt_1e_7": true,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0,
  "checkpoint_ok": true,
  "eval_sample_ok": true,
  "memory_logged": true,
  "practical_gate_pass": true
}
```

Loss trace highlights:

| step | loss |
|---:|---:|
| 50 | `1.72125` |
| 100 | `1.60992` |
| 200 | `1.49908` |
| 300 | `1.36148` |
| 400 | `1.33123` |
| 500 | `1.32578` |
| 600 | `1.28986` |
| 700 | `1.30100` |
| 800 | `1.24533` |
| 900 | `1.23790` |
| 1000 | `1.24072` |

FD/JVP audits:

| step | FD relative error | r=t degenerate target-v max | JVP peak MB | FD peak MB |
|---:|---:|---:|---:|---:|
| 1 | `6.64177e-05` | `0.0` | `3694.71` | `2413.04` |
| 100 | `1.16406e-04` | `0.0` | `3693.57` | `2411.89` |
| 200 | `3.94297e-04` | `0.0` | `3693.57` | `2411.89` |
| 300 | `2.14860e-04` | `0.0` | `3696.04` | `2414.37` |
| 400 | `3.48543e-04` | `0.0` | `3693.22` | `2411.54` |
| 500 | `1.95972e-04` | `0.0` | `3695.27` | `2413.60` |
| 600 | `2.16102e-04` | `0.0` | `3694.09` | `2412.42` |
| 700 | `3.54355e-04` | `0.0` | `3694.86` | `2413.19` |
| 800 | `3.49382e-04` | `0.0` | `3695.05` | `2413.38` |
| 900 | `3.52655e-04` | `0.0` | `3695.27` | `2413.61` |
| 1000 | `3.96541e-04` | `0.0` | `3694.86` | `2413.19` |

---

## Image-space eval results

Official scale-aware eval, inverse decode scale `1.7140430386576422`:

| step | sample std in trainer/normalized space | raw-equivalent std after decode scale | pre-decode scale | FID ↓ | MMD2/KID ↓ | KID x1000 ↓ | images |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | `1.583183` | `2.713643` | `1.714043` | `389.4661` | `0.508815` | `508.815` | 1024 |
| 1000 | `1.562140` | `2.677575` | `1.714043` | `391.8537` | `0.514579` | `514.579` | 1024 |

Raw FLUX B3-medium comparison:

| run | step | sample std in raw FLUX space | pre-decode scale | FID ↓ | MMD2/KID ↓ | KID x1000 ↓ | images |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw 1k | 500 | `1.593152` | `1.0` | `346.9461` | `0.402373` | `402.373` | 1024 |
| raw 1k | 1000 | `1.585804` | `1.0` | `348.3859` | `0.406783` | `406.783` | 1024 |
| raw 5k | 1000 | `1.586091` | `1.0` | `350.3428` | `0.411324` | `411.324` | 1024 |
| raw 5k | 2000 | `1.570263` | `1.0` | `359.3986` | `0.433757` | `433.757` | 1024 |
| raw 5k | 3000 | `1.553130` | `1.0` | `371.9172` | `0.464737` | `464.737` | 1024 |
| raw 5k | 4000 | `1.536760` | `1.0` | `391.8534` | `0.509279` | `509.279` | 1024 |
| raw 5k | 5000 | `1.523735` | `1.0` | `420.7126` | `0.574157` | `574.157` | 1024 |

Eval-only ablation on the **same scale-aware step-1000 sample_latents**:

| decode variant | sample std in saved sample space | raw-equivalent std | pre-decode scale | FID ↓ | MMD2/KID ↓ | KID x1000 ↓ | images |
|---|---:|---:|---:|---:|---:|---:|---:|
| official inverse-scale decode | `1.562140` | `2.677575` | `1.714043` | `391.8537` | `0.514579` | `514.579` | 1024 |
| no-inverse-scale decode ablation | `1.562140` | `1.562140` | `1.0` | `345.6951` | `0.399690` | `399.690` | 1024 |

The ablation is not a valid final scale policy by itself; it decodes normalized-space samples as if they were raw FLUX latents. It is useful diagnostically because it isolates the image-quality drop to decode/sample amplitude rather than the training/JVP/eval plumbing.

---

## Interpretation

1. **The engineering objective is achieved.** The full chain works with full-cache FLUX latents, class labels, mixed precision, live-param fp32 JVP target, r=t sampler, FD audit, checkpointing, EMA sampling, image decode, FID/MMD/KID, and memory logging.

2. **Scalar normalized training is numerically safe.** FD/JVP accuracy remains in the expected `1e-4 ~ 4e-4` range with TF32 off / math SDPA; r=t degeneration remains exactly correct.

3. **The official inverse-scale image quality is worse in this 1k smoke.** The normalized sampler is over-dispersed at 1k: it outputs `std≈1.56` in normalized latent space. Inverting the data scale before decode turns this into raw-equivalent `std≈2.68`, whereas true FLUX raw data std is `≈1.714`.

4. **The no-inverse ablation strongly suggests a scale-calibration issue.** Same model and same samples decode to `FID 345.7` if not multiplied by `1.714`, close to/slightly better than the raw 1k baseline (`FID 348.4`). Therefore the immediate problem is not “FLUX VAE bad” and not “JVP broken”; it is that the generated sample amplitude is not calibrated for the chosen inverse decode policy this early in training.

5. **Do not overclaim.** This is a 1k smoke with 1024-image FID/MMD/KID. It validates plumbing and reveals a scale policy issue; it is not a final FLUX backend quality comparison.

---

## Recommended next step

Before launching longer FLUX training, implement and run a **scale-calibrated eval/sampling diagnostic** rather than simply continuing the official inverse-scale run:

1. Add eval-time latent statistics logging after any inverse transform:
   - saved sample-space mean/std,
   - raw-decode-space mean/std,
   - true cache reference mean/std.
2. Add a controlled `sample_pre_decode_rescale_to_ref_std` diagnostic option, e.g. for eval only:
   - `z_decode = z_sample * (target_raw_std / sample_std)`
   - target raw std `1.7140430386576422` for scalar FLUX.
3. Then compare, on the same checkpoint/samples:
   - official inverse scale `1.714`;
   - identity decode scale `1.0`;
   - adaptive ref-std scale `1.714 / sample_std` if sample space is normalized;
   - eventually per-channel inverse normalization.
4. Only after that decide whether the next training run should be:
   - scalar normalized with retuned sampler/LR,
   - per-channel `latent_norm: true` plus inverse per-channel decode,
   - or raw FLUX latents with explicit endpoint/sampler noise scale.

For paper framing: keep this as a backend/scale-policy stress result. It supports the claim that FLUX/Qwen-style modern VAEs are viable baselines, but backend-specific latent scale and sampler calibration must be handled explicitly.

---

## Update — adaptive ref-std decode diagnostic completed

更新时间：`2026-05-28T13:22Z`

Detailed handoff:

```text
/workspace/PDM/handoff/FLUX2_SCALE_CALIBRATED_EVAL_2026-05-28.md
```

After the initial scale-aware smoke, an eval-only adaptive scale diagnostic was run on the same step-1000 samples. The decode scale was chosen to match generated decode-space std to the measured real FLUX raw latent std:

```text
sample_space_std = 1.5621399879455566
reference_raw_flux_std = 1.7140430386576422
adaptive_scale = 1.7140430386576422 / 1.5621399879455566 = 1.0972403573842702
```

Same samples, 1024-image real ImageNet-256 FID/MMD protocol:

| decode variant | pre-decode scale | decode-space std | FID ↓ | MMD2/KID ↓ | KID x1000 ↓ |
|---|---:|---:|---:|---:|---:|
| official inverse scale | `1.714043` | `2.677575` | `391.8537` | `0.514579` | `514.579` |
| identity ablation | `1.0` | `1.562140` | `345.6951` | `0.399690` | `399.690` |
| adaptive ref-std match | `1.097240` | `1.714043` | `354.0362` | `0.421643` | `421.643` |

Interpretation update:

- Matching std fixes the official inverse-scale over-amplification and improves FID by `~37.8` versus official inverse scale.
- It still does not beat identity decode or the raw-FLUX short baseline.
- Therefore scalar normalization is not enough evidence to launch a long FLUX run; next should be per-channel latent normalization plus inverse per-channel decode, or an explicit FLUX noise/sampler scale policy.
