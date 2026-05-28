# FLUX scale-calibrated eval diagnostic — 2026-05-28

## TL;DR

按上一轮结论，没有直接启动更长的 FLUX scalar-normalized 训练，而是先对同一个 **scale-aware B3 step-1000** 样本做了 eval-only 的尺度校准诊断。

结论：**std-match 自适应 decode scale 能明显修复 official inverse scale 的过放大问题，但仍没有超过 identity decode ablation。**

| decode variant, same step-1000 samples | pre-decode scale | decode-space std | std/ref | FID ↓ | MMD2/KID ↓ | KID x1000 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| official inverse of train multiplier | `1.714043` | `2.677575` | `1.562` | `391.8537` | `0.514579` | `514.579` |
| identity eval-only ablation | `1.0` | `1.562140` | `0.911` | `345.6951` | `0.399690` | `399.690` |
| adaptive ref-std match | `1.0972403574` | `1.714043` | `1.000` | `354.0362` | `0.421643` | `421.643` |

Interpretation:

1. Official inverse scale was indeed too large for these early generated samples: `1.562 * 1.714 = 2.678`, far above real FLUX raw latent std `1.714`.
2. Matching generated decode-space std to real FLUX raw std improves official inverse-scale FID by `~37.82` points (`391.85 -> 354.04`).
3. However, identity decode remains best on this 1024-image smoke (`345.70`). Therefore scalar std matching alone is **not** a final solution.
4. This still supports the main diagnosis: current FLUX issue is sampler/decode scale calibration and/or latent distribution shape, not VAE reconstruction or JVP correctness.

Do **not** report this as a final FLUX result. It is a scale-policy diagnostic on a 1k-step / 1024-image smoke.

---

## Code changes made

Two eval scripts now record sample-space and decode-space latent stats:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/decode_flux2vae_latents.py
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/eval_flux2vae_imagespace.py
```

New fields include:

```text
sample_space_mean/std/rms/absmax
pre_decode_scale/pre_decode_shift
decode_space_mean/std/rms/absmax
reference_raw_flux_mean/std/rms
decode_space_std_over_reference_raw_flux_std
sample_space_std_over_reference_raw_flux_std
```

Reference FLUX raw latent stats embedded for diagnostics only:

```json
{
  "reference_raw_flux_mean": -0.009054178792269978,
  "reference_raw_flux_std": 1.7140430386576422,
  "reference_raw_flux_rms": 1.7140669521708671,
  "source": "compute_latent_cache_stats.py linspace32 shards / 130191 ImageNet-256 train latents"
}
```

Validation:

```bash
python3 -m py_compile \
  experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/eval_flux2vae_imagespace.py \
  experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/decode_flux2vae_latents.py
```

---

## Run details

Base run:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/smoke_flux2vae_b3medium_h512_d12_b128_1k_scaleaware_1024img/
```

Input samples:

```text
.../eval/step_00001000/sample_latents.safetensors
```

Adaptive output directory, local/ignored:

```text
.../eval/step_00001000_refstd_decode_ablate/
```

Log, local/ignored:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/058_flux2vae_scaleaware_step1000_refstd_decode_ablate_20260528T131922Z.log
```

Result JSONs, local/ignored:

```text
.../eval/step_00001000_refstd_decode_ablate/flux2vae_imagespace_eval_record.json
.../eval/step_00001000_refstd_decode_ablate/fid_mmd_metrics_real_imagenet256_1024_refstd_decode.json
.../eval/step_00001000_refstd_decode_ablate/decoded_flux2vae_png_realmetrics_1024_refstd_decode_ablate/decode_summary.json
```

Command:

```bash
cd /workspace/PDM
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
ROOT=$EXP/results/smoke_flux2vae_b3medium_h512_d12_b128_1k_scaleaware_1024img
AB=$ROOT/eval/step_00001000_refstd_decode_ablate
SCALE=1.0972403573842702
mkdir -p "$AB"
ln -sf ../step_00001000/sample_latents.safetensors "$AB/sample_latents.safetensors"

python "$EXP/scripts/eval_flux2vae_imagespace.py" \
  --sample-dir "$AB" \
  --output-dir "$ROOT" \
  --step 1000 \
  --device cuda:0 \
  --batch-size 8 \
  --max-images 1024 \
  --images-subdir decoded_flux2vae_png_realmetrics_1024_refstd_decode_ablate \
  --pre-decode-scale "$SCALE" \
  --pre-decode-shift 0.0 \
  --save-grid \
  --decode-manifest-mode first_last \
  --fid-timeout-sec 7200 \
  --fid-command-template "python $EXP/scripts/eval_imagespace_fid_mmd.py --images-dir {images_dir} --ref-stats /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_stats.npz --ref-features /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_mmd8192_features.npz --output-json $AB/fid_mmd_metrics_real_imagenet256_1024_refstd_decode.json --batch-size 64 --num-workers 4 --device cuda:0 --sqrt-device cuda:0 --mmd-device cuda:0 --mmd-max-ref 8192"
```

Scale derivation:

```text
sample_space_std = 1.5621399879455566
target_raw_flux_std = 1.7140430386576422
adaptive_scale = target_raw_flux_std / sample_space_std = 1.0972403573842702
```

---

## Adaptive eval result

Single-line summary from `flux2vae_imagespace_eval_record.json`:

```json
{
  "created_at_utc": "2026-05-28T13:19:26+00:00",
  "pre_decode_scale": 1.0972403573842702,
  "sample_space_mean": -0.030830515548586845,
  "sample_space_std": 1.5621399879455566,
  "sample_space_rms": 1.5624442100524902,
  "decode_space_mean": -0.03382848589887273,
  "decode_space_std": 1.714043038657642,
  "decode_space_rms": 1.714376843430978,
  "reference_raw_flux_mean": -0.009054178792269978,
  "reference_raw_flux_std": 1.7140430386576422,
  "reference_raw_flux_rms": 1.7140669521708671,
  "fid": 354.03623173360427,
  "mmd2": 0.4216426250589018,
  "kid_x1000": 421.64262505890184,
  "decode_elapsed_sec": 98.7861391659826,
  "inner_fid_elapsed_sec": 13.223148401128128,
  "elapsed_sec": 116.30091026215814
}
```

---

## Comparison against existing FLUX short runs

| run / variant | train scale policy | eval decode policy | step | FID ↓ | MMD2/KID ↓ | note |
|---|---|---|---:|---:|---:|---|
| raw FLUX B3-medium 1k | raw latents | identity | 1000 | `348.3859` | `0.406783` | previous raw baseline |
| scalar-normalized B3 | `latent_multiplier=1/1.714` | official inverse `1.714` | 1000 | `391.8537` | `0.514579` | semantically correct inverse but over-amplifies early samples |
| scalar-normalized B3 | `latent_multiplier=1/1.714` | identity ablation `1.0` | 1000 | `345.6951` | `0.399690` | best smoke metric, but not a principled final inverse |
| scalar-normalized B3 | `latent_multiplier=1/1.714` | adaptive std-match `1.09724` | 1000 | `354.0362` | `0.421643` | principled eval diagnostic; fixes std but not all distribution mismatch |

---

## Decision / next step

Do not spend GPU on a longer scalar-normalized FLUX run yet.

Recommended next engineering step:

1. Implement **per-channel normalization inverse decode** for FLUX eval:
   - train with existing `data.latent_norm: true` and channel stats from `compute_latent_cache_stats.py`;
   - decode with `z_raw = z_norm * channel_std + channel_mean`.
2. Run a short `B3-medium b128 1k` smoke with per-channel normalization.
3. Compare four variants on the same 1024-image eval protocol:
   - raw FLUX baseline,
   - scalar-normalized official inverse,
   - scalar-normalized adaptive eval diagnostic,
   - per-channel normalized official inverse.

Rationale:

- Scalar std matching improved official inverse but did not beat identity/raw.
- The remaining mismatch is likely not one scalar; FLUX latent distribution has channel-wise / spatial / tail structure that the scalar policy does not normalize.
- VAE reconstruction remains healthy, so FLUX route should stay alive as a modern-VAE baseline, especially for later T2I/general-image experiments.
