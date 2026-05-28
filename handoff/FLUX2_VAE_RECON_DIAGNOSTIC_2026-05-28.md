# FLUX.2 VAE Reconstruction Diagnostic — 2026-05-28

## Status

Completed on `2026-05-28` in `/workspace/PDM`.

The diagnostic was requested before continuing FLUX-latent B3 training, to decide whether current FLUX ImageNet-256 B3 degradation is mainly a VAE reconstruction/domain-ceiling problem or a training/configuration problem.

## Environment / route

- Input images: real ImageNet-256 ADM-cropped uint8 cache.
- VAE: `diffusers.AutoencoderKLFlux2` from `diffusers/FLUX.2-dev-bnb-4bit`, subfolder `vae`.
- Encode convention: `vae.encode(x).latent_dist.mode()`.
- Latent convention: raw latent, no SD-style external scale/shift.
- Decode convention: raw `vae.decode(z).sample`, then `[-1,1] -> uint8` conversion.
- Device: `cuda:0`, RTX PRO 6000 Blackwell Server Edition.
- Main run dtype: strict `fp32`, TF32 disabled.
- Main run samples: `1024` first ImageNet train crops.

## New script

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/diagnose_flux2vae_reconstruction.py
```

The script produces:

- reconstructed PNGs;
- pixel reconstruction metrics;
- global/per-channel latent stats;
- optional FID/MMD/KID against real ImageNet-256 reference stats;
- preview grid and original/reconstruction contact sheet;
- JSON and Markdown summaries.

## Main command

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

## Local outputs

Do not commit/upload these by default:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_reconstruction_diagnostic_1024_fp32_strict/
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_reconstruction_diagnostic_smoke8/
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/054_flux2vae_reconstruction_diag_smoke8_20260528T115709Z.log
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/054_flux2vae_reconstruction_diag_1024_fp32_strict_20260528T115802Z.log
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/055_imagenet_real1024_baseline_fid_mmd_20260528T120341Z.log
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/056_flux2vae_recon_vs_input1024_paired_feature_diag_20260528T120424Z.log
```

Output size at completion:

```text
flux2vae_reconstruction_diagnostic_1024_fp32_strict: ~173 MB
flux2vae_reconstruction_diagnostic_smoke8: ~8.6 MB
```

## Metrics

All rows below use 1024 first ImageNet train crops unless noted. The full reference is the existing `1,281,167`-image ImageNet-256 ADM-crop Inception-2048 reference with 8192 features for MMD/KID.

| comparison | FID ↓ | MMD2/KID ↓ | KID x1000 ↓ | images | interpretation |
|---|---:|---:|---:|---:|---|
| FLUX.2 reconstruction vs full real ref | `44.325107` | `1.83545e-05` | `0.01835` | 1024 | VAE reconstruction sample-limited ceiling |
| Real first-1024 inputs vs full real ref | `44.400165` | `-8.34277e-06` | `-0.00834` | 1024 | finite-sample baseline |
| FLUX.2 reconstruction vs same first-1024 inputs | `2.744717` | `-6.12531e-04` | `-0.61253` | 1024 | paired reconstruction distribution drift |

Pixel metrics:

| metric | value |
|---|---:|
| MSE `[0,1]` | `0.0008338014` |
| RMSE `[0,1]` | `0.02887562` |
| MAE `[0,1]` | `0.01673313` |
| global PSNR | `30.78937 dB` |
| mean per-image PSNR | `33.07361 dB` |
| median per-image PSNR | `32.90166 dB` |
| max abs uint8 error | `227` |

Latent / decode stats:

| stat | value |
|---|---:|
| latent shape | `[B, 32, 32, 32]` |
| latent mean | `-0.006279866` |
| latent std | `1.720978` |
| latent min | `-13.357826` |
| latent max | `12.272459` |
| latent RMS | `1.720990` |
| decoded float min / max before clipping | `-2.125535 / 1.737441` |
| encode+decode throughput | `25.769 images/sec` |
| total wall time | `289.64 sec` |
| CUDA peak | `12.44 GB` |

Paired Inception feature drift:

```json
{
  "paired_feature_cosine_mean": 0.991840691139065,
  "paired_feature_l2_mean": 2.167182747933199
}
```

## Conclusion

The FLUX.2 VAE reconstruction ceiling is **not** the main explanation for the current FLUX B3-medium ImageNet-256 FID deterioration.

Why:

- Reconstruction-vs-full-reference FID is `44.3251`.
- Real first-1024-vs-full-reference finite-sample FID is `44.4002`.
- These are effectively equal at this sample size.
- Reconstruction-vs-same-inputs FID is only `2.7447`, with paired feature cosine mean `0.99184`.

Therefore, FLUX.2 VAE can reconstruct these ImageNet-256 ADM crops well under the raw latent convention used by the cache. The current B3-medium `FID 350 -> 421` short-run problem is much more likely from the latent modeling/training side:

1. latent normalization/scale mismatch (`std ~= 1.72`, 32x32 spatial grid);
2. hyperparameters copied from PAE without FLUX-specific retuning;
3. patchification/model-capacity/sampler/LR mismatch;
4. class-conditional prior not learned in the short training window.

## Recommended next step

Do **not** abandon FLUX/Qwen as modern VAE baselines. Instead:

1. Add a FLUX latent normalization/standardization branch to B3 training.
2. Run a short normalized-FLUX B3 smoke and compare against the existing unnormalized 1k/5k runs.
3. If writing paper framing, state that VAE backends are empirical backends; the innovation claim remains HT + MeanFlow/per-patch trajectory decoupling.
4. Later run the same reconstruction-ceiling diagnostic for PAE and Qwen for honest backend comparison.

## Caveats

- This is a 1024-sample diagnostic, not official 50k FID.
- It only tests deterministic reconstruction through `posterior.mode()`, not generative prior quality.
- It does not prove FLUX is ImageNet-optimal; it only rules out a severe reconstruction-ceiling failure under current preprocessing and raw-latent convention.
