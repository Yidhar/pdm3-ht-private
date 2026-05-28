# FLUX.2 VAE route summary — 2026-05-28

This handoff summarizes the FLUX.2 VAE backend work that was completed after the original experiment note had stopped at "upload launched". The detailed record is now backfilled in:

```text
experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/notes/EXPERIMENT_RECORD.md
```

## 1. Status snapshot

| Component | Status | Key result |
|---|---|---|
| FLUX.2 VAE latent cache | complete | ImageNet train `1,281,167` samples, `313` shards, `[32,32,32]` bf16 latents + flips |
| HF latent dataset | complete / public | <https://huggingface.co/datasets/LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents-public>, `167.936 GB` |
| Trainer dataset scan | complete | loader sees `1,281,167` samples, `313` shards, no skipped files |
| Real ImageNet-256 reference stats | complete | Inception pool-2048 stats for all `1,281,167` cropped train images, plus `8192` MMD/KID ref features |
| FLUX image-space eval wiring | complete | sampled latent -> FLUX.2 VAE decode -> PNG -> real ImageNet FID/MMD/KID works |
| B3 mixed precision/JVP route | complete | bf16 backbone + fp32 live-param JVP target works; FD audits ~`1e-4` to `1e-3`; `r=t` degeneration exact |
| B3-medium 5k shorttrain | complete | engineering pass, but image-space metrics worsen after 1k under current config |

## 2. Latent cache details

Source summary:

```text
experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/results/flux2_from_cropped_full_summary.json
experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/results/hf_flux2_vae_latents_upload_result_20260528.json
```

- Backend: `AutoencoderKLFlux2_d32` from `diffusers/FLUX.2-dev-bnb-4bit`, subfolder `vae`.
- Posterior: deterministic `mode()`.
- Encode model dtype: fp32.
- Saved dtype: bf16.
- Per-sample latent shape: `[32,32,32]`.
- Stored tensors: `labels`, `latents`, `latents_flip`.
- Cache path: `/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full`.
- Public dataset repo: `LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents-public`.
- Validated size: `167,935,991,352` bytes (`167.936 GB`).

Important: FLUX.2 latent spatial resolution is `32x32`, while the current PAE cache is `16x16`. Training configs, model patch sizes, memory estimates, and decode/eval scripts must remain backend-specific.

## 3. Real image-space eval wiring

Reference stats source:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/imagenet256_refstats_full_summary.json
```

Reference stats:

| metric | value |
|---|---:|
| images | 1,281,167 |
| feature dim | 2048 |
| shards | 313 |
| elapsed sec | 2,592.585 |
| samples/sec | 494.166 |
| CUDA peak MB | 43,027.3 |
| MMD/KID reference features | 8,192 |

Local reference files:

```text
/workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_stats.npz
/workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_mmd8192_features.npz
```

These are local data artifacts and should not be committed to GitHub.

## 4. B3/MeanFlow FLUX results

### Calibration

Source:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_b3medium_calibration_summary.json
```

Best selected calibration point for short training:

```json
{
  "hidden": 512,
  "depth": 12,
  "heads": 8,
  "batch": 128,
  "params_m": 58.8192,
  "final_loss": 3.5324220657348633,
  "mean_rt": 0.766015625,
  "loop_peak_mb": 16783.7314453125,
  "last_fd_rel": 8.00312485497428e-05,
  "samples_per_sec_effective": 59.89824844510914,
  "mixed_ok": true,
  "live": true,
  "degen": 0.0
}
```

### B3-medium 1k

Source:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_b3medium_h512_b128_1k_imageeval_summary.json
```

- Model: h512/d12/8 heads, patch size 2, `58.819M` params.
- Batch: 128.
- Final loss: `2.2663`.
- Mean realized `r=t`: `0.7516`.
- Last FD rel: `1.752e-4`; max FD rel: `1.197e-3`.
- JVP target uses live params; EMA not used for target; target detached; class conditioning active; degeneration exact.

Image-space eval:

| step | FID | MMD2/KID | images | status |
|---:|---:|---:|---:|---|
| 500 | 346.9461 | 0.402373 | 1024 | ok |
| 1000 | 348.3859 | 0.406783 | 1024 | ok |

### B3-medium 5k

Source run directory:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/shorttrain_flux2vae_b3medium_h512_d12_b128_5k_1024img
```

Final engineering checks:

- Final step: `5000`.
- Final loss: `1.9944`.
- Batch: `128`.
- Peak memory: `~16.78 GB`.
- Last FD rel: `5.316e-4`.
- `no_nan_or_inf=true`.
- `r=t` degeneration exact: `target-v=0`.
- Local checkpoints: `latest.pt`, `step_00003000.pt`, `step_00004000.pt`, `step_00005000.pt`.

Image-space eval with 1024 generated images and 8192 reference features:

| step | FID | MMD2/KID | images | status |
|---:|---:|---:|---:|---|
| 1000 | 350.3428 | 0.411324 | 1024 | ok |
| 2000 | 359.3986 | 0.433757 | 1024 | ok |
| 3000 | 371.9172 | 0.464737 | 1024 | ok |
| 4000 | 391.8534 | 0.509279 | 1024 | ok |
| 5000 | 420.7126 | 0.574157 | 1024 | ok |

## 5. Interpretation to preserve

FLUX VAE route engineering smoke completed. Current ImageNet-256 class-conditional shorttrain does not improve FID under the present configuration, but this is not conclusive because the VAE was not ImageNet-256-tuned and the run is only a short conditional-generation/domain-transfer stress test. Keep FLUX/Qwen as modern VAE/T2I baselines.

Paper-framing guardrail:

- PAE / FLUX / Qwen are VAE backends, not the core innovation.
- The core method claim should stay around HT + MeanFlow per-patch / trajectory decoupling.
- VAE-agnostic claims must be empirical and conditional.
- If PAE wins on ImageNet-256 because it is ImageNet-tuned, report that honestly.
- Do not treat the current FLUX ImageNet-256 5k degradation as a final negative verdict for FLUX/Qwen in T2I or broader image settings.

## 6. Recommended next diagnostics

1. **VAE reconstruction ceiling** on ImageNet-256: real image -> VAE encode -> decode -> FID/MMD/KID, plus optional LPIPS/PSNR/SSIM. Run this for FLUX.2 and PAE side by side.
2. **Latent statistics**: FLUX.2 channel/global mean/std, scale, norm distribution, and whether a latent multiplier/normalization is needed before B3 training.
3. **Training retune if continuing FLUX class-cond ImageNet**: LR, model capacity, sampler steps, class dropout, EMA, latent normalization, and patch size should be retuned for `[32,32,32]` latents instead of copied from PAE.
4. **Future T2I route**: keep FLUX/Qwen VAEs as the more relevant modern-VAE baselines for text-to-image/general-image experiments.

## 7. Artifact policy

Do not upload/commit by default:

```text
/workspace/PDM/data/vae_latents/**
/workspace/PDM/data/reference_stats/**
experiments/**/results/**/decoded_*png*/
experiments/**/results/**/checkpoints/
experiments/**/logs/**
*.pt
*.ckpt
*.safetensors
*.parquet
```

Safe to keep in Git/HF lightweight docs:

```text
experiments/**/configs/*.yaml
experiments/**/scripts/*.py
experiments/**/scripts/*.sh
experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/notes/EXPERIMENT_RECORD.md
handoff/FLUX2_VAE_ROUTE_SUMMARY_2026-05-28.md
```

## 8. Reconstruction-ceiling diagnostic completed — 2026-05-28

A FLUX.2 VAE reconstruction diagnostic was completed after the B3-medium 5k run worsened in image-space FID.

New script:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/diagnose_flux2vae_reconstruction.py
```

Main local result directory, not for Git upload:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_reconstruction_diagnostic_1024_fp32_strict/
```

Key 1024-sample metrics:

| comparison | FID ↓ | MMD2/KID ↓ | KID x1000 ↓ |
|---|---:|---:|---:|
| FLUX.2 recon vs full ImageNet-256 ref | `44.3251` | `1.83545e-05` | `0.01835` |
| Real first-1024 vs full ImageNet-256 ref | `44.4002` | `-8.34277e-06` | `-0.00834` |
| FLUX.2 recon vs same first-1024 inputs | `2.7447` | `-6.12531e-04` | `-0.61253` |

Latent and pixel summary:

```json
{
  "latent_shape": "[B, 32, 32, 32]",
  "latent_mean": -0.0062798662546468265,
  "latent_std": 1.7209782867032213,
  "pixel_mse_0_1": 0.0008338014284686945,
  "global_psnr_db": 30.789373651757323,
  "mean_image_psnr_db": 33.073608754982615,
  "paired_feature_cosine_mean": 0.991840691139065
}
```

Interpretation update:

- The bad current FLUX B3-medium ImageNet-256 FID trend is **not primarily explained by FLUX.2 VAE reconstruction failure**.
- Reconstruction-vs-full-reference FID is effectively equal to the real first-1024 finite-sample baseline.
- The likely bottleneck is FLUX latent modeling/training: normalization/scale, LR/model/sampler retune, patchification, or short-run prior learning.
- Keep FLUX/Qwen as modern VAE/T2I baselines; do not treat the current ImageNet class-conditional FLUX B3 run as a final negative verdict.

Detailed handoff:

```text
handoff/FLUX2_VAE_RECON_DIAGNOSTIC_2026-05-28.md
```
