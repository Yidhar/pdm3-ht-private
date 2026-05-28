# FLUX.2 B3 decode/output-scale sweep — 2026-05-28

## Purpose

After the 50K VAE reconstruction-ceiling audit showed FLUX.2 VAE is **not** worse than PAE on ImageNet-val-256 reconstruction rFID, the next question was whether the poor FLUX B3 generated-image FID is mainly an output-scale / pre-decode latent amplitude problem.

This diagnostic reuses the existing per-channel-normalized FLUX B3 step-1000 generated latents and sweeps the scalar applied before the per-channel inverse FLUX decode:

```text
z_raw = (z_sample * pre_decode_scale + pre_decode_shift) * channel_std + channel_mean
```

If global std mismatch were the primary problem, the scale that matches real FLUX raw latent std should strongly improve image-space FID/MMD.

## Code

New helper:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/sweep_flux2vae_decode_scale.py
```

Design note: the helper creates one isolated `sample_dir` per scale because `eval_flux2vae_imagespace.py` writes fixed filenames under `sample_dir` (`flux2vae_imagespace_eval_record.json`, FID logs, decoded image subdir). Reusing one directory would overwrite records.

Syntax check:

```text
python -m py_compile experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/sweep_flux2vae_decode_scale.py
```

## Input

Existing generated latents:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/smoke_flux2vae_b3medium_h512_d12_b128_1k_channelnorm_1024img/eval/step_00001000/sample_latents.safetensors
```

Source model/run:

```text
smoke_flux2vae_b3medium_h512_d12_b128_1k_channelnorm_1024img
step = 1000
num images = 1024
sample_steps = 4
use_ema = true
sample normalized latent std = 1.562104
```

Per-channel inverse stats:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_latent_cache_stats/flux2_latent_stats_linspace32shards_130191_latents_trainer_stats.pt
```

Reference image-space metrics:

```text
ref stats    = /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_stats.npz
ref features = /workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_mmd8192_features.npz
metric       = ImageNet-256 train ADM Inception pool-2048 FID + poly3 MMD/KID, 1024 generated images
```

The runs used `--local-files-only` for the FLUX.2 VAE load, so no repeated network/model fetch was needed.

## Artifacts

Local result dirs, ignored by git:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_decode_scale_sweep_channelnorm_step1000_20260528T201155Z/
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_decode_scale_sweep_channelnorm_step1000_lowrange_20260528T202427Z/
```

Logs, ignored by git:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/066_flux2vae_decode_scale_sweep_channelnorm_step1000_20260528T201155Z.log
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/067_flux2vae_decode_scale_sweep_channelnorm_step1000_lowrange_20260528T202427Z.log
```

Both runs exited `0`.

## Results

Combined 1024-image image-space sweep:

| pre-decode scale | decode std | decode/ref std | FID ↓ | MMD2/KID ↓ | note |
|---:|---:|---:|---:|---:|---|
| `0.20` | `0.539710` | `0.314875` | `447.8535` | `0.567919` | too compressed; bad |
| `0.30` | `0.806882` | `0.470748` | `329.5382` | `0.337264` | improved |
| `0.35` | `0.940596` | `0.548758` | **`322.3539`** | **`0.324473`** | best in sweep |
| `0.40` | `1.074353` | `0.626794` | `322.4108` | `0.328168` | statistically tied with 0.35 on FID |
| `0.45` | `1.208138` | `0.704847` | `323.1042` | `0.336568` | close |
| `0.50` | `1.341944` | `0.782911` | `328.7377` | `0.355028` | degrading |
| `0.55` | `1.475764` | `0.860984` | `334.8103` | `0.372971` | degrading |
| `0.64` | `1.716666` | `1.001530` | `344.6390` | `0.400106` | std-matched to real FLUX raw std, not best |
| `0.70` | `1.877281` | `1.095236` | `351.2454` | `0.417133` | bad |
| `0.80` | `2.144988` | `1.251420` | `361.2771` | `0.442044` | bad |
| `1.00` | `2.680442` | `1.563813` | `372.7879` | `0.469827` | official per-channel inverse baseline |

Delta vs official inverse baseline:

```text
official inverse scale 1.00: FID 372.7879, MMD2 0.469827
best sweep scale 0.35:       FID 322.3539, MMD2 0.324473
absolute FID improvement:    -50.4340
relative FID improvement:    -13.53%
absolute MMD2 improvement:   -0.145354
```

## Interpretation

1. **Output amplitude matters, but it is not the whole failure.**
   Compressing the sampled normalized latents before per-channel inverse decode improves FID/MMD substantially versus the official inverse decode.

2. **Real-std matching is not optimal.**
   The nominal std-matching scale is about `1 / sample_std = 1 / 1.562104 ≈ 0.64016`, which gives decode/ref std `~1.00`. It only reaches FID `344.6390`, much worse than the best scale `0.35`/`0.40`.

3. **Best FID occurs with under-dispersed raw latents.**
   The best scale `0.35` has decode/ref std `0.5488`, far below the real FLUX raw latent std. Therefore the generated latent distribution/content is not merely globally over-scaled. The model/sampler is producing a poor latent prior distribution, and amplitude compression only masks part of the error.

4. **Do not present eval-time scale `0.35` as a method improvement.**
   It is a diagnostic/preview calibration. The resulting FID is still catastrophically high (`~322` for 1024 images), so the FLUX route needs sampler/training retune rather than a simple decode scale fix.

## Decision

FLUX.2 route remains alive because the VAE ceiling is good, but current B3 step-1000 generated latents are not good enough. The next FLUX action should be sampler/training diagnosis, in this order:

1. **Checkpoint resampling sweep** from the existing step-1000 checkpoint: sample_steps `4/8/16`, EMA vs live if cheap, fixed labels/seed, decode with scale `0.35` and optionally official scale `1.0` for anchor.
2. If sample_steps do not help: run short retune with lower LR (`5e-5` or `2e-5`) and/or longer training, keeping per-channel latent_norm and mixed-precision JVP path unchanged.
3. Keep PAE as the main ImageNet-tuned baseline while treating FLUX.2 as the modern-VAE / future T2I backend.
