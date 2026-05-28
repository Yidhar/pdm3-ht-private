# FLUX.2 per-channel latent_norm B3 smoke — 2026-05-28

更新时间：`2026-05-28T17:15Z`

## 1. Purpose

This run tests the stronger FLUX normalization route requested after the scalar normalized / scale-aware smoke:

```text
train:  z_norm = (z_raw - channel_mean) / channel_std
image eval inverse decode: z_raw = z_norm * channel_std + channel_mean
```

It is a short B3-medium engineering smoke, not a publishable quality run and not an official 50k FID.

## 2. Code / config added

Updated decoder/eval code:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/decode_flux2vae_latents.py
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/eval_flux2vae_imagespace.py
```

New config and launcher:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_flux2vae_fullcache_medium512_b128_1k_channelnorm_imageeval.yaml
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_flux2vae_medium512_b128_1k_channelnorm_imageeval.sh
```

Local runtime outputs, ignored by Git:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/059_flux2vae_medium512_b128_1k_channelnorm_imageeval_20260528T165131Z.log
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/smoke_flux2vae_b3medium_h512_d12_b128_1k_channelnorm_1024img/
```

## 3. Stats file used

Trainer/eval stats path:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_latent_cache_stats/flux2_latent_stats_linspace32shards_130191_latents_trainer_stats.pt
```

Stats metadata:

| item | value |
|---|---:|
| stats samples | `130,191` |
| created_at_utc | `2026-05-28T12:26:01+00:00` |
| mean tensor shape | `[1,32,1,1]` |
| std tensor shape | `[1,32,1,1]` |
| channel mean min/max | `-0.126136 / 0.106348` |
| channel mean average | `-0.00905418` |
| channel mean std across channels | `0.0442244` |
| channel std min/max | `1.668024 / 1.842393` |
| channel std average | `1.713103` |
| channel std across channels | `0.0355904` |

Reference raw FLUX global stats from the same diagnostic family:

| metric | value |
|---|---:|
| raw mean | `-0.0090541788` |
| raw std | `1.71404304` |
| raw RMS | `1.71406695` |

## 4. Training setup

| item | value |
|---|---:|
| dataset | `/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full` |
| samples | `1,281,167` |
| shards | `313` |
| latent shape | `[32,32,32]` |
| class conditional | yes, `1000` classes |
| `data.latent_norm` | `true` |
| `data.latent_multiplier` | `1.0` |
| model | B3-medium h512/d12/8 heads, patch size 2 |
| params | `58.819192M` total / `58.688120M` trainable |
| batch | `128` |
| max steps | `1000` |
| precision | `bf16_backbone_fp32_jvp` |
| JVP target params | live params, not EMA |
| SDPA | math |
| TF32 | off |
| sampler | sample-level `r=t` with `equal_prob=0.75`, `ltg` time sampler |
| eval | 1024 fixed labels, FLUX VAE decode, real ImageNet-256 Inception stats + 8192 MMD ref features |

## 5. Mechanical result

**PASS.** The mixed-precision/JVP/class-cond/FD/memory route is healthy.

| check | result |
|---|---:|
| final step | `1000` |
| final loss | `1.2403079271` |
| min / max loss | `1.1986927986 / 3.7330143452` |
| practical gate | `true` |
| loss finite | `true` |
| grad finite | `true` |
| target finite | `true` |
| class conditioning all steps | `true` |
| mixed modes ok | `true` |
| JVP target uses live params | `true` |
| EMA used for JVP target | `false` |
| target detached all steps | `true` |
| mean realized `r=t` fraction | `0.7491875` |
| `r=t` degenerate `target-v` max | `0.0` |
| final FD rel err | `4.07025e-4` |
| loop peak memory | `16,758.665 MB` |
| total elapsed | `1090.71 sec` |

FD audit points stayed in the expected fp32-JVP range:

| step | FD rel err | `r=t` degenerate `target-v` max |
|---:|---:|---:|
| 1 | `6.64425e-05` | `0.0` |
| 100 | `1.16429e-04` | `0.0` |
| 200 | `3.98718e-04` | `0.0` |
| 300 | `2.11510e-04` | `0.0` |
| 400 | `3.48023e-04` | `0.0` |
| 500 | `1.93653e-04` | `0.0` |
| 600 | `2.15457e-04` | `0.0` |
| 700 | `3.54793e-04` | `0.0` |
| 800 | `3.38910e-04` | `0.0` |
| 900 | `3.70892e-04` | `0.0` |
| 1000 | `4.07025e-04` | `0.0` |

## 6. Image-space eval

Per-channel inverse decode was active. Decode path recorded:

```text
pre_decode_scale = 1.0
pre_decode_shift = 0.0
pre_decode_stats_path = /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_latent_cache_stats/flux2_latent_stats_linspace32shards_130191_latents_trainer_stats.pt
```

Image-space eval results:

| step | sample std in normalized trainer space | decode-space raw std | decode std / ref raw std | FID ↓ | MMD2/KID ↓ | KID x1000 ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 500 | `1.583158` | `2.716459` | `1.584825` | `370.2665` | `0.464136` | `464.136` |
| 1000 | `1.562104` | `2.680442` | `1.563813` | `372.7879` | `0.469827` | `469.827` |

The important diagnostic is not only FID: the generated normalized samples have std `~1.56` rather than `~1.0`. After the mathematically correct per-channel inverse, the raw decode-space std becomes `~2.68`, still much larger than true raw FLUX std `~1.714`.

## 7. Five-way FLUX 1k comparison

All rows use B3-medium h512/d12 b128, 1000 steps, 1024 generated images, and the same real ImageNet-256 reference protocol unless noted.

| variant | training latent policy | eval decode policy | decode-space std | FID ↓ | MMD2/KID ↓ | interpretation |
|---|---|---|---:|---:|---:|---|
| raw FLUX baseline | raw latents | identity raw decode | n/a in old record | `348.3859` | `0.406783` | original FLUX route; mechanically healthy but not tuned |
| scalar normalized, official inverse | `z/1.714043` | multiply by `1.714043` | `2.677575` | `391.8537` | `0.514579` | correct scalar inverse but over-amplifies early generated samples |
| scalar normalized, identity ablation | `z/1.714043` | no inverse scale | `1.562140` | `345.6951` | `0.399690` | best smoke metric, but not the official raw-space inverse |
| scalar normalized, adaptive ref-std eval | `z/1.714043` | multiply by `1.097240` to match raw std | `1.714043` | `354.0362` | `0.421643` | confirms scale overshoot is a major eval issue |
| per-channel latent_norm, official inverse | `(z-mean)/std` | `z_norm*channel_std+channel_mean` | `2.680442` | `372.7879` | `0.469827` | mechanically correct; better than scalar official inverse, still worse than raw/identity/adaptive |

## 8. Interpretation

1. The per-channel normalized route is **mechanically implemented and trainable**:
   - DataLoader over full FLUX cache works.
   - Class labels are injected.
   - `bf16` backbone + `fp32` live-param JVP target works.
   - `r=t` sampler degeneracy remains exact.
   - FD/JVP audit remains healthy with TF32 off / math SDPA.
   - FLUX image-space eval forwards and applies the per-channel inverse transform.

2. The run does **not** solve the FLUX ImageNet-256 smoke quality problem. It reduces the damage relative to scalar official inverse (`FID 372.8` vs `391.9`), but remains worse than the raw 1k baseline and the scalar identity/adaptive eval-only variants.

3. The current bottleneck is likely sampler/output-scale calibration or insufficient training/retuning in FLUX normalized space. The generated normalized latent std is still `~1.56`, so applying the correct inverse transform maps it to an over-dispersed raw latent distribution (`~2.68` vs true `~1.714`).

4. Do **not** interpret this as “FLUX VAE is bad.” The separate reconstruction diagnostic remains healthy (`recon vs same inputs FID 2.7447`, `recon vs full ref FID 44.3251`, close to the real finite-sample baseline). This is a latent prior / sampler / training-config issue.

5. Do **not** spend long-run FLUX budget solely on the current official inverse per-channel setup until one of the following is tested:
   - sampler/noise endpoint scale policy in normalized space;
   - eval-time variance calibration in normalized space before inverse decode;
   - longer warmup / LR / EMA / sampler-step retune for `[32,32,32]` FLUX latents;
   - side-by-side PAE route where ImageNet-tuned latent backend may remain the main ImageNet result.

## 9. Artifact policy

Safe for Git:

```text
scripts/decode_flux2vae_latents.py
scripts/eval_flux2vae_imagespace.py
configs/b3_meanflow_flux2vae_fullcache_medium512_b128_1k_channelnorm_imageeval.yaml
scripts/run_flux2vae_medium512_b128_1k_channelnorm_imageeval.sh
handoff/*.md
experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/notes/EXPERIMENT_RECORD.md
```

Do not commit runtime artifacts:

```text
experiments/**/results/**
experiments/**/logs/**
/workspace/PDM/data/**
*.pt
*.safetensors
*.png
```
