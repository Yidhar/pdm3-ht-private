# FLUX.2 VAE latent_std diagnostic — 2026-05-28

## TL;DR

**Confirmed:** the FLUX.2 VAE ImageNet-256 cache has raw latent standard deviation around `1.714` / RMS around `1.714`. This is a real property of the cached FLUX latents, not a one-off reconstruction diagnostic artifact.

This is **not** evidence that FLUX.2 VAE is bad. The reconstruction diagnostic remains healthy. The issue is that the current B3/MeanFlow FLUX route reused a PAE/unit-RMS-oriented training recipe:

```text
x_data = raw FLUX latent, std ≈ 1.714
noise endpoint = N(0, 1)
sampler start = N(0, 1)
image decode = identity raw latent decode, no inverse normalization
```

That creates an endpoint-scale mismatch and is the highest-priority suspect for the current FLUX B3 short-run FID/MMD degradation.

---

## Files checked

FLUX cache:

```text
/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full
```

Cache summary:

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
  "avg_new_samples_per_sec": 48.608707756964044,
  "cuda_peak_allocated_mb": 41673.30126953125
}
```

New diagnostic script:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/compute_latent_cache_stats.py
```

Local diagnostic outputs, intentionally under ignored `results/`:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_latent_cache_stats/flux2_latent_stats_subset16shards_65536_eachkey.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_latent_cache_stats/flux2_latent_stats_linspace32shards_130191_latents.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_latent_cache_stats/pae_latent_stats_linspace32shards_131072_latents.json
```

---

## Latent stats

### FLUX first 16 shards, `latents + latents_flip`

Scanned `131,072` samples-equivalent across both keys.

| metric | value |
|---|---:|
| global mean | `-0.009236385995357865` |
| global std | `1.714024316313405` |
| global RMS | `1.7140492022517588` |
| min / max | `-16.5 / 17.375` |
| absmax | `17.375` |
| per-channel mean summary | min `-0.12542458`, max `0.10442029`, std across channels `0.04378430` |
| per-channel std summary | min `1.66632737`, max `1.84289220`, mean `1.71309371`, std across channels `0.03566815` |

### FLUX full-cache linspace 32 shards, `latents` only

Scanned `130,191` samples uniformly across the full 313-shard cache.

| metric | value |
|---|---:|
| global mean | `-0.009054178792269978` |
| global std | `1.7140430386576422` |
| global RMS | `1.7140669521708671` |
| min / max | `-16.375 / 17.375` |
| absmax | `17.375` |
| per-channel mean summary | min `-0.12613564`, max `0.10634796`, std across channels `0.04422440` |
| per-channel std summary | min `1.66802397`, max `1.84239272`, mean `1.71310276`, std across channels `0.03559041` |

This matches the prior 1024-image reconstruction diagnostic latent stats:

```json
{
  "latent_mean": -0.0062798662546468265,
  "latent_std": 1.7209782867032213,
  "latent_rms": 1.7209897443111422
}
```

### PAE comparison, linspace 32 shards from local partial PAE cache

PAE local cache is currently partial on this machine, but enough for scale comparison.

| metric | value |
|---|---:|
| scanned samples | `131,072` |
| global mean | `0.21101030088928027` |
| global std | `0.977483105972915` |
| global RMS | `0.999999284771665` |
| min / max | `-3.40625 / 4.8125` |
| absmax | `4.8125` |
| per-channel mean summary | min `-1.36527343`, max `3.32757903`, std across channels `0.89839119` |
| per-channel std summary | min `0.32194730`, max `0.46212851`, mean `0.38389356`, std across channels `0.03149976` |

Interpretation: PAE has global RMS near 1, so the old B3/MeanFlow recipe naturally matches a unit Gaussian noise endpoint much better than raw FLUX latents do.

---

## Training-chain check

Current FLUX B3 config:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_flux2vae_fullcache_medium512_b128_5k_imageeval.yaml
```

Relevant config:

```yaml
data:
  data_path: /workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full
  expected_total: 1281167
  latent_norm: false
  latent_multiplier: 1.0
  flip_prob: 0.5
```

Trainer dataset path:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py
```

Relevant logic:

```python
if self.latent_norm:
    feature = (feature.float().unsqueeze(0) - self._latent_mean) / self._latent_std
    feature = feature.squeeze(0)
feature = feature * self.latent_multiplier
```

Therefore current FLUX 5k training used **raw FLUX latents**: no normalization and multiplier `1.0`.

MeanFlow transport:

```text
external/PAE/PAE/pae_with_generator/transport/meanflow_transport.py
```

Relevant logic:

```python
x_data = x_data.float()
noise = torch.randn_like(x_data)
v = noise - x_data
z_t = (1.0 - t) * x_data + t * noise
```

Therefore the training endpoint noise is unit Gaussian.

FLUX image-space eval:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/eval_flux2vae_imagespace.py
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/decode_flux2vae_latents.py
```

Current eval wrapper hardcodes:

```python
pre_decode_scale=1.0
pre_decode_shift=0.0
```

The decoder can apply scalar scale/shift, but eval currently does not pass inverse scaling automatically. It also does not yet support per-channel inverse normalization from trainer stats.

---

## Link to current FLUX 5k behavior

FLUX B3-medium 5k short run:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/shorttrain_flux2vae_b3medium_h512_d12_b128_5k_1024img
```

Earlier summary:

| step | generated latent sample std | FID | MMD2/KID |
|---:|---:|---:|---:|
| 1000 | `1.5861` | `350.3428` | `0.411324` |
| 2000 | `1.5703` | `359.3986` | `0.433757` |
| 3000 | `1.5531` | `371.9172` | `0.464737` |
| 4000 | `1.5368` | `391.8534` | `0.509279` |
| 5000 | `1.5237` | `420.7126` | `0.574157` |

Real FLUX raw latent std is `~1.714`, while generated sample std drifted down to `~1.52`. This is consistent with the scale mismatch being harmful, though it does not by itself prove it is the only issue.

---

## Verdict

1. `latent_std ≈ 1.714` is real and stable across:
   - first-shards scan with both `latents` and `latents_flip`,
   - uniformly sampled full-cache shards,
   - prior reconstruction diagnostic.
2. This is not a FLUX.2 VAE reconstruction failure. Reconstruction diagnostic remains good:
   - FLUX recon vs full real ref FID `44.3251`, essentially equal to real first-1024 vs full ref FID `44.4002`.
3. The current bug/risk is **not** “FLUX latent_std is wrong”; it is that the current B3/MeanFlow recipe uses raw FLUX latents with std `~1.714` but still assumes a unit Gaussian endpoint and identity decode.
4. The FLUX route needs an explicit scale policy before more meaningful training comparison:
   - scalar scale-only smoke, or
   - per-channel latent normalization plus inverse decode.

---

## Recommended next experiment

Start with the lowest-risk scalar smoke:

```text
train latent_multiplier = 1 / 1.7140430386576422 = 0.5834159221481118
eval/decode pre_decode_scale = 1.7140430386576422
pre_decode_shift = 0
```

Keep all other B3-medium settings fixed for a short `1k` smoke first. Check:

1. normalized training/sample latent std near `1`,
2. FD/JVP audit still healthy,
3. `r=t` degeneration still exact,
4. decoded image-space FID/MMD no longer monotonically degrades like the raw 5k run.

If scalar smoke improves, then implement the cleaner per-channel route:

```text
latent_norm: true
latent_stats_path: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/flux2vae_latent_cache_stats/flux2_latent_stats_linspace32shards_130191_latents_trainer_stats.pt
latent_multiplier: 1.0
```

and extend FLUX eval/decode with inverse per-channel stats:

```python
z_raw = z_norm * std + mean
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
