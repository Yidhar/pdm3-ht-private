# VAE rFID 50K protocol — PAE vs FLUX.2 — 2026-05-28

## 1. Purpose

User-requested rerun after the initial 100-image engineering diagnostic:

```text
用 50K samples 重跑 rFID(论文标准协议),这样 PAE 的 rFID 应该回到接近论文的 0.26,FLUX 也得到可对照的数字
```

This run is the publishable/protocol-grade reconstruction-ceiling audit for the VAE backend question:

```text
ImageNet validation 50,000 ADM-center-crop 256 originals
  -> PAE DINOv2-L d32 encode/decode
  -> rFID(original distribution, PAE reconstruction distribution)

same 50,000 originals
  -> FLUX.2 VAE encode/decode
  -> rFID(original distribution, FLUX reconstruction distribution)
```

Primary metric is **rFID only**. MMD/KID was intentionally skipped for this 50K audit because the decision question is the paper-standard reconstruction FID ceiling.

Interpretation guardrail: this is **only a VAE reconstruction-ceiling test**. It does not validate the MeanFlow latent prior, sampler, conditional training, or generated-sample FID.

## 2. Bottom line

The run completed successfully with `exit_status=0`.

| VAE | rFID vs same 50K originals ↓ | ratio vs PAE | delta vs PAE | PSNR ↑ | MAE ↓ |
|---|---:|---:|---:|---:|---:|
| PAE DINOv2-L d32 | `0.2648149351` | `1.000000` | `0.000000` | `24.238722 dB` | `0.0346135` |
| FLUX.2 VAE | `0.1562118224` | `0.589891` | `-0.108603` | `30.544935 dB` | `0.0173406` |

Key conclusion:

- PAE returns `rFID = 0.264815`, matching the expected paper-level `~0.26` sanity target. This strongly suggests the crop/evaluator/checkpoint path is aligned enough for the intended comparison.
- FLUX.2 returns `rFID = 0.156212`, **better than PAE under this same 50K ImageNet-val reconstruction protocol**.
- Therefore the stop condition `FLUX rFID >> PAE rFID` is not met. It is the opposite: FLUX.2 reconstruction ceiling is not the blocker.
- The current poor FLUX B3 generated-sample FID should remain classified as a **latent prior / MeanFlow training / sampler / output-scale retuning problem**, not a VAE encode-decode ceiling problem.

Decision label from the script:

```text
automatic_label = flux_rfid_not_much_worse_than_pae
```

## 3. Protocol details

| item | value |
|---|---|
| split | ImageNet-1K validation |
| sample count | `50,000` |
| crop | ADM-style center crop to `256 x 256` |
| source cache | `/workspace/PDM/data/cropped_uint8/imagenet1k_validation_256_adm_safetensors` |
| source shards | `13` safetensor shards |
| source indices | `0..49999` |
| image tensor | uint8 RGB `[N, 3, 256, 256]` |
| comparison | original 50K distribution vs reconstruction 50K distribution |
| Inception feature dim | `2048` |
| feature path | direct uint8 tensor -> Inception features; no PNG intermediates |
| direct-vs-PNG validation | N=128 direct path matched PNG path to about `1e-4` rFID before 50K |
| MMD/KID | skipped intentionally |
| TF32 | off (`allow_tf32=false`) |
| sqrt/Frechet device | `cuda:0` |

Important implementation note:

- The direct path quantizes each reconstruction to uint8 RGB before Inception, matching the lossless PNG path semantically while avoiding 150K PNG writes.
- The N=128 smoke comparison gave identical results to the PNG path at practical precision:
  - PAE direct `16.20480394` vs PNG `16.20487111`;
  - FLUX direct `5.49861068` vs PNG `5.49867913`.

## 4. Code / command

New direct-feature script:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_vae_rfid_ceiling_50k_direct.py
```

The older PNG diagnostic script was also generalized for N-image runs, but the 50K protocol run used the direct-feature script to avoid excessive PNG IO.

Launch command recorded in the output directory as `run_val50k_direct.sh`:

```bash
cd /workspace/PDM
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
OUT=$EXP/results/vae_rfid_protocol_val50k_pae_vs_flux2_direct_20260528T182712Z

python "$EXP/scripts/run_vae_rfid_ceiling_50k_direct.py" \
  --crop-cache-dir /workspace/PDM/data/cropped_uint8/imagenet1k_validation_256_adm_safetensors \
  --max-images 50000 \
  --start-index 0 \
  --output-dir "$OUT" \
  --device cuda:0 \
  --pae-model-dtype fp32 \
  --flux-model-dtype fp32 \
  --pae-batch-size 64 \
  --flux-batch-size 64 \
  --feature-batch-size 256 \
  --sqrt-device cuda:0 \
  --log-every-batches 50
```

## 5. Artifacts

Local result directory:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/vae_rfid_protocol_val50k_pae_vs_flux2_direct_20260528T182712Z
```

Main result files:

```text
exit_status.txt                         # 0
finished_at_utc.txt                     # 2026-05-28T19:26:45Z
run_config.json
sample_meta.json
original_feature_stats.json
rfid_metrics.json
rfid_ceiling_summary.json
rfid_ceiling_summary.md
```

Log:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/065_vae_rfid_protocol_val50k_pae_vs_flux2_direct_20260528T182712Z.log
```

These runtime artifacts live under `experiments/**/results/` and `experiments/**/logs/` and should **not** be committed.

## 6. Runtime / memory

| phase | elapsed | throughput |
|---|---:|---:|
| original 50K feature extraction | `77.25 sec` | `647.25 img/s` |
| PAE encode/decode + feature stats | `1653.95 sec` | `30.23 img/s` |
| FLUX.2 encode/decode + feature stats | `1810.51 sec` | `27.62 img/s` |
| total | `3568.51 sec` | about `59.48 min` |

CUDA peak memory:

```text
20022.27 MB
```

GPU was saturated during reconstruction phases; after completion, GPU load returned to zero.

## 7. Model settings

### PAE

| item | value |
|---|---|
| model | PAE DINOv2-L d32 |
| config | `/workspace/PDM/external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml` |
| checkpoint | `/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt` |
| dtype | fp32 |
| latent shape | `[B, 32, 16, 16]` |

### FLUX.2 VAE

| item | value |
|---|---|
| repo | `diffusers/FLUX.2-dev-bnb-4bit` |
| subfolder | `vae` |
| class | `AutoencoderKLFlux2` |
| diffusers | `0.38.0` |
| dtype | fp32 |
| posterior | deterministic `mode()` |
| latent shape | `[B, 32, 32, 32]` |

## 8. Feature statistics

Original 50K feature distribution:

| field | value |
|---|---:|
| feature mean | `0.2461670813` |
| sigma trace | `178.2278243406` |
| sqrt(trace/dims) proxy | `0.2950005167` |

Reconstruction feature distributions:

| VAE | feature mean | sigma trace | sqrt(trace/dims) proxy |
|---|---:|---:|---:|
| PAE | `0.2459596178` | `176.6202702416` | `0.2936671012` |
| FLUX.2 | `0.2471450854` | `177.9879249057` | `0.2948019105` |

## 9. Latent / decode stats

| VAE | latent mean | latent std | latent rms | latent absmax | decoded float mean | decoded float std | decoded absmax |
|---|---:|---:|---:|---:|---:|---:|---:|
| PAE | `0.2124836513` | `0.9771646155` | `0.9999999939` | `4.818028` | `0.4417967194` | `0.2757142291` | `4.679924` |
| FLUX.2 | `-0.0108509157` | `1.7200218357` | `1.7200560623` | `17.176630` | `-0.1122174250` | `0.5565955101` | `2.528065` |

Pixel reconstruction metrics:

| VAE | global PSNR | mean image PSNR | global MSE | global RMSE | global MAE | median image MAE |
|---|---:|---:|---:|---:|---:|---:|
| PAE | `24.238722 dB` | `25.927338 dB` | `0.0037681464` | `0.0613852293` | `0.0346135341` | `0.0307727084` |
| FLUX.2 | `30.544935 dB` | `32.603953 dB` | `0.0008820769` | `0.0296997801` | `0.0173405968` | `0.0149246324` |

## 10. Interpretation for the FLUX route

The 50K protocol result resolves the VAE ceiling question:

```text
PAE rFID   = 0.2648149351
FLUX rFID  = 0.1562118224
FLUX/PAE   = 0.5898905298
FLUX - PAE = -0.1086031128
```

Therefore:

1. **Do not stop the FLUX.2 route on reconstruction-ceiling grounds.** FLUX.2 reconstructs ImageNet-val 256 crops at least as well as PAE under this evaluator/protocol.
2. **PAE remains a valid ImageNet-tuned baseline/backend**, and the PAE 0.2648 number is an important sanity check that the 50K protocol is aligned with the expected PAE paper rFID.
3. **The bad FLUX B3 smoke samples are not explained by VAE reconstruction quality.** The likely causes remain generated latent distribution mismatch, sampler/output amplitude, training length/LR/capacity, or backend-specific normalization/decode policy.
4. **Paper framing should stay VAE-agnostic.** PAE can be presented as the ImageNet-tuned main backend; FLUX/Qwen can be presented as modern VAE/T2I backend ablations. This 50K audit supports the claim that the core HT + MeanFlow mechanism should not be tied to PAE reconstruction quality alone.

## 11. Next FLUX-side ticket

After this audit, the next useful FLUX experiment is no longer another reconstruction ceiling run. It should target the latent-prior/training mismatch:

- Keep FLUX.2 VAE route alive.
- Use the 50K rFID result as the ceiling/reference in docs.
- For B3/MeanFlow FLUX training, focus on:
  1. sampler/output-scale control for `[32, 32, 32]` latents;
  2. generated latent std distribution vs real FLUX latent std (`~1.72` raw in this 50K run);
  3. LR/training-length/capacity retune;
  4. class-conditional conditioning/dropout sanity;
  5. image-space FID/MMD with the existing real ImageNet-256 reference stats.

## 12. Git / artifact hygiene

Commit-safe code/docs:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_vae_rfid_ceiling_50k_direct.py
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_vae_rfid_ceiling_100.py
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/inception_metrics_lib.py
handoff/VAE_RFID_PROTOCOL_50K_PAE_VS_FLUX2_2026-05-28.md
handoff/FLUX2_VAE_ROUTE_SUMMARY_2026-05-28.md
handoff/PROGRESS_2026-05-28_GITHUB_HF_ARTIFACTS.md
experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/notes/EXPERIMENT_RECORD.md
```

Do not commit:

```text
experiments/**/results/**
experiments/**/logs/**
data/**
*.png
*.npz
*.pt
*.safetensors
*.parquet
```
