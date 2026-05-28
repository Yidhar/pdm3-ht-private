# VAE rFID ceiling 100-image diagnostic — PAE vs FLUX.2 — 2026-05-28

## 1. Purpose

User-requested decision check:

```text
100 ImageNet-256 ADM crops -> PAE encode/decode -> rFID
same 100 crops          -> FLUX.2 encode/decode -> rFID
```

Decision rule to preserve:

- If `FLUX rFID >> PAE rFID`: stop the FLUX ImageNet-256 VAE route and report that the FLUX VAE reconstruction ceiling is worse than PAE on ImageNet-256.
- If `FLUX rFID ~= PAE rFID`: the immediate blocker is training / sampler / latent scale, not VAE reconstruction ceiling; record a ticket and defer until after the PAE/mainline work.

This run is an engineering ceiling diagnostic, not a publishable 50k rFID/FID number.

## 2. Script added

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_vae_rfid_ceiling_100.py
```

What the script does:

1. Loads the same `N` cropped ImageNet-256 ADM uint8 images from the local safetensor cache.
2. Writes those images to `original_png/`.
3. Runs PAE DINOv2-L d32 deterministic `encode -> decode`, writes `pae_recon_png/`.
4. Runs FLUX.2 VAE deterministic posterior `mode() -> decode`, writes `flux2_recon_png/`.
5. Computes Inception pool-2048 rFID between each reconstruction directory and the same original-image directory.
6. Records pixel metrics, latent stats, runtime, and optional full-reference context metrics.

Runtime PNGs / summaries are under `experiments/**/results/` and are intentionally not committed.

## 3. Run command

```bash
cd /workspace/PDM
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
python "$EXP/scripts/run_vae_rfid_ceiling_100.py" \
  --max-images 100 \
  --start-index 0 \
  --output-dir "$EXP/results/vae_rfid_ceiling_100_pae_vs_flux2" \
  --device cuda:0 \
  --pae-model-dtype fp32 \
  --flux-model-dtype fp32 \
  --pae-batch-size 8 \
  --flux-batch-size 8 \
  --fid-batch-size 64 \
  --fid-num-workers 2
```

Log:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/060_vae_rfid_ceiling_100_pae_vs_flux2_20260528T173728Z.log
```

Result directory:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/vae_rfid_ceiling_100_pae_vs_flux2/
```

Key result files, local only:

```text
run_config.json
sample_meta.json
rfid_metrics.json
rfid_ceiling_summary.json
rfid_ceiling_summary.md
original_png/*.png
pae_recon_png/*.png
flux2_recon_png/*.png
```

## 4. Sample / settings

| item | value |
|---|---:|
| source cache | `/workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors` |
| crop shard | `images_uint8_shard000000.safetensors` |
| source indices | `0..99` |
| image tensor shape | `[100, 3, 256, 256]` |
| PAE config | `PAE_DINOv2L_d32.yaml` |
| PAE checkpoint | `dinov2-large.pt` |
| FLUX VAE | `diffusers/FLUX.2-dev-bnb-4bit`, subfolder `vae`, class `AutoencoderKLFlux2` |
| posterior | deterministic `mode()` |
| model dtype | fp32 for both VAEs |
| TF32 | off |
| Inception feature dim | 2048 |
| elapsed | `56.1859 sec` |
| CUDA peak memory | `5089.24 MB` |

Sanity smoke with `max_images=2` also passed before the 100-image run; both PAE and FLUX reconstructions were 256x256.

## 5. Primary same-100-image rFID result

| VAE | rFID vs same 100 originals ↓ | ratio vs PAE | delta vs PAE | PSNR ↑ | MAE ↓ |
|---|---:|---:|---:|---:|---:|
| PAE DINOv2-L d32 | `15.851984` | `1.0000` | `0.0000` | `24.264884 dB` | `0.0341667` |
| FLUX.2 VAE | `4.866488` | `0.306996` | `-10.985496` | `30.733994 dB` | `0.0168896` |

Additional same-100 metrics:

| VAE | MMD2/KID vs originals ↓ | KID x1000 ↓ | max abs uint8 diff |
|---|---:|---:|---:|
| PAE DINOv2-L d32 | `-0.00627811` | `-6.27811` | `255` |
| FLUX.2 VAE | `-0.00666353` | `-6.66353` | `180` |

Notes:

- The unbiased MMD/KID estimate can be slightly negative at this small sample size; use rFID / PSNR / qualitative ceiling interpretation here.
- The rFID comparison is paired at the dataset level: both VAEs reconstruct exactly the same 100 original images.

## 6. Latent / decode stats

| VAE | latent shape | latent mean | latent std | decoded shape | decoded float mean | decoded float std |
|---|---|---:|---:|---|---:|---:|
| PAE DINOv2-L d32 | `[B, 32, 16, 16]` | `0.210789` | `0.977532` | `[B, 3, 256, 256]` | `0.430466` | `0.275792` |
| FLUX.2 VAE | `[B, 32, 32, 32]` | `-0.005467` | `1.730345` | `[B, 3, 256, 256]` | `-0.134762` | `0.556446` |

Runtime throughput:

| VAE | elapsed sec | samples/sec |
|---|---:|---:|
| PAE DINOv2-L d32 | `12.5017` | `7.9989` |
| FLUX.2 VAE | `14.5289` | `6.8828` |

## 7. Decision

The observed result is **not** `FLUX rFID >> PAE rFID`.

It is the opposite on this 100-image ceiling check:

```text
PAE rFID   = 15.851984
FLUX rFID  =  4.866488
FLUX/PAE   =  0.306996
FLUX - PAE = -10.985496
```

Therefore:

- Do **not** close the FLUX.2 VAE route on reconstruction-ceiling grounds.
- The current bad FLUX B3 ImageNet-256 smoke FID is much more likely a latent-prior / MeanFlow training / sampler / output-scale issue, not a FLUX VAE encode-decode ceiling issue.
- Record this as a deferred FLUX training-retune ticket after the PAE/mainline conditional-generation work.
- Keep paper framing honest: PAE can remain the ImageNet-tuned baseline/backend, while FLUX/Qwen remain modern VAE/T2I backend baselines. This diagnostic only says FLUX.2 reconstruction ceiling is not obviously worse than PAE on the same ADM ImageNet-256 crops.

## 8. Ticket for later

After the PAE mainline reaches its next milestone, revisit FLUX.2 latent prior training with backend-specific retuning:

1. sampler / output amplitude policy for `[32,32,32]` FLUX latents;
2. LR and training length retune;
3. class dropout / CFG policy if conditional route continues;
4. patch size / model capacity for FLUX latent spatial size;
5. image-space eval with the existing real ImageNet-256 reference stats;
6. optional T2I-specific route where FLUX/Qwen VAE relevance is higher than ImageNet-256 class-conditional reconstruction tuning.

## 9. Git/artifact hygiene

Commit-safe:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_vae_rfid_ceiling_100.py
handoff/VAE_RFID_CEILING_100_PAE_VS_FLUX2_2026-05-28.md
handoff/FLUX2_VAE_ROUTE_SUMMARY_2026-05-28.md
handoff/PROGRESS_2026-05-28_GITHUB_HF_ARTIFACTS.md
experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/notes/EXPERIMENT_RECORD.md
```

Do not commit:

```text
experiments/**/results/**
experiments/**/logs/**
*.png
*.npz
*.pt
*.safetensors
```
