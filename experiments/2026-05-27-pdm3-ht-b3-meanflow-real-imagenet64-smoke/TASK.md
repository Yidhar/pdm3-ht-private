# Experiment 9: Phase0/B3 MeanFlow Real ImageNet64 PAE-Latent Smoke

Date: 2026-05-27

## Objective

Use the real HF ImageNet-1k smoke64 PAE_DINOv2L_d32 latent shard generated in Experiment 8 as the dataset for the Phase0/B3 MeanFlow trainer smoke.

## Input latent dataset

`/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke64`

Expected schema:

- `latents`: `[64, 32, 16, 16]`, BF16
- `latents_flip`: `[64, 32, 16, 16]`, BF16
- `labels`: `[64]`, int64

## Smoke requirements

- live-param fp32 JVP target
- bf16 backbone train forward/backward
- r=t sampler active, default 75% equal
- FD audit active
- memory logging active
- EMA not used for target JVP


## Outcome

Result: PASS

- Completed optimizer steps: `8` / `8`
- Final loss: `3.6721441745758057`
- Practical gate: `True`
- Strict gate: `True`
- FD rel err full JVP: `8.165405772234868e-05`
- All-r=t target-v max abs: `0.0`
- Live-param JVP target: `True`
- EMA used for target JVP: `False`
- Target detached: `True`
- Mean realized r=t fraction: `0.6875`
- Memory logged / loop peak: `True` / `63.06005859375 MB`
