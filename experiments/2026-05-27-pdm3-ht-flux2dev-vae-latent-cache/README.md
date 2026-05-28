# FLUX.2 / Diffusers VAE latent cache from cropped ImageNet-256

This directory holds scripts and logs for building non-PAE VAE latent caches from the shared cropped uint8 ImageNet cache.

Main builder:

```bash
scripts/build_diffusers_vae_latents_from_cropped_cache.py
```

Default FLUX.2 settings are in:

```bash
configs/flux2dev_vae_cache.env
```

Current intended pipeline:

1. Keep `/workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors` as reusable decoded/cropped image cache.
2. Encode that cache with `AutoencoderKLFlux2` using deterministic posterior mode.
3. Save bf16 latents and flipped bf16 latents in shard-compatible safetensors.
4. Record shape, throughput, memory, and any VAE-specific caveats.
5. Later plug the resulting cache into B3/MeanFlow real-data training through backend-configurable latent loader.

Important framing: this is a VAE-agnostic validation path. PAE is a backend, not the claimed innovation.

## Reporting rule for backend comparison

The VAE swap experiment is explicitly allowed to show backend-dependent performance. If PAE is best (for example FID 1.0 vs FLUX/Qwen around 1.5), the result should be reported as: the HT + MeanFlow mechanism transfers to other latent spaces, but ImageNet-tuned PAE remains the strongest backend for ImageNet-256. If FLUX/Qwen fail or degrade severely, the VAE-agnostic claim must be weakened rather than hidden.
## Full-cache follow run

Because the cropped ImageNet-256 cache may still be growing, the preferred full-cache path is now the follow/resume launcher:

```bash
BATCH_SIZE=256 experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/scripts/launch_flux2_from_cropped_follow_full_setsid.sh
```

This writes its outer log path to `results/flux2_from_cropped_follow_full_outer.logpath`, PID to `flux2_from_cropped_follow_full.pid`, and output shards to `/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full`. The script encodes currently durable cropped shards, exits the inner builder at the current crop total, then polls and resumes until `1,281,167` ImageNet train samples are encoded.

