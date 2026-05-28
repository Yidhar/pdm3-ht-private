# PDM-3-HT VAE-Agnostic Latent Backend Experiment

Date: 2026-05-27

## Framing correction

PDM-3-HT must be framed as **VAE-agnostic**. PAE is not an innovation component; it is a latent backend / baseline tool.

Correct research framing:

```text
latent backend (PAE / FLUX.2 VAE / Qwen Image VAE / future VAEs)
+
HT per-patch spatial-time mechanism
+
MeanFlow trajectory mechanism
```

For paper narrative, the novelty should be sold as the **HT + MeanFlow per-patch integration / two-level decoupling**, validated over multiple latent spaces. PAE should be presented as one strong ImageNet-tuned latent baseline, not as the core proposed component.

## Required honesty / reporting rule

If PAE gets better FID than Qwen/FLUX VAE, report it directly instead of hiding it. The possible outcomes are:

- **A / best**: PAE, Qwen-VAE, and FLUX.2-VAE all work; PAE is numerically best because it was tuned for ImageNet-256. Use this to support generality while acknowledging backend-specific tuning.
- **B / medium**: all three work with close numbers. This is the cleanest VAE-agnostic ablation.
- **C / worst**: Qwen/FLUX do not work or degrade substantially. This weakens the VAE-agnostic claim; report that PDM-3-HT currently works reliably only on the PAE latent backend.

## Proposition 1 repositioning

Do **not** claim a PAE-specific MCR guarantee for JVP stability as the core theory. Reposition as:

```text
Any well-behaved / manifold-regularized VAE latent should provide an analogous smooth latent manifold for JVP-based MeanFlow targets. PAE is one verified instance; FLUX.2 and Qwen Image VAE are empirical swap-in tests.
```

## Engineering objective

Build ImageNet-1k 256x256 latent caches from the already materialized ADM-cropped uint8 safetensors cache:

```text
/workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors
```

Primary target for this experiment:

```text
FLUX.2 dev VAE / AutoencoderKLFlux2
repo used for accessible VAE weights: diffusers/FLUX.2-dev-bnb-4bit
output: /workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full
```

Secondary planned target:

```text
Qwen Image VAE / AutoencoderKLQwenImage
output: /workspace/PDM/data/vae_latents/AutoencoderKLQwenImage_d16/imagenet256_train_full
```

## Cache schema

Use the same training-friendly shard schema as PAE latent cache where possible:

```text
latents       bf16 [N, C, H, W]
latents_flip  bf16 [N, C, H, W]
labels        int64 [N]
```

For FLUX.2 VAE shape probe on ImageNet-256:

```text
AutoencoderKLFlux2 posterior mode shape: [N, 32, 32, 32]
```

Use deterministic posterior `mode()` by default, not `sample()`, for reproducible cache and comparability with deterministic PAE encoding.
