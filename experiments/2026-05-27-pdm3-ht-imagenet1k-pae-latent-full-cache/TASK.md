# Task — Full ImageNet-1k train → PAE latent cache

Date: 2026-05-27

## Objective

Build the production full-cache dataset for Phase 0/B3 and later PDM-3-HT work:

- Source: `ILSVRC/imagenet-1k`, split `train`, ImageNet-1k train size 1,281,167.
- Transform: ADM-style center crop to 256×256 RGB.
- Encoder: frozen `PAE_DINOv2L_d32`.
- Output tensors per sample:
  - `latents`: `[32,16,16]`, bf16
  - `latents_flip`: horizontal-flip bf16 latent, `[32,16,16]`
  - `labels`: int64 class id
- Final directory: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full`.

## Required engineering properties

1. Resume-safe full build.
2. Atomic shard write + schema verification before publishing.
3. Manifest/progress/build-summary JSON artifacts.
4. Batch/progress/memory logging through `tee` or nohup log.
5. Clean experiment directory management.
6. Avoid storing full raw ImageNet copy unless explicitly needed.

## Background measurements

Previous real HF streaming measurements showed the old serial PIL pipeline is input/preprocess-bound:

- smoke1024 steady throughput ≈ 9.89 samples/s.
- batch saturation sweep ≈ 8–9.2 samples/s end-to-end.
- PAE encode-only upper bound ≈ 142 samples/s.
- Full train estimate under old pipeline: ≈ 36–44 h, final storage ≈ 42 GB.

This experiment introduces a production builder with single-crop + batch flip + threaded preprocessing/prefetch while retaining the same output schema.
