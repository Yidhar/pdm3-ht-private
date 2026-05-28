# Experiment 8 — HF ImageNet-1k → PAE latent build

This experiment prepares the data prerequisite for the Phase0/B3 real-latent MeanFlow baseline.

User-specified source:

- `https://huggingface.co/datasets/ILSVRC/imagenet-1k`

Target latent format is compatible with:

- `/workspace/PDM/external/PAE/pae_with_generator/dataset/img_latent_dataset.py`

## Status

Overall: **PASS**

- HF login/access verified for account `LAXMAYDAY`.
- `ILSVRC/imagenet-1k` train streaming sample access verified.
- PAE checkpoint downloaded and inspected.
- Required dependency gap fixed (`transformers` upgraded to provide `Dinov2WithRegistersModel`; `omegaconf` installed).
- PAE model loads and encodes 256px tensors to `[N, 32, 16, 16]` BF16 latents.
- Safetensors schema validated with repository `ImgLatentDataset`.
- Upstream `extract_latents.py` indentation compile error fixed.
- Real HF ImageNet train smoke64 → PAE_DINOv2L_d32 latent build completed.

## Main scripts

- Build/access/smoke script: `scripts/build_pae_latents_from_hf_imagenet.py`
- Smoke64 wrapper: `scripts/run_smoke64.sh`
- Dataset verifier: `scripts/verify_img_latent_dataset.py`
- Defaults: `configs/imagenet1k_pae_dinov2l_d32_smoke64.env`

## Persistent assets

- PAE checkpoint:
  `/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt`
- Real smoke64 latent output:
  `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke64`
- Later full train latent output candidate:
  `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train`

## Real HF ImageNet smoke64 result

Command/log:

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-build/scripts/run_smoke64.sh \
  2>&1 | tee experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-build/logs/110_hf_imagenet_smoke64_pae_latent_build.log
```

Build summary:

- dataset: `ILSVRC/imagenet-1k`
- split: `train`
- seen / encoded: `64` / `64`
- shards: `1`
- elapsed seconds: `8.960774183273315`
- CUDA peak MB: `1552.083984375`
- output dir: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke64`

Shard schema:

- `latents`: `[64, 32, 16, 16]`, BF16
- `latents_flip`: `[64, 32, 16, 16]`, BF16
- `labels`: `[64]`, I64

Verification:

```bash
python experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-build/scripts/verify_img_latent_dataset.py \
  /workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke64 --latent-norm
```

Result: **PASS**

- `ImgLatentDataset` length: `64`
- first latent shape: `[32, 16, 16]`
- first latent dtype: `torch.bfloat16`
- first label: `726`

## Notes / quirks

- Initial post-login access check reached `ACCESS_CHECK_OK`, but Python crashed during `datasets` streaming finalization (`PyGILState_Release`).
- The experiment script now supports `--hard-exit`; the hard-exit access check and real smoke64 build completed cleanly.
- `load_state_dict(strict=False)` for the PAE checkpoint reports `missing=0`, `unexpected=5`; unexpected keys are only `teacher_latent_compressor.*` and do not block encoding.

## Downstream use

The smoke64 latent directory was consumed by Experiment 9:

- `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-real-imagenet64-smoke`

Experiment 9 result: B3 MeanFlow real-latent 8-step smoke **PASS**.

## Next recommended step

Do **not** start full ImageNet latent extraction blindly. Recommended progression:

1. Build `imagenet256_train_smoke1024` or `imagenet256_train_smoke10000` PAE latents.
2. Run a 50–200 step B3 MeanFlow short train on the larger real-latent set.
3. Only then approve full ImageNet train latent extraction.
