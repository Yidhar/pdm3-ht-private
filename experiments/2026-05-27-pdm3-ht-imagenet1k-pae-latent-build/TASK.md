# Experiment 8: HF ImageNet-1k → PAE_DINOv2L_d32 Latent Build Smoke

Date: 2026-05-27

## User directive

Source dataset: https://huggingface.co/datasets/ILSVRC/imagenet-1k

Need to build PAE latents ourselves before Phase0/B3 real-latent MeanFlow smoke/training.

## Objective

Prepare and validate a reproducible pipeline:

HF `ILSVRC/imagenet-1k` raw images → PAE_DINOv2L_d32 encoder → `safetensors` latent shards compatible with `ImgLatentDataset` → B3 real-latent smoke.

## Guardrails

- Do not start blind full ImageNet download/encoding.
- First validate HF auth/dataset access, PAE checkpoint availability, PAE model load, and tiny latent shard smoke.
- Store large persistent assets outside the experiment directory.
- Keep experiment directory for scripts/configs/logs/results/records only.

## Proposed persistent paths

- PAE checkpoint: `/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt`
- Latent output smoke: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke64`
- Later full latent output: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train`

## Outcome on 2026-05-27

- HF ImageNet repo metadata visible, but sample access blocked by gated auth: machine is not logged in.
- PAE checkpoint downloaded from `yuezhengrong/PAE-collections` and inspected.
- Dependency gap fixed: `transformers==4.57.6`, `omegaconf==2.3.0`.
- PAE random encode self-test PASS: `[1,3,256,256] -> [1,32,16,16]` BF16.
- Synthetic PAE safetensors format smoke PASS with `ImgLatentDataset`.
- Real ImageNet smoke64 is ready to run after HF auth.


## Outcome after HF login

- HF auth verified as account `LAXMAYDAY`.
- `ILSVRC/imagenet-1k` train streaming access check PASS.
- Real ImageNet train smoke64 encoded with PAE_DINOv2L_d32:
  - output: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke64`
  - seen / encoded: `64` / `64`
  - shard count: `1`
  - latent schema: `latents` and `latents_flip` `[64,32,16,16]` BF16; `labels` `[64]` I64
  - CUDA peak: `1552.083984375 MB`
- `ImgLatentDataset(..., latent_norm=True)` verification PASS.
- Downstream Experiment 9 consumed this real smoke64 latent dataset and passed B3 MeanFlow trainer smoke.
