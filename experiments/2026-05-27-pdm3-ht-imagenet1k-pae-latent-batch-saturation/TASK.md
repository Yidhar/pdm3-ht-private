# Experiment 11: PAE ImageNet latent extraction batch saturation throughput

Date: 2026-05-27  
Status: **PASS / completed**

## User directive

测一些吃满后的吞吐量来计算全量缓存的时间。

## Objective

Run real HF ImageNet-1k train → PAE_DINOv2L_d32 latent extraction with larger batch sizes, measure near-saturation/high-batch throughput, verify saved latent subsets, and extrapolate full ImageNet train cache time/storage.

## Scope

- Dataset: `ILSVRC/imagenet-1k`, split `train`, streaming.
- Encoder: `PAE_DINOv2L_d32`.
- Dtype: bf16 model and bf16 saved latents.
- Output schema: `latents`, `latents_flip`, `labels` compatible with `ImgLatentDataset`.
- Batch sweep: 64, 128, 256, 512, 1024; 2048 real samples each.
- Encode-only probe: 1024, 2048, 3072, 4096 synthetic preprocessed-shape tensors for GPU/VRAM upper-bound measurement.
- Do not start full extraction.

## Guardrails

- Each run is a timing subset only.
- Persistent subset outputs under `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/batch_saturation_2026-05-27/`.
- Record every command/log/result under this experiment directory.

## Results

### Real streaming sweep

- PASS for b64/b128/b256/b512/b1024.
- Best current real sweep end-to-end throughput: `9.1987 samples/s` at batch64.
- Larger batches did not improve total throughput.
- Best real encode-only component: `147.77 samples/s` at batch1024.
- b1024 peak VRAM: `22.05 GiB`, only `23.2%` of the 94.97 GiB GPU.
- Preprocessing consumed roughly `71–78%` of measured elapsed time.

### Encode-only probe

- PASS for batch1024/2048/3072.
- batch4096 OOM.
- Best encode-only pair throughput: `142.07 samples/s` at batch2048.
- Largest passing batch: 3072, `63.44 GiB`, `66.8%` VRAM, `141.90 samples/s`.

### Verification

All five saved real-output latent directories verified PASS with BF16 `[2048,32,16,16]` normal and flip latents and I64 labels.

## Full-cache extrapolation

Main wall-clock estimate for current serial HF streaming/PIL implementation:

- Use previous Exp10 b32 steady `9.8899 samples/s` for optimistic current-pipeline estimate.
- Use this sweep b64 `9.1987 samples/s` for conservative current-pipeline estimate.

Time:

- 1M samples: `28.09–30.20 h` current-pipeline range.
- Full ImageNet train 1,281,167 samples: `35.98–38.69 h` current-pipeline range.
- Worst current sweep point b128: `43.70 h` for full train.
- Encode-only lower bound if input pipeline were hidden: `~2.4–2.5 h` for full train.

Storage:

- 1M samples: `~32.78 GB / 30.53 GiB`.
- Full ImageNet train: `~41.99 GB / 39.11 GiB`.

## Decision

Do **not** assume higher batch alone will make full caching fast. The current implementation is input-pipeline-bound. For full cache under current code, budget **36–39 hours** and optionally reserve **~44 hours** for variance. To approach encoder limits, implement local/multiworker/overlapped preprocessing first.
