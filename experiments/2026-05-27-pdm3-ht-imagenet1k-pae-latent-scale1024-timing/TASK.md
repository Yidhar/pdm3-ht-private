# Experiment 10: HF ImageNet-1k → PAE_DINOv2L_d32 Latent Scale1024 Timing

Date: 2026-05-27

## User directive

先建立更大的真实 latent，并计算外推到 1M samples 的时间。

## Objective

Build a larger real ImageNet-1k train latent subset than smoke64 and measure extraction throughput for extrapolating wall time/storage to 1M samples.

## Scope

- Dataset: `ILSVRC/imagenet-1k`, split `train`, streaming.
- Encoder: PAE_DINOv2L_d32.
- Output: safetensors shards compatible with `ImgLatentDataset`.
- Subset: 1024 samples, batch size 32, shard size 1024.
- Compute timing from script `build_summary.elapsed_seconds`, external `/usr/bin/time`, and output payload size.

## Guardrails

- Do not start full ImageNet extraction.
- Keep persistent latent data under `/workspace/PDM/data/pae_latents/...`.
- Keep experiment records/logs/configs under this experiment directory.


## Outcome

Result: PASS

- Built real ImageNet train smoke1024 PAE latents at `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke1024`.
- `ImgLatentDataset` verification PASS.
- Build elapsed: `103.54050254821777` sec for `1024` samples.
- Shell total: `142.404707856` sec.
- Steady throughput: `9.889849622114143` original samples/sec.
- 1M steady-state extrapolated time: `28.08715889437331` hours.
- 1M storage estimate: `32.7766796875` GB decimal / `30.525661713909358` GiB.
