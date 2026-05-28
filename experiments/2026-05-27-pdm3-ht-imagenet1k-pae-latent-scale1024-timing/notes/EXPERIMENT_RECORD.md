# Experiment Record — ImageNet PAE latent scale1024 timing

Date: 2026-05-27

## Setup

Experiment directory:

- `/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-scale1024-timing`

Persistent output:

- `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke1024`

Config:

- `configs/imagenet1k_pae_dinov2l_d32_smoke1024_b32.env`

## Initial wrapper issue

First attempt used `/usr/bin/time`, but this container does not have `/usr/bin/time`.
The wrapper was patched to record end-to-end shell seconds using `date +%s%N`.

Failed first log:

- `logs/010_hf_imagenet_smoke1024_pae_latent_timing.log`

Successful timing log:

- `logs/011_hf_imagenet_smoke1024_pae_latent_timing.log`

## Build command

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-scale1024-timing/scripts/run_smoke1024_timing.sh \
  2>&1 | tee experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-scale1024-timing/logs/011_hf_imagenet_smoke1024_pae_latent_timing.log
```

## Build result

Result: PASS

- Seen / encoded: `1024` / `1024`
- Shards: `1`
- Shard path: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke1024/latents_rank00_shard000.safetensors`
- Schema: `latents` and `latents_flip` `[1024,32,16,16]` BF16; `labels` `[1024]` I64
- Labels min/max: `0` / `999`
- Save seconds: `0.17846965789794922`
- CUDA peak: `2051.349609375 MB`
- Script build elapsed: `103.54050254821777` sec
- Shell total elapsed: `142.404707856` sec

## Verification

Command/log:

- `logs/020_verify_smoke1024_imglatentdataset.log`

Result: PASS

- len: `1024`
- first shape: `[32,16,16]`
- first dtype: `torch.bfloat16`
- first label: `726`

## Throughput

- Steady seconds/sample: `0.10111377201974392`
- Steady samples/sec: `9.889849622114143`
- Normal+flip PAE encode ops/sec: `19.779699244228286`
- Small-run shell samples/sec: `7.190773503327372`
- Fixed startup overhead estimate: `38.864205307782214` sec

## 1M extrapolation

One continuous run estimate:

- steady-state: `101113.77201974392` sec = `28.08715889437331` h = `1.1702982872655545` d
- with one fixed overhead: `101152.6362250517` sec = `28.097954506958803` h = `1.1707481044566168` d

Naive scaling of this exact 1024 shell job:

- `139067.09751562498` sec = `38.62974930989583` h = `1.6095728879123261` d

Use the steady/fixed-overhead estimate for a single long extraction job. Plan 30–40 h for safety.

## Storage extrapolation

- Observed shard bytes: `33563320`
- Bytes/sample from shard: `32776.6796875`
- 1M storage: `32776679687.5` bytes = `32.7766796875` GB = `30.525661713909358` GiB
- 1024-shard count for 1M: `977`

## Next

Before full extraction, run a batch-size sweep (`batch_size=64/128`) or a `smoke10000` timing build to reduce uncertainty and improve throughput estimate.
