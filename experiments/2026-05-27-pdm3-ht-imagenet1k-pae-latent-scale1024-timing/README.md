# Experiment 10 — ImageNet PAE latent scale1024 timing

Status: **PASS**

Built a larger real HF ImageNet-1k train subset into PAE_DINOv2L_d32 BF16 latent shards and extrapolated timing/storage to 1M original samples.

## Input / config

- Dataset: `ILSVRC/imagenet-1k`
- Split: `train`
- Samples: `1024`
- Batch size: `32`
- Shard size: `1024`
- Image size: `256`
- Model dtype: `bf16`
- Save dtype: `bf16`
- Output: `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke1024`

## Output schema

- `latents`: `[1024, 32, 16, 16]`, BF16
- `latents_flip`: `[1024, 32, 16, 16]`, BF16
- `labels`: `[1024]`, I64

## Result

- Build status: `True`
- Seen / encoded: `1024` / `1024`
- Num shards: `1`
- CUDA peak: `2051.349609375 MB`
- Script build elapsed: `103.54050254821777` sec
- Shell total elapsed: `142.404707856` sec
- Estimated fixed overhead: `38.864205307782214` sec
- Steady throughput: `9.889849622114143` original samples/sec
- Effective normal+flip encode throughput: `19.779699244228286` PAE encode ops/sec

## Verification

`ImgLatentDataset(..., latent_norm=True)` verification: **PASS**

- len: `1024`
- first latent shape: `[32, 16, 16]`
- first latent dtype: `torch.bfloat16`
- first label: `726`

## 1M extrapolation

Primary estimate for one continuous extraction job:

- steady-state time: `101113.77201974392` sec = `28.08715889437331` hours = `1.1702982872655545` days
- with one fixed startup overhead: `101152.6362250517` sec = `28.097954506958803` hours = `1.1707481044566168` days

Naive end-to-end scaling of this small 1024 job:

- `139067.09751562498` sec = `38.62974930989583` hours = `1.6095728879123261` days

The naive number overcounts model load/random-self-test/dataset-init overhead unless the run is split into many tiny jobs.

Planning recommendation: reserve **30–40 hours** for 1M samples with this exact script/config, allowing network/HF streaming variability and shard I/O jitter. The measured steady-state extrapolation itself is about **28.1 hours**.

## 1M storage estimate

Observed safetensors shard bytes/sample: `32776.6796875` bytes.

For 1M original samples, including normal and flip latents:

- `32776679687.5` bytes
- `32.7766796875` GB decimal
- `30.525661713909358` GiB

At shard size 1024 this would create about `977` shards. A larger shard size should be considered for the full run.

## Notes

- The measured script `elapsed_seconds` starts after model load/random self-test/dataset construction and measures the actual dataset iteration + preprocess + normal/flip PAE encode + shard save loop.
- The shell total includes startup/model load/random self-test/dataset init.
- CUDA peak was only about `2051.349609375 MB`; batch-size sweep may improve throughput before committing to full extraction.
