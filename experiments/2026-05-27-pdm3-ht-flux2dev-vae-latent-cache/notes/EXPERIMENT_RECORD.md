# Experiment record — VAE-agnostic latent cache

## 2026-05-27 — Framing revision accepted

- PAE is no longer framed as an innovation component.
- PDM-3-HT novelty should be HT + MeanFlow per-patch integration / two-level decoupling.
- VAE backend should be swappable: PAE, FLUX.2 VAE, Qwen Image VAE.
- FID outcomes must be reported honestly:
  - A: all work, PAE best due ImageNet tuning;
  - B: all work, close numbers;
  - C: Qwen/FLUX fail or degrade, weakening VAE-agnostic claim.

## 2026-05-27 — FLUX.2 CPU shape probe

`AutoencoderKLFlux2` from `diffusers/FLUX.2-dev-bnb-4bit`, subfolder `vae`, was loadable in this environment.

Observed with one 256x256 input on CPU:

```json
{
  "latent_dist_type": "DiagonalGaussianDistribution",
  "mode_shape": [1, 32, 32, 32],
  "mode_dtype": "torch.float32",
  "mode_mean": -0.09721769392490387,
  "mode_std": 1.3858556747436523,
  "sample_shape": [1, 32, 32, 32]
}
```

Decision: cache deterministic `posterior.mode()` unless a downstream FLUX pipeline convention requires sampled latents.

## 2026-05-27 — Generic diffusers VAE cache builder CPU one-sample validation

Implemented:

```text
scripts/build_diffusers_vae_latents_from_cropped_cache.py
```

Validation command encoded 1 cropped ImageNet-256 sample with `AutoencoderKLFlux2` on CPU to avoid disturbing the running PAE GPU cache.

Result:

```json
{
  "ok": true,
  "encoded_total": 1,
  "first_latent_shape": [1, 32, 32, 32],
  "latent_mode": "mode",
  "model_dtype": "fp32",
  "save_dtype": "bf16",
  "encode_time_total_sec": 10.906434059143066,
  "output_dir": "/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_cpu_one_sample"
}
```

CPU speed is not representative. GPU smoke is still needed once it will not interfere with the active PAE pass.

## 2026-05-27 — VAE-agnostic trade-off / outcome taxonomy locked

User-level reporting rule recorded as a hard constraint: if PAE reaches e.g. FID 1.0 while Qwen-VAE/FLUX.2-VAE reach e.g. FID 1.5 because they are not ImageNet-256 tuned, we must report that honestly. This does **not** invalidate the experiment, but it changes the strength of the VAE-agnostic claim.

Outcome taxonomy for paper framing:

- **A / best**: PAE, Qwen-VAE, and FLUX.2-VAE all work; PAE is best because it was ImageNet-256 tuned. Use this to sell generality while acknowledging backend-specific tuning.
- **B / medium**: all three work with close numbers. This is the cleanest ablation.
- **C / worst**: Qwen/FLUX fail or degrade substantially. Then the honest conclusion is that PDM-3-HT is not yet truly VAE-agnostic and currently works reliably only on PAE.

Implementation implication: B3/MeanFlow loader and evaluation must keep `vae_backend`, latent shape, decoder, FID, and MMD separated in logs/results so backend-specific degradation cannot be accidentally hidden.

## 2026-05-27 — FLUX.2 GPU smoke8192 throughput, batch64 effective

Ran `run_flux2_smoke8192.sh`; due current config sourcing, attempted external `BATCH_SIZE=256` was overridden and the actual effective batch size was 64. This exposed an engineering issue: run scripts must preserve explicit env overrides for batch tuning.

Result summary:

```json
{
  "ok": true,
  "status": "complete",
  "encoded_total": 8192,
  "first_latent_shape": [
    64,
    32,
    32,
    32
  ],
  "avg_new_samples_per_sec": 47.50006533797097,
  "elapsed_sec": 172.4629180431366,
  "encode_time_total_sec": 152.17204332351685,
  "h2d_time_total_sec": 7.7229390144348145,
  "save_time_total_sec": 4.055973052978516,
  "cuda_peak_allocated_mb": 10665.30126953125,
  "num_new_shards": 2
}
```

Extrapolation at 47.500 samples/s (each sample writes original + flipped latent):

- 1,000,000 images: 5.85 h.
- ImageNet train 1,281,167 images: 7.49 h.
- FLUX.2 full latent+flip cache estimated size: 167.9 GB.

Next action: patch run scripts so explicit batch-size overrides work, then test larger batch sizes (128/256) before launching full cache.

## 2026-05-27 — Config override bug fixed for batch tuning

Observed during FLUX.2 smoke8192 that `BATCH_SIZE=256 scripts/run_flux2_smoke8192.sh` was silently reset to 64 because `configs/flux2dev_vae_cache.env` used unconditional `export BATCH_SIZE=64`. Patched the config to use shell-default form, e.g. `export BATCH_SIZE="${BATCH_SIZE:-64}"`, and did the same for output paths/device/dtype settings. This allows clean batch-size and output-dir sweeps without editing scripts.

## 2026-05-27 — FLUX.2 batch sweep completed


Measured batch-size points for `AutoencoderKLFlux2` latent cache from cropped ImageNet-256. Samples/sec counts images while encoding both original and horizontal-flip latents.

| point | target | avg samples/s | peak MB | est 1M h | est 1.281M h |
|---|---:|---:|---:|---:|---:|
| bs64_8192 | 8192 | 47.500 | 10665.3 | 5.85 | 7.49 |
| bs128_8192 | 8192 | 50.208 | 21001.3 | 5.53 | 7.09 |
| bs256_8192 | 8192 | 50.743 | 41673.3 | 5.47 | 7.01 |
| bs512_4096 | 4096 | 46.608 | 83016.7 | 5.96 | 7.64 |

Decision: **use batch size 256** for a dedicated FLUX.2 full cache run. Batch 128 is nearly as good and uses half the memory; batch 512 almost fills the GPU (~83GB) but is slower, so it is not worth the risk.

Machine-readable summary: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/results/flux2_batch_sweep_summary.json`

## 2026-05-27 — FLUX.2 full cache follow run launched

Launched FLUX.2 VAE full-cache in follow/resume mode while the cropped ImageNet-256 cache is still being produced.

- Script: `experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/scripts/run_flux2_from_cropped_follow_full.sh`
- Launcher: `experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/scripts/launch_flux2_from_cropped_follow_full_setsid.sh`
- PID: `1660571`
- Log: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/logs/flux2_from_cropped_follow_full_setsid_20260527T194322Z.outer.log`
- Output: `/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full`
- Batch size: `256`
- Expected latent: `[32, 32, 32]`, saved as bf16; deterministic posterior `mode()`.
- Reason: avoid waiting for crop cache completion and avoid a partial one-shot run. The follow loop resumes as new crop shards appear until all `1,281,167` train samples are encoded.
- GPU scheduling: PAE follow watcher remains SIGSTOPed so this FLUX.2 pass gets the GPU exclusively.


## HF FLUX.2 VAE latent dataset upload launched (2026-05-28T02:00:09+00:00)

- repo: `LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents` (https://huggingface.co/datasets/LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents)
- repo_type: `dataset`, private: `true`
- supervisor PID: `1688293`
- log: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/logs/hf_upload_flux2_vae_latents_20260528T015927Z.log`
- behavior: waits for full cache validation, then uploads only `latents_rank00_shard*.safetensors` plus `README.md`, `run_config.json`, `build_summary.json`, `progress.json`, `manifest.jsonl`, and `progress.jsonl`.
- explicit excludes: raw ImageNet parquet, cropped uint8 shards, PAE latents, checkpoints, logs, temp files.

## 2026-05-28 handoff document uploaded to private HF code repo

Created and uploaded a compact handoff bundle for running PAE-backed work on a second machine.

Local files:

```text
/workspace/PDM/HANDOFF_2026-05-28_PDM3_HT_PRIVATE_HF_AND_PAE_PARALLEL.md
/workspace/PDM/.hf_publish/handoff_bundle/README.md
/workspace/PDM/.hf_publish/handoff_bundle/state/handoff_state_20260528T022421Z.json
/workspace/PDM/.hf_publish/handoff_upload_result.json
```

Private HF code repo paths:

```text
https://huggingface.co/LAXMAYDAY/pdm3-ht-20260528-code/blob/main/HANDOFF_2026-05-28_PDM3_HT_PRIVATE_HF_AND_PAE_PARALLEL.md
https://huggingface.co/LAXMAYDAY/pdm3-ht-20260528-code/tree/main/handoff
```

Upload policy: handoff README, compact log snippets, and state JSON only. No raw ImageNet, no cropped uint8 cache, no PAE/FLUX latent shards, no checkpoints/model weights, no `/data`, no `/external`.

## 2026-05-28 — Full FLUX.2 cache complete and consumed by B3 route

FLUX.2 VAE cache completed at `2026-05-28T02:56:00Z`:

```json
{
  "latent_files": 313,
  "latent_samples": 1281167,
  "expected": 1281167,
  "remaining_latents": 0,
  "last_shard": "latents_rank00_shard000312.safetensors",
  "last_shape": [3215, 32, 32, 32]
}
```

The B3 real-data trainer now has FLUX-specific configs using `downsample_ratio=8` and `patch_size=2`. One-shard smoke and all-shard 20-step shorttrain both passed the practical gate with fp32 live-param JVP target and bf16 backbone. See:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/smoke_flux2vae_4096/train_summary.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/shorttrain_flux2vae_fullcache_20/train_summary.json
```

## 2026-05-28 — FLUX.2 latent HF dataset upload blocked by private storage quota

The full local FLUX.2 latent cache is valid and complete, but upload to the private HF dataset repo was stopped after repeated LFS preupload failures:

```text
403 Forbidden: Private repository storage limit reached, please upgrade your plan to increase your private storage limit.
```

Stopped uploader PID `1688293` to avoid wasting CPU/network. Local result marker:

```text
/workspace/PDM/.hf_publish/flux2_vae_latents_upload_result.json
```

This does not affect local FLUX-route training. To share the full `~168 GB` latent cache, either increase private HF storage, upload to an org/account with quota, split to another storage backend, or transfer directly between machines.


## 2026-05-28T03:18Z — Public FLUX.2 VAE latent upload launched

User requested that the full cache be temporarily put into a public dataset repo. The previous private dataset upload was blocked by HF private storage quota, so the uploader was patched to accept `PDM_FLUX2_VAE_LATENT_REPO_PRIVATE=0` and relaunched against a public repo.

- repo: `LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents-public`
- url: https://huggingface.co/datasets/LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents-public
- repo_type: `dataset`, private: `false`
- launch JSON: `/workspace/PDM/.hf_publish/flux2_vae_latents_public_upload_launch.json`
- log: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/logs/hf_upload_flux2_vae_latents_public_20260528T031802Z.log`
- uploader PID at launch: `1717825`
- validated before upload: `313` shards, `1,281,167` samples, `167.936` GB decimal, no tmp/bad files
- allowlist only: `README.md`, `run_config.json`, `build_summary.json`, `progress.json`, `manifest.jsonl`, `progress.jsonl`, `latents_rank00_shard*.safetensors`
- excluded: raw ImageNet parquet, cropped uint8 cache, PAE latents, checkpoints, logs, temp files

Early log status: upload started and entered `upload_large_folder`; pre-upload progress began (e.g. tens of GB pre-uploaded). Continue monitoring the log/result JSON.

## HF FLUX.2 VAE latent dataset upload (2026-05-28T03:23:32+00:00)

- status: `complete`
- repo: `LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents-public`
- url: https://huggingface.co/datasets/LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents-public
- repo_type: `dataset`, private: `false`
- uploaded scope: FLUX.2 VAE latent cache only (`latents_rank00_shard*.safetensors` + metadata/card)
- excluded: raw ImageNet parquet, cropped uint8 cache, PAE latents, checkpoints, logs, temp files
- validated samples/shards: `1281167` / `313`
- result json: `/workspace/PDM/.hf_publish/flux2_vae_latents_upload_result.json`
