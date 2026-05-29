# B3 PAE MeanFlow — step-100k multi-NFE 5K Inception FID sweep

更新时间：`2026-05-29T07:05Z`

## Why this was run

User requested a sampler sensitivity check on the old b96 constant-LR `step_00100000.pt` checkpoint:

```text
测试 2/4 nfe，之前的 baseline 是 1 nfe
```

Important local-code caveat: in this trainer/eval code, the sampler control is `--sample-steps`; this is the NFE count for the deterministic Euler MeanFlow sampler. The already-existing local step-100k 5K anchor was **not** a 1-NFE file: its metadata says `sample_steps: 32`. Therefore this sweep ran:

```text
sample_steps / NFE = 1, 2, 4
```

The requested `2/4` points are included, and `1` was added to make the comparison self-contained and avoid mixing it with the old `32`-step anchor.

## Scope / protocol

Checkpoint:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00100000.pt
```

Config:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_full_fast_h100_b96.yaml
```

Common settings:

```text
num generated / real: 5000 / 5000
EMA: true
precision: bf16_autocast
sample batch size: 64
seed: 2026052910
real seed: 20260529
PAE decode: CUDA bf16
Inception: CUDA torchvision InceptionV3 pool2048
```

## Results

| sample_steps / NFE | FID ↓ | Δ vs NFE1 | Inception RBF MMD ↓ | Inception poly3 KID ↓ | sample sec | decode+Inception sec | finite |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| `1` | `56.96605626689188` | `+0.000000` | `0.03449982203769153` | `0.04272002348840154` | `31.09` | `120.13` | yes |
| `2` | `53.47678397375779` | `-3.489272` | `0.030919553701898694` | `0.03747335398130902` | `39.65` | `118.28` | yes |
| `4` | `49.93874684403772` | `-7.027309` | `0.02770172222198064` | `0.03236527203857964` | `54.86` | `113.98` | yes |

Existing local anchor for reference, **not 1-NFE**:

| sample_steps / NFE | FID ↓ | MMD ↓ | KID ↓ | seed |
|---:|---:|---:|---:|---:|
| `32` | `47.925748666494485` | `0.02595176471424887` | `0.02984040431452506` | `2026052910` |

## Interpretation

- Increasing NFE improves FID monotonically on this old step-100k checkpoint:
  - NFE1 → NFE2: `-3.49` FID;
  - NFE1 → NFE4: `-7.03` FID.
- NFE4 is close to the old 32-step anchor but still about `+2.01` FID worse at this 5K diagnostic sample count.
- The result suggests part of the high FID at low NFE is sampler truncation error. It does **not** by itself rescue the old b96 constant-`2e-4` route as the main fair evaluation, because that route remains hyperparameter-misaligned with the external MeanFlow reference setup.
- For future apples-to-apples reporting, label these as `sample_steps/NFE`, and do not call the previous `47.9257` anchor “1 NFE”; it was `sample_steps=32`.

## Artifacts

Local summary:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00100000/multi_nfe_fid_summary.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00100000/multi_nfe_fid_summary.md
```

Per-NFE metrics:

```text
eval/step_00100000/inception_eval_5k_nfe1/inception_metrics.json
eval/step_00100000/inception_eval_5k_nfe2/inception_metrics.json
eval/step_00100000/inception_eval_5k_nfe4/inception_metrics.json
```

Per-NFE latent metadata:

```text
eval/step_00100000/sample_latents_5000_nfe1.json
eval/step_00100000/sample_latents_5000_nfe2.json
eval/step_00100000/sample_latents_5000_nfe4.json
```

The `.safetensors` latent files are intentionally **not** committed to GitHub; if needed, upload those to HF artifact repo.

## Operational note

The active b128 MeanFlow-reference 50k control was temporarily stopped to free H100 memory. It had reached roughly:

```text
step: 1678
loss: 0.4163530468940735
lr: 1.678e-05
```

No checkpoint had been written yet, so the b128 control restarted from scratch after the sweep.

Restarted train process after the sweep:

```text
pid: 77429
config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_50k.yaml
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_lr1e4_cosine_50k_restart_after_multi_nfe_20260529T070105Z.log
```

The 50k FID controller and post-50k HF upload controller remain active and are still waiting for the new `step_00050000` metrics.
