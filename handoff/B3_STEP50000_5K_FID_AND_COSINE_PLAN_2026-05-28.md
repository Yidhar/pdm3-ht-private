# B3 PAE MeanFlow — step-50k 5K true Inception FID anchor + cosine LR plan

更新时间：`2026-05-28T18:38Z`

## TL;DR

- The exact `step_00050000` PAE B3 b96 checkpoint was retained and evaluated with a **5,000 generated / 5,000 real true Inception** pass.
- Result: **FID = `55.528315`**, Inception RBF MMD = `0.0328459`, Inception poly3 KID = `0.0389490`.
- This supersedes the normal `64`-sample trainer eval for health decisions. The `64`-sample FID is only a smoke/wiring check and should not be used as a convergence gate.
- Training was resumed after the GPU eval from the latest local checkpoint (`step_00052000.pt`); the main run remains the base PAE + LightningDiT + MeanFlow route. Representation Fréchet Loss / FD-loss is still deferred.
- HF artifact watcher uploaded the refreshed step-50k eval artifacts to `LAXMAYDAY/pdm3-ht-model-artifacts` under `b3_meanflow_realdata/fullcache_b96/step_00050000`.
- Cosine LR decay is planned for later/checkpoint-controlled use; it has **not** been applied to the live run yet.

## Exact metric record

Local metric file:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00050000/inception_eval_5k/inception_metrics.json
```

Key values:

| metric | value |
|---|---:|
| generated samples | `5000` |
| real samples | `5000` |
| feature method | `torchvision_inception_v3_imagenet1k_v1_pool2048` |
| FID ↓ | `55.52831543442829` |
| FID mean term | `20.92536628842696` |
| FID trace generated | `181.11671259223343` |
| FID trace real | `195.0878562239439` |
| FID trace sqrt product | `170.80080983508802` |
| Inception RBF MMD ↓ | `0.032845868596164784` |
| Inception poly3 KID ↓ | `0.03894902108381171` |
| generated feature shape | `[5000, 2048]` |
| real feature shape | `[5000, 2048]` |
| decode + Inception elapsed | `107.224755 sec` |

The eval command used GPU PAE decode and GPU Inception, with `--max-save-images 0 --grid-count 0 --no-save-npz`, so no raw real ImageNet image payloads or decoded image NPZ payloads were intended for upload.

## Sampling record

Local sample files:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00050000/sample_latents_5000.safetensors
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00050000/sample_latents_5000.json
```

Sample metadata:

| field | value |
|---|---:|
| checkpoint step | `50000` |
| checkpoint has EMA | `true` |
| checkpoint has optimizer | `true` |
| sample shape | `[5000, 32, 16, 16]` |
| sample steps | `32` |
| sample batch size | `64` |
| precision mode | `bf16_autocast` |
| EMA sampling | `true` |
| seed | `2026052850` |
| finite | `true` |
| latent mean | `0.20773913` |
| latent std | `0.96754885` |
| sampling elapsed | `321.559445 sec` |
| CUDA peak memory | `5835.43 MiB` |

## HF artifact upload

HF model artifact repo:

```text
https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
```

Remote prefix:

```text
b3_meanflow_realdata/fullcache_b96/step_00050000
```

Relevant upload commits:

- `b3cd2c26c56829f793914e3f088d91cb2498ec6e`: initial normal step-50k eval upload.
- `663fe3dfb6f32b2a4380327cc256940bf3fa2975`: refresh containing the 5K latent sample artifact.
- `019298e6dee68c5b2963015e60eb5b5a8210e194`: refresh containing the 5K Inception metric JSON/MD.

The watcher excludes raw real ImageNet images, `inception_features.npz`, and decoded real/generated NPZ payloads by default. The `sample_latents_5000.safetensors` artifact is allowed and uploaded for reproducible downstream decode/eval.

## Runtime / resume state after eval

The step-50k checkpoint was retained for eval as:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints_retained_for_eval/step_00050000_5k_anchor.pt
```

The normal checkpoint directory after cleanup retained the latest resume target:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/latest.pt -> step_00052000.pt
```

Training resumed after GPU eval from `latest.pt` and continued past step `52k`. The 5K eval was therefore anchored to exact step `50000`, while the live trainer did not roll back and resumed from the newer `52000` checkpoint.

## Interpretation

The earlier `30k → 40k` compact/small-FID bump and the `64`-sample step-50k FID should not be over-interpreted. The meaningful early anchor is this 5K true Inception pass:

```text
step 50k, 5K true Inception FID = 55.5283
```

This is far below the noisy `64`-sample FID scale and confirms that the image-space pipeline is connected and the model is learning under the current PAE B3 b96 MeanFlow route. It does **not** make the run publishable; this is still only a 5K early anchor, not a 50K paper FID. The first PAE-paper-comparable long-run target remains approximately `1.07M` steps.

## Controller bugfix

The 5K anchor controller now uses `/proc/<pid>/cmdline` exact argv matching for trainer process groups instead of substring matching through `ps | awk`. The old command text contained the literal trainer script path and could match the controller's own process group, causing the controller to terminate itself. This is fixed in:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_b3_5k_fid_anchor_at_step.sh
```

## Cosine LR decay plan for later

Current live config at this handoff still uses a constant optimizer LR:

```yaml
optimizer:
  lr: 0.0002
```

MeanFlow reference settings discussed for comparison were approximately:

```text
batch 128, Adam/AdamW lr 1e-4, cosine decay, warmup, ~800k steps
```

The active run uses global batch `96`. By a linear batch-size scaling rule, the reference base LR becomes:

```text
1e-4 * 96 / 128 = 7.5e-5
```

Recommended later schedule family:

```text
base_lr:      7.5e-5
min_lr:       7.5e-6   # 10% of base
warmup_steps: 13333    # batch-scaled from a 10k/128-style warmup, or use 10k if matching existing code simplicity
end_step:     1070000
scheduler:    warmup + cosine decay
```

Formula:

```python
if step < warmup_steps:
    lr = base_lr * step / warmup_steps
else:
    p = min(1.0, max(0.0, (step - warmup_steps) / (end_step - warmup_steps)))
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos(pi * p))
```

Schedule values with `base_lr=7.5e-5`, `min_lr=7.5e-6`, `warmup_steps=13333`, `end_step=1070000`:

| step | LR |
|---:|---:|
| `13,333` | `7.500e-5` |
| `50,000` | `7.480e-5` |
| `52,000` | `7.478e-5` |
| `60,000` | `7.468e-5` |
| `100,000` | `7.389e-5` |
| `200,000` | `6.993e-5` |
| `400,000` | `5.505e-5` |
| `600,000` | `3.543e-5` |
| `800,000` | `1.780e-5` |
| `1,000,000` | `8.228e-6` |
| `1,070,000` | `7.500e-6` |

Because the live run has already trained to `>50k` at `2e-4`, do not retroactively rewarm. If/when switching this run to the batch-scaled cosine, use a checkpoint-controlled transition:

1. choose a natural boundary (`60k`, `100k`, or after two 5K-FID anchors if there is plateau evidence);
2. resume from `latest.pt` with a scheduler-aware train script;
3. explicitly override optimizer param-group LR after loading the checkpoint, because PyTorch optimizer state restores the old `2e-4` LR;
4. optionally linearly ramp down from `2e-4` to the cosine target over `2k–5k` steps to avoid a discontinuous optimizer shock;
5. log LR into every `train_step` record.

Recommended config fragment once scheduler support is implemented:

```yaml
optimizer:
  lr: 0.000075
  min_lr: 0.0000075
  scheduler: cosine
  warmup_steps: 13333
  decay_end_step: 1070000
  resume_lr_policy: scheduler_over_checkpoint
  lr_transition_steps: 5000
```

Decision guidance:

- Since the step-50k 5K FID is healthy (`55.53`), there is no urgent need to hot-change LR before the next planned checkpoint/eval boundary.
- If future 5K anchors plateau or worsen on two consecutive anchors, switch to the batch-scaled cosine plan immediately.
- If future anchors continue improving strongly, keep the current run stable and schedule the cosine switch at a cleaner milestone such as `100k`, or use this plan for the next long-run restart/branch.
