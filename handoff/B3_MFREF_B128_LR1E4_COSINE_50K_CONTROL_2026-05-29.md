# B3 MeanFlow reference-hparam 50k control — 2026-05-29

更新时间：`2026-05-29T06:30Z`

## Why this branch exists

The previous H100 route was useful as an engineering/control run, but it was **not hyperparameter-aligned** with the external MeanFlow reference recipe:

- previous route: batch `96`, constant LR `2e-4`;
- MeanFlow reference-like route requested here: batch `128`, Adam LR `1e-4`, warmup, cosine decay.

Therefore, the previous route can remain a diagnostic/control baseline, but it should **not** be treated as the primary fair evaluation of the MeanFlow/PAE method.  The user requested stopping the old route while the additional cost was still small and launching a paper/reference-hparam 50k comparison.

## Action taken

### Old b96 constant-2e-4 route stopped

Old result directory:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template
```

Stop point:

```text
step: 103354
time: 2026-05-29T06:18:29Z
last_loss: 0.4281488358974457
batch: 96
LR: constant 2e-4
```

The previously queued step-150k controller was cancelled because this branch is no longer the main decision route.

Recorded status JSON:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00150000/5k_anchor_control.json
```

Previous exact 5K true-Inception anchors remain useful controls:

| route | step | batch | LR | 5K FID ↓ | MMD ↓ | KID ↓ |
|---|---:|---:|---|---:|---:|---:|
| old control | 50k | 96 | constant `2e-4` | `55.52831543442829` | `0.032845868596164784` | `0.03894902108381171` |
| old control | 100k | 96 | constant `2e-4` | `47.925748666494485` | `0.02595176471424887` | `0.02984040431452506` |

## Trainer change: LR scheduler support

`train_b3_meanflow_realdata.py` now supports both backward-compatible constant LR and warmup/cosine LR.  The training loop sets the optimizer LR each step and logs:

```text
optimizer_lr
optimizer_lr_schedule
```

Supported flat config form:

```yaml
optimizer:
  lr: 0.000075
  min_lr: 0.0000075
  scheduler: cosine
  warmup_steps: 13333
  decay_end_step: 1070000
```

Supported nested config form:

```yaml
optimizer:
  lr: 0.000075
  scheduler:
    type: cosine
    min_lr: 0.0000075
    warmup_steps: 13333
    end_step: 1070000
```

`python3 -m py_compile` passed on the updated trainer.

## New configs

### Active exact/reference-like 50k control

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_50k.yaml
```

Key settings:

```yaml
train:
  exp_name: fullcache_realdata_mfref_b128_lr1e4_cosine_50k
  max_steps: 50000
  global_batch_size: 128
  global_seed: 20260529
  checkpoint_every: 10000
  keep_last_checkpoints: 2
  resume_from: null
  restore_rng: false
optimizer:
  lr: 0.0001
  min_lr: 0.00001
  scheduler: cosine
  warmup_steps: 10000
  decay_end_step: 800000
  beta1: 0.9
  beta2: 0.95
  weight_decay: 0.0
  max_grad_norm: 1.0
```

### Hardware-safe fallback, not started

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b96_lr7p5em5_cosine_50k.yaml
```

This uses the batch-scaled LR rule for batch 96:

```text
1e-4 * 96 / 128 = 7.5e-5
```

Use it only if the batch-128 exact/reference run fails/OOMs.

### Fit probe

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_fitprobe.yaml
```

The 3-step fit probe completed successfully; batch 128 fits on the H100 80GB.

Example probe observation:

```text
step: 3
batch: 128
lr: 3.0e-08
loss: ~1.9986
peak_memory_mb: ~74649.6
```

## Active run status

Active result directory:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_50k
```

Active train PID/log at launch:

```text
PID: 74778
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_lr1e4_cosine_50k_train_20260529T062339Z.log
pid file: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/mfref_b128_lr1e4_cosine_50k.pid
```

Status check at `2026-05-29T06:28Z`:

```text
step: 280
loss: 1.6233808994293213
lr: 2.8e-06
observed speed: ~0.96 steps/sec
peak_memory_mb from metrics: ~74652.0
nvidia-smi: ~77.6 GiB used, ~97% util, ~585 W, 55 C
```

The high early loss is expected because this is a fresh official-zero initialization with a very small warmup LR; do not compare its early loss directly against the already-trained old route.

Rough ETA for 50k, based on early speed:

```text
2026-05-29 ~20:15–21:30 UTC
```

## 50k 5K GPU FID controller

A one-shot controller is waiting for the `step_00050000.pt` checkpoint and will then run the requested 5K true-Inception anchor.

```text
controller PID: 75011
controller script: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_50k_5k_gpu_fid_final_20260529T062518Z.sh
controller log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_50k_5k_gpu_fid_final_20260529T062518Z.log
status JSON: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_50k/eval/step_00050000/5k_anchor_control.json
```

Expected output artifacts:

```text
eval/step_00050000/sample_latents_5000.safetensors
eval/step_00050000/sample_latents_5000.json
eval/step_00050000/inception_eval_5k/inception_metrics.json
eval/step_00050000/inception_eval_5k/inception_metrics.md
```

Controller behavior:

1. wait for `checkpoints/step_00050000.pt`;
2. wait for the trainer's normal small eval;
3. wait briefly for natural trainer exit because `max_steps=50000`;
4. if still active, terminate the exact trainer PGID to free H100;
5. sample 5K EMA latents;
6. run GPU PAE decode + Inception metrics;
7. do **not** resume training automatically, because this is a 50k comparison/control run.

## Comparison framing

Primary horizontal comparison requested by user:

```text
old control: b96 constant 2e-4, step 50k, FID 55.528315
new control: b128 lr1e-4 warmup10k cosine-to-800k, step 50k, FID TBD
```

Caveat: this is step-count aligned, not sample-budget aligned.

```text
old b96 50k sample presentations = 4.8M
new b128 50k sample presentations = 6.4M
sample-budget-equivalent b128 step for old 50k = 37.5k
```

The user explicitly requested the paper/reference hparams at 50k, so the active comparison is accepted as the primary decision point.  If strict sample-budget fairness becomes necessary later, add a 37.5k b128 anchor in a future run.

Also remember that this is still far below the PAE-paper-scale sample budget:

```text
PAE 80ep reference: 100k steps * batch1024 = 102.4M sample presentations
new b128 50k: 6.4M sample presentations = 6.25% of that budget
```

## Next actions

1. Monitor the active b128 run for OOM/instability; memory is high but fit probe and current run are stable.
2. Let the 50k controller finish the 5K true-Inception FID.
3. Compare the new 50k FID/MMD/KID against the old 50k anchor.
4. If b128 exact/reference 50k is better, adopt the reference-hparam route as the main branch and consider extending to 100k/200k.
5. If it is worse, treat it as a negative early signal but not final proof: warmup/cosine can be slower early.  Decide between continuing to 100k, using sample-budget-matched 37.5k, or testing the b96 batch-scaled fallback.
6. Start a distinct HF artifact watcher for this run only after deciding cleanup policy; do not let the old b96 watcher delete or mix artifacts for this reference branch.
