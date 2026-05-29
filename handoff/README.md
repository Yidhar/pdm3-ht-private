# PDM-3-HT handoff index

更新时间：`2026-05-29T07:05:00+00:00`

## 当前有效状态

PAE full latent cache 已完成并公开上传；B3 主线已从旧 b96 constant-2e-4 control 切换到 MeanFlow reference-hparam b128/lr1e-4/warmup+cosine 50k 横向对比。

优先看：

```text
HANDOFF_2026-05-28_PAE_PUBLIC_CACHE_READY_AND_B3_STARTED.md
```

旧 handoff 仍保留作为历史上下文：

```text
HANDOFF_2026-05-28_PDM3_HT_PRIVATE_HF_AND_PAE_PARALLEL.md
```

注意：旧 handoff 中 PAE 为 partial 的描述已经过期。

## 关键链接

- Private code repo: https://huggingface.co/LAXMAYDAY/pdm3-ht-20260528-code
- Public PAE cache repo: https://huggingface.co/datasets/LAXMAYDAY/pdm-pae-dinov2l-d32-imagenet256-train-full

## 当前 B3 run

```text
pid: 77429
config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_50k.yaml
result: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_50k
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_lr1e4_cosine_50k_restart_after_multi_nfe_20260529T070105Z.log
50k 5K controller pid: 75011
```

监控：

```bash
cd /workspace/PDM
EXP=experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
PID=$(cat "$EXP/mfref_b128_lr1e4_cosine_50k.pid")
ps -p "$PID" -o pid,ppid,stat,etime,%cpu,%mem,rss,cmd
nvidia-smi
LOG=$(cat "$EXP/results/mfref_b128_lr1e4_cosine_50k_latest.logpath")
tail -f "$LOG"
cat "$EXP/results/fullcache_realdata_mfref_b128_lr1e4_cosine_50k/eval/step_00050000/5k_anchor_control.json"
```

## 状态 JSON

```text
handoff/state/pae_full_cache_verified_after_hf_upload.json
handoff/state/pae_public_cache_hf_remote_verified.json
handoff/state/b3_fullcache_latest_launch.json
```

<!-- B3_H100_SPEEDUP_LATEST -->

## Latest B3 H100 speedup status

- Active config: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_full_fast_h100.yaml`
- Active PID/log: see `handoff/state/b3_h100_speedup_latest.json`
- Current batch: 32; global SDPA default with JVP/FD math-context patch; TF32 enabled.
- Current throughput: ~57.8 samples/sec vs baseline ~10.6 samples/sec (~5.47x).

<!-- B3_STEP50000_5K_FID_POINTER_20260528 -->

## Latest B3 PAE b96 update — step-50k 5K FID anchor

更新时间：`2026-05-28T18:38Z`

Detailed handoff:

```text
handoff/B3_STEP50000_5K_FID_AND_COSINE_PLAN_2026-05-28.md
```

Key result: exact `step_00050000` checkpoint, `5000` generated / `5000` real true Inception pass, **FID `55.528315`**, RBF MMD `0.0328459`, poly3 KID `0.0389490`. Normal `64`-sample FID is smoke only. Training resumed from `latest.pt -> step_00052000.pt`; step-50k artifacts were uploaded to HF commit `019298e6dee68c5b2963015e60eb5b5a8210e194` under `LAXMAYDAY/pdm3-ht-model-artifacts/b3_meanflow_realdata/fullcache_b96/step_00050000`.

<!-- B3_STEP100000_5K_FID_POINTER_20260529 -->

## Latest B3 PAE b96 update — step-100k 5K FID and 150k anchor

更新时间：`2026-05-29T06:15Z`

Detailed handoff:

```text
handoff/B3_STEP100000_5K_FID_AND_150K_ANCHOR_2026-05-29.md
```

Key result: step `100000` exact 5K true Inception anchor completed with **FID `47.925749`**, RBF MMD `0.0259518`, poly3 KID `0.0298404`. Step-50k → step-100k improved by `-7.6026` FID (`-13.69%`). This is a yellow flag rather than a green light: it improves, but the absolute value and slope are not enough to declare the route solved.

Important normalization: current `100k @ batch96` is only `9.375%` of the PAE paper's `100k @ batch1024` / 80-epoch sample budget, roughly `7.5` PAE-paper-equivalent epochs. The first sample-budget-comparable point is still around `1.07M` current steps.

Historical action: a step `150000` 5K GPU FID anchor controller had been launched, but it was cancelled after the user requested the reference-hparam 50k control branch.

```text
pid: 74230
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/b3_150k_5k_gpu_fid_then_resume_20260529T061233Z_setsid.log
status json: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00150000/5k_anchor_control.json
final status: cancelled_by_user_for_hparam_control
```


<!-- B3_STEP100K_MULTI_NFE_FID_POINTER_20260529 -->

## Latest B3 PAE b96 control update — step-100k multi-NFE 5K FID sweep

更新时间：`2026-05-29T07:05Z`

Detailed handoff:

```text
handoff/B3_STEP100K_MULTI_NFE_FID_2026-05-29.md
```

User requested NFE `2/4` on the old step-100k checkpoint. Because the existing local 5K anchor metadata was `sample_steps=32` rather than true 1-NFE, the sweep ran `sample_steps/NFE = 1, 2, 4` with the same seed `2026052910`.

| sample_steps/NFE | 5K FID ↓ | MMD ↓ | KID ↓ |
|---:|---:|---:|---:|
| `1` | `56.966056` | `0.0344998` | `0.0427200` |
| `2` | `53.476784` | `0.0309196` | `0.0374734` |
| `4` | `49.938747` | `0.0277017` | `0.0323653` |
| `32` existing anchor | `47.925749` | `0.0259518` | `0.0298404` |

Interpretation: NFE helps monotonically; NFE4 is close to but still ~`+2.01` FID worse than the previous 32-step anchor at this 5K diagnostic sample count. Do not label the previous `47.9257` anchor as 1-NFE.

Operational note: the b128 reference-hparam run was temporarily stopped at step `1678` to free GPU for this sweep, then restarted from scratch at `2026-05-29T07:01Z` as PID `77429`; 50k controller PID `75011` remains active.

<!-- B3_MFREF_B128_50K_POINTER_20260529 -->

## Latest B3 update — MeanFlow reference-hparam b128/lr1e-4 cosine 50k control

更新时间：`2026-05-29T06:30Z`

Detailed handoff:

```text
handoff/B3_MFREF_B128_LR1E4_COSINE_50K_CONTROL_2026-05-29.md
```

Decision: the old b96 constant-`2e-4` run is now an engineering/control baseline, not the primary fair evaluation route, because it is not aligned with the external MeanFlow reference hparams. It was stopped at step `103354`; the old step-150k anchor was cancelled.

Active route:

```text
config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_50k.yaml
result: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_50k
train PID: 77429
controller PID: 75011
train log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_lr1e4_cosine_50k_restart_after_multi_nfe_20260529T070105Z.log
50k controller log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_50k_5k_gpu_fid_final_20260529T062518Z.log
```

Reference/control comparison target: old b96 constant-`2e-4` step-50k 5K FID `55.528315` vs new b128 lr`1e-4` warmup10k cosine-to-800k step-50k 5K FID TBD.

Post-50k HF upload/cleanup controller PID `75434` is waiting for the new branch 5K Inception metrics, then uploads to `b3_meanflow_realdata/mfref_b128_lr1e4_cosine_50k`.
