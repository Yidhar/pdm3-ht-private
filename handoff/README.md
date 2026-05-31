# PDM-3-HT handoff index

<!-- B3_FITPROBE_100M_SUBSET64K_20K_POINTER_20260531 -->

## Latest lower-bound probe — B3 100M subset64k fitprobe step-20k

更新时间：`2026-05-31T03:28:54Z`

Detailed handoff:

```text
handoff/B3_FITPROBE_100M_SUBSET64K_20K_2026-05-31.md
```

Result summary: `fitprobe_100m_subset64k_b384_lr5e4` completed to step `20,000` on the first `65,536` PAE/ImageNet-256 latents with a ~`101M` B3 MeanFlow model. True image-space Inception FID used PAE decode, `sample_steps/NFE=32`, subset-real reference, and latent-subset label marginal. Step-20k 5K FID: EMA `58.643213`, raw `61.628071`. Step-10k -> 20k EMA improved only `63.142481 -> 58.643213`; the route learns but plateaus far above the PAE reconstruction lower-bound `1.995241`.

HF artifact prefix:

```text
repo: https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
path: b3_meanflow_realdata/fitprobe_100m_subset64k_b384_lr5e4
commit: ad334901017f194d2bf2b6898d9a1c705db7fd39
```

Decision: do not continue this exact `100M / first-64k / b384 / lr5e-4 / equal_prob=0.75` recipe as the lower-bound route. Next probe should be easier/smaller, e.g. first-8k or class-balanced 5k–10k, with matched subset-real FID, `equal_prob=0.5`, and `lr=2e-4/3e-4`.


<!-- PAE_RECON_FID_LOWER_BOUND_POINTER_20260530 -->

## Latest PAE reconstruction FID lower-bound — 5K matched real-vs-reconstruction

更新时间：`2026-05-30T16:08:58Z`

Detailed handoff:

```text
handoff/PAE_RECONSTRUCTION_FID_LOWER_BOUND_5K_2026-05-30.md
```

Result: matched `real cropped ImageNet-256` vs `PAE decode(cached latent)` with 5K/5K true torchvision InceptionV3 ImageNet1K pool-2048 gives **FID `1.9952405483597886`**.  Integrity check: labels matched for all `5000` sampled global indices.  Interpretation: PAE cache/decoder is not the current B3 FID bottleneck; current generated FID `~48` is mainly model/training recipe gap, not PAE reconstruction ceiling.  HF artifact prefix: `b3_meanflow_realdata/pae_reconstruction_fid_lower_bound_5k_seed20260529`; commit `482dc1dc3d66828768716fb54fd4881c324f9648`.


更新时间：`2026-05-29T07:32:07Z`

## 当前有效状态

PAE full latent cache 已完成并公开上传；B3 主线已从旧 b96 constant-2e-4 control 切换到 MeanFlow reference-hparam b128/lr1e-4/warmup+cosine，并且比较口径已修正为 processed-samples / equivalent-epoch。当前 primary anchor 是 b128 step 37.5k（等效 old b96 step 50k），b128 step 50k 仅作为 b96 66.7k-equivalent secondary reference。

优先看：

```text
handoff/B3_MFREF_B128_EQEPOCH_37K5_50K_CONTROL_2026-05-29.md
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
route: MeanFlow reference-hparam b128/lr1e-4 warmup+cosine, equivalent-epoch anchors
pid: 78353
primary config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5.yaml
resume-to-50k config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_eqepoch_resume_to50k.yaml
result: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_37k5_train_setsid_20260529T072459Z.log
primary 37.5k 5K-FID controller pid: 78122
secondary 50k 5K-FID controller pid: 78123
post-50k HF upload controller pid: 78124
```

Primary fair comparison is now:

```text
old b96 step 50k: 50,000 × 96 = 4.8M samples = 3.746584 epochs, FID 55.528315
new b128 step 37.5k: 37,500 × 128 = 4.8M samples = 3.746584 epochs, FID TBD
```

Secondary reference:

```text
new b128 step 50k = 6.4M samples = 4.995446 epochs = old b96 step 66,667-equivalent
```

监控：

```bash
cd /workspace/PDM
EXP=experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
PID=$(cat "$EXP/mfref_b128_eqepoch.pid")
ps -p "$PID" -o pid,ppid,stat,etime,%cpu,%mem,rss,cmd
nvidia-smi
LOG=$(cat "$EXP/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_latest.logpath")
tail -f "$LOG"
RES="$EXP/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k"
cat "$RES/eval/step_00037500/5k_anchor_control.json"
cat "$RES/eval/step_00050000/5k_anchor_control.json"
cat "$RES/eval/step_00050000/hf_post50k_upload_status.json"
```

## 状态 JSON

```text
handoff/state/pae_full_cache_verified_after_hf_upload.json
handoff/state/pae_public_cache_hf_remote_verified.json
handoff/state/b3_fullcache_latest_launch.json
handoff/state/b3_step50000_5k_fid_status_20260528T1838Z.json
handoff/state/b3_step100k_multi_nfe_fid_status_20260529T0702Z.json
handoff/state/b3_mfref_b128_eqepoch_37k5_50k_status_20260529T0730Z.json
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


<!-- B3_MFREF_B128_EQEPOCH_POINTER_20260529 -->

## Latest B3 update — MeanFlow b128/lr1e-4 cosine equivalent-epoch control

更新时间：`2026-05-29T07:32:07Z`

Detailed handoff:

```text
handoff/B3_MFREF_B128_EQEPOCH_37K5_50K_CONTROL_2026-05-29.md
```

Decision: raw step-count comparison was corrected.  The primary fair comparison is old b96 step `50k` (`4.8M` samples, FID `55.528315`) vs new b128 step **`37.5k`** (`4.8M` samples).  New b128 step `50k` remains useful, but it is **old b96 step `66.7k` equivalent**, not old b96 50k equivalent.

Active route:

```text
result: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k
train PID: 78353
primary 37.5k controller PID: 78122
secondary 50k controller PID: 78123
HF remote prefix: b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k
```

<!-- B3_MFREF_B128_50K_POINTER_20260529 -->

## Historical/superseded B3 update — MeanFlow reference-hparam b128/lr1e-4 cosine raw-step 50k control

更新时间：`2026-05-29T06:30Z`

Detailed handoff:

```text
handoff/B3_MFREF_B128_LR1E4_COSINE_50K_CONTROL_2026-05-29.md
```

Superseded note: this raw-step 50k branch was cancelled after the equivalent-epoch correction.  See `handoff/B3_MFREF_B128_EQEPOCH_37K5_50K_CONTROL_2026-05-29.md` for the active route.

Decision: the old b96 constant-`2e-4` run is now an engineering/control baseline, not the primary fair evaluation route, because it is not aligned with the external MeanFlow reference hparams. It was stopped at step `103354`; the old step-150k anchor was cancelled.

Historical raw-step route (no longer active):

```text
config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_50k.yaml
result: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_50k
train PID: 77429
controller PID: 75011
train log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_lr1e4_cosine_50k_restart_after_multi_nfe_20260529T070105Z.log
50k controller log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_50k_5k_gpu_fid_final_20260529T062518Z.log
```

Original raw-step comparison target is superseded.  Do not use b128 raw step-50k as the primary old-b96-50k comparison; use b128 step 37.5k from the eqepoch branch instead.  The old raw-step controllers were cancelled/marked superseded.
