# TASK — Phase 0 / B3 MeanFlow Real-Data Training Loop

- Date: 2026-05-27
- Working dir: `/workspace/PDM`
- Experiment dir: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer`

## Objective

并行于 ImageNet-1k PAE latent full-cache 构建，完成 B3 MeanFlow training loop 的真实数据版本：

1. DataLoader 可扩展读取 1.28M PAE latent safetensors shards。
2. class-conditional label 注入到 LightningDiT / MeanFlow model kwargs。
3. 训练 recipe 固定为 Phase-0/B3 当前决策：bf16 backbone train forward/backward + fp32 live-param JVP target。
4. r=t sampler 使用 MeanFlow 标准 `equal_prob=0.75`。
5. 周期性 FD audit、memory logging、checkpoint 保存/恢复。
6. 周期性 eval hook：采样 latent + 可选外部 FID command；smoke 阶段允许 dry-run/skip FID。
7. 先用固定 4096 real PAE latent cache 做 smoke，不等待 full cache 完成。

## Non-goals for this step

- 不在本步骤跑完整 ImageNet-1k/FID 主训。
- 不阻塞正在进行的 full PAE cache。
- 不要求已有 PAE decoder/FID pipeline ready；但 training loop 必须暴露 sampling + FID hook。

## Current upstream decisions carried forward

- JVP target 使用 live 参数，不使用 EMA。
- Target path 使用 fp32/no-autocast；train backbone 使用 bf16 autocast。
- r=t samples 理论退化到 target=v；FD audit 需要避免全 r=t 掩盖 JVP edge case。
- `force_drop_ids` 每个 batch 只采样一次，并同时传给 JVP target 和 train forward，避免 classifier-free dropout mismatch。

<!-- B3_LONGRUN_SCOPE_UPDATE_20260528 -->

## Scope update — 2026-05-28 long-run baseline

Current route remains the base MeanFlow path only:

```text
ImageNet-1k ADM-cropped 256
 -> PAE DINOv2-L d32 latent cache
 -> LightningDiT B3/XL-like backbone
 -> MeanFlow objective
 -> EMA sampling
 -> PAE decode
 -> image-space Inception FID/MMD/KID diagnostics
```

Representation Fréchet Loss / FD-loss is intentionally deferred. The immediate goal is to see how far the base PAE + LightningDiT + MeanFlow route converges before adding FD-loss.

Long-run comparison target:

- First PAE-paper-comparable regime: approximately `1.07M` training steps.
- Intermediate steps (`20k`, `30k`, `100k`) are for smoke/eval pipeline validation, sample sanity, and trend monitoring.
- Local safety checkpoints may be written every `10k` steps, but HF long-term full-checkpoint archives are sparse (`100k` multiples plus final `1.07M`) and stale local `.pt` files are cleaned to control disk pressure.


<!-- B3_STEP50000_5K_FID_AND_COSINE_STATUS_20260528 -->

## Status update — step-50k 5K FID anchor and later cosine plan

- Exact `step_00050000` PAE B3 b96 checkpoint evaluated with `5000` EMA generated samples vs `5000` real ImageNet-256 references.
- True Inception result: **FID `55.528315`**, RBF MMD `0.0328459`, poly3 KID `0.0389490`.
- Treat the normal `64`-sample trainer FID as smoke only; do not use it for convergence decisions.
- Main route remains: PAE DINOv2-L d32 latent cache + LightningDiT B3/XL-like backbone + MeanFlow objective + EMA sampling + PAE decode + image-space metrics.
- Representation Fréchet Loss / FD-loss remains deferred.
- Training resumed from `latest.pt -> step_00052000.pt` after the 5K GPU eval.
- HF model artifacts refreshed under `LAXMAYDAY/pdm3-ht-model-artifacts/b3_meanflow_realdata/fullcache_b96/step_00050000`, latest metric commit `019298e6dee68c5b2963015e60eb5b5a8210e194`.
- The 5K controller was fixed to use exact `/proc/<pid>/cmdline` argv matching for trainer PGIDs, avoiding self-termination from substring matching.

Later LR plan:

- Current live LR is still constant `2e-4`; no hot LR change has been applied.
- Batch-scaled MeanFlow cosine plan for batch `96`: `base_lr=7.5e-5`, `min_lr=7.5e-6`, `warmup_steps=13333`, `end_step=1070000`.
- If applying to the current run after a checkpoint, do not rewarm; override optimizer param-group LR after checkpoint load and optionally ramp from `2e-4` to the cosine target over `2k–5k` steps.
- Trigger for early switch: two consecutive 5K anchors plateau/worsen, or instability after ruling out sampling/eval noise. Otherwise keep the current healthy run stable and consider switching at a clean milestone such as `100k` or in a new branch.

<!-- B3_STEP100000_5K_FID_AND_150K_GATE_20260529 -->

## Status update — step-100k 5K FID and 150k gate

- Step `100000` exact 5K true Inception anchor completed: **FID `47.925748666494485`**, MMD `0.02595176471424887`, KID `0.02984040431452506`.
- Step `50000` exact 5K anchor was **FID `55.52831543442829`**, so 50k → 100k improved by `-7.6025667679338085` FID (`-13.69%`).
- Interpretation is yellow-flag: improving, not collapsed, but absolute FID and slope are not yet reassuring.
- Do not directly compare current `100k @ batch96` against PAE paper `100k @ batch1024` / 80ep. Current 100k has only `96/1024 = 9.375%` of that sample budget, roughly `7.5` PAE-paper-equivalent epochs.
- A step `150000` 5K GPU FID anchor controller is running and waiting for `step_00150000.pt`.

```text
controller PID: 74002
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/b3_150k_5k_gpu_fid_then_resume_20260529T060838Z.log
status: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00150000/5k_anchor_control.json
```

Next gate:

- If step-150k FID improves by several points, continue the current constant-`2e-4` run as the main control.
- If step-150k is flat/worse, branch from step-100k/150k and test lower/cosine LR and sampler sensitivity before declaring the base route failed.
