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

