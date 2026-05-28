# TASK — Experiment 5: Mixed Precision Engineering / Synthetic PAE Latent JVP

- 日期：2026-05-27
- 任务书：`/workspace/PDM/2026-05-27-pdm3-ht-research-brief-v2.1.md`
- 基线：Experiment 4 的 `pae_stats_synthetic` + 官方 PAE LightningDiTBlock 子模块 + per-token AdaLN wrapper。

## 用户建议 / 本轮目标

先做 mixed precision 工程：在当前 synthetic latent 框架里跑一次 Phase-3 主训实际精度配方：

```text
bf16/autocast backbone forward/backward + fp32 JVP/finite-difference target path
```

目标产出：

1. `N=1024, C=32` synthetic PAE latent stress 下的 fp32 JVP path FD relative error。
2. mixed train step 的 peak memory 数字。
3. 与纯 fp32 Experiment 4 main run 的显存/正确性对照。
4. 判定 Phase 0 MeanFlow baseline (B3) 应采用的精度路径。

## Gate / 记录规则

- JVP/FD correctness path 必须强制 `fp32 + math SDPA + TF32 disabled`。
- Backbone train path 使用 `torch.autocast(cuda, dtype=torch.bfloat16)`，loss 以 `.float()` MSE 计算。
- `du` / `target = v - du` 不参与反传，必须 detach；训练梯度只穿过 bf16 backbone forward 的 `u`。
- 记录：FD rel err、du/v、r=t degeneracy、forward/JVP/train peak memory、loss/grad finite、错误消息。
- 所有日志通过 `tee` 写入 `logs/`；指标写入 `results/`；总结写入 `notes/` 与 `results/`。

## 状态

- [x] 建立目录结构
- [x] 实现 mixed precision 模式
- [x] py_compile / smoke run
- [x] 正式 mixed precision run
- [x] 对照 fp32 baseline 并汇总
- [x] 清理缓存
