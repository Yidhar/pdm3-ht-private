# Experiment 6 — Phase 0 / B3 MeanFlow Mixed-Precision Integration Smoke

本实验把 Experiment 5 的结论从 synthetic per-patch stress 推进到 Phase 0 / B3 MeanFlow baseline 的 sample-level scalar `(r,t)` 训练语义：

- B3 标准 MeanFlow sampler：按 **sample/image-level** 采样 `t` 与 `r`，默认 75% 样本 `r=t`、25% 样本 `r<t`；不是 per-token / per-patch mixture。
- JVP target path：`torch.func.jvp` 在 fp32/no-autocast 下运行，target `v-du` detach。
- 训练 backbone path：普通 forward/backward 使用 bf16 autocast。
- JVP 参数源：live model params；EMA 仅可存在于训练循环中用于真实 baseline 评估/采样，不参与 target JVP。
- Kernel/精度：math SDPA，FD audit 禁用 TF32；JVP path 不包 checkpoint。

## 文件

- `scripts/run_b3_meanflow_mixed_precision_smoke.py`：B3 mixed-precision smoke 脚本。
- `logs/`：运行日志，使用 `tee` 保留。
- `results/`：JSON metrics 与 Markdown summary。
- `notes/EXPERIMENT_RECORD.md`：实验过程与结果记录。

## 范围说明

当前 PAE 官方 training stack 的 `transport.training_losses` 是标准 FM/velocity loss，仓库内没有现成 B3 MeanFlow baseline。因此本目录实现的是 **Phase 0 / B3 MeanFlow integration smoke scaffold**：它使用官方 PAE LightningDiT 子模块与 PAE latent stats synthetic 输入来验证 mixed precision、JVP target、采样、FD 与显存 telemetry 是否能在 B3 sample-level 语义下跑通。

不声明：完整 ImageNet baseline、FID/FDr、XL 规模长训稳定性。


## 主 smoke 结果（2026-05-27）

运行目录：`results/smoke_b3_mixed_w128d2_g16_b4/`

命令日志：`logs/smoke_b3_mixed_w128d2_g16_b4.log`

结果文件：

- `results/smoke_b3_mixed_w128d2_g16_b4/b3_meanflow_mixed_precision_smoke_metrics.json`
- `results/smoke_b3_mixed_w128d2_g16_b4/b3_meanflow_mixed_precision_smoke_summary.md`

关键指标：

| 指标 | 值 |
|---|---:|
| gate | `PASS` |
| FD rel err full JVP | `9.76049585019447e-05` |
| bf16-vs-fp32 ordinary forward rel | `0.004396095609430949` |
| all-r=t target-v max abs | `0.0` |
| actual r=t target-v max abs | `0.0` |
| short train steps | `8/8` |
| final loss | `3.9109983444213867` |
| mean realized r=t fraction | `0.71875` |
| JVP peak memory | `43.8955 MB` |
| FD peak memory | `29.4805 MB` |
| bf16 forward audit peak memory | `30.6465 MB` |
| train loop peak memory | `65.5625 MB` |

结论：Experiment 5 的 `bf16 backbone + fp32 JVP target` 配方已在 Phase 0 / B3 sample-level MeanFlow smoke scaffold 中跑通。JVP target 使用 live 参数，不使用 EMA；EMA object 在短训中存在并在 optimizer step 后更新，但不参与 target 构造。
