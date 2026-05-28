# PDM-3-HT Experiment 4 — Real LightningDiT Block / PAE Latent / Short-Train Stress

本实验接在 Experiment 3 之后，目标是从自写 proxy 推进到更接近真实工程路径的 stress test：

- 使用官方 PAE repo 携带的 LightningDiT 实现组件：`LightningDiTBlock`、SDPA attention、RMSNorm、SwiGLU。
- 对官方 block 做最小 wrapper：把 image-level scalar conditioning 改成 per-token `(r_i,t_i,Δ_i)` conditioning。
- 使用 PAE-f16d32 的 `C=32,H=W=32,N=1024` latent shape；加载 PAE 官方 latent stats。
- 保持 Experiment 3 已验证的 full-bundle JVP 公式。
- 增加短训级 stress：多步 AdamW，记录 loss/grad/memory/time。

注意：若没有可用的 ImageNet PAE latent shard，本实验会使用基于官方 `Latent-stats/PAE_DINOv2L.pt` 的 `pae_stats_synthetic` latent。它是 PAE-stat-conditioned stress input，不等于真实 ImageNet latent cache；脚本会在结果中显式记录。

## 实验完成摘要（2026-05-27）

最终汇总见：

- `results/AGGREGATE_SUMMARY.md`
- `results/aggregate_metrics.json`
- `notes/EXPERIMENT_RECORD.md`

结论：**PASS with engineering caveats**。

主路径 `fp32 + official PAE LightningDiTBlock submodules + per-token AdaLN wrapper + PAE-stat synthetic normalized latent + math SDPA + TF32 disabled + no checkpoint` 通过 N=1024 full-bundle JVP / central FD / r=t 退化 / 16-step 短训 gate。

关键指标：

- Main width=128/depth=2：FD relative error `6.71513e-05`，`r=t` target-v max abs `0`，短训 `16/16`，final loss `3.37412`，train peak `614.148 MB`。
- Scale width=256/depth=4：FD relative error `9.64663e-05`，短训 `8/8`，train peak `2352.25 MB`。
- bf16/autocast：forward/JVP finite，但 FD error `0.568673`，backward dtype mismatch；诊断失败。
- checkpoint+JVP：custom autograd JVP not implemented；诊断失败。
- default SDPA backend：`_scaled_dot_product_efficient_attention` forward AD not implemented；主路径强制 math SDPA。

注意：本实验仍没有真实 ImageNet PAE latent shard；使用的是官方 PAE DINOv2L latent mean/std 约束的 synthetic normalized latent。
