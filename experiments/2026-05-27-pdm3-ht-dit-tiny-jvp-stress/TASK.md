# 实验任务记录：PDM-3-HT Experiment 3 — LightningDiT-Tiny / PAE-shape Full-Bundle JVP Stress Test

- 日期：2026-05-27
- 工作目录：`/workspace/PDM`
- 实验目录：`experiments/2026-05-27-pdm3-ht-dit-tiny-jvp-stress/`
- 任务来源：在 Experiment 1/2 通过 toy JVP correctness/local fallback gate 后，进入更接近真实 PDM-3-HT 的架构与 shape 压测。

## 实验目标

验证 full-bundle Per-Patch MeanFlow JVP 在更接近真实设置的 proxy 上是否可行：

1. 使用 PAE-shape token：`C=32`，grid 至少覆盖 `16x16=256` 与 `32x32=1024`。
2. 使用 LightningDiT-Tiny-like proxy：self-attention + MLP + per-token AdaLN/time conditioning。
3. 测试 full-bundle JVP：
   ```python
   delta = t - r
   z_dot = delta[..., None] * v
   r_dot = 0
   t_dot = delta
   u, du_dc = jvp(model, (zt, r, t), (z_dot, r_dot, t_dot))
   u_tgt = v - du_dc
   ```
4. 对 full-bundle JVP 做 central finite-difference audit。
5. 做 training-like backward step：`loss = mse(u, sg(v - du_dc))`，确认能反传到参数，无 NaN/Inf。
6. 记录 wall time、peak memory、norm ratio、FD relative error、gradient norm。
7. 尝试 `bf16_autocast` 稳定性；若失败，记录为风险而不是掩盖。

## 与前两步实验的关系

- Experiment 1 已证明 tiny coupled attention 中 full-bundle JVP 与 finite difference 一致，diagonal 漏掉显著 cross-token coupling。
- Experiment 2 已证明 local/stop-context JVP 在 toy 中只能作为诊断/弱 fallback，不应替代 full-bundle 主公式。
- 本实验开始回答更关键的 go/no-go：full-bundle JVP 在 PAE-like token shape 与 DiT-like block 上是否数值正确、可训练、显存可承受。

## 目录规范

```text
experiments/2026-05-27-pdm3-ht-dit-tiny-jvp-stress/
  TASK.md
  README.md
  scripts/
  results/
  logs/
  notes/
```

## 初始验收标准

- fp32 full-bundle JVP vs central finite difference relative error：优先目标 `< 1e-2`，最低接受 `< 5e-2`。
- `r=t` degeneracy：`target-v` max abs 接近 0。
- N=256 与 N=1024 的 full-bundle JVP 均能完成。
- training-like backward step 能完成，gradient finite，无 NaN/Inf。
- 记录 fp32 与 bf16/autocast 的差异；bf16/autocast 若失败，标记为 mixed-precision 风险。
- 记录 peak memory，用于判断后续放大 depth/width 的可行性。

## 完成记录（2026-05-27）

- 状态：**完成 / PASS with mixed-precision caveat**
- 主结果时间：2026-05-27T10:58:58Z
- 主结果：`results/dit_tiny_jvp_stress_metrics.json`、`results/dit_tiny_jvp_stress_summary.md`
- 综合摘要：`results/experiment3_combined_summary.md`
- 实验记录：`notes/EXPERIMENT_RECORD.md`
- 主日志：`logs/run_dit_tiny_jvp_stress_w128d2_fp32_bf16.log`
- 追加日志：`logs/run_scale_w256d4_n1024_fp32.log`、`logs/run_mf_mixture_eq075_n1024_fp32.log`

### Gate

- 主 fp32 gate：PASS
- width=128/depth=2，N=256/1024，gaussian/correlated latent：全部完成 full-bundle JVP、FD audit、train-step backward。
- 主 fp32 最大 FD relative error：`6.90423e-05`。
- 所有 fp32 最大 FD relative error：`7.10713e-05`。
- 所有 fp32 r=t target-v 最大误差：`0`。
- N=1024 scale-up width=256/depth=4：通过，FD relative error `6.27395e-05`，train peak memory `1181.450 MB`。
- MeanFlow 75% r=t mixture：通过，FD relative error `7.10713e-05`，du/v `0.111454`。
- bf16_autocast：无 NaN/Inf 且 grad finite，但 FD relative error 约 `0.408~0.436`，标记为 mixed-precision correctness 风险；后续应保持 JVP path fp32。

### 结论

full-bundle JVP 在当前 DiT-Tiny/PAE-shape proxy 上不是 go/no-go 阻塞点。建议继续推进到更真实 LightningDiT block / PAE latent cache / 多步短训，同时保持 local JVP 和 JVP-free/FD-consistency 作为对照 fallback。

