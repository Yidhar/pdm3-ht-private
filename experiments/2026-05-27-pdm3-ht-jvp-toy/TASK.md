# 实验任务记录：PDM-3-HT Full-Bundle JVP Toy Harness

- 日期：2026-05-27
- 任务来源：PDM-3-HT v2.1 研究报告第一步实验
- 实验目标：验证 Per-Patch MeanFlow full-bundle JVP 公式与实现是否正确。
- 核心验证：
  1. `z_dot = (t-r) * v`
  2. `t_dot = t-r`
  3. `r_dot = 0`
  4. `u_tgt = v - du_dc`
  5. full-bundle JVP 与 finite difference 一致
  6. `r=t` 时退化为 Flow Matching target，即 `u_tgt = v`
  7. full-bundle JVP 与 diagonal approximation 的差距被量化

## 目录规范

```text
experiments/2026-05-27-pdm3-ht-jvp-toy/
  TASK.md                 # 本任务记录
  README.md               # 实验说明、运行方式、结论摘要
  scripts/                # 可复现实验代码
  results/                # JSON/Markdown 结果
  logs/                   # stdout/stderr 日志
  notes/                  # 额外分析记录
```

## 验收标准

- finite-difference relative error：目标 < 5%~10%
- full-bundle JVP 无 NaN/Inf
- `r=t` degeneracy error 约为 0
- 输出 cross-ratio，量化 diagonal approximation 漏掉的 cross-patch coupling


---

## 实验完成记录

- 完成时间：2026-05-27T10:25:34Z
- 脚本：`scripts/run_jvp_toy.py`
- 日志：`logs/run_jvp_toy_float32.log`
- 原始指标：`results/jvp_toy_metrics.json`
- 结果摘要：`results/jvp_toy_summary.md`
- 实验记录：`notes/EXPERIMENT_RECORD.md`

### Gate 结果

PASS。

- 最大 best finite-difference relative error：`8.21285e-05`
- `r=t` target-v 最大误差：`0`
- NaN/Inf：`False`
- diagonal approximation cross_ratio：约 `0.25~0.31`

结论：full-bundle JVP 实现通过 toy 验证；diagonal per-patch approximation 漏掉显著 cross-patch coupling，不应作为严格主公式。
