# 实验任务记录：PDM-3-HT Local / Stop-Context JVP Toy Experiment

- 日期：2026-05-27
- 任务来源：第一步 Full-Bundle JVP Toy Harness 通过后的下一实验
- 实验目标：在 toy coupled attention model 中比较 full-bundle JVP、diagonal JVP、local/stop-context JVP 的差距，判断 local approximation 是否能保留大部分 cross-patch coupling。

## 背景

第一步实验结论：

- full-bundle JVP 与 finite difference 一致，gate PASS。
- diagonal approximation 与 full-bundle JVP 的 cross_ratio 约 0.25~0.31，说明 diagonal 漏掉显著 cross-patch coupling。

本实验继续验证：

> 如果只保留局部邻域 token 的 tangent，即 local / stop-context JVP，是否比 diagonal 更接近 full-bundle JVP？

## 待验证问题

1. radius=0 的 local JVP 是否等价于 diagonal JVP？
2. radius=1/2 的 local-window JVP 相比 diagonal 是否显著降低 full-JVP approximation error？
3. local JVP 能捕获 full-bundle JVP 的多少比例？
4. 对 N=4/16/64 token，局部窗口覆盖率与 approximation error 的关系如何？

## 目录规范

```text
experiments/2026-05-27-pdm3-ht-jvp-local-toy/
  TASK.md
  README.md
  scripts/
  results/
  logs/
  notes/
```

## 验收标准

- full-bundle JVP 与 finite difference 仍然一致，relative error < 5%~10%。
- radius=0 与 diagonal 的误差接近 0。
- radius=1/2 相比 diagonal 至少在部分 token scale 上降低 full-JVP approximation error。
- 记录覆盖率、误差、capture ratio 与结论。

## 完成记录（2026-05-27）

- 状态：**完成 / PASS**
- 运行时间记录：2026-05-27T10:32:25Z
- 执行日志：`logs/run_jvp_local_toy_float32.log`
- 结果摘要：`results/jvp_local_summary.md`
- 完整指标：`results/jvp_local_metrics.json`
- 实验记录：`notes/EXPERIMENT_RECORD.md`

### Gate

- full-bundle JVP vs finite difference 最大相对误差：`8.21285e-05` → PASS
- r=t degeneracy 最大 target-v 误差：`0` → PASS
- NaN/Inf：`False` → PASS
- radius=0 对齐 diagonal baseline → PASS
- radius=1/2 在 N=16/64 上相对 diagonal 降低误差 → PASS

### 结论

local/stop-context JVP 能随 radius 增大逐步接近 full-bundle JVP，但在 N=64 时小半径收益有限：R=1 误差降低约 `9.43%`，R=2 约 `24.91%`，R=3 约 `42.63%`。因此它适合作为诊断/弱 fallback，不适合作为替代 full-bundle JVP 的主公式。下一步建议做 JVP-free / FD-consistency toy experiment。

