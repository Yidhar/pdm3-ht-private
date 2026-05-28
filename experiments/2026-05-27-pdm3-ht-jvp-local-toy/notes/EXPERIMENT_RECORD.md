# 实验记录：PDM-3-HT Experiment 2 — Local / Stop-Context JVP Toy
- 记录时间：2026-05-27T10:32:25Z
- 工作目录：`/workspace/PDM`
- 实验目录：`experiments/2026-05-27-pdm3-ht-jvp-local-toy/`
- 任务：比较 full-bundle JVP、diagonal/self-token JVP、local/stop-context JVP，验证 local tangent 是否能比 diagonal 更接近 full-bundle JVP。

## 执行命令

```bash
set -euo pipefail
EXP_ROOT=/workspace/PDM/experiments/2026-05-27-pdm3-ht-jvp-local-toy
python "$EXP_ROOT/scripts/run_jvp_local_toy.py" \
  --device auto \
  --dtype float32 \
  --n-tokens 4,16,64 \
  --radii 0,1,2,3 \
  2>&1 | tee "$EXP_ROOT/logs/run_jvp_local_toy_float32.log"
```

## 环境

- torch：`2.8.0+cu128`
- cuda_available：`True`
- device 参数：`auto`；实际结果 device：`cuda`
- dtype：`float32` / `torch.float32`
- seed：`20260527`
- fd_eps：`0.01`

## 结果文件

- `logs/run_jvp_local_toy_float32.log`
- `results/jvp_local_metrics.json`
- `results/jvp_local_summary.md`
- `notes/EXPERIMENT_RECORD.md`

## Gate 检查

- full-bundle JVP vs finite difference 最大相对误差：`8.21285e-05` → **PASS**（远低于 5%~10% 阈值）
- r=t degeneracy 最大 `target-v` 误差：`0` → **PASS**
- NaN/Inf：`False` → **PASS**
- radius=0 与 diagonal/self-token JVP 数值一致：**PASS**
- radius=1/2 相比 diagonal 至少在 N=16/64 上降低误差：**PASS**

## 核心指标表

| N | grid | radius | avg coverage | err vs full JVP | target err vs full | error ratio vs diag | error reduction vs diag |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 | 2x2 | 0 | 0.25 | 0.314482 | 0.0752639 | 1 | 3.17979e-12 |
| 4 | 2x2 | 1 | 1 | 0 | 0 | 0 | 1 |
| 4 | 2x2 | 2 | 1 | 0 | 0 | 0 | 1 |
| 4 | 2x2 | 3 | 1 | 0 | 0 | 0 | 1 |
| 16 | 4x4 | 0 | 0.0625 | 0.249729 | 0.0608364 | 1 | 4.00435e-12 |
| 16 | 4x4 | 1 | 0.390625 | 0.187066 | 0.0455712 | 0.749078 | 0.250922 |
| 16 | 4x4 | 2 | 0.765625 | 0.111123 | 0.0270708 | 0.444976 | 0.555024 |
| 16 | 4x4 | 3 | 1 | 0 | 0 | 0 | 1 |
| 64 | 8x8 | 0 | 0.015625 | 0.307697 | 0.0659489 | 1 | 3.24984e-12 |
| 64 | 8x8 | 1 | 0.118164 | 0.278672 | 0.0597281 | 0.905671 | 0.0943286 |
| 64 | 8x8 | 2 | 0.282227 | 0.231057 | 0.0495226 | 0.750924 | 0.249076 |
| 64 | 8x8 | 3 | 0.472656 | 0.176524 | 0.0378345 | 0.573694 | 0.426306 |

## 观察与解释

1. **full-bundle JVP 仍然可信**：finite-difference audit 最大相对误差只有约 `8.21e-05`，说明本实验中 full-bundle JVP 计算链路稳定。
2. **radius=0 验证了 diagonal baseline**：R=0 的 `err_vs_full_jvp` 与 `diag_err_vs_full_jvp` 完全对齐，确认 local 实现可作为 radius sweep 的可信基线。
3. **local window 能降低误差，但收益随 token 数变大而变弱**：
   - N=4 / 2x2：R=1 已覆盖全局，所以误差降为 0，不代表局部性优势。
   - N=16 / 4x4：R=1 覆盖 0.391、误差降低 0.251；R=2 覆盖 0.766、误差降低 0.555；R=3 覆盖 1、误差降低 1。
   - N=64 / 8x8：R=1 覆盖 0.118、误差降低 0.0943；R=2 覆盖 0.282、误差降低 0.249；R=3 覆盖 0.473、误差降低 0.426。
4. **cross-patch coupling 不完全局部**：N=64 时 R=1 只将 JVP 误差从 `0.3077` 降到 `0.2787`，误差降低约 `9.43%`；即使 R=3 覆盖约 `47.3%` token，仍有 `0.1765` 相对误差。
5. **target error 小于 du error，但趋势一致**：由于目标是 `u_tgt = v - du/dc`，target 相对误差随 radius 增大下降；N=64 从 R=0 的 `0.06595` 降到 R=3 的 `0.03783`。

## 结论

**实验执行 Gate：PASS。** 该 toy harness 成功验证了 local/stop-context JVP 的实现与对比指标。

**策略结论：local-window JVP 可作为诊断/弱 fallback，但不应替代 full-bundle JVP 主公式。** 原因是：在 N=64 时，较小 radius 对 full JVP 的近似改善有限，说明注意力耦合包含明显长程成分。若真实 DiT/PAE-HT 中出现 full JVP 不稳定，更优先的 fallback 应是 JVP-free consistency、SplitMeanFlow 或 FD-first staged loss，而不是把 local JVP 作为严格近似主线。

## 下一步建议

建议进入 **Experiment 3：JVP-free / FD-consistency toy experiment**，验证不用 JVP 的 finite-difference / consistency 目标是否能在 toy setting 中保持稳定，并与 full-JVP target 做误差和梯度稳定性对比。
