# 实验记录：Full-Bundle JVP Toy Harness

## 1. 任务

验证 PDM-3-HT Per-Patch MeanFlow 的第一步核心实现：

```python
delta = t_vec - r_vec
z_dot = delta[..., None] * v
r_dot = zeros_like(r_vec)
t_dot = delta
u_tgt = v - du_dc
```

其中 `du_dc` 由 `torch.func.jvp` 对整个 `(z_t, r_vec, t_vec)` 进行一次 full-bundle JVP 得到。

## 2. 文件目录

```text
experiments/2026-05-27-pdm3-ht-jvp-toy/
  TASK.md
  README.md
  scripts/run_jvp_toy.py
  logs/run_jvp_toy_float32.log
  results/jvp_toy_metrics.json
  results/jvp_toy_summary.md
  notes/EXPERIMENT_RECORD.md
```

## 3. 运行命令

```bash
cd /workspace/PDM
python experiments/2026-05-27-pdm3-ht-jvp-toy/scripts/run_jvp_toy.py \
  --device auto \
  --dtype float32 \
  --n-tokens 4,16,64 \
  2>&1 | tee experiments/2026-05-27-pdm3-ht-jvp-toy/logs/run_jvp_toy_float32.log
```

## 4. 实验配置

- Torch：2.8.0+cu128
- CUDA：可用
- device：cuda
- dtype：float32
- seed：20260527
- batch size：2
- channels：8
- tiny attention width：32
- heads：4
- depth：2
- token 数：4, 16, 64

## 5. 结果摘要

| N tokens | best finite-difference rel err | cross_ratio full-vs-diag JVP | target cross_ratio | r=t target-v max abs | NaN/Inf |
|---:|---:|---:|---:|---:|---|
| 4 | 3.73859e-05 | 0.314482 | 0.0752639 | 0 | False |
| 16 | 7.96652e-05 | 0.249729 | 0.0608364 | 0 | False |
| 64 | 8.21285e-05 | 0.307697 | 0.0659489 | 0 | False |

## 6. 结论

本实验 **通过第一步 gate**：

1. full-bundle JVP 与 central finite difference 高度一致，最大 best relative error 为 `8.21285e-05`，远低于 5%~10% 阈值。
2. `r=t` 时 `delta=0`，`du_dc=0`，target 精确退化为 Flow Matching target `v`。
3. full-bundle JVP 无 NaN/Inf。
4. full-bundle 与 diagonal approximation 的 JVP 差异明显：cross_ratio 约 `0.25~0.31`，说明 attention coupling 下 diagonal per-patch 近似确实漏掉了不可忽略的 cross-patch 项。

## 7. 下一步

建议下一步不是直接上大模型，而是：

1. 在当前 toy harness 中增加 `local/stop-context JVP` 模式。
2. 把同样的 audit 接入 DiT-Tiny / LightningDiT-Tiny 的 per-token AdaLN proxy。
3. 在 PAE latent shape proxy 上测试 JVP 的显存、速度、NaN、cross_ratio 分布。

