# PDM-3-HT Local / Stop-Context JVP Toy Experiment

本实验在 tiny coupled self-attention model 上比较：

1. full-bundle JVP：所有 token tangent 同时参与。
2. diagonal JVP / radius=0：每个输出 token 只保留自身输入 token tangent。
3. local-window JVP：每个输出 token 只保留空间邻域内 token tangent，例如 Chebyshev radius=1/2。

## 运行

```bash
cd /workspace/PDM
python experiments/2026-05-27-pdm3-ht-jvp-local-toy/scripts/run_jvp_local_toy.py
```

结果输出：

```text
experiments/2026-05-27-pdm3-ht-jvp-local-toy/results/
```
