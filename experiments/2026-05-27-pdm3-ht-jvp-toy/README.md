# PDM-3-HT Full-Bundle JVP Toy Harness

本实验是 PDM-3-HT v2.1 的第一步：在 tiny coupled transformer toy model 上验证 Per-Patch MeanFlow full-bundle JVP 公式。

## 核心公式

令：

```python
delta = t_vec - r_vec
z_dot = delta[..., None] * v
r_dot = zeros_like(r_vec)
t_dot = delta
u_tgt = v - du_dc
```

其中 `du_dc` 由 `torch.func.jvp` 对整个 `(z_t, r_vec, t_vec)` 做一次 full-bundle JVP 得到。

## 运行

```bash
cd /workspace/PDM
python experiments/2026-05-27-pdm3-ht-jvp-toy/scripts/run_jvp_toy.py
```

结果会写入：

```text
experiments/2026-05-27-pdm3-ht-jvp-toy/results/
```

