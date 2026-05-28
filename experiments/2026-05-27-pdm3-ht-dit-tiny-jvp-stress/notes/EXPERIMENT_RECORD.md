# 实验记录：Experiment 3 综合结果：LightningDiT-Tiny / PAE-shape Full-Bundle JVP Stress Test

- 主结果生成时间：2026-05-27T10:58:58Z
- 追加 scale-up 生成时间：2026-05-27T10:59:21Z
- 追加 MeanFlow-mixture 生成时间：2026-05-27T11:00:19Z
- torch：`2.8.0+cu128`；cuda_available：`True`；device：`cuda`
- 说明：本实验使用 LightningDiT-like tiny proxy，不是官方 LightningDiT；它保留 PAE-shape `C=32`、全局 attention、per-token AdaLN/time conditioning、full-bundle JVP 这些关键风险因子。

## 执行摘要

- **主 fp32 gate：PASS**。width=128/depth=2 下，N=256 与 N=1024、gaussian 与 correlated latent 均通过 full-bundle JVP FD audit 和 training-like backward。
- 主 fp32 最大 FD relative error：`6.90423e-05`，远低于 `<1e-2` strict gate。
- 所有 fp32 记录最大 FD relative error：`7.10713e-05`。
- 所有 fp32 `r=t` degeneracy 最大 target-v 误差：`0`。
- N=1024 full-bundle JVP 与 train-step 均 finite，无 NaN/Inf。
- 追加 scale-up：width=256/depth=4/heads=8/N=1024 通过，FD rel err `6.27395e-05`，train peak memory `1181.450 MB`。
- 追加 MeanFlow 75% `r=t` mixture：N=1024 通过，FD rel err `7.10713e-05`，`du/v` 从主设置约 `0.213` 降至 `0.111454`。
- `bf16_autocast` 稳定性：无 NaN/Inf，梯度 finite；但 FD relative error 高达 `0.408~0.436`，不能作为 correctness path，真实训练应保持 JVP path fp32。
- 本轮最大 JVP peak memory：`213.121 MB`；最大 train-step peak memory：`1181.450 MB`。

## 主实验命令

```bash
EXP_ROOT=/workspace/PDM/experiments/2026-05-27-pdm3-ht-dit-tiny-jvp-stress
python "$EXP_ROOT/scripts/run_dit_tiny_jvp_stress.py" \
  --output-dir "$EXP_ROOT/results" \
  --device auto \
  --grids 16,32 \
  --modes fp32,bf16_autocast \
  --latent-modes gaussian,correlated \
  --batch-size 1 \
  --channels 32 \
  --width 128 \
  --heads 4 \
  --depth 2 \
  --fd-eps 1e-2 \
  --fd-max-tokens 1024 \
  --lr 1e-4 \
  --grad-clip 1.0 \
  2>&1 | tee "$EXP_ROOT/logs/run_dit_tiny_jvp_stress_w128d2_fp32_bf16.log"
```

## 追加实验命令

### scale-up N=1024 / width=256 / depth=4

```bash
python "$EXP_ROOT/scripts/run_dit_tiny_jvp_stress.py" \
  --output-dir "$EXP_ROOT/results/scale_w256d4" \
  --device auto --grids 32 --modes fp32 --latent-modes gaussian \
  --batch-size 1 --channels 32 --width 256 --heads 8 --depth 4 \
  --fd-eps 1e-2 --fd-max-tokens 1024 --lr 1e-4 --grad-clip 1.0 \
  2>&1 | tee "$EXP_ROOT/logs/run_scale_w256d4_n1024_fp32.log"
```

### MeanFlow mixture / 75% r=t

```bash
python "$EXP_ROOT/scripts/run_dit_tiny_jvp_stress.py" \
  --output-dir "$EXP_ROOT/results/mf_mixture_eq075" \
  --device auto --grids 32 --modes fp32 --latent-modes gaussian \
  --batch-size 1 --channels 32 --width 128 --heads 4 --depth 2 \
  --fd-eps 1e-2 --fd-max-tokens 1024 --equal-prob 0.75 \
  --lr 1e-4 --grad-clip 1.0 \
  2>&1 | tee "$EXP_ROOT/logs/run_mf_mixture_eq075_n1024_fp32.log"
```

## 综合指标表

| tag | width/depth | eq_prob | grid | N | latent | mode | status | fd rel err | r=t err | du/v | JVP MB | train MB | grad finite | grad norm | NaN/Inf |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| main-w128d2 | 128/2 | 0 | 16 | 256 | gaussian | fp32 | ok | 6.90423e-05 | 0 | 0.207664 | 20.75 | 65.2065 | True | 0.644589 | False |
| main-w128d2 | 128/2 | 0 | 16 | 256 | gaussian | bf16_autocast | ok | 0.435701 | 0 | 0.198355 | 28.2578 | 60.5327 | True | 0.673863 | False |
| main-w128d2 | 128/2 | 0 | 32 | 1024 | gaussian | fp32 | ok | 5.78311e-05 | 0 | 0.213067 | 113.605 | 335.634 | True | 0.545651 | False |
| main-w128d2 | 128/2 | 0 | 32 | 1024 | gaussian | bf16_autocast | ok | 0.408732 | 0 | 0.213246 | 123.605 | 306.313 | True | 0.530191 | False |
| main-w128d2 | 128/2 | 0 | 16 | 256 | correlated | fp32 | ok | 6.30149e-05 | 0 | 0.2074 | 28.875 | 65.2065 | True | 0.8948 | False |
| main-w128d2 | 128/2 | 0 | 16 | 256 | correlated | bf16_autocast | ok | 0.407845 | 0 | 0.226138 | 28.2578 | 60.5327 | True | 0.890203 | False |
| main-w128d2 | 128/2 | 0 | 32 | 1024 | correlated | fp32 | ok | 6.77057e-05 | 0 | 0.213172 | 113.605 | 335.634 | True | 0.583318 | False |
| main-w128d2 | 128/2 | 0 | 32 | 1024 | correlated | bf16_autocast | ok | 0.416029 | 0 | 0.213774 | 123.605 | 306.313 | True | 0.590207 | False |
| scale-w256d4 | 256/4 | 0 | 32 | 1024 | gaussian | fp32 | ok | 6.27395e-05 | 0 | 0.204823 | 213.121 | 1181.45 | True | 0.679237 | False |
| mf75-w128d2 | 128/2 | 0.75 | 32 | 1024 | gaussian | fp32 | ok | 7.10713e-05 | 0 | 0.111454 | 104.23 | 334.259 | True | 0.559183 | False |

## 结论

1. **full-bundle JVP 在 DiT-Tiny/PAE-shape proxy 上通过当前 go/no-go gate。** fp32 FD audit 全部约 `5.8e-05~7.1e-05`，N=1024 train-step 可反传，梯度 finite，无 NaN/Inf。
2. **N=1024/C=32 本身不是阻塞点。** 即使 scale-up 到 width=256/depth=4，train-step peak memory 约 `1.18 GB`，说明继续增大 depth/width 或接入更真实 LightningDiT block 有空间。
3. **MeanFlow 75% r=t mixture 会降低 JVP 项相对强度。** 在 N=1024/w128d2 下，`du/v` 约从 `0.213` 降到 `0.111`，这符合大量 r=t 样本退化为 FM 的预期，可能改善早期稳定性。
4. **mixed precision 需要谨慎。** `bf16_autocast` 路径没有 NaN，训练梯度 finite，但 FD audit 误差约 `0.41~0.44`，不能作为严格 JVP correctness 路径。后续真实训练应采用 fp32 JVP path，并单独验证 autocast 边界。
5. **本实验仍是 proxy。** 尚未覆盖官方 LightningDiT、FlashAttention/checkpointing/compile、真实 PAE latent、真实 LTG sampler、多步训练曲线和大模型显存；下一步应做真实模块集成 stress 或短训。

## 下一步建议

- Experiment 4：接入真实/更接近真实的 LightningDiT block，测试 width/depth sweep、activation checkpointing、FlashAttention/SDPA 兼容性。
- 同时加入 local JVP sampled-ablation 与 JVP-free/FD-consistency arm，比较 gradient cosine 与 loss stability。
- 若 PAE encoder/latent cache 可用，替换 gaussian/correlated proxy 为真实 PAE latent subset。
