# EXPERIMENT_RECORD — Experiment 4: Real LightningDiT Block / PAE Latent / Short-Train Stress

- 日期：2026-05-27
- 工作目录：`/workspace/PDM/experiments/2026-05-27-pdm3-ht-lightningdit-block-pae-shorttrain-stress/`
- 任务书：`/workspace/PDM/2026-05-27-pdm3-ht-research-brief-v2.1.md`
- 上一步基线：Experiment 3 证明 proxy LightningDiT-like / PAE-shape full-bundle JVP 可行。


## 0. 协作/子 agent 状态

- 按 `AGENTS.md` 检查 Switchyard：`switchyard host list` 返回 `command not found`，本环境不可用。
- 尝试用无上下文 Codex 子审阅脚本，MCP wrapper 因 sandbox policy 返回 permission error；未能取得有效 peer review。
- 因此本实验由当前 agent 本地执行，并把所有命令、日志、metrics 落盘。

## 1. 目标复述

本实验进入更真实的工程路径：

1. 使用官方 PAE repo 的 `models/lightningdit.py` 组件，尤其是 `LightningDiTBlock`、`Attention`、`RMSNorm`、`SwiGLU`、`FinalLayer`、`VisionRotaryEmbeddingFast`。
2. 不直接使用官方 `LightningDiTBlock.forward(x, c)`，因为官方实现的 AdaLN conditioning 是 image-level `c: [B,D]` 并沿 `dim=1` chunk；本实验需要 per-token `(r_i,t_i,Δ_i)`，所以实现最小 wrapper：保留官方 attention / MLP / norm / AdaLN module，令 `adaLN_modulation(c_tokens)` 输出 `[B,N,6D]` 并沿 `dim=-1` chunk。
3. 使用 PAE-f16d32 形状 `B x N x C = 1 x 1024 x 32`，加载官方 `Latent-stats/PAE_DINOv2L.pt` 的 mean/std。
4. 当前环境没有真实 ImageNet PAE spatial latent shard；因此主实验输入为 `pae_stats_synthetic`：使用官方 PAE latent mean/std 生成 raw latent 后，按 PAE generator config 的 `latent_norm: true` 归一化。**不能称为真实 ImageNet latent cache**。
5. 复用 Experiment 3 的 full-bundle JVP：

```python
delta = t - r
z_dot = delta[..., None] * v
r_dot = torch.zeros_like(r)
t_dot = delta
u, du = jvp(model, (zt, r, t), (z_dot, r_dot, t_dot))
target = (v - du).detach()
```

## 2. 环境与依赖

- Torch：`2.8.0+cu128`
- GPU：`NVIDIA RTX PRO 6000 Blackwell Server Edition`
- CUDA：available
- 官方 PAE repo：`/workspace/PDM/external/PAE/pae_with_generator`
- 官方 PAE commit：`51f8fa6`
- PAE latent stats：`/workspace/PDM/external/PAE_hf/Latent-stats/PAE_DINOv2L.pt`
  - `mean`: `[1,32,1,1]`, original dtype `torch.bfloat16`
  - `std`: `[1,32,1,1]`, original dtype `torch.bfloat16`
- 已确认没有真实 spatial ImageNet PAE latent shard；只有 LightningDiT demo vectors `10000 x 32`，不足以代表 `32 x 32 x 32` spatial latent。

工程设置：

- 默认禁用 TorchDynamo / `torch.compile`：本实验审计 forward-mode AD/JVP，不审计 compile；避免 compiled wrapper 干扰。
- 默认禁用 TF32：`torch.backends.cuda.matmul.allow_tf32=False`，`torch.backends.cudnn.allow_tf32=False`。原因：早期 smoke 中启用/未禁用 TF32 时，`fd_eps=1e-2` 的 central FD relative error 可到 `0.055`；禁用 TF32 后同配置降到 `5.52e-05`。
- 默认 `sdpa_kernel=math`：仍调用官方 `F.scaled_dot_product_attention` 分支，但强制 math backend；默认 CUDA efficient/flash backend 当前不支持 forward AD/JVP。

## 3. 脚本与输出

脚本：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-lightningdit-block-pae-shorttrain-stress/scripts/run_lightningdit_block_pae_shorttrain_stress.py
```

主汇总：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-lightningdit-block-pae-shorttrain-stress/results/AGGREGATE_SUMMARY.md
/workspace/PDM/experiments/2026-05-27-pdm3-ht-lightningdit-block-pae-shorttrain-stress/results/aggregate_metrics.json
```

## 4. 执行命令记录

### 4.1 py_compile

```bash
EXP_ROOT=/workspace/PDM/experiments/2026-05-27-pdm3-ht-lightningdit-block-pae-shorttrain-stress
python -m py_compile "$EXP_ROOT/scripts/run_lightningdit_block_pae_shorttrain_stress.py"
find "$EXP_ROOT" -type d -name __pycache__ -prune -exec rm -rf {} +
```

结果：通过。

### 4.2 Smoke

```bash
EXP_ROOT=/workspace/PDM/experiments/2026-05-27-pdm3-ht-lightningdit-block-pae-shorttrain-stress
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:$PYTHONPATH \
python "$EXP_ROOT/scripts/run_lightningdit_block_pae_shorttrain_stress.py" \
  --output-dir "$EXP_ROOT/results/smoke" \
  --device auto \
  --grids 8 \
  --modes fp32 \
  --latent-modes pae_stats_synthetic \
  --batch-size 1 \
  --channels 32 \
  --width 64 \
  --heads 4 \
  --depth 1 \
  --train-steps 2 \
  --equal-prob 0.75 \
  --time-sampler ltg \
  --fd-eps 1e-2 \
  --fd-max-tokens 1024 \
  --lr 1e-4 \
  --grad-clip 1.0 \
  2>&1 | tee "$EXP_ROOT/logs/smoke_run.log"
```

核心输出：

- FD relative error：`5.51952e-05`
- `du/v`：`0.127252`
- `r=t` target-v max abs：`0`
- train：`2/2` optimizer steps
- final loss：`3.78317`
- train peak：`22.7544 MB`

### 4.3 Main fp32 short train gate

```bash
EXP_ROOT=/workspace/PDM/experiments/2026-05-27-pdm3-ht-lightningdit-block-pae-shorttrain-stress
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:$PYTHONPATH \
python "$EXP_ROOT/scripts/run_lightningdit_block_pae_shorttrain_stress.py" \
  --output-dir "$EXP_ROOT/results/main_w128d2_fp32" \
  --device auto \
  --grids 32 \
  --modes fp32 \
  --latent-modes pae_stats_synthetic \
  --batch-size 1 \
  --channels 32 \
  --width 128 \
  --heads 4 \
  --depth 2 \
  --train-steps 16 \
  --equal-prob 0.75 \
  --time-sampler ltg \
  --ltg-mu 0.0 \
  --ltg-sigma 1.0 \
  --fd-eps 1e-2 \
  --fd-max-tokens 1024 \
  --lr 1e-4 \
  --grad-clip 1.0 \
  2>&1 | tee "$EXP_ROOT/logs/run_main_w128d2_fp32_shorttrain.log"
```

核心输出：

- `N=1024`, width=128, heads=4, depth=2
- FD relative error：`6.71513e-05`
- `du/v`：`0.143234`
- `r=t` target-v max abs：`0`
- JVP time：`0.627087 sec`
- JVP peak：`106.667 MB`
- train：`16/16` optimizer steps
- final loss：`3.37412`
- train peak：`614.148 MB`
- strict gate：`PASS`

### 4.4 Scale fp32 probe

```bash
EXP_ROOT=/workspace/PDM/experiments/2026-05-27-pdm3-ht-lightningdit-block-pae-shorttrain-stress
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:$PYTHONPATH \
python "$EXP_ROOT/scripts/run_lightningdit_block_pae_shorttrain_stress.py" \
  --output-dir "$EXP_ROOT/results/scale_w256d4_fp32" \
  --device auto \
  --grids 32 \
  --modes fp32 \
  --latent-modes pae_stats_synthetic \
  --batch-size 1 \
  --channels 32 \
  --width 256 \
  --heads 8 \
  --depth 4 \
  --train-steps 8 \
  --equal-prob 0.75 \
  --time-sampler ltg \
  --ltg-mu 0.0 \
  --ltg-sigma 1.0 \
  --fd-eps 1e-2 \
  --fd-max-tokens 1024 \
  --lr 1e-4 \
  --grad-clip 1.0 \
  2>&1 | tee "$EXP_ROOT/logs/run_scale_w256d4_fp32.log"
```

核心输出：

- `N=1024`, width=256, heads=8, depth=4
- FD relative error：`9.64663e-05`
- `du/v`：`0.119047`
- `r=t` target-v max abs：`0`
- train：`8/8` optimizer steps
- final loss：`3.60646`
- train peak：`2352.25 MB`
- strict gate：`PASS`

### 4.5 bf16/autocast diagnostic

```bash
EXP_ROOT=/workspace/PDM/experiments/2026-05-27-pdm3-ht-lightningdit-block-pae-shorttrain-stress
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:$PYTHONPATH \
python "$EXP_ROOT/scripts/run_lightningdit_block_pae_shorttrain_stress.py" \
  --output-dir "$EXP_ROOT/results/bf16_diag_w128d2" \
  --device auto \
  --grids 32 \
  --modes bf16_autocast \
  --latent-modes pae_stats_synthetic \
  --batch-size 1 \
  --channels 32 \
  --width 128 \
  --heads 4 \
  --depth 2 \
  --train-steps 8 \
  --equal-prob 0.75 \
  --time-sampler ltg \
  --ltg-mu 0.0 \
  --ltg-sigma 1.0 \
  --fd-eps 1e-2 \
  --fd-max-tokens 1024 \
  --lr 1e-4 \
  --grad-clip 1.0 \
  2>&1 | tee "$EXP_ROOT/logs/run_bf16_diag_w128d2.log"
```

核心输出：

- forward/JVP finite
- FD relative error：`0.568673`，不满足 correctness gate
- `r=t` target-v max abs：`0`
- backward/optimizer：`0/8`，第一步报错：`expected input and grad types to match, or input to be at::Half and grad to be at::Float`
- 结论：bf16/autocast 不能作为 correctness 或短训主路径；需单独工程化 mixed precision JVP/backward。

### 4.6 Checkpoint diagnostic

```bash
EXP_ROOT=/workspace/PDM/experiments/2026-05-27-pdm3-ht-lightningdit-block-pae-shorttrain-stress
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:$PYTHONPATH \
python "$EXP_ROOT/scripts/run_lightningdit_block_pae_shorttrain_stress.py" \
  --output-dir "$EXP_ROOT/results/checkpoint_probe_w128d2_fp32" \
  --device auto \
  --grids 32 \
  --modes fp32 \
  --latent-modes pae_stats_synthetic \
  --batch-size 1 \
  --channels 32 \
  --width 128 \
  --heads 4 \
  --depth 2 \
  --train-steps 2 \
  --equal-prob 0.75 \
  --time-sampler ltg \
  --fd-eps 1e-2 \
  --fd-max-tokens 1024 \
  --lr 1e-4 \
  --grad-clip 1.0 \
  --use-checkpoint \
  2>&1 | tee "$EXP_ROOT/logs/run_checkpoint_probe_w128d2_fp32.log"
```

核心输出：

- eval JVP/FD finite：FD relative error `7.12826e-05`
- short-train checkpoint path：`0/2`，第一步报错：`You must implement the jvp function for custom autograd.Function to use it with forward mode AD.`
- 结论：当前 PyTorch checkpoint custom autograd path 与 forward-mode JVP 不兼容；本实验主路径必须 `no checkpoint`。

### 4.7 Default SDPA backend diagnostic

```bash
EXP_ROOT=/workspace/PDM/experiments/2026-05-27-pdm3-ht-lightningdit-block-pae-shorttrain-stress
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:$PYTHONPATH \
python "$EXP_ROOT/scripts/run_lightningdit_block_pae_shorttrain_stress.py" \
  --output-dir "$EXP_ROOT/results/sdpa_default_probe_w128d2_fp32" \
  --device auto \
  --grids 32 \
  --modes fp32 \
  --latent-modes pae_stats_synthetic \
  --batch-size 1 \
  --channels 32 \
  --width 128 \
  --heads 4 \
  --depth 2 \
  --train-steps 0 \
  --equal-prob 0.75 \
  --time-sampler ltg \
  --fd-eps 1e-2 \
  --fd-max-tokens 1024 \
  --sdpa-kernel default \
  2>&1 | tee "$EXP_ROOT/logs/run_sdpa_default_probe_w128d2_fp32.log"
```

核心输出：

- 报错：`Trying to use forward AD with _scaled_dot_product_efficient_attention that does not support it because it has not been implemented yet.`
- 结论：默认 CUDA efficient/flash SDPA backend 不能用于 full-bundle JVP；主路径强制 `sdpa_kernel=math`。

## 5. Gate 判定

### 主 fp32 gate

通过条件：

- N=1024 full-bundle JVP finite：通过
- central FD relative error `< 1e-2`：`6.71513e-05`，通过
- `r=t` 退化检查 target-v max abs `< 1e-7`：`0`，通过
- short train loop 无 NaN/Inf，梯度 finite：`16/16`，通过
- final loss finite：`3.37412`，通过

判定：**PASS**。

### 扩展 scale gate

- width=256/depth=4/N=1024：FD relative error `9.64663e-05`，`8/8` 短训 finite，train peak `2352.25 MB`。

判定：**PASS**。

### Mixed precision / checkpoint / backend caveats

- bf16/autocast：forward/JVP finite，但 FD error `0.568673`，backward dtype mismatch；**FAIL diagnostic / 不作为 correctness path**。
- checkpoint：checkpoint + JVP train path 报 custom autograd JVP not implemented；**FAIL diagnostic / 主路径不用 checkpoint**。
- default SDPA backend：efficient attention forward AD/JVP not implemented；**FAIL diagnostic / 主路径强制 math SDPA**。
- TF32：禁用是必要工程条件；启用或未禁用时 FD audit 可被数值精度污染。

## 6. 结论

本实验在更真实的官方 LightningDiT block 子模块路径上验证了 Per-Patch MeanFlow full-bundle JVP 的主风险：

- fp32、TF32 disabled、math SDPA、no checkpoint 时，`N=1024,C=32` 的 PAE-shape full-bundle JVP 与 central FD 高一致，短训稳定。
- 使用官方 PAE latent stats 的 synthetic normalized latent 通过；但这不是实际 ImageNet PAE latent shard，不能外推为真实 latent 数据分布已完全覆盖。
- 下一步若要更接近真实训练，应补充真实 PAE spatial latent shard 或直接跑 PAE encoder 抽取小批 latent，并为 mixed precision / efficient attention / checkpoint 另做可微算子替代或 fallback 设计。

## 7. 文件索引

- 脚本：`scripts/run_lightningdit_block_pae_shorttrain_stress.py`
- smoke log：`logs/smoke_run.log`
- main log：`logs/run_main_w128d2_fp32_shorttrain.log`
- scale log：`logs/run_scale_w256d4_fp32.log`
- bf16 log：`logs/run_bf16_diag_w128d2.log`
- checkpoint log：`logs/run_checkpoint_probe_w128d2_fp32.log`
- default SDPA log：`logs/run_sdpa_default_probe_w128d2_fp32.log`
- aggregate：`results/AGGREGATE_SUMMARY.md`
- aggregate metrics：`results/aggregate_metrics.json`
