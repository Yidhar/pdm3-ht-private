# Experiment 5 Record — Mixed Precision Engineering / Synthetic PAE Latent JVP

## 开始记录

- 开始时间：2026-05-27 UTC
- 工作目录：`/workspace/PDM/experiments/2026-05-27-pdm3-ht-mixed-precision-synthetic-jvp`
- 基础脚本来源：Experiment 4 `run_lightningdit_block_pae_shorttrain_stress.py`
- Switchyard 状态：`switchyard host list` 返回 command not found；MCP no-context Codex 子 agent 也因 sandbox policy 未能运行，故本轮本地直接执行并记录。

## 待写入结果

运行完成后补充命令、指标、判定和文件索引。

## 执行记录（2026-05-27 UTC）

### 0. 协作/子 agent 状态

- `switchyard host list` 结果：`/bin/bash: line 1: switchyard: command not found`，无法使用 AGENTS.md 中的 Switchyard bridge。
- 尝试使用无上下文 Codex MCP 子 agent 做只读方案审查，返回 `Permission Error: Operation blocked by sandbox policy`，未产生可用 peer review；本轮改造由本地直接完成并记录。

### 1. 实现

新增脚本：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-mixed-precision-synthetic-jvp/scripts/run_mixed_precision_synthetic_jvp_stress.py
```

在 Experiment 4 synthetic PAE latent stress 框架基础上新增 mode：

```text
bf16_backbone_fp32_jvp
```

语义：

```text
backbone_forward_mode = bf16_autocast
jvp_target_mode      = fp32
fd_mode              = fp32
target_detached      = true
```

训练 step 中：

1. `torch.no_grad()` 下用 fp32 JVP 得到 `du`。
2. `target = (v - du).detach()`。
3. 用 `torch.autocast(cuda, dtype=torch.bfloat16)` 对 backbone 普通 forward 得到 `u`。
4. `loss = mse(u.float(), target.float())`。
5. `loss.backward()` 只穿过 backbone `u`，不穿过 JVP path。

### 2. py_compile / 清理

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-mixed-precision-synthetic-jvp
python -m py_compile "$EXP/scripts/run_mixed_precision_synthetic_jvp_stress.py"
find "$EXP" -type d -name __pycache__ -prune -exec rm -rf {} +
```

结果：通过。

### 3. Smoke run

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-mixed-precision-synthetic-jvp
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:${PYTHONPATH:-} \
python "$EXP/scripts/run_mixed_precision_synthetic_jvp_stress.py" \
  --output-dir "$EXP/results/smoke_mixed_w64d1" \
  --device auto \
  --grids 8 \
  --modes bf16_backbone_fp32_jvp \
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
  2>&1 | tee "$EXP/logs/smoke_mixed_w64d1.log"
```

结果：

```text
FD rel err              = 5.15986e-05
bf16-vs-fp32 fwd rel   = 0.00501426
train                  = 2/2
final loss             = 3.53704
train peak             = 19.2026 MB
JVP peak               = 10.1445 MB
NaN/Inf                = false
```

### 4. Main mixed run — width=128/depth=2

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-mixed-precision-synthetic-jvp
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:${PYTHONPATH:-} \
python "$EXP/scripts/run_mixed_precision_synthetic_jvp_stress.py" \
  --output-dir "$EXP/results/main_mixed_w128d2" \
  --device auto \
  --grids 32 \
  --modes bf16_backbone_fp32_jvp \
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
  2>&1 | tee "$EXP/logs/run_main_mixed_w128d2.log"
```

结果：

```text
FD rel err              = 7.11995e-05
bf16-vs-fp32 fwd rel   = 0.00486624
du/v                   = 0.137126
r=t degen max          = 0
train                  = 16/16
final loss             = 3.61895
fp32 JVP peak          = 106.667 MB
mixed train peak       = 124.499 MB
NaN/Inf                = false
strict/practical gate  = true/true
```

Main step-1 memory breakdown：

```text
target fp32 JVP peak       = 108.323 MB
bf16 backbone forward peak = 86.4478 MB
backward peak              = 119.509 MB
```

### 5. Scale mixed run — width=256/depth=4

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-mixed-precision-synthetic-jvp
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:${PYTHONPATH:-} \
python "$EXP/scripts/run_mixed_precision_synthetic_jvp_stress.py" \
  --output-dir "$EXP/results/scale_mixed_w256d4" \
  --device auto \
  --grids 32 \
  --modes bf16_backbone_fp32_jvp \
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
  2>&1 | tee "$EXP/logs/run_scale_mixed_w256d4.log"
```

结果：

```text
FD rel err              = 9.12391e-05
bf16-vs-fp32 fwd rel   = 0.00620341
du/v                   = 0.123746
r=t degen max          = 0
train                  = 8/8
final loss             = 3.27342
fp32 JVP peak          = 218.643 MB
mixed train peak       = 384.429 MB
NaN/Inf                = false
strict/practical gate  = true/true
```

### 6. 对照与判定

对照 Experiment 4：

| path | FD rel err | train | train peak MB | 备注 |
|---|---:|---:|---:|---|
| Exp4 fp32 main w128d2 | `6.71513e-05` | `16/16` | `614.148` | pass baseline |
| Exp5 split mixed main w128d2 | `7.11995e-05` | `16/16` | `124.499` | pass，显存低约 `4.93x` |
| Exp4 direct bf16_autocast diag | `0.568673` | `0/8` | `280.796` | fail：backward dtype mismatch |
| Exp4 fp32 scale w256d4 | `9.64663e-05` | `8/8` | `2352.25` | pass baseline |
| Exp5 split mixed scale w256d4 | `9.12391e-05` | `8/8` | `384.429` | pass，显存低约 `6.12x` |

最终判定：

```text
PASS: bf16/autocast backbone + fp32 JVP/FD target path is viable on synthetic PAE latent stress.
```

Phase 0 / B3 建议：

1. 默认采用 `bf16_backbone_fp32_jvp`。
2. JVP/FD path 继续固定 fp32；不要直接 bf16/autocast JVP。
3. 保持 `target.detach()`，梯度只穿过 backbone 普通 forward。
4. 继续强制 math SDPA、禁用 TF32、禁用 checkpoint。

### 7. 文件索引

- 汇总：`results/AGGREGATE_SUMMARY.md`
- 汇总 JSON：`results/aggregate_metrics.json`
- Main metrics：`results/main_mixed_w128d2/lightningdit_block_pae_shorttrain_metrics.json`
- Scale metrics：`results/scale_mixed_w256d4/lightningdit_block_pae_shorttrain_metrics.json`
- Smoke metrics：`results/smoke_mixed_w64d1/lightningdit_block_pae_shorttrain_metrics.json`
- Main log：`logs/run_main_mixed_w128d2.log`
- Scale log：`logs/run_scale_mixed_w256d4.log`
- Smoke log：`logs/smoke_mixed_w64d1.log`

---

## 2026-05-27 追加复核：EMA / bf16-vs-fp32 forward / r=t ratio

触发问题：

1. JVP 用的是 EMA 参数还是 live 参数？
2. `bf16-vs-fp32 fwd rel ≈ 5e-3` 是否来自 split mixed 框架？
3. `r=t degen max = 0` 是否可能掩盖 edge case；实际 `r=t` 比例是多少？

### A. JVP 参数源

结论：**Experiment 5 的 JVP/FD/short-train target 都使用 live model 参数，不使用 EMA 参数。**

依据：

- 脚本没有 EMA shadow/copy/update 逻辑。
- `run_one()` 中只构建一个 `model = PerPatchLightningDiTStressModel(...).to(..., dtype=torch.float32)`。
- JVP 调用为 `full_bundle_jvp(model, ...)`，其内部 `torch.func.jvp(fn, ...)` 的 `fn` 直接调用同一个 `model`。
- short train optimizer 是 `torch.optim.AdamW(model.parameters(), ...)`，target JVP 也直接传入同一个 live `model`。

因此本实验是 **live-param JVP**。如果 Phase-0/B3 引入 EMA，建议规则是：

- training loss / target JVP：用 live 参数；
- eval / sampling / checkpoint report：可用 EMA；
- 不要在同一个 loss 中用 EMA target JVP 去监督 live bf16 forward，除非明确作为 teacher-student 设计并单独记录。

### B. `bf16-vs-fp32 fwd rel ≈ 5e-3`

确认：这个数值是 mixed recipe 的预期差异，而不是 JVP 错误。

Experiment 5 的 `bf16_backbone_fp32_jvp` 配方：

- target path：`fp32` JVP/FD，`target=(v-du).detach()`；
- trainable path：`bf16_autocast` ordinary forward 得到 `u`；
- loss：`mse(u.float(), target.float())`。

因此 audit 中额外测量的 `bf16_forward_vs_fp32_rel_err` 是同一 live model、同一输入下：

```text
rel(u_fp32_reference, u_bf16_autocast)
```

主运行数值：

```text
main_mixed_w128d2: 0.00486624
scale_mixed_w256d4: 0.00620341
smoke_mixed_w64d1: 0.00501426
```

解释：这是 bf16/autocast backbone 与 fp32 reference forward 的普通 mixed-precision 数值差，训练会通过反传吸收；真正用于判定 JVP correctness 的指标仍是 fp32 JVP vs fp32 FD：

```text
main FD rel err  = 7.11995e-05
scale FD rel err = 9.12391e-05
```

### C. `r=t` ratio 与 mixed-mask edge 复核

配置层面：三次 mixed run 均使用：

```text
--equal-prob 0.75
```

这意味着 sampler 对每个 token 以 Bernoulli(0.75) 将 `r` 置为 `t`。在 recorded correctness sample 上，实际比例为：

| run | configured p(r=t) | realized r=t fraction | eq/neq tokens |
|---|---:|---:|---:|
| smoke_mixed_w64d1 | `0.75` | `0.8125` | `52/12` |
| main_mixed_w128d2 | `0.75` | `0.7666015625` | `785/239` |
| scale_mixed_w256d4 | `0.75` | `0.7529296875` | `771/253` |

重要澄清：

- 原汇报的 `r=t degen max = 0` 是 **all-token r=t sanity check**：把整张 latent grid 的 `r` 全部设置为 `t`，此时 full-bundle tangent 为 0，所以 `du=0`、`target=v` 精确成立。
- 它不是“在 75/25 per-token mixed grid 中，所有 `r_i=t_i` token 都有 `du_i=0`”的断言。

已追加轻量审计：

```text
script:  scripts/audit_rt_mixed_mask_degeneracy.py
log:     logs/rt_mixed_mask_audit.log
json:    results/rt_mixed_mask_audit/rt_mixed_mask_audit.json
summary: results/rt_mixed_mask_audit/rt_mixed_mask_audit.md
```

补充结果：

| run | all-token r=t target-v max | mixed eq-token target-v max | mixed eq-token du/v | mixed neq-token du/v |
|---|---:|---:|---:|---:|
| smoke_mixed_w64d1 | `0` | `0.000825465` | `0.000162026` | `0.354273` |
| main_mixed_w128d2 | `0` | `0.000395894` | `7.41674e-05` | `0.28345` |
| scale_mixed_w256d4 | `0` | `0.000761986` | `0.000121338` | `0.248543` |

解释：

- **all-token r=t**：严格退化为 Flow Matching，`target=v`，通过。
- **per-token 75/25 mixed grid**：由于 Transformer attention 的 full-bundle JVP 跨 token 耦合，即使某个 token 自身 `r_i=t_i`、`delta_i=0`，它的输出 `u_i` 仍可能受到其他 `delta_j>0` token 的变化影响，所以 mixed equal-token slice 上 `du_i` 不必严格为 0。本次观测到该项很小（约 `7e-5` 到 `1.6e-4` 的 eq-token `du/v`），但不是数学上强制为 0。

Phase-0/B3 记录规则更新：

1. 必须同时记录 `configured_equal_prob` 和每 batch/sample 的 `realized_r_eq_t_fraction`。
2. 必须区分：
   - all-token/all-sample `r=t` degeneracy check；
   - per-token mixed-mask equal-slice diagnostic。
3. 如果目标是复现标准 MeanFlow baseline，应按 **sample/image-level scalar `(r,t)`** 做 75% `r=t` + 25% `r≠t`，而不是 token-level Bernoulli；token-level mixture 属于 PDM-3-HT per-patch 实验条件。

实现更新：`scripts/run_mixed_precision_synthetic_jvp_stress.py` 的 future short-train records 已补充：

```text
latent_meta
r_eq_t_count
r_eq_t_total
r_eq_t_fraction_realized
```

已重新 `py_compile` 并清理 `__pycache__`。
