# Experiment 6 Record — Phase 0 / B3 MeanFlow Mixed-Precision Integration Smoke

## 0. 任务

用户要求：

> 把 Experiment 5 得到的 bf16 backbone + fp32 JVP target 配方正式接入 Phase 0 / B3 MeanFlow baseline，并做一个短训 smoke，验证 live-param fp32 JVP target、bf16 backbone 训练、r=t sampler、FD audit 和 memory logging 全部可用。

## 1. 初始决策

- 现有 PAE 官方 `transport.training_losses` 是标准 Flow Matching/velocity objective，仓库内未发现已实现的 B3 MeanFlow baseline。
- 因此本实验先建立 B3 integration smoke scaffold：sample-level scalar `(r,t)` MeanFlow training loop + 官方 PAE LightningDiT 子模块 + PAE latent stats synthetic 输入。
- 正式沿用 Experiment 5 的 split mixed-precision 配方：bf16 backbone forward/backward；fp32 JVP/FD target；target detach；live params；math SDPA；TF32 disabled；JVP path no checkpoint。

## 2. 待记录运行

运行命令、结果、判定将在 smoke 完成后追加。


## 3. Switchyard / peer delegation 状态

按照 `/workspace/PDM/AGENTS.md` 检查 HYARD/Switchyard：

```json
{"status":"unavailable","job_id":null,"message":"switchyard command not found","next_actions":["continue locally"]}
```

因此本实验在本地继续执行。

## 4. 实现

新增脚本：

```text
scripts/run_b3_meanflow_mixed_precision_smoke.py
```

实现要点：

1. 使用官方 PAE LightningDiT 子模块：`LightningDiT` / `PatchEmbed` / `LightningDiTBlock` / `Attention` / `RMSNorm` / `SwiGLU` / `FinalLayer` / `VisionRotaryEmbeddingFast`。
2. 将 B3 MeanFlow forward signature 扩展为：

   ```python
   u_theta(z_t, r, t, y)
   c = emb(t) + emb(t - r) + emb(y)
   ```

3. sample-level scalar sampler：`r,t` shape `[B]`，默认 75% 样本 `r=t`，25% 样本 `r<t`。
4. JVP target path：

   ```python
   delta = t - r
   z_dot = delta[:, None, None, None] * v
   r_dot = zeros_like(r)
   t_dot = delta
   u, du = jvp(model, (z_t, r, t), (z_dot, r_dot, t_dot))
   target = (v - du).detach()
   ```

5. mixed precision：
   - target/JVP/FD：fp32 no autocast；
   - trainable backbone forward/backward：bf16 autocast；
   - loss 用 `F.mse_loss(u.float(), target.float())`。
6. live-param target：JVP 使用 live model；EMA object 可存在但不参与 target JVP，且在 optimizer step 后更新。
7. math SDPA；TF32 disabled；JVP path 禁止 checkpoint。

说明：默认 `stress_nonzero` 初始化用于让 tiny smoke 的 JVP/FD 非平凡；这不等于完整长训的官方 zero-init 质量设置。

## 5. 编译检查

```bash
python -m py_compile experiments/2026-05-27-pdm3-ht-b3-meanflow-mixed-precision-integration/scripts/run_b3_meanflow_mixed_precision_smoke.py
```

结果：通过。

## 6. Debug preflight smoke

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-mixed-precision-integration
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:${PYTHONPATH:-} python "$EXP/scripts/run_b3_meanflow_mixed_precision_smoke.py"   --output-dir "$EXP/results/debug_smoke_b3_mixed_w64d1_b2"   --device auto   --grid 8   --batch-size 2   --channels 32   --width 64   --heads 4   --depth 1   --train-steps 1   --equal-prob 0.75   --time-sampler ltg   --fd-eps 1e-2   --lr 1e-4   --grad-clip 1.0   2>&1 | tee "$EXP/logs/debug_smoke_b3_mixed_w64d1_b2.log"
```

结果：PASS。

- FD rel err：`6.21388e-05`
- bf16-vs-fp32 fwd rel：`0.00492029`
- all-r=t degen：`0`
- train：`1/1`
- loop peak：`21.0913 MB`

## 7. 主 smoke

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-mixed-precision-integration
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:${PYTHONPATH:-} python "$EXP/scripts/run_b3_meanflow_mixed_precision_smoke.py"   --output-dir "$EXP/results/smoke_b3_mixed_w128d2_g16_b4"   --device auto   --grid 16   --batch-size 4   --channels 32   --width 128   --heads 4   --depth 2   --train-steps 8   --equal-prob 0.75   --time-sampler ltg   --fd-eps 1e-2   --lr 1e-4   --grad-clip 1.0   2>&1 | tee "$EXP/logs/smoke_b3_mixed_w128d2_g16_b4.log"
```

结果文件：

```text
results/smoke_b3_mixed_w128d2_g16_b4/b3_meanflow_mixed_precision_smoke_metrics.json
results/smoke_b3_mixed_w128d2_g16_b4/b3_meanflow_mixed_precision_smoke_summary.md
logs/smoke_b3_mixed_w128d2_g16_b4.log
```

### 7.1 Gate

```json
{
  "status_ok": true,
  "fd_ok_lt_1e_2": true,
  "fd_rel_err_full_jvp": 9.76049585019447e-05,
  "degenerate_ok_lt_1e_7": true,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0,
  "target_detached": true,
  "live_param_jvp_target": true,
  "sample_level_sampler": true,
  "mixed_modes_ok": true,
  "train_ok": true,
  "memory_logged": true,
  "no_nan_or_inf": true,
  "practical_gate_pass": true,
  "strict_gate_pass": true
}
```

### 7.2 核心指标

| 指标 | 值 |
|---|---:|
| FD rel err full JVP | `9.76049585019447e-05` |
| bf16-vs-fp32 ordinary forward rel | `0.004396095609430949` |
| du/v norm ratio | `0.05477528763633035` |
| all-r=t target-v max abs | `0.0` |
| actual r=t target-v max abs | `0.0` |
| JVP peak memory | `43.8955078125 MB` |
| FD peak memory | `29.48046875 MB` |
| bf16 forward audit peak memory | `30.646484375 MB` |
| train loop peak memory | `65.5625 MB` |
| train steps | `8/8` |
| final loss | `3.9109983444213867` |
| loss finite | `true` |
| grad finite | `true` |
| u/du/target finite | `true` |
| target detached all steps | `true` |
| JVP param source all live | `true` |
| train mean realized r=t fraction | `0.71875` |

### 7.3 Step telemetry

| step | loss | r=t fraction | target JVP MB | bf16 backbone MB | backward MB | grad norm | r=t target-v max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `4.1924333572387695` | `0.5` | `45.525390625` | `47.2841796875` | `58.97900390625` | `3.804748773574829` | `0.0` |
| 2 | `4.064338684082031` | `1.0` | `61.73486328125` | `62.24267578125` | `65.5625` | `3.48045015335083` | `0.0` |
| 3 | `4.1449737548828125` | `0.5` | `61.73486328125` | `62.24267578125` | `65.5625` | `3.7784547805786133` | `0.0` |
| 4 | `4.117317199707031` | `1.0` | `61.73486328125` | `62.24267578125` | `65.5625` | `3.7269763946533203` | `0.0` |
| 5 | `4.032303333282471` | `0.5` | `61.73486328125` | `62.24267578125` | `65.5625` | `3.535240888595581` | `0.0` |
| 6 | `3.929927349090576` | `0.75` | `61.73486328125` | `62.24267578125` | `65.5625` | `3.4743056297302246` | `0.0` |
| 7 | `4.028849124908447` | `0.5` | `61.73486328125` | `62.24267578125` | `65.5625` | `3.425520420074463` | `0.0` |
| 8 | `3.9109983444213867` | `1.0` | `61.73486328125` | `62.24267578125` | `65.5625` | `3.5698113441467285` | `0.0` |

## 8. 结论

PASS：Experiment 5 的 `bf16 backbone + fp32 JVP target` 配方已在 Phase 0 / B3 sample-level MeanFlow smoke scaffold 中可用。

确认项：

- live-param fp32 JVP target：通过；`jvp_param_source=live`，EMA 不参与 target。
- bf16 backbone train path：通过；短训 8/8 optimizer steps，loss/grad finite。
- sample-level r=t sampler：通过；configured 0.75，8-step mean realized 0.71875。
- FD audit：通过；FD rel err `9.7605e-05`。
- r=t degeneracy：通过；all-r=t 与实际 r=t 样本 target-v max 均为 `0.0`。
- memory logging：通过；JVP/FD/backbone/backward/loop peak 均记录。

限制：这不是完整 ImageNet B3 baseline/FID 复现；上游 `transport.training_losses` 仍需后续改成生产级 MeanFlow loss 或新建正式 trainer。当前结果给出的是进入 Phase 0/B3 生产训练循环前的 mixed-precision 与 target 构造 gate。
