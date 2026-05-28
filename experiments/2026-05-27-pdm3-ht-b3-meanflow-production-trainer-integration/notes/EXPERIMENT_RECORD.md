# Experiment 7 Record — Phase 0 / B3 MeanFlow Production Trainer Integration

## 2026-05-27 — 任务与执行

### 背景

Experiment 6 已在 synthetic latent scaffold 中验证：

- bf16/autocast backbone forward/backward 可用；
- fp32/no-autocast JVP target path 可用；
- FD rel err 约 `1e-4`；
- `r=t` 退化时 target 精确回到 `v`；
- bf16-vs-fp32 forward drift 约 `5e-3`，符合 mixed-precision 训练预期。

本实验将该配方接入 `external/PAE/pae_with_generator` 的非破坏式 Phase 0 / B3 MeanFlow trainer 路径。

### 协作/工具状态

- Switchyard 检查：不可用；此前返回 `status=unavailable`、`message=switchyard command not found`、`next_actions=[continue locally]`，因此本地执行。
- Codex 无上下文子 agent 尝试：工具 sandbox policy 拦截，未得到有效分析结果；继续本地推进。

### 工程改动

新增：

```text
external/PAE/pae_with_generator/models/meanflow_lightningdit.py
external/PAE/pae_with_generator/transport/meanflow_transport.py
external/PAE/pae_with_generator/train_meanflow_dit.py
external/PAE/pae_with_generator/configs/DiT_B3MeanFlowTiny_PAE_DINOv2L_d32_smoke.yaml
```

修改：

```text
external/PAE/pae_with_generator/transport/__init__.py
```

设计要点：

1. `B3MeanFlowLightningDiT` 复用官方 PAE `LightningDiT` block，把 conditioning 改为 `emb(t) + emb(t-r) + emb(y)`。
2. `B3MeanFlowTransport` 生成 sample-level scalar `(r,t)`，默认 `equal_prob=0.75`。
3. target path 使用 live model parameters，fp32/no-autocast，通过 `torch.func.jvp` 得到 `du`，target 为 `(v-du).detach()`。
4. trainable backbone path 使用 bf16/autocast forward/backward。
5. EMA 可存在并 step 后更新，但 `ema_used_for_target_jvp=False`。
6. class dropout 的 `force_drop_ids` 每个 batch 固定一次，并复用在 target JVP 和 bf16 train forward。
7. FD audit 使用 central finite difference 对 full JVP 进行校验。
8. FD audit 如果随机抽到全 `r=t`，强制一个 audit sample 为 `r<t`，防止 `du=0` / `FD=0` 掩盖错误；训练 sampler 本身不被改动。
9. `transport/__init__.py` 增加 optional `torchdiffeq` guard，避免 MeanFlow import 被标准 ODE/SDE transport 的可选依赖阻塞。

### 数据与配置

Synthetic tiny PAE latent shard：

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration/data/tiny_pae_latents_safetensors/part-000.safetensors
```

数据形状：`[64, 32, 16, 16]`，labels 范围约 `[20, 993]`，latent 标准差约 `1.0`。

Smoke config：

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration/configs/b3_meanflow_tiny_smoke.yaml
```

关键配置：

```yaml
train:
  max_steps: 8
  global_batch_size: 4
  sdpa_kernel: math
  allow_tf32: false
  use_ema: true
model:
  model_type: B3MeanFlowLightningDiT-Tiny/1
  in_chans: 32
  use_checkpoint: false
  meanflow_init_scheme: stress_nonzero
  meanflow_final_linear_init: xavier
meanflow:
  precision_recipe: bf16_backbone_fp32_jvp
  equal_prob: 0.75
  time_sampler: ltg
  fd_eps: 0.01
  fd_audit_every: 4
```

### 执行命令

静态检查：

```bash
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:${PYTHONPATH:-} \
python -m py_compile \
  external/PAE/pae_with_generator/transport/meanflow_transport.py \
  external/PAE/pae_with_generator/train_meanflow_dit.py \
  external/PAE/pae_with_generator/models/meanflow_lightningdit.py \
  external/PAE/pae_with_generator/transport/__init__.py
```

短训：

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:${PYTHONPATH:-} \
python /workspace/PDM/external/PAE/pae_with_generator/train_meanflow_dit.py \
  --config "$EXP/configs/b3_meanflow_tiny_smoke.yaml" \
  2>&1 | tee "$EXP/logs/smoke_b3_meanflow_tiny_prod_trainer.log"
```

### 最终结果

最终日志：

```text
[2026-05-27 13:53:13] DONE gate practical=True strict=True final_loss=3.580087661743164 loop_peak=63.06005859375MB
```

核心结果：

| 指标 | 值 |
|---|---:|
| practical gate | `True` |
| strict gate | `True` |
| completed steps | `8 / 8` |
| final loss | `3.580087661743164` |
| loss all finite | `True` |
| grad all finite | `True` |
| u/du/target all finite | `True` |
| target detached all steps | `True` |
| live-param JVP target | `True` |
| sample-level sampler | `True` |
| mixed modes ok | `True` |
| no NaN/Inf | `True` |
| mean realized `r=t` fraction | `0.6875` |
| max actual `r=t` target-v abs | `0.0` |
| bf16-vs-fp32 fwd rel mean | `0.005162052709372032` |
| bf16-vs-fp32 fwd rel min/max | `0.004896900220522285 / 0.005438045536788979` |
| loop peak memory | `63.06005859375 MB` |

FD audit：

| step | fd_rel_err | degen target-v max | forced non-degen | JVP MB | FD MB | r=t frac |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `4.5551111559665746e-05` | `0.0` | `False` | `62.89990234375` | `47.48486328125` | `0.75` |
| 4 | `4.852922736423381e-05` | `0.0` | `True` | `62.90087890625` | `47.48583984375` | `0.75` |
| 8 | `8.228138473050667e-05` | `0.0` | `True` | `62.90087890625` | `47.48583984375` | `0.75` |

Memory：

| 项 | 峰值 |
|---|---:|
| train target JVP peak max | `59.6083984375 MB` |
| bf16 backbone forward peak max | `60.61767578125 MB` |
| backward peak max | `63.06005859375 MB` |
| FD audit JVP peak max | `62.90087890625 MB` |
| FD audit central-FD peak max | `47.48583984375 MB` |
| loop peak max | `63.06005859375 MB` |

### 解释

- FD rel err `8.23e-05` 说明 fp32 JVP target path 与 central FD 一致，JVP 计算本身正确。
- bf16-vs-fp32 fwd rel mean `5.16e-3` 是 mixed-precision 常规前向漂移；target 用 fp32 计算、训练 backbone 用 bf16/autocast，符合预期。
- all-`r=t` degen target-v max 为 `0.0`，说明退化路径正确：`t-r=0` 时 tangent 为 0，`du=0`，target 精确等于 `v`。
- FD audit 的 forced non-degenerate 字段解决了短 batch 下 75% `r=t` 随机抽到全退化样本时 FD rel err 可能为 0 的掩盖问题。

### 产物

```text
logs/smoke_b3_meanflow_tiny_prod_trainer.log
results/smoke_b3_meanflow_tiny_prod_trainer/log.txt
results/smoke_b3_meanflow_tiny_prod_trainer/metrics.jsonl
results/smoke_b3_meanflow_tiny_prod_trainer/meanflow_train_metrics.json
results/smoke_b3_meanflow_tiny_prod_trainer/meanflow_train_summary.md
patches/meanflow_production_trainer_integration.patch
```

### 限制与下一步

限制：这是 tiny synthetic safetensors smoke，不是真实 ImageNet latent 长训，不声明 FID/FDr。

建议下一步：把同一 trainer 配置切到真实 PAE latent 小 shard，做 Phase 0/B3 real-latent smoke（例如 50–200 steps），重点观察真实 latent 下 FD audit、loss finite、显存、吞吐与 EMA/checkpoint 行为，然后再进入 B3 baseline 主训预算评估。
