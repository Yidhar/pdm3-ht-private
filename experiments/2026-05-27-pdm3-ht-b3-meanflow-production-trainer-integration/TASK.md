# TASK — Experiment 7: Phase 0 / B3 MeanFlow Production Trainer Integration

## 状态

**完成时间**：2026-05-27 13:53 UTC  
**最终结论**：`PASS`（`practical_gate_pass=True`，`strict_gate_pass=True`）

## 目标

在 Experiment 6 smoke scaffold 已 PASS 的基础上，把 `bf16 backbone + fp32 JVP target` 配方接入到 `external/PAE/pae_with_generator` 下一个非破坏式 Phase 0 / B3 MeanFlow trainer 路径，并用短训 smoke 验证。

固定工程配方：

```text
B3/global sample-level MeanFlow
sample/image-level scalar (r,t): 75% r=t, 25% r<t
live-param fp32 no-autocast JVP target: target=(v-du).detach()
bf16/autocast trainable backbone forward/backward
EMA 可存在并 step 后更新，但不参与 JVP target
math SDPA，TF32 disabled，JVP path 不使用 activation checkpoint
FD audit + memory logging
```

## 约束

- 原 `train_dit.py` 与标准 FM `transport/transport.py` 不改变训练语义。
- 正式路径应能读取 PAE latent safetensors 数据；本实验用 tiny synthetic safetensors shard 做 smoke，不声明 ImageNet/FID/FDr。
- 需要记录代码路径、运行命令、日志、JSON metrics 与 Markdown summary。

## 验证项

- [x] 新增 non-invasive B3 MeanFlow LightningDiT wrapper：`external/PAE/pae_with_generator/models/meanflow_lightningdit.py`。
- [x] 新增 B3 MeanFlow transport/loss：`external/PAE/pae_with_generator/transport/meanflow_transport.py`。
- [x] 实现 live-param fp32/no-autocast JVP target：`target=(v-du).detach()`。
- [x] 实现 bf16/autocast trainable backbone forward/backward。
- [x] 新增单卡 Phase0 trainer：`external/PAE/pae_with_generator/train_meanflow_dit.py`。
- [x] class dropout 使用固定 `force_drop_ids`，保证 JVP target 与 train forward 条件一致。
- [x] EMA 存在并在 optimizer step 后更新，但 `ema_used_for_target_jvp=False`。
- [x] sample-level scalar `(r,t)` sampler，默认 `equal_prob=0.75`。
- [x] 对 FD audit 增加 all-`r=t` 随机抽中时的 forced non-degenerate 防护，避免 `fd_rel=0` 掩盖 JVP 正确性。
- [x] smoke 数据集、配置、日志、metrics、summary 写入实验目录。
- [x] `py_compile` 通过。
- [x] 短训 smoke PASS：FD rel err、`r=t` degen、memory、loss/grad finite 全部满足。
- [x] 清理 `__pycache__`（最终回复前执行）。

## 最终 smoke 核心指标

- `completed_steps`: `8 / 8`
- `final_loss`: `3.580087661743164`
- `fd_rel_err_full_jvp`（last/max）: `8.228138473050667e-05`
- `all_r_eq_t_degenerate_target_minus_v_max_abs`: `0.0`
- `max_actual_rt_samples_target_minus_v_abs`: `0.0`
- `mean_realized_r_eq_t_fraction`: `0.6875`（batch size=4、8 steps 的短训采样均值；目标概率为 0.75）
- `bf16_forward_vs_fp32_rel_err`: mean `0.005162052709372032`, min `0.004896900220522285`, max `0.005438045536788979`
- train loop peak memory: `63.06005859375 MB`
- target JVP train peak memory max: `59.6083984375 MB`
- bf16 backbone forward peak memory max: `60.61767578125 MB`
- backward peak memory max: `63.06005859375 MB`
- FD audit JVP peak memory max: `62.90087890625 MB`
- FD audit central-FD peak memory max: `47.48583984375 MB`
- `target_detached_all_steps=True`
- `jvp_param_source_all_live=True`
- `sample_level_sampler=True`
- `mixed_modes_ok=True`
- `no_nan_or_inf=True`

## 主要产物

```text
external/PAE/pae_with_generator/models/meanflow_lightningdit.py
external/PAE/pae_with_generator/transport/meanflow_transport.py
external/PAE/pae_with_generator/train_meanflow_dit.py
external/PAE/pae_with_generator/configs/DiT_B3MeanFlowTiny_PAE_DINOv2L_d32_smoke.yaml
external/PAE/pae_with_generator/transport/__init__.py

experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration/configs/b3_meanflow_tiny_smoke.yaml
experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration/data/tiny_pae_latents_safetensors/part-000.safetensors
experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration/logs/smoke_b3_meanflow_tiny_prod_trainer.log
experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration/results/smoke_b3_meanflow_tiny_prod_trainer/meanflow_train_metrics.json
experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration/results/smoke_b3_meanflow_tiny_prod_trainer/meanflow_train_summary.md
experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration/results/smoke_b3_meanflow_tiny_prod_trainer/metrics.jsonl
experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration/patches/meanflow_production_trainer_integration.patch
```
