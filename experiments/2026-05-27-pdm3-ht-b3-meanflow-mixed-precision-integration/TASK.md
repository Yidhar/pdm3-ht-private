# TASK — Experiment 6: Phase 0 / B3 MeanFlow Mixed-Precision Integration Smoke

## 目标

把 Experiment 5 已验证的 mixed-precision 配方正式接入 Phase 0 / B3 MeanFlow baseline 的最小可运行训练路径，并完成短训 smoke。

配方固定为：

```text
bf16/autocast backbone forward/backward
+ fp32 no-autocast JVP / FD target path
+ target = (v - du).detach()
+ JVP target 使用 live 参数，不使用 EMA
+ sample/image-level scalar (r,t) sampler: 75% r=t, 25% r!=t
+ math SDPA, FD audit 时 TF32 disabled
+ 不在 JVP path 使用 activation checkpoint
```

## 验证项

- [x] 创建独立实验目录，保留脚本、日志、结果、记录。
- [x] 使用官方 PAE LightningDiT 子模块构造 sample-level B3 MeanFlow smoke 模型。
- [x] 接入 live-param fp32 JVP target path。
- [x] 接入 bf16/autocast backbone train path。
- [x] 实现标准 MeanFlow sample-level `r=t` sampler。
- [x] 输出 FD audit：`fd_rel_err_full_jvp`。
- [x] 输出 bf16-vs-fp32 ordinary forward 差异。
- [x] 输出 JVP / FD / backbone forward / backward / loop memory logging。
- [x] 短训 smoke：loss、grad、u/du/target 均 finite。
- [x] 更新 `README.md` 与 `notes/EXPERIMENT_RECORD.md`。
- [x] 清理 `__pycache__`。

## 判定

- PASS：短训完成，live-param fp32 JVP、bf16 backbone、sample-level r=t sampler、FD audit、memory logging 全部可用，且无 NaN/Inf。
- Caveat：该实验是 B3 baseline 的工程 smoke，不声明 ImageNet FID/FDr，也不等价于完整长训。


## 完成结果（2026-05-27）

主 smoke：`results/smoke_b3_mixed_w128d2_g16_b4/`

- Gate：PASS (`practical_gate_pass=True`, `strict_gate_pass=True`)
- FD rel err full JVP：`9.76049585019447e-05`
- bf16-vs-fp32 forward rel：`0.004396095609430949`
- live-param JVP target：通过 (`jvp_param_source=live`)
- target detach：通过 (`target_detached_all_steps=True`)
- sample-level r=t sampler：通过，8-step mean realized fraction `0.71875`
- all-r=t degeneracy：`target-v max abs = 0.0`
- short train：`8/8` optimizer steps, loss/grad/u/du/target 全 finite
- memory logging：JVP `43.8955 MB`；FD `29.4805 MB`；forward audit `30.6465 MB`；train loop peak `65.5625 MB`
