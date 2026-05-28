# TASK — Experiment 4: Real LightningDiT Block / PAE Latent / Short-Train Stress

- 日期：2026-05-27
- 任务书：`/workspace/PDM/2026-05-27-pdm3-ht-research-brief-v2.1.md`
- 上一步：Experiment 3 已证明 full-bundle JVP 在 LightningDiT-like proxy / PAE-shape 上通过。

## 目标

进入更真实的工程风险验证：

1. 使用官方 PAE repo 中的 `models/lightningdit.py` 组件，特别是真实 `LightningDiTBlock` / `Attention` / RMSNorm / SwiGLU / SDPA 路径。
2. 将 scalar image-level AdaLN 改造成 per-token `(r_i,t_i)` conditioning，但保留官方 block 的 attention/MLP/norm/adaLN modules。
3. 使用 PAE-f16d32 形状 `B x 32 x 32 x 32`，并加载官方 PAE latent statistics；若没有真实 latent shard，则记录为 `pae_stats_synthetic`，不冒充 ImageNet 真 latent。
4. 做 full-bundle Per-Patch MeanFlow JVP audit、finite-difference audit、短训 stress。
5. 对照 checkpointing / bf16-autocast / LTG / 75% r=t mixture 的稳定性和显存。

## Gate

主 fp32 gate：

- N=1024 full-bundle JVP finite。
- central FD relative error `< 1e-2`。
- `r=t` 退化检查 target-v max abs `< 1e-7`。
- short train loop 无 NaN/Inf，梯度 finite。
- loss 最后一步 finite；记录 loss 曲线与 peak memory。

bf16 gate 只作为稳定性诊断，不作为 correctness path；若 FD error 高但 finite，需要明确 caveat。

## 文件规范

- 脚本：`scripts/`
- 日志：`logs/`
- 指标/摘要：`results/`
- 记录：`notes/EXPERIMENT_RECORD.md`

## 状态

- [x] 建立目录结构
- [x] 实现实验脚本
- [x] py_compile / smoke run
- [x] 正式 fp32 短训
- [x] bf16/autocast 诊断
- [x] scale/checkpoint 追加
- [x] 汇总记录与清理缓存


## 完成记录（2026-05-27）

- 脚本：`scripts/run_lightningdit_block_pae_shorttrain_stress.py`
- 汇总：`results/AGGREGATE_SUMMARY.md`
- 指标：`results/aggregate_metrics.json`
- 实验记录：`notes/EXPERIMENT_RECORD.md`

### 最终判定

**PASS with engineering caveats**：

- 主路径 `fp32 + official PAE LightningDiTBlock submodules + per-token AdaLN wrapper + PAE-stat synthetic normalized latent + math SDPA + TF32 disabled + no checkpoint` 通过。
- Main fp32 N=1024：FD relative error `6.71513e-05`，`r=t` target-v max abs `0`，短训 `16/16` finite，final loss `3.37412`，train peak `614.148 MB`。
- Scale width=256/depth=4：FD relative error `9.64663e-05`，短训 `8/8` finite，train peak `2352.25 MB`。
- Caveat：没有真实 ImageNet PAE latent shard；本实验使用官方 PAE latent stats synthetic normalized latent。
- Caveat：bf16/autocast、checkpoint+JVP、default CUDA efficient/flash SDPA backend 均未通过诊断，不能作为当前 correctness/train 主路径。
