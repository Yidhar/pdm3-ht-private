# Experiment 5 — Mixed Precision Engineering / Synthetic PAE Latent JVP

本实验从 Experiment 4 复制 synthetic PAE latent + official PAE LightningDiTBlock 子模块 stress 框架，新增工程配方：

```text
bf16/autocast backbone forward/backward
+ fp32 JVP/finite-difference target path
+ math SDPA backend
+ TF32 disabled
+ no checkpoint
```

关键原因：此前 `bf16_autocast` 直接包住 JVP path 的诊断失败：FD rel err 高，且 backward 报 dtype mismatch。本实验将 JVP target path 与训练 backbone path 分离：`du` 在 fp32/no_grad 下计算并 detach，梯度只通过 bf16/autocast 的普通 forward `u`。


## 完成记录（2026-05-27）

**PASS**：`bf16_backbone_fp32_jvp` 在 synthetic PAE latent stress 下通过。

- Main w128d2：FD rel err `7.11995e-05`，train `16/16`，mixed train peak `124.499 MB`，JVP peak `106.667 MB`。
- Scale w256d4：FD rel err `9.12391e-05`，train `8/8`，mixed train peak `384.429 MB`，JVP peak `218.643 MB`。
- 直接 bf16/autocast JVP 仍判定为不可用；B3 应采用 split mixed recipe：fp32 JVP target path + bf16/autocast backbone forward/backward。

主要输出：

- `results/AGGREGATE_SUMMARY.md`
- `results/aggregate_metrics.json`
- `notes/EXPERIMENT_RECORD.md`

追加复核（2026-05-27）：

- JVP/FD/target 均确认使用 live model 参数；本实验没有 EMA shadow 参数。
- `bf16-vs-fp32 fwd rel ≈ 5e-3` 是 split mixed recipe 的普通 forward 数值差，JVP correctness 仍以 fp32 JVP vs fp32 FD rel err 为准。
- sampler 配置为 `--equal-prob 0.75`；recorded correctness sample 的 realized `r=t` fraction 为 smoke `0.8125`、main `0.7666015625`、scale `0.7529296875`。
- 追加 mixed-mask edge audit：
  - `scripts/audit_rt_mixed_mask_degeneracy.py`
  - `logs/rt_mixed_mask_audit.log`
  - `results/rt_mixed_mask_audit/rt_mixed_mask_audit.md`
  - `results/rt_mixed_mask_audit/rt_mixed_mask_audit.json`
- future short-train record 已在脚本中补充 `r_eq_t_count / r_eq_t_total / r_eq_t_fraction_realized` 字段。
