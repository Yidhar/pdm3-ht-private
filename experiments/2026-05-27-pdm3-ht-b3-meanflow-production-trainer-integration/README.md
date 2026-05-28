# Experiment 7 — Phase 0 / B3 MeanFlow Production Trainer Integration

本实验把 Experiment 6 的 B3 MeanFlow mixed-precision smoke scaffold 推进为 `external/PAE/pae_with_generator` 下的非破坏式正式 trainer 路径，并完成 tiny synthetic PAE latent 短训 smoke。

## 一句话结论

**PASS**：Phase 0 / B3 MeanFlow trainer 已能在真实 PAE/LightningDiT 代码路径下运行 `bf16 backbone + fp32 live-param JVP target`，并通过 8-step smoke：FD rel err `8.228e-05`、`r=t` 退化 `target-v=0`、loss/grad finite、memory logging 全部正常。

## 核心原则

- 不改坏原始 PAE `train_dit.py` / 标准 FM `transport.training_losses` 语义。
- 新增 B3 MeanFlow wrapper / transport / trainer：
  - `models/meanflow_lightningdit.py`
  - `transport/meanflow_transport.py`
  - `train_meanflow_dit.py`
- 训练语义：sample/image-level scalar `(r,t)`，默认 75% `r=t`，25% `r<t`。
- 精度语义：bf16/autocast trainable backbone forward/backward + fp32/no-autocast JVP/FD target path。
- JVP target 使用 live params；EMA 只在 optimizer step 后更新，不参与 target。
- `target=(v-du).detach()`；target 全程 detached。
- class dropout 的 `force_drop_ids` 每个 batch 固定采样一次，并同时传给 target JVP forward 与 train forward，避免条件不一致。
- SDPA 使用 math；TF32 disabled；JVP path 不启用 activation checkpoint。

## 文件布局

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration/
├── TASK.md
├── README.md
├── configs/
│   └── b3_meanflow_tiny_smoke.yaml
├── data/
│   └── tiny_pae_latents_safetensors/
│       └── part-000.safetensors
├── logs/
│   └── smoke_b3_meanflow_tiny_prod_trainer.log
├── notes/
│   └── EXPERIMENT_RECORD.md
├── patches/
│   └── meanflow_production_trainer_integration.patch
└── results/
    └── smoke_b3_meanflow_tiny_prod_trainer/
        ├── log.txt
        ├── meanflow_train_metrics.json
        ├── meanflow_train_summary.md
        └── metrics.jsonl
```

## 新增/修改代码

```text
external/PAE/pae_with_generator/models/meanflow_lightningdit.py
external/PAE/pae_with_generator/transport/meanflow_transport.py
external/PAE/pae_with_generator/train_meanflow_dit.py
external/PAE/pae_with_generator/configs/DiT_B3MeanFlowTiny_PAE_DINOv2L_d32_smoke.yaml
external/PAE/pae_with_generator/transport/__init__.py
```

说明：

- `meanflow_lightningdit.py`：复用官方 `LightningDiT` block/embedding/patchify/final-layer 组件，新增 B3 MeanFlow forward signature：`u_theta(z_t, r, t, y, force_drop_ids=None)`。
- `meanflow_transport.py`：实现 B3/global MeanFlow loss、sample-level `(r,t)`、fp32 JVP target、bf16 backbone forward、FD audit、memory telemetry。
- `train_meanflow_dit.py`：新增单卡 Phase0/B3 trainer；保留 EMA 但不用于 JVP target。
- `transport/__init__.py`：仅增加 optional `torchdiffeq` guard，使缺少 `torchdiffeq` 时仍可 import MeanFlow transport；标准 `create_transport()` 若被调用仍会显式报缺依赖。

## 复现实验命令

### 1. 静态检查

```bash
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:${PYTHONPATH:-} \
python -m py_compile \
  external/PAE/pae_with_generator/transport/meanflow_transport.py \
  external/PAE/pae_with_generator/train_meanflow_dit.py \
  external/PAE/pae_with_generator/models/meanflow_lightningdit.py \
  external/PAE/pae_with_generator/transport/__init__.py
```

### 2. 短训 smoke

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-production-trainer-integration
PYTHONPATH=/workspace/PDM/external/PAE/pae_with_generator:${PYTHONPATH:-} \
python /workspace/PDM/external/PAE/pae_with_generator/train_meanflow_dit.py \
  --config "$EXP/configs/b3_meanflow_tiny_smoke.yaml" \
  2>&1 | tee "$EXP/logs/smoke_b3_meanflow_tiny_prod_trainer.log"
```

最终日志：

```text
[2026-05-27 13:53:13] DONE gate practical=True strict=True final_loss=3.580087661743164 loop_peak=63.06005859375MB
```

## 最终指标

| 指标 | 值 |
|---|---:|
| gate practical | `True` |
| gate strict | `True` |
| completed steps | `8 / 8` |
| final loss | `3.580087661743164` |
| FD rel err full JVP, last/max | `8.228138473050667e-05` |
| all-`r=t` degen target-v max abs | `0.0` |
| actual `r=t` samples target-v max abs | `0.0` |
| mean realized `r=t` fraction | `0.6875` |
| bf16-vs-fp32 fwd rel mean | `0.005162052709372032` |
| bf16-vs-fp32 fwd rel min/max | `0.004896900220522285 / 0.005438045536788979` |
| train target JVP peak memory max | `59.6083984375 MB` |
| bf16 backbone fwd peak memory max | `60.61767578125 MB` |
| backward peak memory max | `63.06005859375 MB` |
| FD audit JVP peak memory max | `62.90087890625 MB` |
| FD audit central-FD peak memory max | `47.48583984375 MB` |
| train loop peak memory | `63.06005859375 MB` |
| target detached all steps | `True` |
| JVP param source all live | `True` |
| EMA used for target JVP | `False` |
| sample-level sampler | `True` |
| mixed modes ok | `True` |
| no NaN/Inf | `True` |

FD audit 明细：

| step | fd_rel_err | degen target-v max | forced non-degen | JVP MB | FD MB | r=t frac |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `4.5551111559665746e-05` | `0.0` | `False` | `62.89990234375` | `47.48486328125` | `0.75` |
| 4 | `4.852922736423381e-05` | `0.0` | `True` | `62.90087890625` | `47.48583984375` | `0.75` |
| 8 | `8.228138473050667e-05` | `0.0` | `True` | `62.90087890625` | `47.48583984375` | `0.75` |

解释：step 4/8 的 FD audit 原始随机 batch 全为 `r=t`；audit 逻辑在不改变训练 sampler 的前提下强制一个 audit sample 为 `r<t`，避免 FD/JVP 都为 0 而掩盖 JVP 正确性。

## 环境

- Date: `2026-05-27`
- PyTorch: `2.8.0+cu128`
- GPU: `NVIDIA RTX PRO 6000 Blackwell Server Edition`
- PAE root: `/workspace/PDM/external/PAE/pae_with_generator`
- PAE git commit recorded by trainer: `51f8fa6`
- SDPA: math only
- TF32: disabled

## 范围与限制

这是 Phase 0 / B3 trainer integration smoke，不是完整 ImageNet 长训，不声明 FID/FDr。当前 trainer 是单进程/单卡路径；多卡分布式长训、真实 ImageNet latent 数据吞吐、FID/FDr 评估与 checkpoint/resume 策略仍是后续 Phase 0/Phase 3 工作。
