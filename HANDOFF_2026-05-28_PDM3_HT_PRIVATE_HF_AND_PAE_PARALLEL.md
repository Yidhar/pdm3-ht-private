<!-- PAE_PUBLIC_CACHE_READY_UPDATE_20260528T062420Z -->

> **状态更新 `2026-05-28T06:25:31+00:00`**：PAE DINOv2-L d32 full latent cache 已完成并上传到公开 dataset repo `LAXMAYDAY/pdm-pae-dinov2l-d32-imagenet256-train-full`，远端校验 `313` shards / `1,281,167` samples / `41.992 GB`。旧文档中“PAE partial cache / 需继续构建”的描述已过期。B3 full-cache run 已启动，PID `31771`，log `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/fullcache_realdata_20260528T062419Z.log`。最新补充文档见 `HANDOFF_2026-05-28_PAE_PUBLIC_CACHE_READY_AND_B3_STARTED.md`。

# PDM-3-HT 交接文档：私有 HF 发布、FLUX.2 VAE latent 上传、PAE 并行训练准备

- 生成时间：`2026-05-28T02:24:21+00:00`
- 工作目录：`/workspace/PDM`
- HF 用户：`LAXMAYDAY`
- 当前目标：把可共享的工程代码/任务记录/小日志放到私有 HF；大数据 latent 单独进 dataset repo；另一台机器可并行接 PAE-backed B3/MeanFlow 训练或继续构建 PAE latent。

## 0. 最重要的约束

1. **不要把整个 `/workspace/PDM` 上传。** 本机 `data/` 约数百 GB，里面包含 raw/cropped/cache/latent；`external/` 也不进代码 repo。
2. 私有 code repo 只放：任务书、研究记录、实验脚本、配置、notes、少量 handoff 日志和状态 JSON。
3. latent 必须单独 dataset repo；当前已单独建立并启动上传的是 **FLUX.2 VAE latent** repo。
4. PAE latent 目前是 **partial cache**，不能冒充 full ImageNet-1k cache；等 full cache 完成后再决定是否另开 PAE latent dataset repo。
5. raw ImageNet parquet、cropped uint8 cache、checkpoint/权重、完整日志、tmp/lock 文件都不要上传到 code repo。

## 1. 私有 HF 仓库状态

| 用途 | repo | 类型 | private | 状态 |
|---|---|---:|---:|---|
| 过滤后的工程代码/任务记录/小日志 | [`LAXMAYDAY/pdm3-ht-20260528-code`](https://huggingface.co/LAXMAYDAY/pdm3-ht-20260528-code) | model | yes | 已上传 filtered snapshot；本 handoff 会作为补充上传 |
| FLUX.2 VAE ImageNet-256 latent cache | [`LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents`](https://huggingface.co/datasets/LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents) | dataset | yes | repo 已创建；上传 supervisor 正在等待 full cache 完成后自动上传 |

code repo 已上传记录：

```text
/workspace/PDM/.hf_publish/code_upload_result.json
```

FLUX.2 latent 上传 supervisor 启动记录：

```text
/workspace/PDM/.hf_publish/flux2_vae_latents_upload_launch.json
/workspace/PDM/.hf_publish/upload_flux2_vae_latents.py
```

## 2. 当前 cache/data 状态快照

目标 ImageNet train 样本数：`1,281,167`。

| cache | path | completed shards | completed samples | 进度 | last | shape/key | size | tmp/bad |
|---|---|---:|---:|---:|---|---|---:|---|
| cropped uint8 ImageNet-256 | `/workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors` | 313 | 1,281,167 | 100.00% | `images_uint8_shard000312.safetensors` | last_shape `[3215, 3, 256, 256]` keys `['images', 'labels', 'source_indices']` | 251.908 GB | tmp 0 / bad 0 |
| FLUX.2 VAE latents | `/workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full` | 290 | 1,187,840 | 92.72% | `latents_rank00_shard000289.safetensors` | last_shape `[4096, 32, 32, 32]` keys `['labels', 'latents', 'latents_flip']` | 155.703 GB | tmp 0 / bad 0 |
| PAE DINOv2-L d32 latents | `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full` | 89 | 364,544 | 28.45% | `latents_rank00_shard000088.safetensors` | last_shape `[4096, 32, 16, 16]` keys `['labels', 'latents', 'latents_flip']` | 11.948 GB | tmp 0 / bad 0 |

解释：

- cropped uint8 cache 已经是完整 ImageNet train 256x256 ADM-style center crop，可作为 PAE/FLUX/Qwen VAE cache 的共同输入。
- FLUX.2 VAE cache 正在编码，latent shape 是 `[N,32,32,32]`，与 PAE 的 `[N,32,16,16]` 不同；训练配置不能直接混用。
- PAE cache 当前只有 partial shards，可用于工程 smoke/loader 测试；full train 需要补到 `1,281,167`。

完整机器可读状态 JSON：

```text
/workspace/PDM/.hf_publish/handoff_bundle/state/handoff_state_20260528T022421Z.json
```

## 3. 当前进程和 GPU 调度

截至本 handoff 生成时，相关进程快照见：

```text
/workspace/PDM/.hf_publish/handoff_bundle/log_snippets/process_gpu_hf_snapshot_20260528T022421Z.txt
```

当前关键进程：

- FLUX.2 VAE encoder：`build_diffusers_vae_latents_from_cropped_cache.py`，GPU 负载约满载；输出到 `data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full`。
- FLUX.2 HF upload supervisor：`/workspace/PDM/.hf_publish/upload_flux2_vae_latents.py`，轮询 cache 完整性，full 后只上传 allowlist 文件。
- PAE follow script：`run_pae_from_cropped_follow_full.sh` 当前为暂停/partial 状态；不要和 FLUX.2 encoder 抢同一张 GPU，除非主动切换调度。

监控命令：

```bash
cd /workspace/PDM
ps -eo pid,ppid,stat,etime,%cpu,%mem,rss,cmd | \
  grep -E 'build_diffusers_vae_latents_from_cropped_cache.py|run_flux2_from_cropped_follow_full.sh|upload_flux2_vae_latents.py|run_pae_from_cropped_follow_full.sh|build_pae_latents|train_b3_meanflow' | \
  grep -v grep || true
nvidia-smi
tail -f "$(cat experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/results/flux2_from_cropped_follow_full_outer.logpath)"
tail -f "$(cat experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/results/hf_upload_flux2_vae_latents.logpath)"
```

## 4. 新机器最小恢复路径

### 4.1 拉私有 code repo

二选一：

```bash
# 方法 A：git clone HF model repo
git lfs install
git clone https://huggingface.co/LAXMAYDAY/pdm3-ht-20260528-code /workspace/PDM
cd /workspace/PDM

# 方法 B：hf download 到已有目录
mkdir -p /workspace/PDM
hf download LAXMAYDAY/pdm3-ht-20260528-code --repo-type model --local-dir /workspace/PDM
```

注意：code repo 是 filtered snapshot，不含：

- `/workspace/PDM/data`
- `/workspace/PDM/external`
- PAE checkpoint
- ImageNet raw/cropped cache
- latent shard / safetensors 大文件

所以新机器训练 PAE-backed B3 前，必须准备下面几类依赖。

### 4.2 准备 Python / HF auth / 依赖

```bash
hf auth login
python -m pip install -U huggingface_hub safetensors torch torchvision diffusers transformers accelerate pyyaml pillow numpy tqdm
```

若沿用当前容器环境，优先直接复用当前环境/镜像，避免版本漂移。

### 4.3 准备 `external/PAE` 和 PAE checkpoint

B3 real-data trainer 当前会 import：

```text
/workspace/PDM/external/PAE/pae_with_generator
```

PAE cache builder 需要：

```text
/workspace/PDM/external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml
/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt
```

这些没有上传到 code repo。新机器需要通过内部 rsync/scp、原始上游 repo、或私有权重库恢复。恢复后建议跑一次随机 self-test 或 smoke cache。

## 5. PAE cache / PAE-backed B3 的推荐推进路径

### 路径 A：继续/完成 PAE latent cache，再跑 full B3

如果新机器能拿到完整 cropped cache：

```text
/workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors
```

则直接跑 PAE builder，从 cropped uint8 cache 生成 PAE latent：

```bash
cd /workspace/PDM
EXP=experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized
bash "$EXP/scripts/run_pae_from_cropped_follow_full.sh" 2>&1 | tee "$EXP/logs/pae_from_cropped_follow_full_new_machine_$(date -u +%Y%m%dT%H%M%SZ).log"
```

默认关键配置在：

```text
experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized/configs/optimized_cache.env
```

关键变量：

```bash
EXPECTED_TRAIN_SIZE=1281167
CROP_OUT=/workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors
LATENT_FULL_OUT=/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full
PAE_BATCH_SIZE=${PAE_BATCH_SIZE:-1024}
LATENT_SHARD_SIZE=${LATENT_SHARD_SIZE:-4096}
MODEL_DTYPE=${MODEL_DTYPE:-bf16}
SAVE_DTYPE=${SAVE_DTYPE:-bf16}
```

如果只想 restart/resume 单次 builder 而不是 follow loop：

```bash
cd /workspace/PDM
python experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized/scripts/build_pae_latents_from_cropped_cache.py \
  --crop-cache-dir /workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors \
  --output-dir /workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full \
  --pae-config /workspace/PDM/external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml \
  --pae-ckpt /workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt \
  --image-size 256 --batch-size 1024 --latent-shard-size 4096 \
  --max-samples 1281167 --device cuda:0 --model-dtype bf16 --save-dtype bf16 --resume
```

### 路径 B：没有 cropped cache，但有 ImageNet/HF 授权

先重建 cropped uint8 cache，再跑 PAE builder。raw parquet/cropped uint8 cache 因为体积和授权原因没有上传到 code repo。

```bash
cd /workspace/PDM
EXP=experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized
bash "$EXP/scripts/run_cropped_full_cache.sh"
bash "$EXP/scripts/run_pae_from_cropped_follow_full.sh"
```

### 路径 C：只做训练工程 smoke

如果新机器只拿到 partial PAE latent 或只需验证训练循环，可先跑 B3 smoke/full-loader scan，不要宣称 full result。

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_smoke4096.sh
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/scan_fullcache_dataset.sh
```

## 6. PAE cache 完整性扫描

快速扫描命令：

```bash
cd /workspace/PDM
python - <<'PY'
from pathlib import Path
from safetensors import safe_open
p=Path('/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full')
files=sorted(p.glob('latents_rank00_shard*.safetensors'))
total=0; bad=[]; last_shape=None
for f in files:
    try:
        with safe_open(str(f), framework='pt', device='cpu') as sf:
            total += int(sf.get_tensor('labels').numel())
            if f == files[-1]:
                last_shape=list(sf.get_tensor('latents').shape)
    except Exception as e:
        bad.append((f.name, repr(e)))
tmp=sorted([x.name for x in list(p.glob('.*tmp*'))+list(p.glob('*.tmp*'))+list(p.glob('*.lock'))])
print({'files':len(files),'samples':total,'expected':1281167,'ready':total==1281167,'last':files[-1].name if files else None,'last_shape':last_shape,'tmp':tmp[:10],'bad':bad[:10]})
PY
```

B3 trainer 自带 scan：

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/scan_fullcache_dataset.sh
```

full cache 判定：

- samples 必须等于 `1,281,167`。
- 不应有 `.tmp` / `.lock`。
- PAE latent shape 应为 `[N,32,16,16]`。
- 每 shard 通常 `4096` samples；最后一个 shard 应为余数 `3215`。

## 7. B3 MeanFlow real-data trainer 状态

目录：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
```

重要文件：

```text
README.md
TASK.md
configs/b3_meanflow_realdata_smoke4096.yaml
configs/b3_meanflow_realdata_full.yaml
scripts/train_b3_meanflow_realdata.py
scripts/run_smoke4096.sh
scripts/scan_fullcache_dataset.sh
notes/EXPERIMENT_RECORD.md
```

已完成 smoke：

- `smoke_realdata_4096` passed。
- 4096 real PAE latents, shape `[32,16,16]`, labels `0..999`。
- class-conditioning 已接入：labels 作为 `model_kwargs["y"]`。
- mixed precision：bf16 backbone + fp32 live-param JVP target。
- JVP 使用 live 参数，不使用 EMA target。
- r=t sampler：`equal_prob=0.75`；tiny smoke 实测约 `0.7917`。
- FD audit last rel err：约 `6.78e-05`。
- r=t degenerate target-v max abs：`0.0`。
- checkpoint/eval latent sample 正常；FID hook 有接口但 smoke 未配置 FID command。

smoke 命令：

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_smoke4096.sh
```

full-template 命令，**只在 PAE full cache ready 后运行**：

```bash
cd /workspace/PDM
python experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py \
  --config experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_full.yaml \
  2>&1 | tee experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/fullcache_realdata_$(date -u +%Y%m%dT%H%M%SZ).log
```

如果要真正评估 FID，需要在 YAML 中设置：

```yaml
eval:
  enabled: true
  every_steps: 10000
  fid_command_template: "python path/to/fid.py --samples {sample_dir} --out {output_dir} --step {step}"
```

## 8. FLUX.2 VAE latent repo 后续

上传 supervisor 会在 full cache 完整后自动：

1. 检查 samples=`1,281,167`、shards=`313`、无 tmp/lock、safetensors 可读。
2. 检查 keys/shape：`latents` `[N,32,32,32]`，`latents_flip` `[N,32,32,32]`，`labels` `[N]`。
3. 写 dataset README。
4. 只上传 allowlist：`README.md`、`run_config.json`、`build_summary.json`、`progress.jsonl`、`manifest.jsonl`、`latents_rank00_shard*.safetensors`。

监控上传：

```bash
cd /workspace/PDM
tail -f "$(cat experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/results/hf_upload_flux2_vae_latents.logpath)"
```

上传完成后会写：

```text
/workspace/PDM/.hf_publish/flux2_vae_latents_upload_result.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/results/hf_flux2_vae_latents_upload_result_20260528.json
```

另一台机器下载 FLUX.2 VAE latents：

```bash
mkdir -p /workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full
hf download LAXMAYDAY/pdm3-ht-20260528-flux2-vae-latents \
  --repo-type dataset \
  --local-dir /workspace/PDM/data/vae_latents/AutoencoderKLFlux2_d32/imagenet256_train_full
```

提醒：FLUX.2 latent shape `[32,32,32]`，PAE latent shape `[32,16,16]`；B3 config 需要 backend-specific latent shape/model patch settings。

## 9. 当前机器上 PAE 继续跑的注意事项

如果决定在当前机器继续 PAE cache，先确认 FLUX.2 encoding/upload 不再占 GPU，或手动做资源调度。

当前已有 partial PAE shards：

- samples：`364,544` / `1,281,167`
- shards：`89`
- output：`/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full`

若暂停的 follow PID 仍在，可恢复：

```bash
kill -CONT 1631005
```

若 PID 已不存在，重启：

```bash
cd /workspace/PDM
EXP=experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized
LOG="$EXP/logs/pae_from_cropped_follow_full_restart_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "/workspace/PDM/$LOG" > "$EXP/results/pae_from_cropped_full.logpath"
setsid bash "$EXP/scripts/run_pae_from_cropped_follow_full.sh" > "$LOG" 2>&1 < /dev/null &
echo $! > "$EXP/pae_from_cropped_full.pid"
```

## 10. 研究叙事 / paper framing 锁定

不要再把 PAE 写成核心创新组件。当前更诚实、更稳的 framing：

- 核心创新：**per-patch HT spatial-time mechanism** + **MeanFlow trajectory mechanism**。
- PAE 是 latent backend / baseline 工具，不是 novelty 本体。
- VAE backend 应该可换：PAE、FLUX.2 VAE、Qwen Image VAE。
- Proposition 1 不写成 “PAE 特有保证”，而写成 “manifold-regularized VAE/backbone 预计提供类似 JVP 稳定性；PAE 是已验证实例”。
- backend 对比要诚实报告：
  - A：三个 VAE 都 work，PAE 最好（ImageNet-tuned）→ 强调普适性 + backend specialization。
  - B：三个 VAE 数字接近 → 最强 ablation。
  - C：FLUX/Qwen 不 work 或明显变差 → VAE-agnostic claim 必须收窄。

## 11. handoff bundle 内容

本 handoff bundle 在本机：

```text
/workspace/PDM/.hf_publish/handoff_bundle
```

上传到 code repo 后位于：

```text
handoff/
HANDOFF_2026-05-28_PDM3_HT_PRIVATE_HF_AND_PAE_PARALLEL.md
```

小日志/状态文件：

| 文件 | 说明 |
|---|---|
| `handoff/log_snippets/b3_smoke4096_tail_20260528T022421Z.log` | b3_smoke4096_tail; source `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/010_smoke4096.log`; 4414 bytes |
| `handoff/log_snippets/flux2_vae_encode_tail_20260528T022421Z.log` | flux2_vae_encode_tail; source `/workspace/PDM/experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/logs/flux2_from_cropped_follow_full_setsid_20260527T194322Z.outer.log`; 78639 bytes |
| `handoff/log_snippets/flux2_vae_hf_upload_tail_20260528T022421Z.log` | flux2_vae_hf_upload_tail; source `/workspace/PDM/experiments/2026-05-27-pdm3-ht-flux2dev-vae-latent-cache/logs/hf_upload_flux2_vae_latents_20260528T015927Z.log`; 4570 bytes |
| `handoff/log_snippets/pae_from_cropped_tail_20260528T022421Z.log` | pae_from_cropped_tail; source `/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized/logs/010_pae_from_cropped_follow_full_gap65536_20260527T185110Z.log`; 31794 bytes |
| `handoff/log_snippets/process_gpu_hf_snapshot_20260528T022421Z.txt` | process_gpu_hf_snapshot; source `generated`; 1460 bytes |
| `handoff/state/handoff_state_20260528T022421Z.json` | 本次状态快照 JSON |
| `handoff/state/code_upload_result.json` | code repo filtered upload 记录 |
| `handoff/state/flux2_vae_latents_upload_launch.json` | FLUX.2 latent upload supervisor 启动记录 |

## 12. 下一步 checklist

当前机器：

- [ ] 继续监控 FLUX.2 VAE encoding，预计完成后 upload supervisor 自动上传 dataset repo。
- [ ] FLUX.2 upload 完成后检查 `.hf_publish/flux2_vae_latents_upload_result.json`。
- [ ] 避免同时用同一 GPU 跑 PAE 和 FLUX.2；若 FLUX 完成，可恢复/重启 PAE cache。
- [ ] PAE full cache 达到 `1,281,167` 后再运行 B3 full train。

新机器：

- [ ] 拉私有 code repo。
- [ ] 恢复 `external/PAE` 和 PAE checkpoint。
- [ ] 准备 cropped uint8 cache 或 raw ImageNet/HF 授权。
- [ ] 完成 PAE latent full cache 或先跑 smoke/scan。
- [ ] PAE cache 完整后跑 B3 real-data full config。
- [ ] 如果要做 FLUX/Qwen VAE backend ablation，先改 backend-specific latent shape/config，再跑同样 B3/HT/MeanFlow 训练流程。

---

这份文档和 handoff bundle 可以放在私有库；它们只包含文档、状态 JSON 和短日志 tail，不含 raw data、latent shards、checkpoint 或权重。
