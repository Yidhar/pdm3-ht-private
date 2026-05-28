# PDM-3-HT 交接补充：PAE public cache ready，B3 full-cache 已启动

- 更新时间：`2026-05-28T06:25:31+00:00`
- 工作目录：`/workspace/PDM`
- HF 用户：`LAXMAYDAY`
- 本补充 supersedes 旧 handoff 中“PAE latent partial cache / 需继续构建 PAE cache”的状态。
- 当前目标：直接进入 **B3 MeanFlow real-data full-cache run**，同时保留新机器 cold restore 路径。

## 1. 状态结论

PAE 侧已经 ready：

| 项 | 状态 |
|---|---|
| PAE external code | 本机存在：`/workspace/PDM/external/PAE/pae_with_generator` |
| PAE config | 本机存在：`/workspace/PDM/external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml` |
| PAE ckpt | 本机存在：`/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt` |
| PAE full latent cache | 本机完整，ready=true |
| PAE public HF cache repo | 已创建、公开、远端校验通过 |
| B3 full-cache run | 已按 full config 直接启动 |

## 2. PAE public cache repo

公开 dataset repo：

```text
https://huggingface.co/datasets/LAXMAYDAY/pdm-pae-dinov2l-d32-imagenet256-train-full
```

远端校验：

```json
{
  "repo_id": "LAXMAYDAY/pdm-pae-dinov2l-d32-imagenet256-train-full",
  "private": false,
  "sha": "963c63d140712dfd300d837867e779f7c5d53a68",
  "total_files": 317,
  "safetensors_count": 313,
  "safetensors_bytes": 41991766672,
  "has_README": true,
  "has_cache_summary": true,
  "has_manifest": true
}
```

说明：

- repo type: `dataset`
- visibility: `public`
- 上传内容仅为 PAE latent cache 与 README/manifest/summary。
- 没有上传 raw ImageNet parquet。
- 没有上传 cropped uint8 cache。
- 没有上传 checkpoint / PAE 权重。

## 3. 本机 PAE full cache 校验

本机路径：

```text
/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full
```

本机扫描结果：

```json
{
  "files": 313,
  "samples": 1281167,
  "expected": 1281167,
  "ready": true,
  "bytes": 41991766672,
  "gb_decimal": 41.992,
  "first": "latents_rank00_shard000000.safetensors",
  "last": "latents_rank00_shard000312.safetensors",
  "first_shape": [
    4096,
    32,
    16,
    16
  ],
  "last_shape": [
    3215,
    32,
    16,
    16
  ],
  "keys": [
    "labels",
    "latents",
    "latents_flip"
  ],
  "bad_count": 0,
  "tmp_count": 0
}
```

full cache 判定：

- `samples == 1,281,167`
- `files == 313`
- latent shape `[N, 32, 16, 16]`
- first shard `[4096, 32, 16, 16]`
- last shard `[3215, 32, 16, 16]`
- keys: `labels`, `latents`, `latents_flip`
- no tmp/lock/bad files

## 4. B3 full-cache run 已启动

启动时间：`2026-05-28T06:24:20+00:00`

PID：

```text
31771
```

PID file：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/fullcache_realdata.pid
```

Config：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_full.yaml
```

Log：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/fullcache_realdata_20260528T062419Z.log
```

训练命令等价于：

```bash
cd /workspace/PDM
PAE_GEN_ROOT=/workspace/PDM/external/PAE/pae_with_generator PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1 XFORMERS_DISABLED=1 DISABLE_XFORMERS=1 python experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py   --config experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_full.yaml
```

当前 full config 要点：

```text
data_path: /workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full
expected_total: 1281167
model_type: B3MeanFlowLightningDiT-XL/1
in_chans: 32
max_steps: 100000
global_batch_size: 4
checkpoint_every: 1000
resume_from: auto
eval.every_steps: 10000
```

## 5. 监控命令

进程/GPU：

```bash
cd /workspace/PDM
PID=$(cat experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/fullcache_realdata.pid)
ps -p "$PID" -o pid,ppid,stat,etime,%cpu,%mem,rss,cmd
nvidia-smi
```

日志：

```bash
cd /workspace/PDM
LOG=$(cat experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_latest.logpath)
tail -f "$LOG"
```

metrics：

```bash
cd /workspace/PDM
tail -f experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/metrics.jsonl
```

checkpoint：

```bash
ls -lah experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints
```

## 6. 新机器恢复：最短路径

如果只需要复现 B3 full-cache run，新机器需要三块：

1. 私有 code repo：

```bash
hf download LAXMAYDAY/pdm3-ht-20260528-code --repo-type model --local-dir /workspace/PDM
```

2. PAE runtime artifacts：

```text
/workspace/PDM/external/PAE/pae_with_generator
/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt
```

这两项目前不在公开 cache repo 内；建议通过私有 artifact repo 或内部 rsync 恢复。

3. PAE public latent cache：

```bash
mkdir -p /workspace/PDM/data/_hf_downloads/pdm-pae-dinov2l-d32-imagenet256-train-full
hf download LAXMAYDAY/pdm-pae-dinov2l-d32-imagenet256-train-full   --repo-type dataset   --local-dir /workspace/PDM/data/_hf_downloads/pdm-pae-dinov2l-d32-imagenet256-train-full   --include "README.md"   --include "cache_summary.json"   --include "manifest.jsonl"   --include "pae_latents/imagenet256_train_full/*.safetensors"

mkdir -p /workspace/PDM/data/pae_latents/PAE_DINOv2L_d32
ln -sfn /workspace/PDM/data/_hf_downloads/pdm-pae-dinov2l-d32-imagenet256-train-full/pae_latents/imagenet256_train_full   /workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full
```

恢复后检查：

```bash
cd /workspace/PDM
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/scan_fullcache_dataset.sh
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_smoke4096.sh
```

## 7. 不要做的事

- 不要再把 PAE cache 当 partial 处理。
- 不要重建 PAE latent cache，除非要验证 builder。
- 不要上传 raw ImageNet / cropped uint8 / checkpoint 到公开 repo。
- 不要把 PAE 权重混入 public latent cache repo。
- 不要同时启动第二个同 config 的 B3 full run；先检查 PID。

## 8. 相关状态文件

```text
handoff/state/pae_full_cache_verified_after_hf_upload.json
handoff/state/pae_public_cache_hf_remote_verified.json
handoff/state/b3_fullcache_latest_launch.json
```

## 9. 当前下一步

1. 监控 B3 startup 是否完成 dataset snapshot/model build。
2. 观察第一个 metrics/step 是否 finite。
3. 第一个 checkpoint 后确认 `latest.pt` 正常。
4. 后续按 `metrics.jsonl` 和 GPU memory 决定是否调整 batch/config。

<!-- B3_STARTUP_MONITOR_20260528T0633Z -->

## 10. B3 startup monitor update

更新时间：`2026-05-28T06:33:28+00:00`

B3 full-cache run 已通过启动阶段和首个 checkpoint gate：

```json
{
  "pid": 31771,
  "last_train_step": 1298,
  "last_train_loss": 0.5098568201065063,
  "last_train_loss_finite": true,
  "last_train_grad_finite": true,
  "last_fd_step": 1000,
  "last_fd_rel_err": 7.258136401943587e-05,
  "last_fd_no_nan_or_inf": true,
  "first_checkpoint_ready": true
}
```

首个 checkpoint：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00001000.pt
```

最新 metrics：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/metrics.jsonl
```

结论：dataset snapshot/model build/FD audit/train loop/checkpoint write 均已通过。继续正常监控即可。

<!-- B3_BACKGROUND_MONITOR_20260528T0634Z -->

## 11. Background monitor

轻量监控脚本已启动，每 `300s` 采样一次 B3 process/GPU/latest metric/checkpoint：

```text
monitor_pid: 35172
monitor_log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/b3_fullcache_monitor_20260528T063433Z.jsonl
monitor_script: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/monitor_b3_fullcache.sh
```

查看：

```bash
tail -f /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/b3_fullcache_monitor_20260528T063433Z.jsonl
```

<!-- B3_H100_SPEEDUP_20260528T0658Z -->

## 12. H100 speedup / utilization update

更新时间：`2026-05-28T07:00:26.394320+00:00`

用户观察到 H100 显存/功耗未跑满后，已把 B3 full-cache 训练切到 H100 高吞吐配置并继续后台训练。

### 当前 active run

```text
pid: 39137
config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_full_fast_h100.yaml
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/fullcache_realdata_fast_h100_b32_default_jvp_math_20260528T065556Z.log
metrics: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/metrics.jsonl
output: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template
latest_checkpoint: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00003000.pt
monitor_pid: 39473
monitor_log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/b3_fullcache_monitor_20260528T065808Z.jsonl
monitor_interval_sec: 120
```

当前配置要点：

```yaml
train.global_batch_size: 32
train.sdpa_kernel: default
train.allow_tf32: true
train.checkpoint_every: 2000
train.resume_from: auto
train.restore_rng: false
data.num_workers: 12
data.prefetch_factor: 6
meanflow.fd_audit_every: 2000
```

`restore_rng=false` 仅用于绕开 resume 时 checkpoint RNG state 被 `map_location=cuda` 后 `torch.set_rng_state` 要 CPU ByteTensor 的兼容问题；model / EMA / optimizer 仍从 `latest.pt` 恢复。

### 关键代码补丁

`external/PAE/pae_with_generator/transport/meanflow_transport.py` 已加 `_math_sdpa_context`，只在 MeanFlow `full_jvp()` / `central_fd()` 路径强制 math SDPA。

原因：直接全局 `sdpa_kernel=default` 会在 step=3001 触发 PyTorch forward-AD 错误：efficient/flash SDPA 不支持 `torch.func.jvp`。补丁后：

- JVP/FD target path：math SDPA，保证 forward-mode AD/FD audit 正确；
- 普通训练 bf16 forward/backward：走全局 default/flash-enabled SDPA，提高 H100 利用率。

### 吞吐对比（samples/sec）

| 配置 | batch | mean sec/step | samples/sec | peak MB | 备注 |
|---|---:|---:|---:|---:|---|
| baseline | 4 | 0.378219 | 10.576 | 15792.3 | 原始 bs4/math/TF32 off |
| b16 patched | 16 | 0.470641 | 33.996 | 19837.3 | default + JVP math patch |
| current b32 | 32 | 0.553180 | 57.847 | 27664.2 | active run |

当前 b32 相对 baseline 样本吞吐提升约 `5.47x`。

最近 GPU sample：

```text
0, NVIDIA H100 80GB HBM3, 30315 MiB, 81559 MiB, 96 %, 544.95 W, 700.00 W
```

### 注意

- 这个提升按 `samples/sec` 计算；由于 batch 从 4 提到 32，单步耗时会变长，但同等样本预算快很多。
- effective batch 已改变，继续沿用 `lr=2e-4`；如后续 loss/质量异常，可退到 b16 或调整 LR/schedule。
- 当前从 `step_00003000.pt` 恢复，`checkpoint_every=2000`，所以下一个预期 checkpoint 是 `step_00004000.pt`。

<!-- B3_H100_B96_SKIP_EQ_JVP_20260528T072336Z -->

## 13. H100 speedup follow-up: b96 + exact r=t JVP skip active

更新时间：`2026-05-28T07:23:36Z`

当前已从最初的低吞吐配置继续升级为 **b96 H100 fast run**，并在 MeanFlow target path 中加入了一个数学等价的优化：对采样到 `r == t` 的样本跳过 JVP，因为此时 `delta=t-r=0`，JVP tangent 为零，`du==0`，所以 target 精确等于 `v`。这不是近似降质。

### Active run

```text
pid: 41529
config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_full_fast_h100_b96.yaml
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/fullcache_realdata_fast_h100_b96_skip_eq_jvp_20260528T071553Z.log
metrics: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/metrics.jsonl
output: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template
latest_durable_checkpoint: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00004000.pt
monitor_pid: 42029
monitor_log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/b3_fullcache_monitor_20260528T071828Z.jsonl
monitor_interval_sec: 120
```

### 当前配置要点

```yaml
train.global_batch_size: 96
train.sdpa_kernel: default
train.allow_tf32: true
train.checkpoint_every: 2000
train.resume_from: auto
train.restore_rng: false
data.num_workers: 12
data.prefetch_factor: 6
data.persistent_workers: true
meanflow.precision_recipe: bf16_backbone_fp32_jvp
meanflow.equal_prob: 0.75
meanflow.fd_audit_every: 2000
```

### 当前吞吐/资源快照

```json
{"count": 548, "first_step": 4001, "last": {"step": 4548, "created_at_utc": "2026-05-28T07:23:34Z", "batch_size": 96, "elapsed_sec": 0.7845330950804055, "loss": 0.38225769996643066, "peak_memory_mb": 59063.015625, "target_jvp_sec": 0.24004452815279365, "target_jvp_effective_batch": 24, "target_jvp_skipped_equal_batch": 72, "backbone_forward_sec": 0.18115895194932818, "backward_sec": 0.30305137066170573, "actual_rt_samples_target_minus_v_max_abs": 0.0}, "last20": {"n": 20, "mean_sec": 0.7809267731383442, "samples_s": 122.93086023187634, "step0": 4529, "step1": 4548}, "last50": {"n": 50, "mean_sec": 0.7848150786850602, "samples_s": 122.32180880220321, "step0": 4499, "step1": 4548}, "last100": {"n": 100, "mean_sec": 0.7902820288483053, "samples_s": 121.47562072226648, "step0": 4449, "step1": 4548}, "last_all": {"n": 548, "mean_sec": 0.8001426122776729, "samples_s": 119.97861197109346, "step0": 4001, "step1": 4548}}
```

GPU snapshot:

```text
0, NVIDIA H100 80GB HBM3, 70371 MiB, 81559 MiB, 100 %, 592.00 W, 700.00 W, 63
```

对比基线：原始 bs4/math SDPA/TF32 off 约 **10.5 samples/s**；当前 b96 最近窗口约 **120 samples/s**，约 **11x+** 样本吞吐提升。GPU 利用率已在采样中接近/达到 100%，显存约 70GB/80GB。功耗没有满 700W 属正常现象，当前主要以 GPU util 和吞吐判断已接近饱和。

### 注意

- 当前最新持久 checkpoint 仍是 `step_00004000.pt`；b96 run 从 4000 resume，下一持久化目标为 `step_00006000.pt`。
- 在 6000 checkpoint 落盘前，不建议为了继续试 b112/b128 中断当前 run，否则 4000 之后进度会丢失。
- 如果 b96 在 6000 前 OOM/非有限，回退 b64 fast config；否则 6000 之后再考虑小步试 b104/b112 或降低每步诊断同步频率来继续挤吞吐。

<!-- GITHUB_HF_ARTIFACTS_PROGRESS_20260528T072956Z -->

## 14. GitHub code sync + HF model artifacts routing note

更新时间：`2026-05-28T07:29:55Z`

用户指定：

- 代码同步目标：<https://github.com/Yidhar/pdm3-ht-private>
- 模型产物统一目标：<https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts>
- 交接沟通方式：文档留言。

当前已完成：

1. b96 训练继续后台跑，未中断；当前状态：`{"count": 1019, "step": 5019, "time": "2026-05-28T07:29:55Z", "loss": 0.34228333830833435, "last100_samples_s": 121.12723326976942, "last100_mean_sec": 0.7925550465285778, "to_6000": 981, "effective_jvp": 20, "skipped_jvp": 76, "rt_degen_max_abs": 0.0}`。
2. 已新增/更新 `.gitignore` 与 `ARTIFACTS.md`，明确 GitHub 只放 code/config/docs/handoff，模型 checkpoint/weights/eval outputs 统一走 HF artifacts repo。
3. 已写交接留言：`handoff/PROGRESS_2026-05-28_GITHUB_HF_ARTIFACTS.md`。
4. 已准备干净代码 staging repo：`/workspace/pdm3-ht-private-sync`，排除 `data/`、`.cache/`、`experiments/**/results/`、`logs/`、checkpoint、`.safetensors`、`.pt` 等大文件/产物。

当前 GitHub push 阻塞：本机没有 GitHub 凭据；`gh auth status` 未登录，HTTPS/SSH 都无法访问私库。拿到 token/SSH key 后，在 staging repo 内执行：

```bash
cd /workspace/pdm3-ht-private-sync
git push -u origin main
```

HF artifacts repo 已验证存在且为 private：`LAXMAYDAY/pdm3-ht-model-artifacts`。当前未上传训练 checkpoint；先等 b96 到 `step_00006000.pt` 落盘后再按需发布模型产物。

<!-- GITHUB_HF_ARTIFACTS_PROGRESS_FOLLOWUP_20260528T073131Z -->

### 14.1 Follow-up: local Git commit ready; HF artifacts doc updated

更新时间：`2026-05-28T07:31:31Z`

- GitHub staging repo 已提交本地 commit，路径 `/workspace/pdm3-ht-private-sync`；具体 hash 用 `git -C /workspace/pdm3-ht-private-sync log -1 --oneline` 查看。
- 已生成可手工导入的 bundle；最新 bundle 用 `ls -lh /workspace/pdm3-ht-private-sync_*.bundle` 查看。
- 已尝试 `git push -u origin main`，失败原因仅为本机无 GitHub credential：`fatal: could not read Username for 'https://github.com': terminal prompts disabled`。
- HF model artifacts repo 文档已更新成功；上传了 `handoff/PROGRESS_2026-05-28_GITHUB_HF_ARTIFACTS.md`。最新 repo sha 用 Hugging Face API 查询。

<!-- GITHUB_PUSH_COMPLETE_20260528T073948Z -->

### 14.2 GitHub push complete

更新时间：`2026-05-28T07:39:46Z`

GitHub 授权已完成并已成功推送到：<https://github.com/Yidhar/pdm3-ht-private>

```text
branch: main
pushed_head_before_this_note: 592238cc4e69bbbb0c3a68cdf11309b92f23e425
remote_main_after_push: 592238cc4e69bbbb0c3a68cdf11309b92f23e425
push_result: 9f9db46..592238c main -> main
```

处理细节：远端已有初始 `main`，所以没有 force push；先 fetch/merge `origin/main`，冲突文件采用本地当前 source snapshot，非冲突远端文件保留。push 前 artifact 检查通过，无 data/results/logs/checkpoints/PAE latent/model weight 被纳入 GitHub。

push 期间 b96 训练仍在跑：`{"last_step": 5748, "time": "2026-05-28T07:39:46Z", "loss": 0.3704003691673279, "last100_samples_s": 120.89864530931222, "to_6000": 252, "rt_degen_max_abs": 0.0, "last_fd_step": 4000, "last_fd_rel": 0.11669899379970686, "last_checkpoint_step": 4000}`。

<!-- B3_B96_STEP6000_DURABLE_20260528T074445Z -->

## 15. B3 b96 reached step 6000 durable checkpoint

更新时间：`2026-05-28T07:44:44Z`

b96 H100 run 已通过下一个持久化 gate，并继续训练。

```json
{"last_step": 6088, "time": "2026-05-28T07:44:43Z", "loss": 0.41259926557540894, "last100_samples_s": 118.34274827062414, "last100_mean_sec": 0.811203064005822, "last100_step_range": [5989, 6088], "rt_degen_max_abs": 0.0, "fd_step": 6000, "fd_time": "2026-05-28T07:43:14Z", "fd_rel": 0.08643620461852597, "fd_no_nan_or_inf": true, "fd_degen_target_minus_v": 0.0, "checkpoint_step": 6000, "checkpoint_time": "2026-05-28T07:43:31Z", "checkpoint_path": "/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00006000.pt"}
```

当前最新持久 checkpoint：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00006000.pt
```

FD audit step 6000 已通过，`no_nan_or_inf=true`，`all_r_eq_t_degenerate_target_minus_v_max_abs=0.0`。`latest.pt` 已指向 `step_00006000.pt`。按 artifact routing 约定，此 checkpoint 没有提交到 GitHub；如需发布模型产物，只上传到：<https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts>。


<!-- B3_B96_STEP10000_EVAL_SMOKE_20260528T0842Z -->

## Step 10000 eval/sample pipeline gate complete

更新时间：`2026-05-28T08:44:24Z`

用户要求的 step `10000` 五项已落地：

1. checkpoint: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00010000.pt`
2. EMA latent sample: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/sample_latents.safetensors`，shape `[64, 32, 16, 16]`，finite `True`
3. image-space decoded samples: generated PNG `64` 张 + real-ref PNG `64` 张；grid:
   - `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/images/generated_grid.png`
   - `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/images/real_ref_grid.png`
4. compact smoke metrics（非 official Inception FID/KID）:
   - `compact_fid_rp128=4.0626297`
   - `compact_mmd_rbf_rp128=0.049283276`
   - `compact_kid_poly3_rp128=0.0034318871`
5. train/eval summary: `handoff/STEP10000_EVAL_SUMMARY_2026-05-28.md`

内置 official FID 仍是 `skipped_no_command`，因为 fast config 当前 `eval.fid_command_template` 为空；本次 compact metrics 目标是提前验证 sample→PAE decode→image metrics 线路，避免等到 100k 后才发现 pipeline 未接通。

主训练未中断，当前最后记录：step `10441` @ `2026-05-28T08:44:23Z`，last200 约 `119.135` samples/s（如有）。

FD 风险：step `10000` built-in FD audit finite/no_nan，但 `fd_rel=0.061354188` 仍高于 practical gate `1e-2`；专项 A/B/C eps sweep 脚本已准备好：`experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/audit_fd_jvp_modes.py`。建议等 GPU 空出或在下一个 durable checkpoint 后短暂停训运行，避免与当前 74GB H100 主训抢显存。


<!-- B3_B96_STEP10000_EVAL_SYNCED_20260528T0849Z -->

## Step 10000 eval smoke artifacts synced

更新时间：`2026-05-28T08:46:29Z`

GitHub code/docs sync complete:

```text
repo: https://github.com/Yidhar/pdm3-ht-private
branch: main
commit: 5f53937046bff6e4b51ee30d2b2416ad15ccd93c
commit_msg: Add B3 step10000 eval smoke and FD audit tools
```

Included in GitHub: new decode helper, FD A/B/C audit helper, and handoff summaries. Artifact exclusion check passed before push; no checkpoint/cache/results/logs were committed.

HF model artifacts sync complete:

```text
repo: https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
repo_sha_after_upload: 616b9c47d548936710111e49d619c4fe7ad26273
eval_artifact_dir: b3_meanflow_realdata/fullcache_b96/step_00010000/eval
summary_doc: handoff/STEP10000_EVAL_SUMMARY_2026-05-28.md
```

Uploaded lightweight step-10000 eval artifacts to HF: `sample_latents.safetensors`, `decoded_samples.npz`, generated/real-ref PNGs, grids, `compact_metrics.json`, `compact_metrics.md`, and `eval_record.json`. The multi-GB training checkpoint remains local only unless explicitly publishing checkpoints to HF.

<!-- B3_B96_DEDICATED_FD_AUDIT_DONE_20260528T0916Z -->

## Dedicated FD/JVP A/B/C audit complete

更新时间：`2026-05-28T09:16Z`

专项 FD/JVP audit 已按用户建议在 durable `step_00012000.pt` 后短暂停训执行；audit 使用 `step_00010000.pt`、固定 seed、小 batch `8`、`model.eval()`、`equal_prob=0` 非退化 batch、class dropout 关闭、JVP/FD fp32，并 sweep `eps=[1e-2, 3e-3, 1e-3]`。

Audit 输出：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fd_jvp_dedicated_audit/step_00010000_after_step_12000_20260528T085000Z/fd_jvp_modes_audit.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fd_jvp_dedicated_audit/step_00010000_after_step_12000_20260528T085000Z/fd_jvp_modes_audit.jsonl
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fd_jvp_dedicated_audit/step_00010000_after_step_12000_20260528T085000Z/fd_jvp_modes_audit.md
```

Compact result table:

| mode | TF32 | global SDPA | eps | fd_rel | abs_err | du_norm | fd_norm | no_nan | backbone_rel | degen_target-v |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| A | off | math | 1e-02 | `0.00752493` | `1.15436` | `153.831` | `153.404` | true | `0.00623756` | `0` |
| A | off | math | 3e-03 | `0.000765322` | `0.1177` | `153.831` | `153.792` | true | `0.00623756` | `0` |
| A | off | math | 1e-03 | `0.00095552` | `0.146983` | `153.831` | `153.825` | true | `0.00623756` | `0` |
| B | off | default | 1e-02 | `0.00752493` | `1.15436` | `153.831` | `153.404` | true | `0.00623485` | `0` |
| B | off | default | 3e-03 | `0.000765322` | `0.1177` | `153.831` | `153.792` | true | `0.00623485` | `0` |
| B | off | default | 1e-03 | `0.00095552` | `0.146983` | `153.831` | `153.825` | true | `0.00623485` | `0` |
| C | on | default | 1e-02 | `0.0401946` | `6.18794` | `153.789` | `153.95` | true | `0.00624116` | `0` |
| C | on | default | 3e-03 | `0.125223` | `19.385` | `153.789` | `154.804` | true | `0.00624116` | `0` |
| C | on | default | 1e-03 | `0.348566` | `56.6146` | `153.789` | `162.421` | true | `0.00624116` | `0` |

结论：A/B（TF32 off，math/default SDPA）在 `eps=3e-3/1e-3` 回到 `~7.7e-4–9.6e-4`，`eps=1e-2` 也低于 `1e-2`；C（当前 fast 口径，TF32 on）显著变差且 eps 越小越差。因此 built-in FD 在 fast H100 b96 run 中出现 `0.05–0.11` 量级，主要判断为 **TF32 + finite-difference sensitivity / 诊断口径问题**，不是 JVP 主链路错误。所有模式 `no_nan=true`，`r=t` 退化目标检查严格为 `0`。

训练恢复备注：audit 后第一次普通 `nohup` resume 在本执行环境中随父进程退出被清理，日志无 traceback，停在 step `12099` 附近；已改用 `setsid` 方式从 `latest.pt -> step_00012000.pt` 重启，避免被父进程生命周期回收。后续后台启动训练建议使用 `setsid ... < /dev/null > log 2>&1 &`。

<!-- B3_B96_FD_AUDIT_SYNCED_AND_RESUMED_20260528T0920Z -->

## FD audit artifacts synced; B3 training resumed stable

更新时间：`2026-05-28T09:20Z`

- GitHub code/docs repo updated: `https://github.com/Yidhar/pdm3-ht-private`, commit `6acab05561162c891b5ea88cb5955de9c1abc98c` (`Record B3 FD audit result and resume note`).
- HF artifact repo updated: `https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts`, repo sha `6dcee800ae305b792cf46563a29544b098e96a79`.
- Uploaded dedicated FD audit artifacts to HF path:

```text
b3_meanflow_realdata/fullcache_b96/step_00010000/fd_jvp_dedicated_audit_after_step_12000/fd_jvp_modes_audit.json
b3_meanflow_realdata/fullcache_b96/step_00010000/fd_jvp_dedicated_audit_after_step_12000/fd_jvp_modes_audit.jsonl
b3_meanflow_realdata/fullcache_b96/step_00010000/fd_jvp_dedicated_audit_after_step_12000/fd_jvp_modes_audit.md
```

Live resume checkpoint after audit: `setsid` training PID `49292` is alive and reparented to PID `1`; resumed from `latest.pt -> step_00012000.pt`; verified beyond previous stop point with last train record step `12367` @ `2026-05-28T09:19:08Z`, last50 throughput `~122.9 samples/s`, H100 snapshot `70371 MiB / 81559 MiB`, `100%` util, `~601 W`.

Next expected gates:

- `step_00014000.pt` checkpoint + built-in FD audit around step `14000`.
- `step_00020000.pt` checkpoint + built-in eval/sample at step `20000`.

<!-- B3_B96_FD_JVP_FINAL_RISK_RECLASSIFICATION_20260528T0940Z -->

## Final FD/JVP risk reclassification after A/B/C audit

更新时间：`2026-05-28T09:40Z`

English record / paper wording:

> Dedicated FD/JVP audit on the step-10000 PAE B3 checkpoint confirms that the fp32 JVP path is numerically correct when TF32 is disabled. With TF32 off, both math SDPA and default SDPA produce FD relative errors in the 7.7e-4 ~ 9.6e-4 range for eps=3e-3/1e-3, while TF32-on finite-difference diagnostics show much larger apparent errors. Therefore, the earlier high built-in FD rel values in the H100 fast b96 run are attributed to TF32-sensitive finite-difference diagnostics, not to an incorrect JVP implementation. The r=t degeneration remains exact with target-v=0.

中文统一口径：

> step-10000 PAE B3 checkpoint 的专项 FD/JVP audit 已确认：在关闭 TF32 的严格数值口径下，fp32 JVP target 路径是正确的；math SDPA 与 default SDPA 在 eps=3e-3/1e-3 下都回到 7.7e-4 ~ 9.6e-4 的 FD rel。此前 H100 fast b96 run 中 built-in FD rel 出现 0.05 ~ 0.11，主要是 TF32 + finite-difference sensitivity / 诊断口径导致，不是 JVP 主链路错误。r=t 退化路径仍严格正确，target-v=0。

Operational decision:

- Keep the PAE B3 b96 mainline on the fast H100 config (`allow_tf32=true`, `sdpa_kernel=default`, `bf16_backbone_fp32_jvp`); do not stop or slow the main run solely because fast built-in FD rel is above the old `1e-2` practical gate.
- Interpret built-in FD rel under the fast config as a finite/no-NaN smoke diagnostic plus TF32-sensitive trend, not as the strict JVP correctness gate.
- Strict FD/JVP correctness audits should use: TF32 off, fixed seed, small fixed non-degenerate batch, `model.eval()`, dropout/class dropout disabled or fixed, JVP/FD fp32, global math/default SDPA comparison, and `eps` sweep (`1e-2`, `3e-3`, `1e-3`).
- Next gates: `step_00014000.pt` checkpoint + built-in FD audit; `step_00020000.pt` checkpoint + EMA sample + PAE decode + image-space metrics.
- Step-20000 image-space evaluation should connect real ImageNet-256 Inception FID/MMD/KID using the cropped uint8 ImageNet cache, not only the compact random-projection smoke metric. If the generated eval batch remains `64` samples, label the Inception numbers as early/sample-count-limited diagnostics rather than publishable 50k FID.

<!-- B3_B96_STEP14000_AND_INCEPTION_EVAL_READY_20260528T0948Z -->

## Step-14000 gate reached; step-20000 Inception eval helper prepared

更新时间：`2026-05-28T09:48Z`

Main training remains alive under `setsid` on H100 and continued past the gate.

Step-14000 checkpoint / built-in FD audit:

```text
checkpoint: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00014000.pt
created_at_utc: 2026-05-28T09:41:25Z
latest.pt -> step_00014000.pt
```

Built-in fast-config FD at step `14000`:

```json
{
  "fd_rel_err_full_jvp": 0.07857135953004488,
  "fd_eps": 0.01,
  "fd_mode": "fp32",
  "jvp_mode": "fp32",
  "allow_tf32_fast_config": true,
  "r_eq_t_count": 70,
  "r_eq_t_total": 96,
  "realized_r_eq_t_fraction": 0.7291666865348816,
  "u_finite": true,
  "du_finite": true,
  "fd_finite": true,
  "no_nan_or_inf": true,
  "target_detached": true,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0
}
```

Interpretation: this `0.07857` is consistent with the settled fast-config TF32 finite-difference diagnostic artifact pattern; it is **not** treated as evidence of a broken JVP path. Strict JVP correctness gating remains the dedicated TF32-off fixed non-degenerate eps-sweep audit.

Step-20000 eval preparation:

- Added script: `experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/decode_and_inception_eval_step.py`.
- Purpose: decode generated PAE latents, read real cropped uint8 ImageNet-256 cache directly, and compute image-space InceptionV3 pool-2048 FID / RBF-MMD / poly3-KID.
- Default devices are CPU (`--decode-device cpu --inception-device cpu`) to avoid disturbing the active H100 training process.
- Inception weights have been downloaded/cached locally via torchvision (`inception_v3_google-0cc3c7bd.pth`).
- Smoke validation completed on step-10000 with 4 generated + 4 real ImageNet-256 images: script status `ok`, feature dim `2048`, finite metrics produced. This validates wiring only; step-20000 should run on the full generated eval batch.

Recommended step-20000 command after trainer writes `eval/step_00020000/sample_latents.safetensors`:

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
python3 "$EXP/scripts/decode_and_inception_eval_step.py" \
  --sample-latents "$EXP/results/fullcache_realdata_singleproc_template/eval/step_00020000/sample_latents.safetensors" \
  --output-dir "$EXP/results/fullcache_realdata_singleproc_template/eval/step_00020000/inception_eval" \
  --real-image-cache /workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors \
  --num-real 0 \
  --real-seed 20260528 \
  --decode-device cpu \
  --decode-batch-size 2 \
  --inception-device cpu \
  --inception-batch-size 16 \
  --torch-num-threads 8 \
  --max-save-images 256 \
  --grid-count 64
```

If trainer eval still emits `64` generated samples, record resulting FID/MMD/KID as early/sample-count-limited Inception diagnostics, not publishable 50k FID.

<!-- B3_B96_STEP20000_EVAL_DONE_20260528T1116Z -->

## Step-20000 checkpoint, EMA sample, FD smoke, and Inception eval completed

更新时间：`2026-05-28T11:16Z`

结论：`step_00020000` 主产物已经完整产出；PAE B3 b96 主线没有停止，已继续向后训练。

Step-20000 checkpoint:

```text
checkpoint: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00020000.pt
created_at_utc: 2026-05-28T11:02:43Z
latest.pt -> step_00020000.pt
```

Trainer built-in EMA latent sample:

```json
{
  "step": 20000,
  "sample_status": "ok",
  "num_samples": 64,
  "sample_steps": 32,
  "use_ema": true,
  "sample_shape": [64, 32, 16, 16],
  "sample_finite": true,
  "sample_mean": 0.19003459811210632,
  "sample_std": 0.9108490347862244,
  "sample_path": "/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00020000/sample_latents.safetensors"
}
```

Built-in fast-config FD/JVP smoke at step `20000`:

```json
{
  "fd_eps": 0.01,
  "fd_mode": "fp32",
  "jvp_mode": "fp32",
  "fd_rel_err_full_jvp": 0.06927925867689227,
  "fd_abs_err_norm": 15.122713088989258,
  "du_norm": 220.61480712890625,
  "fd_norm": 218.2863006591797,
  "r_eq_t_count": 75,
  "r_eq_t_total": 96,
  "realized_r_eq_t_fraction": 0.78125,
  "u_finite": true,
  "du_finite": true,
  "fd_finite": true,
  "no_nan_or_inf": true,
  "target_detached": true,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0
}
```

Interpretation: this fast-config FD rel (`0.069279`) remains in the previously explained TF32 finite-difference diagnostic-artifact regime. The dedicated step-10000 TF32-off audit already established the fp32 JVP target path correctness; strict FD gates should continue to use TF32 off + fixed non-degenerate small batch + eps sweep.

CPU PAE decode + real ImageNet-256 Inception eval completed:

```json
{
  "status": "ok",
  "elapsed_sec": 36.44545849598944,
  "num_generated": 64,
  "num_real": 64,
  "feature_dim": 2048,
  "fid": 319.8114004384611,
  "inception_mmd_rbf": 0.04636890681232764,
  "inception_kid_poly3": 0.0600598865598263
}
```

Image summaries:

```json
{
  "generated_mean": 0.48540279269218445,
  "generated_std": 0.2715531885623932,
  "generated_channel_mean_rgb": [0.5125942826271057, 0.4890085458755493, 0.4558027684688568],
  "generated_channel_std_rgb": [0.2644721567630768, 0.26147812604904175, 0.28310543298721313],
  "real_mean": 0.4458613693714142,
  "real_std": 0.2769797444343567,
  "real_channel_mean_rgb": [0.4725829064846039, 0.4556937515735626, 0.4083418548107147],
  "real_channel_std_rgb": [0.27943283319473267, 0.26707392930984497, 0.27828601002693176]
}
```

Caveat: this is a `64` generated / `64` real sample early diagnostic to verify the image-space sample pipeline and monitor training trend; it is **not** a publishable 50k ImageNet FID.

HF artifact upload completed:

```text
repo: https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
commit: https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/commit/ee0884422f792685559a0ca12083fbf3f516d316
sha: ee0884422f792685559a0ca12083fbf3f516d316
```

Uploaded step-20000 files:

```text
b3_meanflow_realdata/fullcache_b96/step_00020000/eval/eval_record.json
b3_meanflow_realdata/fullcache_b96/step_00020000/eval/sample_latents.safetensors
b3_meanflow_realdata/fullcache_b96/step_00020000/inception_eval/inception_metrics.json
b3_meanflow_realdata/fullcache_b96/step_00020000/inception_eval/inception_metrics.md
b3_meanflow_realdata/fullcache_b96/step_00020000/inception_eval/inception_features.npz
b3_meanflow_realdata/fullcache_b96/step_00020000/inception_eval/decoded_and_real_imagenet256_samples.npz
b3_meanflow_realdata/fullcache_b96/step_00020000/inception_eval/images/generated_grid.png
b3_meanflow_realdata/fullcache_b96/step_00020000/inception_eval/images/real_imagenet256_grid.png
```

Current mainline at doc update:

```json
{
  "latest_step": 21050,
  "created_at_utc": "2026-05-28T11:16:48Z",
  "loss": 0.42084091901779175,
  "elapsed_sec": 0.7927653328515589,
  "batch_size": 96
}
```

Next: keep the fast H100 b96 mainline running; next natural gates are the next periodic checkpoints/FD smoke and a larger/periodic image-space eval if sample count or schedule is increased.

<!-- B3_B96_TREND_SNAPSHOT_20260528T1126Z -->

## B3 b96 loss / FID / MMD trend snapshot

更新时间：`2026-05-28T11:27Z`

为了让 step-10000 与 step-20000 的 image-space 指标可比，补跑了 step-10000 的同口径 full-64 CPU PAE decode + real ImageNet-256 Inception eval：同一 real seed `20260528`、`64` generated / `64` real、torchvision InceptionV3 pool-2048。

Comparable 64-sample Inception diagnostics:

| step | FID ↓ | Inception RBF-MMD ↓ | poly3-KID ↓ | gen image std | real image std |
|---:|---:|---:|---:|---:|---:|
| 10000 | 372.095178 | 0.104331 | 0.147350 | 0.165529 | 0.276980 |
| 20000 | 319.811400 | 0.046369 | 0.060060 | 0.271553 | 0.276980 |

Relative change from step-10000 to step-20000:

```text
FID:            -14.05%
Inception MMD:  -55.56%
KID:            -59.24%
```

Interpretation:

- Image-space diagnostics improved clearly from 10k to 20k: FID down about `14%`, MMD down about `56%`, KID down about `59%`.
- The generated image standard deviation moved from `0.1655` at 10k to `0.2716` at 20k, close to the sampled real reference std `0.2770`; this is consistent with samples becoming less washed-out / closer in low-level contrast statistics.
- Training loss is **not monotonically decreasing** after the early phase. It dropped strongly in the first several thousand steps, then plateaued and drifted slightly upward while image-space metrics improved. Treat loss as a training-health signal, not the sole quality metric.

Loss means by step bin:

| step bin | mean loss |
|---:|---:|
| 1-1k | 0.524857 |
| 1k-2k | 0.430982 |
| 2k-4k | 0.411007 |
| 4k-6k | 0.391033 |
| 6k-8k | 0.385693 |
| 8k-10k | 0.388462 |
| 10k-12k | 0.389656 |
| 12k-14k | 0.392543 |
| 14k-16k | 0.395027 |
| 16k-18k | 0.396849 |
| 18k-20k | 0.398791 |

Current mainline at trend update:

```json
{
  "latest_step": 21822,
  "created_at_utc": "2026-05-28T11:27:04Z",
  "loss": 0.41953331232070923,
  "batch_size": 96
}
```

HF upload for the comparable step-10000 full-64 eval:

```text
commit: https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/commit/c7e1cc80e87cc43f9a8e335cd252279c53d358fb
sha: c7e1cc80e87cc43f9a8e335cd252279c53d358fb
path: b3_meanflow_realdata/fullcache_b96/step_00010000/inception_eval_64_seed20260528/
```

Caveat: these FID/MMD/KID numbers are still early `64`-sample diagnostics, not publishable 50k FID. They are useful for same-run trend monitoring and pipeline verification.

<!-- B3_B96_FDLOSS_DEFER_FID_CONVERGENCE_WATCH_20260528T1202Z -->

## B3 b96 route decision: defer Representation Fréchet FD-loss, measure base MeanFlow FID convergence

更新时间：`2026-05-28T12:02Z`

Decision update:

- **Representation Fréchet Loss / FD-loss is deferred for now.** Do not add FD-loss to the active trainer yet.
- Continue the current **PAE latent + LightningDiT B3/XL + MeanFlow objective** path and measure how far the image-space FID/MMD/KID diagnostics converge before post-training.
- Keep the terminology distinct:
  - `fd_audit` / `fd_rel_err_full_jvp` in trainer logs = finite-difference JVP audit, not Representation Fréchet Loss.
  - `FID/MMD/KID` = image-space evaluation diagnostics.
  - `FD-loss` = future Representation Fréchet Loss post-training, currently inactive.

Current mainline at this decision update:

```json
{
  "type": "train_step",
  "step": 24395,
  "created_at_utc": "2026-05-28T12:02:07Z",
  "batch_size": 96,
  "label_min": 1,
  "label_max": 991,
  "label_unique_count": 93,
  "class_cond_injected": true,
  "force_drop_count": 14,
  "force_drop_total": 96,
  "precision_recipe": "bf16_backbone_fp32_jvp",
  "backbone_forward_mode": "bf16_autocast",
  "jvp_target_mode": "fp32",
  "fd_audit_mode": "fp32",
  "jvp_param_source": "live",
  "ema_used_for_target_jvp": false,
  "target_detached": true,
  "r_t_sampling_granularity": "sample_level",
  "configured_equal_prob": 0.75,
  "r_eq_t_count": 66,
  "r_eq_t_total": 96,
  "realized_r_eq_t_fraction": 0.6875,
  "target_jvp_sec": 0.29190411418676376,
  "target_jvp_peak_memory_mb": 12018.00048828125,
  "target_jvp_effective_batch": 30,
  "target_jvp_skipped_equal_batch": 66,
  "target_requires_grad": false,
  "target_jvp_u_finite": true,
  "du_finite": true,
  "target_finite": true,
  "du_norm": 185.23731994628906,
  "v_norm": 1254.45068359375,
  "du_over_v_norm_ratio": 0.14766409103913206,
  "actual_rt_samples_target_minus_v_max_abs": 0.0,
  "backbone_forward_sec": 0.1810024380683899,
  "backbone_forward_peak_memory_mb": 58975.13720703125,
  "u_finite": true,
  "bf16_forward_vs_fp32_rel_err": 0.007437853805106772,
  "fp32_reference_forward_norm": 610.7955932617188,
  "loss_finite": true,
  "loss_mean": 0.3517903685569763,
  "loss": 0.3517903685569763,
  "backward_sec": 0.3018683339469135,
  "backward_peak_memory_mb": 59064.298828125,
  "grad_norm_before_clip": 0.2101142257452011,
  "grad_clip_threshold": 1.0,
  "grad_is_finite": true,
  "optimizer_step_applied": true,
  "elapsed_sec": 0.8359481291845441,
  "peak_memory_mb": 59064.298828125
}
```

Current comparable FID trend file:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/fid_convergence_summary.md
```

Current comparable 64-sample diagnostics:

| step | FID ↓ | Inception RBF-MMD ↓ | poly3-KID ↓ | gen image std | real image std |
|---:|---:|---:|---:|---:|---:|
| 10000 | 372.095178 | 0.104331 | 0.147350 | 0.165529 | 0.276980 |
| 20000 | 319.811400 | 0.046369 | 0.060060 | 0.271553 | 0.276980 |

A CPU-only watcher has been started to automatically evaluate future trainer samples every 10k steps starting at step 30000 without occupying the H100 training path:

```text
watcher script: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/watch_b3_fid_convergence.sh
watcher pid: 55440
watcher log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/b3_fid_convergence_watch_20260528T120036Z.log
```

Watcher behavior:

- Polls trainer progress every `180s`.
- Detects new `eval/step_*/sample_latents.safetensors` for steps `>=30000`.
- Runs CPU PAE decode + CPU torchvision Inception metrics through `decode_and_inception_eval_step.py`.
- Writes per-step metrics to `eval/step_xxxxx/inception_eval/inception_metrics.json`.
- Maintains:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/fid_convergence_summary.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/fid_convergence_summary.md
```

Next natural gate:

```text
step 30000: checkpoint + EMA sample + CPU PAE decode + 64/64 Inception FID/MMD/KID convergence point
```

Caveat: these are still early small-sample `64 generated / 64 real` convergence diagnostics, not official 50k FID. Use them for same-run trend monitoring and pipeline validation.
