# PDM-3-HT handoff index

更新时间：`2026-05-28T06:25:31+00:00`

## 当前有效状态

PAE full latent cache 已完成并公开上传；B3 full-cache run 已启动。

优先看：

```text
HANDOFF_2026-05-28_PAE_PUBLIC_CACHE_READY_AND_B3_STARTED.md
```

旧 handoff 仍保留作为历史上下文：

```text
HANDOFF_2026-05-28_PDM3_HT_PRIVATE_HF_AND_PAE_PARALLEL.md
```

注意：旧 handoff 中 PAE 为 partial 的描述已经过期。

## 关键链接

- Private code repo: https://huggingface.co/LAXMAYDAY/pdm3-ht-20260528-code
- Public PAE cache repo: https://huggingface.co/datasets/LAXMAYDAY/pdm-pae-dinov2l-d32-imagenet256-train-full

## 当前 B3 run

```text
pid: 31771
config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_full.yaml
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/fullcache_realdata_20260528T062419Z.log
```

监控：

```bash
cd /workspace/PDM
PID=$(cat experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/fullcache_realdata.pid)
ps -p "$PID" -o pid,ppid,stat,etime,%cpu,%mem,rss,cmd
nvidia-smi
LOG=$(cat experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_latest.logpath)
tail -f "$LOG"
```

## 状态 JSON

```text
handoff/state/pae_full_cache_verified_after_hf_upload.json
handoff/state/pae_public_cache_hf_remote_verified.json
handoff/state/b3_fullcache_latest_launch.json
```

<!-- B3_H100_SPEEDUP_LATEST -->

## Latest B3 H100 speedup status

- Active config: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_full_fast_h100.yaml`
- Active PID/log: see `handoff/state/b3_h100_speedup_latest.json`
- Current batch: 32; global SDPA default with JVP/FD math-context patch; TF32 enabled.
- Current throughput: ~57.8 samples/sec vs baseline ~10.6 samples/sec (~5.47x).
