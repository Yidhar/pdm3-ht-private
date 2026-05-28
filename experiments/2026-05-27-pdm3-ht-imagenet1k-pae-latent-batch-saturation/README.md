# Experiment 11 — ImageNet-1k → PAE latent batch/throughput saturation

Date: 2026-05-27  
Status: **PASS / completed**

## 目的

按用户要求“测一些吃满后的吞吐量来计算全量缓存的时间”，在真实 HF ImageNet-1k train streaming 输入上测 PAE_DINOv2L_d32 latent extraction 的大 batch 端到端吞吐，并补做一个 encode-only 高 batch probe，判断当前全量缓存到底是 GPU 饱和、显存饱和，还是输入管线瓶颈。

## 输入/模型

- Dataset: `ILSVRC/imagenet-1k`, split `train`, streaming, HF auth user `LAXMAYDAY` 已可访问。
- Encoder: `PAE_DINOv2L_d32`
- PAE config: `/workspace/PDM/external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml`
- PAE checkpoint: `/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt`
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, ~94.97 GiB VRAM
- Dtype: bf16 model, bf16 saved latents
- Output schema: `latents`, `latents_flip`, `labels`; shape `[N, 32, 16, 16]`, `[N, 32, 16, 16]`, `[N]`

## 目录

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-batch-saturation
├── README.md
├── TASK.md
├── configs/
│   └── batch_saturation_sweep.env
├── logs/
│   ├── 010_batch_saturation_sweep_b64_128_256_512_1024_n2048.log   # failed import/dataclass attempt
│   ├── 011_batch_saturation_sweep_b64_128_256_512_1024_n2048.log   # completed real sweep
│   ├── 020_verify_all_saturation_outputs.log                       # verifier PASS logs
│   └── 030_encode_only_saturation_probe_b1024_2048_3072_4096.log   # encode-only high-batch probe
├── notes/
│   └── EXPERIMENT_RECORD.md
├── results/
│   ├── batch_saturation_extrapolation.json
│   ├── encode_only_saturation_probe.json
│   ├── file_manifest.sha256
│   ├── pae_latent_batch_saturation_manifest.sha256
│   └── summary.json
└── scripts/
    ├── batch_saturation_sweep.py
    ├── pae_encode_only_saturation_probe.py
    └── run_batch_saturation_sweep.sh
```

Persistent latent/timing subset outputs:

```text
/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/batch_saturation_2026-05-27/
├── batch_saturation_summary.json
├── imagenet256_train_saturation_b64_n2048_offset1024/
├── imagenet256_train_saturation_b128_n2048_offset3072/
├── imagenet256_train_saturation_b256_n2048_offset5120/
├── imagenet256_train_saturation_b512_n2048_offset7168/
└── imagenet256_train_saturation_b1024_n2048_offset9216/
```

## Run 1: 真实 HF ImageNet streaming 端到端 batch sweep

命令：

```bash
cd /workspace/PDM
EXP=experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-batch-saturation
bash "$EXP/scripts/run_batch_saturation_sweep.sh" \
  2>&1 | tee "$EXP/logs/011_batch_saturation_sweep_b64_128_256_512_1024_n2048.log"
```

配置：

```text
BATCH_SIZES=64,128,256,512,1024
SAMPLES_PER_BATCH_SIZE=2048
INITIAL_SKIP=1024
MODEL_DTYPE=bf16
SAVE_DTYPE=bf16
```

结果表（原图样本吞吐；每个样本保存 normal + hflip 两份 latent）：

| batch | status | encoded | elapsed s | total samples/s | encode-only samples/s | PAE ops/s | peak GiB | peak VRAM | preprocess/elapsed |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | PASS | 2048 | 222.64 | **9.199** | 132.7 | 265.4 | 2.65 | 2.8% | 77.8% |
| 128 | PASS | 2048 | 251.46 | 8.145 | 133.8 | 267.7 | 3.95 | 4.2% | 74.3% |
| 256 | PASS | 2048 | 227.27 | 9.011 | 138.1 | 276.1 | 6.53 | 6.9% | 75.5% |
| 512 | PASS | 2048 | 233.36 | 8.776 | 144.9 | 289.8 | 11.70 | 12.3% | 77.1% |
| 1024 | PASS | 2048 | 246.65 | 8.303 | **147.8** | 295.5 | 22.05 | 23.2% | 71.1% |

结论：端到端吞吐没有随 batch 变大提升；`batch=64` 在本 sweep 中反而最佳。`encode-only` 已经有 132–148 samples/s，但端到端只有 8–9 samples/s，说明当前脚本主要卡在 HF streaming + PIL decode/crop/flip + Python 串行预处理，而不是 PAE encoder/GPU。

## Run 2: encode-only 高 batch “吃显存” probe

命令：

```bash
cd /workspace/PDM
EXP=experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-batch-saturation
python "$EXP/scripts/pae_encode_only_saturation_probe.py" \
  --batch-sizes 1024,2048,3072,4096 \
  --output-json "$EXP/results/encode_only_saturation_probe.json" \
  2>&1 | tee "$EXP/logs/030_encode_only_saturation_probe_b1024_2048_3072_4096.log"
```

说明：该 probe 使用 random preprocessed-shape tensor `[B,3,256,256]`，只隔离测 PAE encoder 两次 forward（normal + flip equivalent）的上限吞吐；不包含 HF streaming、PIL、crop/flip、safetensors 写盘。

| batch | status | pair sec | samples/s pair | PAE ops/s | peak GiB | peak VRAM |
|---:|---|---:|---:|---:|---:|---:|
| 1024 | PASS | 8.925 | 114.7 | 229.5 | 22.05 | 23.2% |
| 2048 | PASS | 14.416 | **142.1** | 284.1 | 42.74 | 45.0% |
| 3072 | PASS | 21.649 | 141.9 | 283.8 | 63.44 | 66.8% |
| 4096 | FAIL/OOM | — | — | — | OOM around 87.4 GiB process use | — |

结论：PAE encoder 的高 batch 上限大约 142 samples/s（每个原图样本含 normal + flip 两次 encode），batch=2048/3072 已进入平台区；batch=4096 在当前 PyTorch allocator/模型路径下 OOM。即使显存打到 45–67%，吞吐也没有继续提升，说明“吃满显存”不是当前端到端瓶颈的解法。

## 校验

命令：

```bash
cd /workspace/PDM
EXP=experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-batch-saturation
VERIFY=experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-build/scripts/verify_img_latent_dataset.py
for d in /workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/batch_saturation_2026-05-27/imagenet256_train_saturation_b*_n2048_offset*; do
  echo "VERIFY $d"
  python "$VERIFY" "$d" --latent-norm
done 2>&1 | tee "$EXP/logs/020_verify_all_saturation_outputs.log"
```

验证结果：5 个输出目录均 PASS；schema 均为 `latents BF16 [2048,32,16,16]`, `latents_flip BF16 [2048,32,16,16]`, `labels I64 [2048]`。

## 全量缓存时间外推

样本数：

- 1M: `1,000,000`
- ImageNet-1k train full: `1,281,167`

当前串行 HF streaming/PIL 管线的 wall-clock 推荐使用：

- 上一次 Experiment 10 batch32 steady: `9.8899 samples/s` → 比本次 sweep 的大 batch 更好。
- 本次大 batch sweep 最佳: `9.1987 samples/s` at batch64 → 更保守。

| 场景 | samples/s | 1M 时间 | full train 时间 |
|---|---:|---:|---:|
| Exp10 batch32 steady real pipeline | 9.8899 | 28.09 h | 35.98 h |
| Current sweep best real pipeline b64 | 9.1987 | 30.20 h | 38.69 h |
| Current sweep worst real pipeline b128 | 8.1445 | 34.11 h | 43.70 h |
| Real sweep encode-only best b1024 lower bound | 147.77 | 1.88 h | 2.41 h |
| Encode-only probe best b2048 lower bound | 142.07 | 1.96 h | 2.51 h |

**实际建议估计：** 如果不改输入管线，按当前实现全量 ImageNet train 缓存应按 **约 36–39 小时** 预算；保守 worst-case 可放到 **44 小时**。如果改成多 worker 本地/并行输入并完全隐藏预处理，理论 encoder 下限是 **约 2.4–2.5 小时**，但这不是当前脚本能达到的端到端时间。

## 存储外推

当前 observed bytes/sample ≈ `32776.68` bytes（两份 bf16 latent + int64 label + safetensors overhead），与理论 payload `32776` bytes/sample 基本一致。

| 样本数 | decimal GB | GiB |
|---:|---:|---:|
| 1,000,000 | 32.78 GB | 30.53 GiB |
| 1,281,167 | 41.99 GB | 39.11 GiB |

## 关键结论

1. 当前真实 PAE latent extraction **不是 GPU 饱和**，是输入管线瓶颈。
2. 提高 batch 到 1024 没有提高端到端吞吐，反而降低到 8.30 samples/s；只是把 peak VRAM 提到 22.05 GiB。
3. Encode-only 高 batch probe 显示 encoder 平台吞吐约 142 samples/s；batch=4096 OOM，batch=2048/3072 足够接近 encoder 上限。
4. 当前不改工程时，全量 ImageNet train latent cache 预算应取 **36–39 小时**；为了稳妥可预留 **~44 小时**。
5. 真正要接近 2–3 小时级别，需要重写输入/预处理管线：本地 cache 或预下载、多 worker decode/crop、pinned memory、CPU/GPU overlap、normal+flip 合批、异步写盘。

## 后续工程建议

优先级从高到低：

1. 做本地化 ImageNet shard/cache，避免 HF streaming 网络/迭代瓶颈。
2. 用 `DataLoader(num_workers>0, pin_memory=True, persistent_workers=True, prefetch_factor=...)` 或等效 producer/consumer 管线并行 PIL decode/crop/flip。
3. GPU encode 和 CPU preprocessing/保存异步重叠；当前脚本是“攒一批 → encode → 继续攒下一批”的串行模式。
4. 将 normal 和 hflip 拼成 `2B` 一次 encode（如果显存允许）以减少 Python 调用/同步开销。
5. 全量正式跑之前先做一个 `10k–50k` 多 worker pipeline benchmark，再外推。
