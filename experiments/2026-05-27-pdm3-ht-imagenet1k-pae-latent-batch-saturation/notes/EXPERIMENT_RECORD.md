# Experiment Record — PAE latent extraction batch saturation throughput

Date: 2026-05-27

## Setup

Experiment directory:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-batch-saturation
```

Persistent output base:

```text
/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/batch_saturation_2026-05-27
```

Model/data:

- HF dataset: `ILSVRC/imagenet-1k`, `train`, streaming.
- PAE: `PAE_DINOv2L_d32`.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, `94.97 GiB`.
- Dtypes: bf16 model, bf16 saved latents.

## 001 — Sweep script creation

Created:

```text
scripts/batch_saturation_sweep.py
scripts/run_batch_saturation_sweep.sh
configs/batch_saturation_sweep.env
```

Design:

- Load PAE model once.
- Load HF streaming iterator once.
- Initial skip 1024 samples.
- Sequential real-image timed segments for batch sizes 64/128/256/512/1024.
- Save one safetensors shard per segment with `latents`, `latents_flip`, `labels`.
- Record preprocess time, encode time, save time, CUDA peak memory, end-to-end throughput.

## 002 — First run failed and fixed

Log:

```text
logs/010_batch_saturation_sweep_b64_128_256_512_1024_n2048.log
```

Failure:

- Importing the previous builder script with `importlib` failed around dataclass/module registration.

Fix:

```python
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)
```

## 003 — Real ImageNet streaming batch saturation sweep

Command:

```bash
cd /workspace/PDM
EXP=experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-batch-saturation
bash "$EXP/scripts/run_batch_saturation_sweep.sh" \
  2>&1 | tee "$EXP/logs/011_batch_saturation_sweep_b64_128_256_512_1024_n2048.log"
```

Summary path:

```text
/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/batch_saturation_2026-05-27/batch_saturation_summary.json
```

Results:

| batch | status | total samples/s | encode-only samples/s | peak GiB | preprocess/elapsed |
|---:|---|---:|---:|---:|---:|
| 64 | PASS | 9.1987 | 132.72 | 2.65 | 77.8% |
| 128 | PASS | 8.1445 | 133.84 | 3.95 | 74.3% |
| 256 | PASS | 9.0112 | 138.07 | 6.53 | 75.5% |
| 512 | PASS | 8.7763 | 144.89 | 11.70 | 77.1% |
| 1024 | PASS | 8.3032 | 147.77 | 22.05 | 71.1% |

Interpretation:

- Larger batch did not improve end-to-end throughput.
- GPU encode-only component is much faster than full pipeline.
- The real sweep is input pipeline limited.

## 004 — Verification of saved real latent subsets

Command:

```bash
cd /workspace/PDM
EXP=experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-batch-saturation
VERIFY=experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-build/scripts/verify_img_latent_dataset.py
for d in /workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/batch_saturation_2026-05-27/imagenet256_train_saturation_b*_n2048_offset*; do
  echo "VERIFY $d"
  python "$VERIFY" "$d" --latent-norm
done 2>&1 | tee "$EXP/logs/020_verify_all_saturation_outputs.log"
```

Result:

- All five directories PASS.
- Shape/dtype invariant holds: `latents BF16 [2048,32,16,16]`, `latents_flip BF16 [2048,32,16,16]`, `labels I64 [2048]`.

## 005 — Encode-only high-batch saturation probe

Created:

```text
scripts/pae_encode_only_saturation_probe.py
```

Command:

```bash
cd /workspace/PDM
EXP=experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-batch-saturation
python "$EXP/scripts/pae_encode_only_saturation_probe.py" \
  --batch-sizes 1024,2048,3072,4096 \
  --output-json "$EXP/results/encode_only_saturation_probe.json" \
  2>&1 | tee "$EXP/logs/030_encode_only_saturation_probe_b1024_2048_3072_4096.log"
```

Results:

| batch | status | pair samples/s | PAE ops/s | peak GiB | peak VRAM |
|---:|---|---:|---:|---:|---:|
| 1024 | PASS | 114.73 | 229.46 | 22.05 | 23.2% |
| 2048 | PASS | 142.07 | 284.13 | 42.74 | 45.0% |
| 3072 | PASS | 141.90 | 283.80 | 63.44 | 66.8% |
| 4096 | FAIL/OOM | — | — | OOM | — |

Interpretation:

- Encoder-only throughput plateaus around 142 samples/s once batch reaches 2048.
- Batch 3072 uses more memory but does not improve throughput.
- Batch 4096 OOM in current allocator/model path.
- This is an upper-bound probe only; it excludes HF/PIL/save overhead.

## 006 — Extrapolation and final summaries

Created:

```text
results/batch_saturation_extrapolation.json
results/summary.json
```

Current-pipeline full-cache estimates:

- Exp10 b32 steady: 1M `28.09 h`; full train `35.98 h`.
- Current b64 best: 1M `30.20 h`; full train `38.69 h`.
- Current b128 worst: full train `43.70 h`.

Encode-only lower bound:

- 1M `~1.96 h`.
- Full train `~2.51 h`.

Storage:

- 1M `~32.78 GB / 30.53 GiB`.
- Full train `~41.99 GB / 39.11 GiB`.

## 007 — Subagent/delegation note

The local Switchyard bridge requested by AGENTS instructions is not installed in this environment:

```text
SWITCHYARD_UNAVAILABLE
```

A context-free `cexll`/Codex reviewer attempt was also blocked by the tool sandbox policy, so final interpretation was completed locally from recorded JSON outputs.

## Final conclusion

The full cache job should not be launched under the assumption that batch size alone will saturate the GPU. Current full extraction is dominated by input streaming/preprocessing. Under existing code, reserve about `36–39 h` for full ImageNet train, with `~44 h` as conservative variance budget. To materially reduce this, implement local/multiworker/overlapped input pipeline before full-cache production.
