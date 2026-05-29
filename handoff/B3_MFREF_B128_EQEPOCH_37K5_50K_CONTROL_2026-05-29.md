# B3 MeanFlow reference-hparam b128 equivalent-epoch control — 2026-05-29

更新时间：`2026-05-29T07:32:07Z`

## Decision / correction

User correction accepted: **b128 50k is not equivalent to b96 50k**.  From this point, b96 vs b128 comparisons must be normalized by processed samples / equivalent epochs, not raw step count.

Dataset/cache cardinality used for normalization:

```text
N = 1,281,167 ImageNet-train samples
manifest = /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/hf_pae_cache_upload/manifest.jsonl
```

Formulas:

```text
processed_samples = steps × global_batch_size
equivalent_epochs = processed_samples / 1,281,167
b96 -> b128: b128_steps = b96_steps × 96 / 128 = b96_steps × 0.75
b128 -> b96: b96_steps = b128_steps × 128 / 96 = b128_steps × 1.333333
```

Therefore:

| target | samples | epochs | equivalent |
|---|---:|---:|---|
| old b96 50k | 4,800,000 | 3.746584 | new b128 **37,500** |
| new b128 37.5k | 4,800,000 | 3.746584 | old b96 **50,000** |
| new b128 50k | 6,400,000 | 4.995446 | old b96 **66,667** |
| old b96 100k | 9,600,000 | 7.493168 | new b128 **75,000** |
| old b96 1.07M | 102,720,000 | 80.176901 | new b128 **802,500** |
| MeanFlow b128 800k | 102,400,000 | 79.927129 | old b96 **1,066,667** |

New comparison framing:

1. **Primary fair comparison**: old b96 constant-`2e-4` step `50,000` 5K FID `55.528315` vs new b128 reference-hparam step **`37,500`** 5K FID TBD.
2. **Secondary reference point**: new b128 step **`50,000`**, but label it as **b96 `66,667`-equivalent**, not b96 50k-equivalent.

## Superseded raw-step b128 run

The previous b128 50k raw-step-aligned branch was intentionally stopped before any checkpoint, because its 50k point would have processed 6.4M samples and therefore was not a fair comparison to old b96 50k.

```text
superseded result: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_50k
superseded config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_50k.yaml
old train PID: 77429
old 50k FID controller PID: 75011
old post-50k HF controller PID: 75434
status: superseded/cancelled around step ~865, before first checkpoint
```

The old controller JSONs in the superseded result directory were marked `superseded_cancelled`.

## Active run

Active result directory:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k
```

Active configs:

```text
primary exact-equivalent config:
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5.yaml

resume-to-50k secondary config:
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_eqepoch_resume_to50k.yaml
```

Config snippets are committed in this repo under:

```text
handoff/config_snippets/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5.yaml
handoff/config_snippets/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_eqepoch_resume_to50k.yaml
```

Key training settings:

```yaml
model: B3MeanFlowLightningDiT-XL/1
latent: PAE_DINOv2L_d32, 32 channels, ImageNet-256 full train cache
batch: 128
optimizer: Adam(beta1=0.9, beta2=0.95), lr=1e-4
scheduler: warmup 10k steps, cosine decay to min_lr=1e-5, decay_end_step=800k
precision: bf16 backbone + fp32 JVP path
sdpa: default
allow_tf32: true
checkpoint/eval cadence primary: every 7,500 b128 steps
```

Checkpoint/eval cadence is deliberately aligned to old b96 every 10k processed-sample anchors:

```text
b128  7,500 <=> b96 10,000
b128 15,000 <=> b96 20,000
b128 22,500 <=> b96 30,000
b128 30,000 <=> b96 40,000
b128 37,500 <=> b96 50,000  # primary fair FID anchor
```

Launch status:

```text
launch JSON: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/launch_status.json
pid file: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/mfref_b128_eqepoch.pid
current PID at doc update: 78353
train log path file: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_latest.logpath
train log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_37k5_train_setsid_20260529T072459Z.log
```

Latest metric snapshot at doc update:

```json
{
  "step": 398,
  "created_at_utc": "2026-05-29T07:32:06Z",
  "loss": 1.3236348628997803,
  "optimizer_lr": 3.98e-06,
  "optimizer_lr_schedule": "cosine",
  "peak_memory_mb": 74652.00244140625,
  "grad_is_finite": true,
  "loss_finite": true
}
```

Early speed observed after relaunch was about `1.0–1.1 sec/step` with H100 GPU at ~100% util and ~77.6 GiB memory used.  Expected primary anchor timing from the 07:25 UTC relaunch is roughly **2026-05-29 18:15–18:45 UTC** for step 37.5k, plus 5K sampling/eval time; step 50k follows after the 37.5k eval pause and resume.

## Controllers

### 1) Step 37.5k primary 5K FID, then resume to 50k

```text
controller PID: 78122
script: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_37k5_5k_gpu_fid_then_resume_20260529T072317Z.sh
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_37k5_5k_gpu_fid_then_resume_20260529T072317Z.setsid.log
status JSON: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00037500/5k_anchor_control.json
```

Current planned behavior:

1. wait for `checkpoints/step_00037500.pt`;
2. wait for trainer's small 64-sample eval;
3. let/force the 37.5k trainer stop naturally;
4. sample 5K EMA latents with `sample_steps=32`, batch `64`, seed `2026052937`;
5. GPU PAE decode + true Inception FID/MMD/KID;
6. resume training using `b3_meanflow_realdata_mfref_b128_lr1e4_cosine_eqepoch_resume_to50k.yaml`.

### 2) Step 50k secondary 5K FID, final no-resume

```text
controller PID: 78123
script: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_50k_5k_gpu_fid_final_20260529T072317Z.sh
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_50k_5k_gpu_fid_final_20260529T072317Z.setsid.log
status JSON: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00050000/5k_anchor_control.json
```

This produces the secondary anchor:

```text
b128 step 50,000 = 6.4M samples = b96 step 66,667-equivalent
```

### 3) Post-50k HF upload/cleanup

```text
controller PID: 78124
script: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_post50k_hf_upload_cleanup_20260529T072317Z.sh
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_post50k_hf_upload_cleanup_20260529T072317Z.setsid.log
status JSON: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00050000/hf_post50k_upload_status.json
HF repo: LAXMAYDAY/pdm3-ht-model-artifacts
remote prefix: b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k
```

After step-50k metrics exist, it uploads sparse checkpoints/eval artifacts for steps `37500` and `50000`, then prunes non-archive local checkpoint clutter according to the watcher policy.

## Old b96 control anchors to keep for comparison

Old route:

```text
result: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template
batch: 96
LR: constant 2e-4
stopped at step: 103354
```

True 5K Inception anchors:

| old b96 step | sample_steps/NFE | samples | epochs | FID ↓ | MMD ↓ | KID ↓ |
|---:|---:|---:|---:|---:|---:|---:|
| 50,000 | 32 | 4,800,000 | 3.746584 | `55.528315` | `0.0328459` | `0.0389490` |
| 100,000 | 32 | 9,600,000 | 7.493168 | `47.925749` | `0.0259518` | `0.0298404` |

Step-100k multi-NFE sweep on the old control checkpoint:

| sample_steps/NFE | 5K FID ↓ | MMD ↓ | KID ↓ |
|---:|---:|---:|---:|
| 1 | `56.966056` | `0.0344998` | `0.0427200` |
| 2 | `53.476784` | `0.0309196` | `0.0374734` |
| 4 | `49.938747` | `0.0277017` | `0.0323653` |
| 32 existing anchor | `47.925749` | `0.0259518` | `0.0298404` |

Important caveat: the existing old step-100k anchor `47.9257` used `sample_steps=32`; do **not** label it as 1-NFE.

## Monitoring commands

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
RES="$EXP/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k"

date -u '+now=%Y-%m-%dT%H:%M:%SZ'
ps -eo pid,ppid,pgid,sid,stat,etime,%cpu,%mem,rss,cmd   | grep -E 'train_b3_meanflow_realdata.py|mfref_b128_eqepoch|sample_b3_meanflow|decode_and_inception|watch_b3_hf_upload'   | grep -v grep || true
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true

python3 - <<'PY2'
import json, pathlib
p=pathlib.Path('/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/metrics.jsonl')
rows=[]
for line in p.read_text(errors='replace').splitlines() if p.exists() else []:
    try:o=json.loads(line)
    except Exception: continue
    if o.get('type')=='train_step': rows.append(o)
print('rows', len(rows))
if rows:
    last=rows[-1]
    print({k:last.get(k) for k in ['step','created_at_utc','loss','optimizer_lr','optimizer_lr_schedule','peak_memory_mb','grad_is_finite','loss_finite']})
PY2

cat "$RES/eval/step_00037500/5k_anchor_control.json"
cat "$RES/eval/step_00050000/5k_anchor_control.json"
cat "$RES/eval/step_00050000/hf_post50k_upload_status.json"
LOG=$(cat "$EXP/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_latest.logpath")
tail -f "$LOG"
```

## Next actions

1. Keep the active b128 eqepoch run alive; do not restart unless it fails.
2. Treat step **37.5k** as the fair replacement for old b96 50k.
3. Let the 37.5k controller generate the primary 5K true Inception FID and resume to 50k.
4. Let the 50k controller generate the secondary b96-66.7k-equivalent FID.
5. Let the post-50k controller upload artifacts to HF under `b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k`.
6. In reports/charts, label x-axis as processed samples or equivalent epochs; raw step count is secondary metadata only.


## Progress update — 2026-05-29T19:14:37Z

Step **37,500** primary equivalent-epoch anchor has completed. This is the fair sample-equivalent point to old b96 step 50k:

```text
b128 37,500 × 128 = 4,800,000 samples
old b96 50,000 × 96 = 4,800,000 samples
```

### Step 37.5k 5K true Inception FID

Report files:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00037500/inception_eval_5k/inception_metrics.md
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00037500/inception_eval_5k/inception_metrics.json
```

Metrics, 5,000 generated / 5,000 real, EMA, `sample_steps=32` (**script-level NFE=32; one `transport.call_model(...)` per Euler step in `sample_meanflow_latents`; this is not a 1-NFE result**):

| run | processed samples | equivalent | FID ↓ | MMD ↓ | KID ↓ |
|---|---:|---:|---:|---:|---:|
| old b96 step 50k | 4,800,000 | b96 50k | `55.528315` | `0.0328459` | `0.0389490` |
| new b128 step 37.5k | 4,800,000 | b96 50k | `64.769588` | `0.040049` | `0.050480` |

Interpretation: at the fair early 4.8M-sample anchor, the b128 lr1e-4 cosine/reference-hparam run is currently worse than the old b96 constant-2e-4 control by about **+9.241 FID**. Continue to the secondary b128 step 50k / b96 66.7k-equivalent anchor before making final decision on this route.

Image stats:

| split | mean | std | finite |
|---|---:|---:|:---:|
| generated | `0.446426659822464` | `0.2760142982006073` | `True` |
| real ImageNet256 | `0.45189958810806274` | `0.2797693610191345` | `True` |

### Resume-to-50k incident and fix

The 37.5k controller completed sampling/eval and attempted to resume to 50k, but the first resume process exited immediately with:

```text
TypeError: RNG state must be a torch.ByteTensor
```

Failed log:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_resume_to50k_after_37k5_fid_20260529T181348Z.log
```

Cause: `torch.load(..., map_location=cuda)` remapped the saved CPU RNG ByteTensor to CUDA before `torch.set_rng_state`; PyTorch requires the CPU RNG state on CPU.

Fix applied in both local experiment script and private-sync copy:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py
/workspace/pdm3-ht-private-sync/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py
```

The trainer now coerces the saved CPU RNG state back to CPU before restore. `py_compile` passed. Resume-to-50k was relaunched successfully:

```text
pid: 85230
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_resume_to50k_after_rngfix_20260529T191236Z.log
latest observed train step: 37597
latest loss: 0.38400977849960327
ETA step 50k: 2026-05-29T22:47:48Z UTC, approximate from current speed
```

Controllers remain active:

- step 50k 5K FID: waiting for `checkpoints/step_00050000.pt`;
- post-50k HF upload/cleanup: waiting for step-50k `inception_metrics.json`.
