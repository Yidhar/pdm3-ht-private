# B3 MeanFlow reference-hparam b128 equivalent-epoch control — 2026-05-29

更新时间：`2026-05-30T07:07:12Z`

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

### Midpoint FID slope controller added: b128 22.5k/30k, 32-step NFE=32

User requested intermediate points to distinguish "slow warm-up may overtake" vs "effective LR too low is genuinely worse": new b128 around 20k and 30k with the same 32-step FID口径.

Current retained checkpoints for this b128 run are:

```text
step_00022500.pt
step_00030000.pt
step_00037500.pt
```

There is no exact `step_00020000.pt` in the local run state, so the controller evaluates the nearest retained early checkpoint `step_00022500.pt` plus exact `step_00030000.pt`. By processed-sample normalization these correspond to:

| b128 step | processed samples | b96-equivalent step |
|---:|---:|---:|
| 22,500 | 2,880,000 | 30,000 |
| 30,000 | 3,840,000 | 40,000 |

To avoid competing with the single H100 while the resumed 37.5k->50k training and the existing 50k FID controller are active, the midpoint controller is gated on the step-50k 5K Inception metrics file. After 50k FID completes, it will run 22.5k then 30k sequentially with the same settings:

```text
num_generated = 5000
num_real = 5000
EMA = true
sample_steps = 32
script-level NFE = 32
precision_mode = bf16_autocast
real_seed = 20260529
```

Controller:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_22k5_30k_5k_gpu_fid_after50k_20260529T194245Z.sh
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_22k5_30k_5k_gpu_fid_after50k_20260529T194245Z.setsid.log
```

Status:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/midpoint_22k5_30k_after50k_control.json
```

Expected output metrics:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00022500/inception_eval_5k/inception_metrics.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00030000/inception_eval_5k/inception_metrics.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/midpoint_22k5_30k_5k_fid_summary.md
```

## 2026-05-29 final update: b128 30k / 37.5k / 50k 5K FID results are complete

The b128 lr1e-4 cosine/reference-hparam line reached step 50k and all currently available requested 5K Inception FID anchors have completed.

All completed rows below use the same evaluation口径:

```text
5,000 generated / 5,000 real
EMA = true
sample_steps = 32
script-level NFE = 32
precision_mode = bf16_autocast
PAE decode
torchvision InceptionV3 pool-2048 image-space metrics
```

Important NFE convention: in the current scripts, `sample_steps` is passed as `num_steps` to `sample_meanflow_latents(...)`, and the sampler performs one `transport.call_model(...)` per Euler step. Therefore these are `sample_steps=32` / script-level `NFE=32` results, not 1-NFE results.

### Final metric table

| b128 step | processed samples | b96-equivalent step | status | FID ↓ | MMD RBF ↓ | KID poly3 ↓ | gen/real |
|---:|---:|---:|---|---:|---:|---:|---:|
| 22,500 | 2,880,000 | 30,000 | `missing/pruned_before_eval` | `NA` | `NA` | `NA` | `NA` |
| 30,000 | 3,840,000 | 40,000 | `ok` | `72.256046` | `0.044132` | `0.056897` | `5000/5000` |
| 37,500 | 4,800,000 | 50,000 | `ok` | `64.769588` | `0.040049` | `0.050480` | `5000/5000` |
| 50,000 | 6,400,000 | 66,667 | `ok` | `58.183655` | `0.035026` | `0.042753` | `5000/5000` |

Detailed local metric files:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00030000/inception_eval_5k/inception_metrics.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00037500/inception_eval_5k/inception_metrics.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00050000/inception_eval_5k/inception_metrics.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/midpoint_22k5_30k_5k_fid_summary.md
```

### Slope / interpretation

The new b128 lr1e-4 cosine/reference-hparam line is **not diverging**. Its FID is monotonically improving:

```text
30k  -> 37.5k: 72.256046 -> 64.769588  = -7.486458 FID
37.5k -> 50k: 64.769588 -> 58.183655  = -6.585933 FID
30k  -> 50k:  72.256046 -> 58.183655  = -14.072391 FID
```

However, at equal processed samples it is still behind the old b96 constant-2e-4 line:

```text
old b96 step 50k:       50,000 × 96  = 4,800,000 samples, FID = 55.528315
new b128 step 37.5k:    37,500 × 128 = 4,800,000 samples, FID = 64.769588
new - old delta at equal samples: +9.241273 FID worse
```

At b128 step 50k, the new line has processed 6.4M samples, i.e. b96-equivalent step 66.7k, and reaches FID `58.183655`. This is clearly improving, but still has not reached the old b96 step-50k anchor.

Current conclusion:

```text
The b128 lr1e-4 cosine/reference-hparam route has a healthy downward FID slope, so the earlier loss uptrend is not sampling-quality collapse by itself. But this route is currently sample-inefficient versus the old b96 constant-2e-4 line at the fair 4.8M-sample anchor. The decisive next comparison is b128 step 75k, which is b96 step-100k equivalent; compare that against old b96 step-100k FID = 47.925749.
```

### Missing 20k / 22.5k note

The exact requested step-20k checkpoint was never available in this run. Step 22.5k existed earlier, but it was pruned before the midpoint controller evaluated it. Therefore the only retained early midpoint result is step 30k.

Controller status:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/midpoint_22k5_30k_after50k_control.json
status: done
steps_available_evaluated: [30000]
steps_missing_or_pruned: [20000, 22500]
```

Future controllers should protect any requested midpoint checkpoints from cleanup, or run the midpoint eval before HF cleanup begins.

### HF artifact upload state

HF artifact repo:

```text
https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
remote_prefix: b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k
```

Upload/cleanup completed:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00050000/hf_post50k_upload_status.json
status: done
updated_at_utc: 2026-05-29T23:02:23Z
```

Uploaded artifacts recorded in:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/hf_artifact_upload_state.json
```

Completed HF commits:

| artifact | HF path | commit |
|---|---|---|
| checkpoint step 37.5k | `b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/checkpoints/step_00037500.pt` | https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/commit/d0ea647b4b680ca15b29ae7c791a3001d1ade936 |
| checkpoint step 50k | `b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/checkpoints/step_00050000.pt` | https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/commit/9561c2aa65ff3d517702930e3c628648bcee1dbe |
| eval step 37.5k | `b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/step_00037500` | https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/commit/db846081b37dc2fec0a8c196f8dd2bfacbc7d9a1 |
| eval step 50k | `b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/step_00050000` | https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/commit/e2b47d23671e43537c6503c221c3d82f08c8f47d |

Local checkpoint state after cleanup:

```text
step_00030000.pt deleted by cleanup at 2026-05-29T22:57:33Z
step_00037500.pt uploaded then deleted at 2026-05-29T22:59:59Z
step_00050000.pt remains local as latest checkpoint at last check, size about 11G
```

### Recommended next action

If continuing this hyperparameter line, run/evaluate the next decisive anchor:

```text
b128 step 75,000 == b96 step 100,000 equivalent
5K generated / 5K real
EMA
sample_steps=32 / script-level NFE=32
```

Compare directly to the old b96 step-100k anchor:

```text
old b96 step 100k, sample_steps=32, 5K/5K true Inception FID = 47.925749
```

### Additional HF upload: step 30k midpoint eval artifacts

After the final summary, the retained step-30k midpoint evaluation artifacts were uploaded manually to the same HF artifact repo/prefix. The step-30k checkpoint itself had already been deleted by cleanup, but the 5K sample/eval artifacts are now centralized on HF.

```text
repo: https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
path: b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/step_00030000
url:  https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/tree/main/b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/step_00030000
```

Midpoint summary/control were also uploaded:

```text
https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/blob/main/b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/midpoint_22k5_30k_5k_fid_summary.md
https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/blob/main/b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/midpoint_22k5_30k_after50k_control.json
```

Local upload state:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/midpoint_hf_upload_state.json
```

## 2026-05-29 update: b128 lr1e-4 cosine resumed from 50k to 75k

User requested continuing the current b128 lr1e-4 cosine/reference-hparam line to the fair old-b96-100k-equivalent anchor:

```text
b128 step 75,000 × 128 = 9,600,000 processed samples
old b96 step 100,000 × 96 = 9,600,000 processed samples
```

A new resume config was created and launched:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_eqepoch_resume_to75k.yaml
/workspace/pdm3-ht-private-sync/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_eqepoch_resume_to75k.yaml
```

Config summary:

```text
exp_name: fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k
resume_from: auto
restore_rng: true
start checkpoint: checkpoints/latest.pt -> step_00050000.pt
max_steps: 75000
global_batch_size: 128
optimizer.lr: 1e-4
optimizer.scheduler: cosine
optimizer.warmup_steps: 10000
optimizer.decay_end_step: 800000
checkpoint_every: 12500
eval.every_steps: 12500
```

Launch state at start:

```text
train pid: 89393
pidfile: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/mfref_b128_eqepoch_75k.pid
train log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_resume_to75k_20260529T231800Z.log
```

The trainer successfully resumed from step 50k:

```text
[2026-05-29 23:18:21] resumed checkpoint path=.../checkpoints/latest.pt step=50000 restore_rng=True
[2026-05-29 23:19:15] step=50050 loss=0.383651 lr=9.94305e-05 ... class_cond=True
```

Initial runtime observation after launch:

```text
latest observed metrics: step 50099
per-step elapsed around 1.0-1.06s
GPU: H100, about 78.9GB used, 100% util, about 600W observed
```

### Step-75k FID controller

A waiting controller was launched for the requested fair anchor:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_75k_5k_gpu_fid_20260529T231741Z.sh
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_75k_5k_gpu_fid_20260529T231741Z.setsid.log
```

Controller PID at launch:

```text
89391
```

It waits for:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/checkpoints/step_00075000.pt
```

Then it waits for the trainer's small step-75k eval to finish / trainer to exit, and runs the requested true FID pass:

```text
num_generated = 5000
num_real = 5000
EMA = true
sample_steps = 32
script-level NFE = 32
precision_mode = bf16_autocast
sample_batch_size = 64
seed = 2026052975
real_seed = 20260529
```

Expected outputs:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00075000/sample_latents_5000.safetensors
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00075000/sample_latents_5000.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00075000/inception_eval_5k/inception_metrics.json
```

### Step-75k HF upload/cleanup controller

A post-FID HF upload/cleanup controller was also launched:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_post75k_hf_upload_cleanup_20260529T231741Z.sh
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_post75k_hf_upload_cleanup_20260529T231741Z.setsid.log
```

Controller PID at launch:

```text
89392
```

It waits for step-75k `inception_metrics.json`, then uploads the checkpoint/eval artifacts to:

```text
repo: LAXMAYDAY/pdm3-ht-model-artifacts
remote_prefix: b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k
archive_extra_steps: 75000
eval_start_step: 75000
```

It keeps the live latest checkpoint locally by default and prunes superseded non-archive local checkpoints.

### ETA

Based on the observed 50k->75k start speed (~1.0-1.1 sec/step after resume), remaining training from step 50k to 75k is approximately 7.0-7.8 hours, plus final checkpoint/small eval and ~8-12 minutes for the 5K sample+Inception pass.

Approximate expected step-75k FID result time from launch:

```text
2026-05-30 06:30-07:10 UTC
```

This will be the decisive equal-processed-samples comparison against:

```text
old b96 step 100k, sample_steps=32, 5K/5K true Inception FID = 47.925749
```

## 2026-05-30 final update: b128 lr1e-4 cosine step-75k 5K true Inception FID

Step-75k finished and the requested fair old-b96-100k-equivalent evaluation is complete.

Run/eval identity:

```text
line: new b128 lr1e-4 cosine/reference-hparam
step: 75,000
processed samples: 75,000 × 128 = 9,600,000
b96-equivalent: 100,000 steps
EMA: true
num_generated / num_real: 5,000 / 5,000
sample_steps: 32
script-level NFE: 32
precision_mode: bf16_autocast
feature extractor: torchvision InceptionV3 pool-2048, image-space after PAE decode
created_at_utc: 2026-05-30T06:36:51+00:00
elapsed_sec: 119.56876546004787
```

Result:

| b128 step | b96-equivalent step | processed samples | FID ↓ | MMD RBF ↓ | KID poly3 ↓ |
|---:|---:|---:|---:|---:|---:|
| 30,000 | 40,000 | 3.84M | `72.25604594006893` | `0.044131939345276594` | `0.05689739428994445` |
| 37,500 | 50,000 | 4.80M | `64.76958787726045` | `0.040048901451779084` | `0.0504804631539697` |
| 50,000 | 66,667 | 6.40M | `58.18365476925243` | `0.03502626901133388` | `0.04275330488187956` |
| 75,000 | 100,000 | 9.60M | `52.63151203759128` | `0.029638864545267207` | `0.035842746291795624` |

Comparison against old b96 constant-2e-4 line at equal processed samples:

| fair anchor | old b96 constant-2e-4 | new b128 lr1e-4 cosine | delta new-old |
|---|---:|---:|---:|
| 4.8M samples | step 50k FID `55.52831543442829` | step 37.5k FID `64.76958787726045` | `+9.24127244283216` worse |
| 9.6M samples | step 100k FID `47.925748666494485` | step 75k FID `52.63151203759128` | `+4.705763371096795` worse |

Slope of the new b128 line:

```text
30k   -> 37.5k: 72.256046 -> 64.769588 = -7.486458 FID
37.5k -> 50k:   64.769588 -> 58.183655 = -6.585933 FID
50k   -> 75k:   58.183655 -> 52.631512 = -5.552143 FID
30k   -> 75k:   72.256046 -> 52.631512 = -19.624534 FID
```

Conclusion:

```text
The b128 lr1e-4 cosine/reference-hparam line is not collapsed; FID continues to improve monotonically through the 75k / b96-100k-equivalent anchor. However, at the fair equal-processed-samples comparison it has still not caught the old b96 constant-2e-4 route: step-75k b128 is FID 52.6315, while old b96 step-100k is FID 47.9257, so the new line remains +4.7058 FID worse. The gap narrowed from +9.24 at 4.8M samples to +4.71 at 9.6M samples, but old b96 remains the stronger baseline so far.
```

Artifacts:

```text
metrics JSON:
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00075000/inception_eval_5k/inception_metrics.json

metrics MD:
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00075000/inception_eval_5k/inception_metrics.md

sample latents:
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00075000/sample_latents_5000.safetensors
```

Controller states:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00075000/5k_anchor_control.json
status: done
updated_at_utc: 2026-05-30T06:37:36Z

/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00075000/hf_post75k_upload_status.json
status: done
updated_at_utc: 2026-05-30T06:40:23Z
```

HF upload completed:

| artifact | HF path | commit |
|---|---|---|
| checkpoint step 75k | `b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/checkpoints/step_00075000.pt` | https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/commit/e95605a1ccc556a02eb8b8bff5dacba2dbf545f7 |
| eval step 75k | `b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/step_00075000` | https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/commit/c9c32323aae10bd456ccf61b1e8d5e412158d3cf |

Local checkpoint state after post-75k upload/cleanup:

```text
checkpoints/latest.pt -> step_00075000.pt
checkpoints/step_00075000.pt remains local as latest, size about 11G
step_00050000.pt and step_00062500.pt were deleted as non-archive checkpoints superseded by latest
```

No active b128 train/sample/decode/post75k controller processes were present at the post-result check.

## 2026-05-30 update: launched b128 lr1e-4 cosine continuation 75k -> 100k

User requested continuing the current b128 lr1e-4 cosine/reference-hparam line from step 75k to step 100k.

Important framing:

```text
b128 step 100,000 × 128 = 12,800,000 processed samples
b96-equivalent step = 133,333
```

Therefore this is **not** the fair equal-samples comparison to old b96 step 100k; that fair anchor was b128 step 75k and is already complete. The step-100k continuation tests whether longer training on the slower b128 lr1e-4 cosine line keeps closing the quality gap.

New config:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_eqepoch_resume_to100k.yaml
/workspace/pdm3-ht-private-sync/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_mfref_b128_lr1e4_cosine_eqepoch_resume_to100k.yaml
```

Config delta from resume-to-75k:

```text
max_steps: 100000
resume_from: auto
restore_rng: true
start checkpoint: checkpoints/latest.pt -> step_00075000.pt
checkpoint_every: 12500
eval.every_steps: 12500
```

Launch state:

```text
launched_at_utc: 2026-05-30T07:04:34Z
train pid: 93280
pidfile: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/mfref_b128_eqepoch_100k.pid
train log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_resume_to100k_20260530T070434Z.log
```

Trainer resume confirmation:

```text
[2026-05-30 07:04:52] resumed checkpoint path=.../checkpoints/latest.pt step=75000 restore_rng=True
[2026-05-30 07:04:52] optimizer lr=0.0001 scheduler=cosine min_lr=1e-05 warmup_steps=10000 decay_end_step=800000
[2026-05-30 07:05:48] step=75050 loss=0.394192 lr=9.85027e-05 ... class_cond=True
```

Runtime at first post-launch check:

```text
GPU: H100, ~77.7 GiB used, ~97% util, ~588W, temp ~53C
```

Step-100k 5K FID controller:

```text
pid: 93277
script: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_100k_5k_gpu_fid_20260530T070434Z.sh
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_100k_5k_gpu_fid_20260530T070434Z.sh.setsid.log
status: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00100000/5k_anchor_control.json
current status at launch check: waiting_for_checkpoint
```

Step-100k eval plan:

```text
num_generated / num_real: 5000 / 5000
EMA: true
sample_steps: 32
script-level NFE: 32
precision_mode: bf16_autocast
sample seed: 2026053001
real seed: 20260529
```

Step-100k HF upload/cleanup controller:

```text
pid: 93278
script: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_post100k_hf_upload_cleanup_20260530T070434Z.sh
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/mfref_b128_eqepoch_post100k_hf_upload_cleanup_20260530T070434Z.sh.setsid.log
status: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k/eval/step_00100000/hf_post100k_upload_status.json
remote_prefix: b3_meanflow_realdata/mfref_b128_lr1e4_cosine_eqepoch_37k5_50k
archive_extra_steps: 100000
eval_start_step: 100000
```

Expected checkpoints/evals:

```text
step 87,500: trainer checkpoint + small eval, around 2026-05-30 10:50-11:20 UTC if speed holds
step 100,000: trainer checkpoint + small eval + 5K true Inception FID + HF upload/cleanup
ETA for step-100k 5K FID result: roughly 2026-05-30 14:30-15:10 UTC
```

Monitoring commands:

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
RES="$EXP/results/fullcache_realdata_mfref_b128_lr1e4_cosine_eqepoch_37k5_50k"

date -u '+now=%Y-%m-%dT%H:%M:%SZ'
ps -p "$(cat $EXP/mfref_b128_eqepoch_100k.pid)" -o pid,ppid,pgid,sid,stat,etime,%cpu,%mem,rss,cmd || true
ps -p "$(cat $EXP/mfref_b128_eqepoch_100k_fid_controller.pid)" -o pid,ppid,pgid,sid,stat,etime,%cpu,%mem,rss,cmd || true
ps -p "$(cat $EXP/mfref_b128_eqepoch_post100k_hf_upload_cleanup.pid)" -o pid,ppid,pgid,sid,stat,etime,%cpu,%mem,rss,cmd || true

tail -n 80 "$EXP/logs/mfref_b128_eqepoch_resume_to100k_20260530T070434Z.log"
cat "$RES/eval/step_00100000/5k_anchor_control.json" 2>/dev/null || true
cat "$RES/eval/step_00100000/hf_post100k_upload_status.json" 2>/dev/null || true
nvidia-smi --query-gpu=utilization.gpu,power.draw,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
```
