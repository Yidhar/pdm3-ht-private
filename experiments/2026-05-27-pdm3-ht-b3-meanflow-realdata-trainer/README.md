# B3 MeanFlow Real-Data Trainer

This experiment adds a real-data training loop for Phase-0/B3 MeanFlow over PAE latent safetensors.

## Main script

```bash
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py
```

It imports the already validated MeanFlow model and transport from:

```text
/workspace/PDM/external/PAE/pae_with_generator
```

## Smoke run

Uses the fixed 4096-sample real PAE latent cache:

```bash
cd /workspace/PDM
python experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py \
  --config experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_smoke4096.yaml \
  2>&1 | tee experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/010_smoke4096.log
```

Expected outputs:

```text
results/smoke_realdata_4096/
  log.txt
  config_resolved.yaml
  dataset_snapshot.json
  metrics.jsonl
  train_summary.json
  train_summary.md
  checkpoints/latest.pt
  checkpoints/step_00000003.pt
  eval/step_00000003/sample_latents.safetensors
```

## Full-cache config

`configs/b3_meanflow_realdata_full.yaml` points at the full ImageNet-1k PAE latent cache:

```text
/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full
```

It is intended for use after the full cache is complete. The dataset scanner snapshots the file list at startup and records it in `dataset_snapshot.json`.

## Resume

Set in config:

```yaml
train:
  resume_from: auto   # or an explicit checkpoint path
```

`auto` loads `checkpoints/latest.pt` from the configured output/exp directory if present.

## Eval/FID hook

The smoke config samples latent tensors and skips FID. For a real FID pipeline, set:

```yaml
eval:
  enabled: true
  every_steps: 10000
  fid_command_template: "python path/to/fid.py --samples {sample_dir} --out {output_dir} --step {step}"
```

The trainer writes stdout/stderr and attempts to parse a numeric `FID` from command output.

## Current smoke status — 2026-05-27

Smoke `smoke_realdata_4096` completed successfully.

- Gate: `practical_gate_pass=true`
- Dataset: 4096 real PAE latents, 1 shard, shape `[32,16,16]`, labels min/max `0/999`
- Training: 6/6 optimizer steps, all finite
- Class-conditioning: labels injected every step
- MeanFlow recipe: bf16 backbone + fp32 live-param JVP target
- r=t sampler: observed mean r=t fraction `0.7917` for tiny batch smoke; configured probability `0.75`
- FD audit last rel err: `6.78e-05`
- r=t degenerate target-v max abs: `0.0`
- CUDA loop peak memory: `63.06 MiB` for Tiny smoke, while full-cache builder stayed active
- Checkpoints: `step_00000003.pt`, `step_00000006.pt`, `latest.pt`
- Eval: latent samples saved at steps 3 and 6; FID skipped because no FID command configured

Partial full-cache scan at `2026-05-27T17:06:13Z` saw 22 completed shards / 90,112 samples; expected full total is 1,281,167, so the full train config is not ready for full run yet.

## Convenience wrappers

```bash
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_smoke4096.sh
bash experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/scan_fullcache_dataset.sh
```

<!-- B3_LONGRUN_HF_ARCHIVE_POLICY_20260528 -->

## Long-run archive / HF artifact policy — 2026-05-28

The active B3 b96 H100 route is now treated as a long run toward the first PAE-paper-comparable regime:

- Target steps after restart/resume: `1,070,000` (`train.max_steps: 1070000`).
- Rationale: the original PAE paper's first meaningful comparable point is around `1.07M` steps, so `20k/30k/100k` are early diagnostics/pipeline gates, not final quality comparisons.
- Full trainer checkpoints are large (`~11 GiB`) because they include model, EMA, optimizer and RNG state.
- Local safety checkpoint cadence is `10k` steps (`checkpoint_every: 10000`) with `keep_last_checkpoints: 2`, but long-term HF archives are sparse: `100k` multiples plus the final `1.07M` step.
- A background HF watcher uploads only sparse archive checkpoints to `LAXMAYDAY/pdm3-ht-model-artifacts` under `b3_meanflow_realdata/fullcache_b96/` and deletes uploaded/superseded local `.pt` files while keeping the current `latest.pt` target for fast resume.
- Future image-space eval artifacts are uploaded to HF from step `>=30000`, excluding raw real ImageNet images / raw NPZ payloads by default. The allow-list includes the trainer's normal `sample_latents.safetensors`, larger anchor files such as `sample_latents_5000.safetensors` plus their JSON metadata, generated PNGs/grids when present, metric JSON/MD files, and logs.

Operational caveat: the process already running on 2026-05-28 loaded the old config in memory. The updated `max_steps=1070000`, `checkpoint_every=10000` and `fd_audit_every=10000` become fully active after a controlled restart/resume from `checkpoints/latest.pt` (or after the old run reaches its previous stop and is resumed).


<!-- B3_5K_INCEPTION_FID_ANCHOR_20260528 -->

## 5k true Inception FID anchor plan — 2026-05-28

The normal trainer eval intentionally emits only `64` EMA samples every `10k` steps. Those image-space Inception numbers are useful for smoke/trend monitoring, but they have high finite-sample variance and are not comparable to paper-scale FID. At step `50k` or `60k`, run one larger `5,000`-sample Inception pass to establish a more stable early baseline.

Important scope notes:

- This is still a `5k` anchor, not the full publishable `50k` FID commonly reported in papers.
- Use the current base route only: PAE DINOv2-L d32 cache + LightningDiT B3/XL-like backbone + MeanFlow objective + EMA sampling. Representation Fréchet Loss / FD-loss remains deferred.
- On the single-H100 machine, CUDA sampling should be done during a short controlled pause at a checkpoint boundary because the live trainer uses most H100 memory.
- After the latent file is written, training can be resumed and CPU decode/Inception can run in the background if we want to minimize GPU downtime.
- Do not upload raw real ImageNet images or decoded real-image NPZ payloads. Use `--max-save-images 0 --grid-count 0 --no-save-npz` for the 5k pass unless explicitly requested otherwise.

Recommended 5k recipe after a step checkpoint exists:

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
CFG="$EXP/configs/b3_meanflow_realdata_full_fast_h100_b96.yaml"
RES="$EXP/results/fullcache_realdata_singleproc_template"
STEP=50000   # or 60000
CKPT="$RES/checkpoints/step_$(printf '%08d' "$STEP").pt"
EVAL="$RES/eval/step_$(printf '%08d' "$STEP")"

# 1) Short GPU step: generate 5k EMA latents from the checkpoint.
python "$EXP/scripts/sample_b3_meanflow_eval_latents.py" \
  --config "$CFG" \
  --checkpoint "$CKPT" \
  --num-samples 5000 \
  --sample-batch-size 64 \
  --sample-steps 32 \
  --precision-mode bf16_autocast \
  --seed 2026052850 \
  --output-latents "$EVAL/sample_latents_5000.safetensors"

# 2) CPU/background step: PAE decode and real ImageNet-256 Inception metrics.
#    This avoids H100 contention after the trainer is resumed.
nohup python "$EXP/scripts/decode_and_inception_eval_step.py" \
  --sample-latents "$EVAL/sample_latents_5000.safetensors" \
  --output-dir "$EVAL/inception_eval_5k" \
  --num-real 5000 \
  --real-seed 20260528 \
  --decode-device cpu \
  --decode-batch-size 8 \
  --inception-device cpu \
  --inception-batch-size 64 \
  --torch-num-threads 64 \
  --max-save-images 0 \
  --grid-count 0 \
  --no-save-npz \
  > "$EVAL/inception_eval_5k.log" 2>&1 &
```

Decision gate around the existing 64-sample compact/image-space FID trend:

- If the step-50k small eval is `< 307` or basically flat/improving, treat the 30k→40k bump as likely sample-count noise and run the 5k anchor at either `50k` or the next natural `60k` checkpoint.
- If the step-50k small eval is `> 310` and still rising, run the 5k anchor immediately from the step-50k checkpoint before changing LR; use the 5k number to distinguish true degradation from noisy 64-sample diagnostics.

Actual 50k gate status:

- Step `50k` 64-sample image-space Inception eval completed with `FID=286.290236`, `MMD=0.028853`, `KID=0.036241` (`n_gen=n_real=64`).
- This passes the `<307` health gate, so no LR change is recommended solely from the earlier loss increase or the 30k→40k small-FID bump.
- Historical note: this document originally suggested using the next natural `60k` checkpoint, but the user requested an immediate exact step-50k 5K anchor. That exact step-50k anchor has now completed; see the actual-result section below.

One-shot controller for a future optional `60k` anchor:

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
nohup env STEP=60000 NUM_SAMPLES=5000 \
  bash "$EXP/scripts/run_b3_5k_fid_anchor_at_step.sh" \
  > "$EXP/logs/b3_5k_fid_anchor_step_00060000_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 &
```

For a future optional `60k` or later anchor, the controller waits for the chosen `step_XXXXXXXX.pt` and the trainer's normal small eval, stops the trainer process group to free the H100, writes `eval/step_XXXXXXXX/sample_latents_5000.safetensors`, resumes training from `latest.pt` with the on-disk long-run config, then launches CPU decode/Inception into `eval/step_XXXXXXXX/inception_eval_5k/`.

<!-- B3_STEP50000_5K_FID_ACTUAL_20260528 -->

## Actual step-50k 5K true Inception FID anchor — 2026-05-28

The exact step-50k checkpoint was retained and evaluated with a `5,000` generated / `5,000` real ImageNet-256 true Inception pass. This is the first meaningful early image-space anchor for the PAE B3 b96 MeanFlow route; the normal `64`-sample trainer FID remains a smoke/wiring metric only.

| metric | value |
|---|---:|
| checkpoint | `step_00050000` |
| generated / real | `5000 / 5000` |
| FID ↓ | `55.52831543442829` |
| Inception RBF MMD ↓ | `0.032845868596164784` |
| Inception poly3 KID ↓ | `0.03894902108381171` |
| feature method | `torchvision_inception_v3_imagenet1k_v1_pool2048` |
| 5K sampling elapsed | `321.56 sec` |
| GPU decode + Inception elapsed | `107.22 sec` |

Local artifacts:

```text
results/fullcache_realdata_singleproc_template/eval/step_00050000/sample_latents_5000.safetensors
results/fullcache_realdata_singleproc_template/eval/step_00050000/sample_latents_5000.json
results/fullcache_realdata_singleproc_template/eval/step_00050000/inception_eval_5k/inception_metrics.json
results/fullcache_realdata_singleproc_template/eval/step_00050000/inception_eval_5k/inception_metrics.md
```

HF artifacts were refreshed under:

```text
https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/tree/main/b3_meanflow_realdata/fullcache_b96/step_00050000
```

Latest relevant HF commit containing the 5K Inception metrics:

```text
019298e6dee68c5b2963015e60eb5b5a8210e194
```

Training resumed after eval from `checkpoints/latest.pt -> step_00052000.pt`; the live run did not roll back to 50k. The retained exact eval checkpoint is stored locally under `checkpoints_retained_for_eval/step_00050000_5k_anchor.pt`.

<!-- B3_COSINE_LR_PLAN_20260528 -->

## Planned later cosine LR decay — 2026-05-28

No LR scheduler is active in the live trainer at this point; current live config still uses `optimizer.lr: 0.0002`. For a later checkpoint-controlled switch or a new long-run branch, use the batch-scaled MeanFlow reference plan:

```text
reference: batch 128, lr 1e-4, cosine decay
active batch: 96
batch-scaled base_lr = 1e-4 * 96 / 128 = 7.5e-5
min_lr = 7.5e-6
warmup_steps = 13333
end_step = 1070000
```

Cosine formula:

```python
if step < warmup_steps:
    lr = base_lr * step / warmup_steps
else:
    p = min(1.0, max(0.0, (step - warmup_steps) / (end_step - warmup_steps)))
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + cos(pi * p))
```

Schedule values: `50k -> 7.480e-5`, `100k -> 7.389e-5`, `400k -> 5.505e-5`, `800k -> 1.780e-5`, `1.07M -> 7.500e-6`.

For the already-running `2e-4` run, do not rewarm. If switching this run, resume at a natural checkpoint and explicitly override optimizer param-group LR after checkpoint load (`scheduler_over_checkpoint`), optionally ramping down from `2e-4` to the cosine target over `2k–5k` steps. Given the healthy step-50k 5K FID (`55.53`), this is a planned later control knob rather than an emergency hot change.

<!-- B3_STEP100000_5K_FID_AND_150K_ANCHOR_20260529 -->

## Actual step-100k 5K true Inception FID anchor and slope status — 2026-05-29

The exact step-100k checkpoint was evaluated with the same `5000` generated / `5000` real true Inception protocol used at step-50k.

| step | generated / real | FID ↓ | Inception RBF MMD ↓ | Inception poly3 KID ↓ |
|---:|---:|---:|---:|---:|
| `50000` | `5000 / 5000` | `55.52831543442829` | `0.032845868596164784` | `0.03894902108381171` |
| `100000` | `5000 / 5000` | `47.925748666494485` | `0.02595176471424887` | `0.02984040431452506` |

Step-50k → step-100k improved by `-7.6025667679338085` FID, or `-13.691333346698498%`.

Interpretation:

- This is not a collapse signal: the true 5K anchor improved.
- It is also not a full green light: the absolute FID is still high and the slope is shallow enough to justify denser anchors and/or a controlled LR/sampler branch if the next anchor stalls.
- The normal trainer 64-sample FID remains smoke only.

Important sample-budget normalization:

```text
PAE paper 80ep point: 100000 steps × batch1024 = 102.4M image presentations
current run step-100k: 100000 steps × batch96 = 9.6M image presentations
current / PAE sample budget: 96 / 1024 = 9.375%
current step-100k ≈ PAE-paper-equivalent step 9375 ≈ 7.5 epochs
first comparable point ≈ 100000 × 1024 / 96 = 1,066,667 current steps
```

So do not compare `100k @ batch96` directly to PAE's `100k @ batch1024` / 80-epoch result.

Because the step-100k slope is a yellow flag, a step-150k 5K GPU anchor controller was launched:

```text
pid: 74230
script: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/b3_150k_5k_gpu_fid_then_resume_20260529T060838Z.sh
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/b3_150k_5k_gpu_fid_then_resume_20260529T061233Z_setsid.log
status: waiting_for_checkpoint
status json: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00150000/5k_anchor_control.json
```

Decision gate:

- If step-150k improves by several FID points, keep the current constant-`2e-4` run as the main control.
- If step-150k is flat/worse, branch from a saved checkpoint and test a lower/cosine LR schedule and/or sampler sensitivity sweep (`32` vs `64/128` steps, plus CFG scale/interval if supported) rather than hot-changing the live run without a control.
