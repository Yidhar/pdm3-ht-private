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
- Preferred anchor timing is therefore the next natural `60k` checkpoint, using a controlled short trainer pause for GPU sampling and then immediate resume.

One-shot controller for the preferred `60k` anchor:

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
nohup env STEP=60000 NUM_SAMPLES=5000 \
  bash "$EXP/scripts/run_b3_5k_fid_anchor_at_step.sh" \
  > "$EXP/logs/b3_5k_fid_anchor_step_00060000_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 &
```

The controller waits for `step_00060000.pt` and the trainer's normal small eval, stops the trainer process group to free the H100, writes `eval/step_00060000/sample_latents_5000.safetensors`, resumes training from `latest.pt` with the on-disk long-run config, then launches CPU decode/Inception into `eval/step_00060000/inception_eval_5k/`.
