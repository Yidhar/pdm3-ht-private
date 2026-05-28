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
- Future image-space eval artifacts are uploaded to HF from step `>=30000`, excluding raw real ImageNet images / raw NPZ payloads by default.

Operational caveat: the process already running on 2026-05-28 loaded the old config in memory. The updated `max_steps=1070000`, `checkpoint_every=10000` and `fd_audit_every=10000` become fully active after a controlled restart/resume from `checkpoints/latest.pt` (or after the old run reaches its previous stop and is resumed).

