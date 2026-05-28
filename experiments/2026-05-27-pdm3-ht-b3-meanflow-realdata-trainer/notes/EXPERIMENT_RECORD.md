# Experiment Record — B3 Real-Data Training Loop

## 2026-05-27 start

User request: while full ImageNet-1k PAE latent cache is running, write the real-data B3 training loop engineering layer:

- DataLoader over 1.28M latents
- label/class conditional injection
- eval loop with sampling + FID hook
- checkpoint save/resume

Initial full-cache status observed before starting this experiment:

- full-cache process active under experiment `2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache`
- progress around 70k/1,281,167 samples encoded
- throughput around 37 samples/s end-to-end
- smoke4096 latent cache already verified

Switchyard note: `switchyard` command was not present in this environment, so no peer HYARD delegation could be launched; proceeded locally.

## Commands

Commands and results will be appended below.

## 2026-05-27 implementation

Created:

- `scripts/train_b3_meanflow_realdata.py`
- `configs/b3_meanflow_realdata_smoke4096.yaml`
- `configs/b3_meanflow_realdata_full.yaml`
- result directories under `results/`

Implemented features:

- `ShardIndexedLatentDataset`: snapshots safetensor shard list and uses cumulative shard offsets instead of a 1.28M-entry Python map.
- DataLoader knobs: workers, prefetch, persistent workers, shuffle/drop_last, random `latents` vs `latents_flip`.
- class-conditional injection: batch labels are passed as `model_kwargs['y']`; `force_drop_ids` is sampled once per train batch and reused through JVP target and bf16 train forward.
- MeanFlow precision: `bf16_backbone_fp32_jvp`, live-param JVP target, target detached, no EMA target.
- r=t sampler: `equal_prob=0.75` from transport.
- metrics JSONL, FD audit, CUDA peak memory logging.
- checkpoint save/resume state: model, EMA, optimizer, RNG, config, dataset snapshot; `latest.pt` symlink points to latest step checkpoint.
- eval hook: latent Euler sampler saves `sample_latents.safetensors`; optional external `fid_command_template` can be configured. Smoke leaves it empty and records `skipped_no_command`.

## 2026-05-27 smoke4096 command

```bash
cd /workspace/PDM
python experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py \
  --config experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_smoke4096.yaml \
  2>&1 | tee experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/010_smoke4096.log
```

## 2026-05-27 smoke4096 result

Output directory:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/smoke_realdata_4096
```

Key result:

```json
{
  "practical_gate_pass": true,
  "dataset_total_samples": 4096,
  "dataset_num_shards": 1,
  "latent_shape": [32, 16, 16],
  "final_step": 6,
  "optimizer_steps_this_run": 6,
  "final_loss": 3.4510116577148438,
  "mean_realized_r_eq_t_fraction": 0.7916666666666666,
  "fd_rel_err_full_jvp_last": 6.783736791015577e-05,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0,
  "loop_peak_memory_mb": 63.06005859375,
  "checkpoint_ok": true,
  "eval_sample_ok": true,
  "fid_status": "skipped_no_command"
}
```

Artifacts verified:

- `metrics.jsonl` written.
- `dataset_snapshot.json` written.
- checkpoints: `step_00000003.pt`, `step_00000006.pt`, `latest.pt` symlink.
- checkpoint keys verified: `model`, `ema`, `optimizer`, `rng_state`, `config`, `dataset_summary`, `step`.
- eval latent samples saved at steps 3 and 6.
- class conditional labels verified per step (`class_cond_injected=True`).
- FD audits at steps 1, 3, 6 all passed `<1e-2`; last rel err `6.78e-05`.

## 2026-05-27 partial full-cache dataset scan

Command imported the experiment dataset and scanned active full-cache directory without training.

Result at `2026-05-27T17:06:13Z`:

```json
{
  "num_shards": 22,
  "total_samples": 90112,
  "expected_total_from_config": 1281167,
  "ready_for_full_train": false,
  "latent_shape": [32, 16, 16],
  "skipped_files_count": 0,
  "scan_elapsed_sec": 0.14347953093238175
}
```

Interpretation: full-cache writer is still running; current scanner correctly snapshots only completed `.safetensors` shards and does not include the pending unsaved shard samples.

<!-- B3_LONGRUN_HF_ARCHIVE_POLICY_20260528T1248Z -->

## 2026-05-28 long-run archive/HF cleanup policy

User requested longer checkpoint intervals plus automatic HF upload and local checkpoint cleanup, with the note that the original PAE paper's first meaningful comparable point is around `1.07M` steps.

Applied policy for the B3 b96 route:

- `train.max_steps: 1070000`
- `train.checkpoint_every: 10000`
- `train.keep_last_checkpoints: 2`
- `meanflow.fd_audit_every: 10000`
- HF model artifact repo: `LAXMAYDAY/pdm3-ht-model-artifacts`
- Remote prefix: `b3_meanflow_realdata/fullcache_b96`
- HF archive cadence: sparse `100k` multiples plus final `1,070,000`; do not upload every 10k local safety checkpoint.
- Local policy: keep the current `latest.pt` target for fast resume; delete uploaded archive checkpoints once superseded; delete non-archive checkpoints once a newer local latest exists.

The step-20000 full checkpoint was uploaded successfully to HF and then removed locally:

```text
remote: b3_meanflow_realdata/fullcache_b96/checkpoints/step_00020000.pt
commit: https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/commit/fc5bb486b2a82244728e4cfc2195b6cd4c234772
sha: fc5bb486b2a82244728e4cfc2195b6cd4c234772
```

Next archive/eval gate: step `30000` checkpoint + EMA sample + PAE decode + image-space Inception FID/MMD/KID diagnostics. These early small-sample metrics remain convergence diagnostics, not official 50k ImageNet FID.

Caveat: the live process started before the config edit and therefore still uses its old in-memory schedule until a controlled restart/resume from `latest.pt` or natural resume after the old `100000`-step stop.

