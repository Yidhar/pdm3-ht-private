# B3 b96 step 10000 eval / sample pipeline summary

更新时间：`2026-05-28T08:44:24Z`

## Status

Step `10000` gate 已完成，且主训练未中断继续运行。这个 gate 现在覆盖用户要求的五项：checkpoint、EMA latent sample、image-space decoded samples、compact FID/MMD/KID smoke metrics、train/eval summary。

## 1. Checkpoint

| item | value |
|---|---|
| checkpoint | `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00010000.pt` |
| exists | `True` |
| latest checkpoint record | `10000` @ `2026-05-28T08:38:20Z` |
| current training last seen | step `10441` @ `2026-05-28T08:44:23Z` |

## 2. EMA sample / latent eval

Built-in eval at step `10000` completed with EMA enabled.

| field | value |
|---|---:|
| sample_status | `ok` |
| use_ema | `True` |
| sample_shape | `[64, 32, 16, 16]` |
| sample_finite | `True` |
| sample_mean | `0.1781466` |
| sample_std | `0.93171239` |
| sample_elapsed_sec | `4.2959515` |
| label_unique_count | `64` |
| fid_status | `skipped_no_command` |
| official_fid | `None` |

Latent sample path:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/sample_latents.safetensors
```

## 3. Image-space decoded samples

CPU PAE decode smoke completed without disturbing H100 training.

| item | value |
|---|---|
| decode status | `ok` |
| device / dtype | `cpu` / `torch.float32` |
| elapsed_sec | `53.670049` |
| generated count | `64` |
| real ref count | `64` |
| generated grid | `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/images/generated_grid.png` |
| real ref grid | `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/images/real_ref_grid.png` |
| generated PNG dir | `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/images/generated` |
| real-ref PNG dir | `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/images/real_ref` |
| decoded npz | `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/decoded_samples.npz` |

Generated image stats:

| split | shape | mean | std | min | max | finite |
|---|---|---:|---:|---:|---:|---|
| generated | `[64, 256, 256, 3]` | `0.47714785` | `0.16552858` | `0` | `1` | `True` |
| real_ref | `[64, 256, 256, 3]` | `0.44521365` | `0.27499324` | `0` | `1` | `True` |

Latent stats:

| split | shape | mean | std | min | max | finite |
|---|---|---:|---:|---:|---:|---|
| generated | `[64, 32, 16, 16]` | `0.17814662` | `0.93171155` | `-3.179611` | `5.1052089` | `True` |
| real_ref | `[64, 32, 16, 16]` | `0.20653252` | `0.97844017` | `-2.9375` | `4.4375` | `True` |

## 4. Compact metrics

> 注意：这里的 compact metrics 是 deterministic random-projection image-space smoke metrics，用于确认 image decode / metrics wiring 已接通；不是 official Inception FID/KID。当前 config 的 `eval.fid_command_template` 为空，所以 official FID 在内置 eval 中仍为 `skipped_no_command`。

| metric | value |
|---|---:|
| compact_fid_rp128 | `4.0626297` |
| compact_mmd_rbf_rp128 | `0.049283276` |
| compact_kid_poly3_rp128 | `0.0034318871` |
| compact_mmd_rbf_bandwidth2 | `10.914266` |
| feature_method | `uint8_rgb_downsample_random_projection` |
| feature_seed | `20260528` |

Metrics files:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/compact_metrics.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/compact_metrics.md
```

## 5. Train/eval runtime summary

| window | mean sec/step | samples/s @ b96 | steps/hour | range |
|---:|---:|---:|---:|---|
| last 50 | `0.817788` | `117.390` | `4402.1` | `10392`→`10441` |
| last 200 | `0.805808` | `119.135` | `4467.6` | `10242`→`10441` |
| last 1000 | `0.801577` | `119.764` | `4491.1` | `9442`→`10441` |


Latest live train record:

```json
{
  "step": 10441,
  "created_at_utc": "2026-05-28T08:44:23Z",
  "loss": 0.373961865901947,
  "elapsed_sec": 0.7954787518829107,
  "batch_size": 96
}
```

## 6. FD audit risk note

Built-in FD audit remains finite and validates the r=t degenerate path, but step `10000` is still above the previous practical FD gate `1e-2`:

```json
{
  "step": 10000,
  "created_at_utc": "2026-05-28T08:37:57Z",
  "fd_eps": 0.01,
  "fd_rel_err_full_jvp": 0.06135418799846591,
  "fd_abs_err_norm": 8.050538063049316,
  "du_norm": 131.42738342285156,
  "fd_norm": 131.2141571044922,
  "r_eq_t_count": 81,
  "r_eq_t_total": 96,
  "realized_r_eq_t_fraction": 0.84375,
  "u_finite": true,
  "du_finite": true,
  "fd_finite": true,
  "target_detached": true,
  "no_nan_or_inf": true,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0
}
```

Dedicated A/B/C eps sweep script is ready here:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/audit_fd_jvp_modes.py
```

Recommended command once GPU memory is available or after a durable checkpoint short stop/resume:

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
python3 "$EXP/scripts/audit_fd_jvp_modes.py" \
  --config "$EXP/configs/b3_meanflow_realdata_full_fast_h100_b96.yaml" \
  --checkpoint "$EXP/results/fullcache_realdata_singleproc_template/checkpoints/step_00010000.pt" \
  --batch-size 8 \
  --eps 1e-2 3e-3 1e-3 \
  --modes A B C \
  --output-dir "$EXP/results/fd_jvp_dedicated_audit/step_00010000_live"
```

Current action: do **not** run the second DiT-XL audit process concurrently while the H100 training process is occupying ~74GB unless explicitly accepting the OOM/slowdown risk. Safe path is to run it after a durable checkpoint pause/resume or when training frees GPU memory.

## 7. Code / artifact routing

New helper scripts to sync to GitHub code repo:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/decode_and_compact_eval_step.py
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/audit_fd_jvp_modes.py
```

Artifact routing remains:

- GitHub `<https://github.com/Yidhar/pdm3-ht-private>`: code/config/docs/handoff only.
- HF `<https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts>`: model/eval artifacts/docs. Step-10000 decoded images/compact metrics are lightweight enough to upload there; multi-GB checkpoint upload should be treated as explicit model artifact publish.
