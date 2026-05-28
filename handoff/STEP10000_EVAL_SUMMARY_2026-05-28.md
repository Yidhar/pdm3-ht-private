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


<!-- B3_B96_STEP10000_EVAL_SYNCED_20260528T0849Z -->

## Step 10000 eval smoke artifacts synced

更新时间：`2026-05-28T08:46:29Z`

GitHub code/docs sync complete:

```text
repo: https://github.com/Yidhar/pdm3-ht-private
branch: main
commit: 5f53937046bff6e4b51ee30d2b2416ad15ccd93c
commit_msg: Add B3 step10000 eval smoke and FD audit tools
```

Included in GitHub: new decode helper, FD A/B/C audit helper, and handoff summaries. Artifact exclusion check passed before push; no checkpoint/cache/results/logs were committed.

HF model artifacts sync complete:

```text
repo: https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
repo_sha_after_upload: 616b9c47d548936710111e49d619c4fe7ad26273
eval_artifact_dir: b3_meanflow_realdata/fullcache_b96/step_00010000/eval
summary_doc: handoff/STEP10000_EVAL_SUMMARY_2026-05-28.md
```

Uploaded lightweight step-10000 eval artifacts to HF: `sample_latents.safetensors`, `decoded_samples.npz`, generated/real-ref PNGs, grids, `compact_metrics.json`, `compact_metrics.md`, and `eval_record.json`. The multi-GB training checkpoint remains local only unless explicitly publishing checkpoints to HF.

<!-- B3_B96_DEDICATED_FD_AUDIT_DONE_20260528T0916Z -->

## Dedicated FD/JVP A/B/C audit complete

更新时间：`2026-05-28T09:16Z`

专项 FD/JVP audit 已按用户建议在 durable `step_00012000.pt` 后短暂停训执行；audit 使用 `step_00010000.pt`、固定 seed、小 batch `8`、`model.eval()`、`equal_prob=0` 非退化 batch、class dropout 关闭、JVP/FD fp32，并 sweep `eps=[1e-2, 3e-3, 1e-3]`。

Audit 输出：

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fd_jvp_dedicated_audit/step_00010000_after_step_12000_20260528T085000Z/fd_jvp_modes_audit.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fd_jvp_dedicated_audit/step_00010000_after_step_12000_20260528T085000Z/fd_jvp_modes_audit.jsonl
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fd_jvp_dedicated_audit/step_00010000_after_step_12000_20260528T085000Z/fd_jvp_modes_audit.md
```

Compact result table:

| mode | TF32 | global SDPA | eps | fd_rel | abs_err | du_norm | fd_norm | no_nan | backbone_rel | degen_target-v |
|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| A | off | math | 1e-02 | `0.00752493` | `1.15436` | `153.831` | `153.404` | true | `0.00623756` | `0` |
| A | off | math | 3e-03 | `0.000765322` | `0.1177` | `153.831` | `153.792` | true | `0.00623756` | `0` |
| A | off | math | 1e-03 | `0.00095552` | `0.146983` | `153.831` | `153.825` | true | `0.00623756` | `0` |
| B | off | default | 1e-02 | `0.00752493` | `1.15436` | `153.831` | `153.404` | true | `0.00623485` | `0` |
| B | off | default | 3e-03 | `0.000765322` | `0.1177` | `153.831` | `153.792` | true | `0.00623485` | `0` |
| B | off | default | 1e-03 | `0.00095552` | `0.146983` | `153.831` | `153.825` | true | `0.00623485` | `0` |
| C | on | default | 1e-02 | `0.0401946` | `6.18794` | `153.789` | `153.95` | true | `0.00624116` | `0` |
| C | on | default | 3e-03 | `0.125223` | `19.385` | `153.789` | `154.804` | true | `0.00624116` | `0` |
| C | on | default | 1e-03 | `0.348566` | `56.6146` | `153.789` | `162.421` | true | `0.00624116` | `0` |

结论：A/B（TF32 off，math/default SDPA）在 `eps=3e-3/1e-3` 回到 `~7.7e-4–9.6e-4`，`eps=1e-2` 也低于 `1e-2`；C（当前 fast 口径，TF32 on）显著变差且 eps 越小越差。因此 built-in FD 在 fast H100 b96 run 中出现 `0.05–0.11` 量级，主要判断为 **TF32 + finite-difference sensitivity / 诊断口径问题**，不是 JVP 主链路错误。所有模式 `no_nan=true`，`r=t` 退化目标检查严格为 `0`。

训练恢复备注：audit 后第一次普通 `nohup` resume 在本执行环境中随父进程退出被清理，日志无 traceback，停在 step `12099` 附近；已改用 `setsid` 方式从 `latest.pt -> step_00012000.pt` 重启，避免被父进程生命周期回收。后续后台启动训练建议使用 `setsid ... < /dev/null > log 2>&1 &`。

<!-- B3_B96_FD_AUDIT_SYNCED_AND_RESUMED_20260528T0920Z -->

## FD audit artifacts synced; B3 training resumed stable

更新时间：`2026-05-28T09:20Z`

- GitHub code/docs repo updated: `https://github.com/Yidhar/pdm3-ht-private`, commit `6acab05561162c891b5ea88cb5955de9c1abc98c` (`Record B3 FD audit result and resume note`).
- HF artifact repo updated: `https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts`, repo sha `6dcee800ae305b792cf46563a29544b098e96a79`.
- Uploaded dedicated FD audit artifacts to HF path:

```text
b3_meanflow_realdata/fullcache_b96/step_00010000/fd_jvp_dedicated_audit_after_step_12000/fd_jvp_modes_audit.json
b3_meanflow_realdata/fullcache_b96/step_00010000/fd_jvp_dedicated_audit_after_step_12000/fd_jvp_modes_audit.jsonl
b3_meanflow_realdata/fullcache_b96/step_00010000/fd_jvp_dedicated_audit_after_step_12000/fd_jvp_modes_audit.md
```

Live resume checkpoint after audit: `setsid` training PID `49292` is alive and reparented to PID `1`; resumed from `latest.pt -> step_00012000.pt`; verified beyond previous stop point with last train record step `12367` @ `2026-05-28T09:19:08Z`, last50 throughput `~122.9 samples/s`, H100 snapshot `70371 MiB / 81559 MiB`, `100%` util, `~601 W`.

Next expected gates:

- `step_00014000.pt` checkpoint + built-in FD audit around step `14000`.
- `step_00020000.pt` checkpoint + built-in eval/sample at step `20000`.
