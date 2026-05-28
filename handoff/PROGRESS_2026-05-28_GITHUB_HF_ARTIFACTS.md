# Progress note: GitHub code sync + HF model artifacts routing

- Time UTC: `2026-05-28T07:29:55Z`
- Working tree: `/workspace/PDM`
- GitHub code target: <https://github.com/Yidhar/pdm3-ht-private>
- HF model artifact target: <https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts>

## Current B3 training status while sync is prepared

```json
{"count": 1019, "step": 5019, "time": "2026-05-28T07:29:55Z", "loss": 0.34228333830833435, "last100_samples_s": 121.12723326976942, "last100_mean_sec": 0.7925550465285778, "to_6000": 981, "effective_jvp": 20, "skipped_jvp": 76, "rt_degen_max_abs": 0.0}
```

GPU snapshot:

```text
0, NVIDIA H100 80GB HBM3, 70371 MiB, 81559 MiB, 100 %, 587.47 W, 700.00 W, 62
```

Active run:

```text
pid: 41529
config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_full_fast_h100_b96.yaml
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/fullcache_realdata_fast_h100_b96_skip_eq_jvp_20260528T071553Z.log
latest_durable_checkpoint: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00004000.pt
monitor_pid: 42029
monitor_log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/b3_fullcache_monitor_20260528T071828Z.jsonl
```

Next durability milestone remains `step_00006000.pt`. Do not interrupt b96 before 6000 unless it OOMs or reports non-finite loss/grad.

## Code sync preparation

Prepared a clean GitHub-oriented source snapshot that includes code/config/docs/handoff and excludes runtime data/artifacts:

- Excluded: `data/`, `.cache/`, `experiments/**/results/`, `experiments/**/logs/`, checkpoints, PAE latent shards, raw/cropped ImageNet caches, model weights.
- Added/updated `.gitignore` to enforce that split.
- Added `ARTIFACTS.md` so future operators route model products to the HF model artifact repo instead of GitHub.

Local staging repo:

```text
/workspace/pdm3-ht-private-sync
```

## Current blocker

GitHub push cannot complete from this machine yet because no GitHub credentials are present:

- `gh auth status`: not logged in.
- HTTPS `git ls-remote`: cannot read username with prompts disabled.
- SSH `git ls-remote`: permission denied publickey.

Once a GitHub token/SSH key with access to `Yidhar/pdm3-ht-private` is available, push the prepared staging repo:

```bash
cd /workspace/pdm3-ht-private-sync
git remote -v
git push -u origin main
```

## Artifact routing note

The HF model artifact repo exists and is private. Use it for future B3 checkpoints/model cards/eval outputs:

```text
LAXMAYDAY/pdm3-ht-model-artifacts
https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
```

No model checkpoint has been uploaded in this note; current latest durable training checkpoint remains local at `step_00004000.pt` until b96 reaches `step_00006000.pt`.

## Follow-up — 2026-05-28T07:31:31Z

Prepared local Git commit for the GitHub code repo:

```text
staging_repo: /workspace/pdm3-ht-private-sync
branch: main
commit: see `git -C /workspace/pdm3-ht-private-sync log -1 --oneline`
bundle: see latest `/workspace/pdm3-ht-private-sync_*.bundle`
```

A direct push was attempted and failed only because the machine has no GitHub credential:

```text
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```

The HF model artifacts repository README and this handoff note were updated successfully:

```text
repo: LAXMAYDAY/pdm3-ht-model-artifacts
latest repo sha: check with `HfApi().repo_info("LAXMAYDAY/pdm3-ht-model-artifacts", repo_type="model").sha`
uploaded doc: handoff/PROGRESS_2026-05-28_GITHUB_HF_ARTIFACTS.md
```

## Follow-up — 2026-05-28T07:39:46Z — GitHub push complete

GitHub authorization is now active as `Yidhar`, and the prepared code snapshot was pushed successfully.

```text
repo: https://github.com/Yidhar/pdm3-ht-private
branch: main
pushed_head_before_this_note: 592238cc4e69bbbb0c3a68cdf11309b92f23e425
remote_main_after_push: 592238cc4e69bbbb0c3a68cdf11309b92f23e425
push_result: 9f9db46..592238c main -> main
```

Notes:

- The remote already had an initial `main` commit, so I fetched and merged `origin/main` instead of force-pushing.
- Add/add conflicts were resolved in favor of the local current source snapshot, while preserving non-conflicting remote-only files through the merge.
- Artifact exclusion check passed before push: no `data/`, `results/`, `logs/`, `.pt`, `.ckpt`, `.safetensors`, or `.parquet` files were tracked.

Training status during push:

```json
{"last_step": 5748, "time": "2026-05-28T07:39:46Z", "loss": 0.3704003691673279, "last100_samples_s": 120.89864530931222, "to_6000": 252, "rt_degen_max_abs": 0.0, "last_fd_step": 4000, "last_fd_rel": 0.11669899379970686, "last_checkpoint_step": 4000}
```

GPU snapshot:

```text
0, NVIDIA H100 80GB HBM3, 70371 MiB, 81559 MiB, 100 %, 584.89 W, 700.00 W, 60
```

## Follow-up — 2026-05-28T07:44:44Z — B3 b96 reached step 6000 durable checkpoint

The active b96 H100 run reached the next durability gate and continued training.

```json
{"last_step": 6088, "time": "2026-05-28T07:44:43Z", "loss": 0.41259926557540894, "last100_samples_s": 118.34274827062414, "last100_mean_sec": 0.811203064005822, "last100_step_range": [5989, 6088], "rt_degen_max_abs": 0.0, "fd_step": 6000, "fd_time": "2026-05-28T07:43:14Z", "fd_rel": 0.08643620461852597, "fd_no_nan_or_inf": true, "fd_degen_target_minus_v": 0.0, "checkpoint_step": 6000, "checkpoint_time": "2026-05-28T07:43:31Z", "checkpoint_path": "/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00006000.pt"}
```

Runtime pointers:

```text
pid: 41529
config: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_full_fast_h100_b96.yaml
log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/fullcache_realdata_fast_h100_b96_skip_eq_jvp_20260528T071553Z.log
latest_checkpoint: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00006000.pt
monitor_pid: 42029
monitor_log: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/logs/b3_fullcache_monitor_20260528T071828Z.jsonl
```

GPU snapshot:

```text
0, NVIDIA H100 80GB HBM3, 74421 MiB, 81559 MiB, 72 %, 562.66 W, 700.00 W, 62
```

The new durable local model checkpoint is `step_00006000.pt` and `latest.pt` points to it. FD audit at step 6000 passed with finite values. Per artifact-routing policy, this checkpoint was **not** committed to GitHub. It should be uploaded only to the HF model artifacts repo if/when a model artifact publish is requested:

```text
https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
```


<!-- B3_B96_STEP10000_EVAL_SMOKE_20260528T0842Z -->

## Step 10000 eval/sample pipeline gate complete

更新时间：`2026-05-28T08:44:24Z`

用户要求的 step `10000` 五项已落地：

1. checkpoint: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00010000.pt`
2. EMA latent sample: `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/sample_latents.safetensors`，shape `[64, 32, 16, 16]`，finite `True`
3. image-space decoded samples: generated PNG `64` 张 + real-ref PNG `64` 张；grid:
   - `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/images/generated_grid.png`
   - `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/eval/step_00010000/images/real_ref_grid.png`
4. compact smoke metrics（非 official Inception FID/KID）:
   - `compact_fid_rp128=4.0626297`
   - `compact_mmd_rbf_rp128=0.049283276`
   - `compact_kid_poly3_rp128=0.0034318871`
5. train/eval summary: `handoff/STEP10000_EVAL_SUMMARY_2026-05-28.md`

内置 official FID 仍是 `skipped_no_command`，因为 fast config 当前 `eval.fid_command_template` 为空；本次 compact metrics 目标是提前验证 sample→PAE decode→image metrics 线路，避免等到 100k 后才发现 pipeline 未接通。

主训练未中断，当前最后记录：step `10441` @ `2026-05-28T08:44:23Z`，last200 约 `119.135` samples/s（如有）。

FD 风险：step `10000` built-in FD audit finite/no_nan，但 `fd_rel=0.061354188` 仍高于 practical gate `1e-2`；专项 A/B/C eps sweep 脚本已准备好：`experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/audit_fd_jvp_modes.py`。建议等 GPU 空出或在下一个 durable checkpoint 后短暂停训运行，避免与当前 74GB H100 主训抢显存。


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

<!-- B3_B96_FD_JVP_FINAL_RISK_RECLASSIFICATION_20260528T0940Z -->

## Final FD/JVP risk reclassification after A/B/C audit

更新时间：`2026-05-28T09:40Z`

English record / paper wording:

> Dedicated FD/JVP audit on the step-10000 PAE B3 checkpoint confirms that the fp32 JVP path is numerically correct when TF32 is disabled. With TF32 off, both math SDPA and default SDPA produce FD relative errors in the 7.7e-4 ~ 9.6e-4 range for eps=3e-3/1e-3, while TF32-on finite-difference diagnostics show much larger apparent errors. Therefore, the earlier high built-in FD rel values in the H100 fast b96 run are attributed to TF32-sensitive finite-difference diagnostics, not to an incorrect JVP implementation. The r=t degeneration remains exact with target-v=0.

中文统一口径：

> step-10000 PAE B3 checkpoint 的专项 FD/JVP audit 已确认：在关闭 TF32 的严格数值口径下，fp32 JVP target 路径是正确的；math SDPA 与 default SDPA 在 eps=3e-3/1e-3 下都回到 7.7e-4 ~ 9.6e-4 的 FD rel。此前 H100 fast b96 run 中 built-in FD rel 出现 0.05 ~ 0.11，主要是 TF32 + finite-difference sensitivity / 诊断口径导致，不是 JVP 主链路错误。r=t 退化路径仍严格正确，target-v=0。

Operational decision:

- Keep the PAE B3 b96 mainline on the fast H100 config (`allow_tf32=true`, `sdpa_kernel=default`, `bf16_backbone_fp32_jvp`); do not stop or slow the main run solely because fast built-in FD rel is above the old `1e-2` practical gate.
- Interpret built-in FD rel under the fast config as a finite/no-NaN smoke diagnostic plus TF32-sensitive trend, not as the strict JVP correctness gate.
- Strict FD/JVP correctness audits should use: TF32 off, fixed seed, small fixed non-degenerate batch, `model.eval()`, dropout/class dropout disabled or fixed, JVP/FD fp32, global math/default SDPA comparison, and `eps` sweep (`1e-2`, `3e-3`, `1e-3`).
- Next gates: `step_00014000.pt` checkpoint + built-in FD audit; `step_00020000.pt` checkpoint + EMA sample + PAE decode + image-space metrics.
- Step-20000 image-space evaluation should connect real ImageNet-256 Inception FID/MMD/KID using the cropped uint8 ImageNet cache, not only the compact random-projection smoke metric. If the generated eval batch remains `64` samples, label the Inception numbers as early/sample-count-limited diagnostics rather than publishable 50k FID.

<!-- B3_B96_STEP14000_AND_INCEPTION_EVAL_READY_20260528T0948Z -->

## Step-14000 gate reached; step-20000 Inception eval helper prepared

更新时间：`2026-05-28T09:48Z`

Main training remains alive under `setsid` on H100 and continued past the gate.

Step-14000 checkpoint / built-in FD audit:

```text
checkpoint: /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template/checkpoints/step_00014000.pt
created_at_utc: 2026-05-28T09:41:25Z
latest.pt -> step_00014000.pt
```

Built-in fast-config FD at step `14000`:

```json
{
  "fd_rel_err_full_jvp": 0.07857135953004488,
  "fd_eps": 0.01,
  "fd_mode": "fp32",
  "jvp_mode": "fp32",
  "allow_tf32_fast_config": true,
  "r_eq_t_count": 70,
  "r_eq_t_total": 96,
  "realized_r_eq_t_fraction": 0.7291666865348816,
  "u_finite": true,
  "du_finite": true,
  "fd_finite": true,
  "no_nan_or_inf": true,
  "target_detached": true,
  "all_r_eq_t_degenerate_target_minus_v_max_abs": 0.0
}
```

Interpretation: this `0.07857` is consistent with the settled fast-config TF32 finite-difference diagnostic artifact pattern; it is **not** treated as evidence of a broken JVP path. Strict JVP correctness gating remains the dedicated TF32-off fixed non-degenerate eps-sweep audit.

Step-20000 eval preparation:

- Added script: `experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/decode_and_inception_eval_step.py`.
- Purpose: decode generated PAE latents, read real cropped uint8 ImageNet-256 cache directly, and compute image-space InceptionV3 pool-2048 FID / RBF-MMD / poly3-KID.
- Default devices are CPU (`--decode-device cpu --inception-device cpu`) to avoid disturbing the active H100 training process.
- Inception weights have been downloaded/cached locally via torchvision (`inception_v3_google-0cc3c7bd.pth`).
- Smoke validation completed on step-10000 with 4 generated + 4 real ImageNet-256 images: script status `ok`, feature dim `2048`, finite metrics produced. This validates wiring only; step-20000 should run on the full generated eval batch.

Recommended step-20000 command after trainer writes `eval/step_00020000/sample_latents.safetensors`:

```bash
EXP=/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
python3 "$EXP/scripts/decode_and_inception_eval_step.py" \
  --sample-latents "$EXP/results/fullcache_realdata_singleproc_template/eval/step_00020000/sample_latents.safetensors" \
  --output-dir "$EXP/results/fullcache_realdata_singleproc_template/eval/step_00020000/inception_eval" \
  --real-image-cache /workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors \
  --num-real 0 \
  --real-seed 20260528 \
  --decode-device cpu \
  --decode-batch-size 2 \
  --inception-device cpu \
  --inception-batch-size 16 \
  --torch-num-threads 8 \
  --max-save-images 256 \
  --grid-count 64
```

If trainer eval still emits `64` generated samples, record resulting FID/MMD/KID as early/sample-count-limited Inception diagnostics, not publishable 50k FID.

