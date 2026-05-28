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
