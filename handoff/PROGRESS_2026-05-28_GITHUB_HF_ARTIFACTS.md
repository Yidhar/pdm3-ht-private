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
