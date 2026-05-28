---
license: other
private: true
---

# PDM-3-HT Model Artifacts

Private artifact repository for PDM-3-HT / HT + MeanFlow experiments.

## Policy

This repo is intended for **selected model artifacts only**:

- selected checkpoints after a run is validated;
- compact training/eval summaries;
- optional small previews/manifests if explicitly needed.

Do **not** upload by default:

- raw ImageNet data;
- full VAE/PAE latent caches;
- decoded eval PNG folders;
- every intermediate checkpoint from exploratory runs;
- huge logs.

The code/config/docs live in the private GitHub repo:

```text
https://github.com/Yidhar/pdm3-ht-private
```

## Current local artifact candidates

Current active run at creation time:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/shorttrain_flux2vae_b3medium_h512_d12_b128_5k_1024img
```

Upload only after the run finishes and a checkpoint/eval point is selected.

Example upload pattern, adjust path/step after selection:

```bash
hf upload LAXMAYDAY/pdm3-ht-model-artifacts \
  /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/shorttrain_flux2vae_b3medium_h512_d12_b128_5k_1024img/checkpoints/<SELECTED_CHECKPOINT> \
  checkpoints/flux2vae_b3medium_h512_d12_b128_5k/<SELECTED_CHECKPOINT> \
  --repo-type model --private \
  --commit-message "Add selected FLUX.2 B3-medium checkpoint"
```

<!-- PDM3_HT_ARTIFACT_ROUTING_20260528 -->

## Artifact routing / handoff note — 2026-05-28T07:31:08Z

- Code/docs target: https://github.com/Yidhar/pdm3-ht-private
- Model artifact target: this private HF model repo, `LAXMAYDAY/pdm3-ht-model-artifacts`.
- Current B3 training run is still local; do not upload partial/unstable checkpoints automatically. Next intended durable checkpoint gate is `step_00006000.pt`.
- Keep raw ImageNet, cropped caches, PAE latent cache shards, and runtime logs out of this model artifact repo unless explicitly requested.
- Current local progress note: `/workspace/PDM/handoff/PROGRESS_2026-05-28_GITHUB_HF_ARTIFACTS.md`.
