# Artifact Routing

Code/config/docs are pushed to the private GitHub repo:

```text
https://github.com/Yidhar/pdm3-ht-private
```

Selected model artifacts are routed to the private Hugging Face model repo:

```text
https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
```

## What goes to Hugging Face model repo

Upload only selected artifacts after validation:

- selected checkpoint(s), not every checkpoint;
- compact `train_summary.json` / eval summary JSON / README;
- small manifest files if needed.

Avoid uploading by default:

- raw ImageNet;
- full PAE/FLUX latent caches;
- decoded image folders;
- every exploratory checkpoint;
- full logs and JSONL traces.

## Current candidate run

Active FLUX.2 route local output:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/shorttrain_flux2vae_b3medium_h512_d12_b128_5k_1024img
```

After completion, choose the best validated checkpoint/eval step before uploading.

Example:

```bash
hf upload LAXMAYDAY/pdm3-ht-model-artifacts \
  /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/shorttrain_flux2vae_b3medium_h512_d12_b128_5k_1024img/checkpoints/<SELECTED_CHECKPOINT> \
  checkpoints/flux2vae_b3medium_h512_d12_b128_5k/<SELECTED_CHECKPOINT> \
  --repo-type model --private \
  --commit-message "Add selected FLUX.2 B3-medium checkpoint"
```
