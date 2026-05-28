# PDM-3-HT private research/code snapshot

This is a filtered Hugging Face snapshot for the PDM-3-HT research workspace.

## What is included

- Research briefs and reports.
- Experiment task/README files.
- Experiment scripts, configs, patches, and notes.
- Lightweight result summaries (`.json`, `.md`, `.sha256`, `.yaml`, `.yml`, `.csv`).

## What is intentionally excluded

- Raw ImageNet parquet files.
- Cropped ImageNet-256 uint8 cache.
- PAE / FLUX.2 / Qwen latent cache shards.
- Model checkpoints and downloaded weights.
- External dependency clones.
- Logs, PID files, temporary files, and large binaries.

The VAE latent cache is published separately as a private Hugging Face dataset repo.

See `PUBLISH_MANIFEST.json` for the exact file list and selection policy.
