# PDM-3-HT Private Code Repository

Private code/config/documentation repository for the PDM-3-HT research track.

## Scope of this Git repo

Included here:

- research briefs/reports;
- experiment task docs and notes;
- training/eval/cache scripts;
- YAML/env configs;
- lightweight patches and handoff docs.

Excluded by design:

- raw datasets;
- ImageNet crops;
- PAE/FLUX latent caches;
- checkpoints/model weights;
- decoded image folders;
- heavy logs and per-step metrics JSONL;
- third-party source trees under `external/`.

The exclusion is intentional: code is versioned in GitHub, model/data artifacts are handled separately.

## Private model artifact repository

Selected model artifacts should be stored in the private Hugging Face model repo:

```text
https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
```

Do not bulk-upload every checkpoint. Upload only selected validated artifacts plus compact summaries/manifests.

## Main active route

Current active engineering route:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer
```

Key scripts:

```text
scripts/train_b3_meanflow_realdata.py
scripts/decode_flux2vae_latents.py
scripts/eval_flux2vae_imagespace.py
scripts/eval_imagespace_fid_mmd.py
scripts/build_imagenet256_reference_stats.py
```

Current FLUX.2 route configs include:

```text
configs/b3_meanflow_flux2vae_fullcache_medium512_b128_1k_imageeval.yaml
configs/b3_meanflow_flux2vae_fullcache_medium512_b128_5k_imageeval.yaml
```

## Important precision invariant

MeanFlow target path:

```text
bf16 backbone + fp32 live-param JVP target
```

Preserve these invariants unless intentionally ablated:

- JVP target uses live parameters, not EMA;
- EMA is used only for eval sampling;
- target is detached;
- `r=t` sampler remains active, usually `equal_prob=0.75`;
- FD audit remains enabled for correctness monitoring.

## Third-party dependencies

The local working tree may contain third-party repos under `/workspace/PDM/external`, for example LightningDiT and PAE. They are intentionally not vendored into this private code repo. Re-clone or mount them separately when reproducing on a new machine.

## Artifact hygiene

Before pushing, check:

```bash
git status --short
git ls-files -z | xargs -0 -r du -b | sort -nr | head
```

No raw data, full latent caches, checkpoints, `.safetensors`, `.pt`, `.npz`, decoded PNG folders, or large logs should be committed here.
