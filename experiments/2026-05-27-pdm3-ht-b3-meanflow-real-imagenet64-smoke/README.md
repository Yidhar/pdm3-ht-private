# Experiment 9 — B3 MeanFlow real ImageNet64 PAE-latent smoke

This validates that the Phase0/B3 MeanFlow trainer can train directly on the real PAE latents produced from HF ImageNet-1k smoke64.

## Status

Overall: **PASS**

Final trainer line:

```text
DONE gate practical=True strict=True final_loss=3.6721441745758057 loop_peak=63.06005859375MB
```

## Input

Real latent directory from Experiment 8:

- `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke64`

Schema:

- `latents`: `[64, 32, 16, 16]`, BF16
- `latents_flip`: `[64, 32, 16, 16]`, BF16
- `labels`: `[64]`, I64

## Config and command

Config:

- `configs/DiT_B3MeanFlowTiny_PAE_DINOv2L_d32_imagenet64_smoke.yaml`

Command/log:

```bash
cd /workspace/PDM/external/PAE/pae_with_generator
python train_meanflow_dit.py --config /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-real-imagenet64-smoke/configs/DiT_B3MeanFlowTiny_PAE_DINOv2L_d32_imagenet64_smoke.yaml \
  2>&1 | tee /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-real-imagenet64-smoke/logs/010_b3_meanflow_real_imagenet64_smoke.log
```

## Precision/JVP recipe validated

- trainable backbone forward/backward: bf16/autocast
- JVP target path: fp32/no-autocast
- JVP parameter source: live model parameters
- EMA target JVP: disabled (`False`)
- target: detached
- sampler: sample-level `r=t` default mix active
- FD audit: fp32 central finite difference
- memory logging: active
- TF32: disabled
- SDPA: math path

## Gate

| item | value |
|---|---:|
| train ok | `True` |
| practical gate pass | `True` |
| strict gate pass | `True` |
| FD rel err full JVP | `8.165405772234868e-05` |
| FD ok < 1e-2 | `True` |
| all-r=t target-v max abs | `0.0` |
| degenerate ok < 1e-7 | `True` |
| target detached | `True` |
| live-param JVP target | `True` |
| sample-level sampler | `True` |
| mixed modes ok | `True` |
| memory logged | `True` |
| no NaN/Inf | `True` |

## Core metrics

| item | value |
|---|---:|
| completed optimizer steps | `8` |
| final loss | `3.6721441745758057` |
| min loss | `3.484823703765869` |
| max loss | `3.672729015350342` |
| loss all finite | `True` |
| grad all finite | `True` |
| target detached all steps | `True` |
| JVP param source all live | `True` |
| mean realized r=t fraction | `0.6875` |
| max actual r=t target-v abs | `0.0` |
| loop peak memory MB | `63.06005859375` |
| total elapsed seconds | `1.6858452630694956` |

## FD audits

| step | fd rel err | all-r=t target-v max | forced non-degen | JVP MB | FD MB | r=t frac |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `4.220718926064614e-05` | `0.0` | `False` | `62.89990234375` | `47.48486328125` | `0.75` |
| 4 | `5.038698494655051e-05` | `0.0` | `True` | `62.90087890625` | `47.48583984375` | `0.75` |
| 8 | `8.165405772234868e-05` | `0.0` | `True` | `62.90087890625` | `47.48583984375` | `0.75` |

## Train losses

| step | loss | realized r=t fraction |
|---:|---:|---:|
| 1 | `3.5579824447631836` | `0.75` |
| 2 | `3.510148286819458` | `1.0` |
| 3 | `3.672729015350342` | `0.5` |
| 4 | `3.6101555824279785` | `1.0` |
| 5 | `3.6400887966156006` | `0.5` |
| 6 | `3.5460257530212402` | `0.75` |
| 7 | `3.484823703765869` | `0.75` |
| 8 | `3.6721441745758057` | `0.25` |

## Interpretation

- The real ImageNet smoke64 PAE latent shard is compatible with the Phase0/B3 trainer.
- FD rel err `8.165405772234868e-05` validates the fp32 live-param JVP target path.
- `r=t` degeneracy is correct: all-r=t target reduces to `v` with max abs `0.0`.
- Mixed precision drift (`bf16` train forward vs `fp32` target forward) remains in the expected ~5e-3 range in the train summary, while the JVP itself is validated by FD.
- EMA exists/updates after steps, but is not used for target JVP.

## Next recommended step

Build a larger real PAE latent subset (`smoke1024` or `smoke10000`) and run a 50–200 step B3 MeanFlow short train before committing to full ImageNet latent extraction.
