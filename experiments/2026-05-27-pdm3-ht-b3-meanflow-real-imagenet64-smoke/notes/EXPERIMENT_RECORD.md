# Experiment Record — B3 MeanFlow real ImageNet64 PAE-latent smoke

Date: 2026-05-27

## Setup

Experiment directory:

- `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-real-imagenet64-smoke`

Input latent directory from Experiment 8:

- `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke64`

Input was verified as:

- `latents`: `[64, 32, 16, 16]`, BF16
- `latents_flip`: `[64, 32, 16, 16]`, BF16
- `labels`: `[64]`, I64

Config:

- `configs/DiT_B3MeanFlowTiny_PAE_DINOv2L_d32_imagenet64_smoke.yaml`

Run output:

- `/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-real-imagenet64-smoke/results/B3MeanFlowTiny_PAE_DINOv2L_d32_imagenet64_smoke`

## Command

```bash
cd /workspace/PDM/external/PAE/pae_with_generator
python train_meanflow_dit.py --config /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-real-imagenet64-smoke/configs/DiT_B3MeanFlowTiny_PAE_DINOv2L_d32_imagenet64_smoke.yaml \
  2>&1 | tee /workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-real-imagenet64-smoke/logs/010_b3_meanflow_real_imagenet64_smoke.log
```

## Result

Result: PASS

Final line:

```text
DONE gate practical=True strict=True final_loss=3.6721441745758057 loop_peak=63.06005859375MB
```

Core metrics:

- Completed optimizer steps: `8` / `8`
- Final loss: `3.6721441745758057`
- Loss all finite: `True`
- Grad all finite: `True`
- Target detached all steps: `True`
- JVP parameter source all live: `True`
- EMA used for target JVP: `False`
- Sample-level sampler: `True`
- Mean realized r=t fraction: `0.6875`
- Max actual r=t target-v abs: `0.0`
- Loop peak memory MB: `63.06005859375`
- No NaN/Inf: `True`

Gate:

- practical_gate_pass: `True`
- strict_gate_pass: `True`
- fd_rel_err_full_jvp: `8.165405772234868e-05`
- all_r_eq_t_degenerate_target_minus_v_max_abs: `0.0`
- live_param_jvp_target: `True`
- mixed_modes_ok: `True`
- memory_logged: `True`

FD audits:

| step | fd rel err | all-r=t target-v max | forced non-degen | JVP MB | FD MB | r=t frac |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `4.220718926064614e-05` | `0.0` | `False` | `62.89990234375` | `47.48486328125` | `0.75` |
| 4 | `5.038698494655051e-05` | `0.0` | `True` | `62.90087890625` | `47.48583984375` | `0.75` |
| 8 | `8.165405772234868e-05` | `0.0` | `True` | `62.90087890625` | `47.48583984375` | `0.75` |

## Interpretation

- This confirms the B3 MeanFlow trainer can consume real PAE_DINOv2L_d32 latents generated from HF ImageNet-1k smoke64.
- The mixed precision recipe is wired correctly: bf16 backbone train path plus fp32 live-param JVP target.
- `r=t` degeneracy is handled correctly; target equals `v` in all-r=t audit.
- FD audits validate the JVP target path at short-train scale.

## Next

Recommended next experiment: build larger real PAE latent subset (`smoke1024` or `smoke10000`) and run 50–200 B3 steps with the same logging/audit gates.
