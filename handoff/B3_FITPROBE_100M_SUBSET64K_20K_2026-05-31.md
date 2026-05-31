# B3 100M subset64k fitprobe — step-20k result and HF artifact update

更新时间：`2026-05-31T03:28:54Z`

## TL;DR

本次按用户要求把较小训练集 fast-fit 路线先跑完并留档：

- Run：`fitprobe_100m_subset64k_b384_lr5e4`
- 模型：B3 MeanFlow / LightningDiT custom，约 `101.27M` params（`hidden_size=512, depth=21, heads=8, patch_size=1`）
- 数据：PAE ImageNet-256 latent cache 的前 `65,536` 样本 contiguous subset
- 训练：`20,000` steps，batch `384`，cosine LR `5e-4 -> 5e-5`，warmup `500`，`equal_prob=0.75`
- 评估：PAE decode 到 image-space 后，用 torchvision InceptionV3 ImageNet1K pool-2048 做 true FID
- 结论：能学，但在该 small-subset / 100M / high-LR recipe 下快速进入平台；EMA FID `10k -> 20k` 只从 `63.1425` 到 `58.6432`。这离 PAE reconstruction lower-bound FID `1.9952` 很远，说明瓶颈是模型/flow/训练配方，不是 PAE decode/cache 下限。

## Artifact status

Hugging Face unified model artifact repo:

```text
repo: https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts
remote prefix: b3_meanflow_realdata/fitprobe_100m_subset64k_b384_lr5e4
commit: ad334901017f194d2bf2b6898d9a1c705db7fd39
commit URL: https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/commit/ad334901017f194d2bf2b6898d9a1c705db7fd39
```

Uploaded selected/curated artifacts only:

```text
README.md
fitprobe_summary.json
SHA256SUMS.json
SHA256SUMS.txt
config/b3_meanflow_realdata_100m_subset64k_b384_lr5e4_fitprobe.yaml
checkpoints/step_00020000_slim_no_optimizer.pt
eval/step_00002000/... metrics/grid/json
eval/step_00004000/... metrics/grid/json
eval/step_00006000/... metrics/grid/json
eval/step_00010000/... metrics/grid/json
eval/step_00020000/... metrics/grid/json
eval/step_00020000/sample_latents_5000_nfe32_subsetlabels_ema.safetensors
eval/step_00020000/sample_latents_5000_nfe32_subsetlabels_raw.safetensors
```

Intentionally **not** uploaded:

```text
full optimizer checkpoints
all intermediate full checkpoints
decoded_and_real_imagenet256_samples.npz
inception_features.npz
real ImageNet image dumps / real-image grids
```

The uploaded step-20k checkpoint is slimmed to model/EMA/config/dataset metadata and excludes optimizer/scaler/RNG state.

Local full run path:

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fitprobe_100m_subset64k_b384_lr5e4
```

## Code/config changes recorded in GitHub

This handoff also records the code support used by the fitprobe:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/configs/b3_meanflow_realdata_100m_subset64k_b384_lr5e4_fitprobe.yaml
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/sample_b3_meanflow_eval_latents.py
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/decode_and_inception_eval_step.py
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/run_fitprobe_anchor_and_resume.sh
```

Notable implementation additions:

- Trainer dataset supports `data.max_samples` to truncate the sorted latent shard stream to the first N global samples.
- Sampler supports `--label-source latent-subset` and related args so generated label marginal matches the subset labels.
- Decode/eval supports `--real-max-samples N`, restricting real FID references to global indices `[0, N)`.
- Generic controller `run_fitprobe_anchor_and_resume.sh` can wait for a checkpoint, stop train, run EMA/raw sample+decode+true-FID, then resume if needed.

## Training config summary

```yaml
data:
  max_samples: 65536
  expected_total: 65536
  subset_mode: first_n_contiguous
train:
  exp_name: fitprobe_100m_subset64k_b384_lr5e4
  max_steps: 20000
  global_batch_size: 384
  checkpoint_every: 2000
  keep_last_checkpoints: 12
  use_ema: true
model:
  hidden_size: 512
  depth: 21
  num_heads: 8
  patch_size: 1
  use_rope: true
  use_rmsnorm: true
  use_swiglu: true
optimizer:
  lr: 0.0005
  min_lr: 0.00005
  scheduler: cosine
  warmup_steps: 500
  decay_end_step: 20000
meanflow:
  precision_recipe: bf16_backbone_fp32_jvp
  equal_prob: 0.75
  time_sampler: ltg
```

Approximate size:

```text
total params: ~101.27M
trainable params: ~101.14M
```

## FID / metric protocol

All anchors below use:

```text
PAE decode to image-space
torchvision InceptionV3 ImageNet1K pool-2048 true FID
sample_steps / script-level NFE = 32
real reference = first 65,536 cropped ImageNet-256 subset, not full ImageNet
label source = latent-subset label marginal
```

Important caveat: `2k/4k/6k` anchors use `2K` generated samples; `10k/20k` anchors use `5K` generated samples. Cross-sample-count comparisons have noise, but the trend is still useful.

| step | mode | samples | FID ↓ | MMD_RBF ↓ | KID_poly3 ↓ |
|---:|---|---:|---:|---:|---:|
| 2,000 | EMA | 2K | 180.86566893061683 | 0.09147492517350453 | 0.12487298337346742 |
| 4,000 | EMA | 2K | 108.82244341828391 | 0.057576398037170184 | 0.0771563919850986 |
| 4,000 | raw | 2K | 103.01700794748547 | 0.05377397575155918 | 0.0704296022782458 |
| 6,000 | EMA | 2K | 92.4871382129187 | 0.045732939779151494 | 0.058601107639874694 |
| 6,000 | raw | 2K | 101.57515388916136 | 0.05203098962005104 | 0.06900815830508122 |
| 10,000 | raw | 5K | 77.80089238912399 | 0.046339529607105856 | 0.05950594781799179 |
| 10,000 | EMA | 5K | 63.14248149776478 | 0.0378653834895728 | 0.04687677322364392 |
| 20,000 | raw | 5K | 61.62807074039921 | 0.0353458204251087 | 0.04430557547555347 |
| 20,000 | EMA | 5K | 58.64321275926517 | 0.03321789329154545 | 0.041432888387396005 |

Step-10k to step-20k, comparable 5K EMA anchor:

```text
FID: 63.14248149776478 -> 58.64321275926517
absolute change: -4.49926873849961
relative change: -7.13%
```

Step-10k to step-20k, comparable 5K raw anchor:

```text
FID: 77.80089238912399 -> 61.62807074039921
absolute change: -16.17282164872478
relative change: -20.79%
```

EMA remains better than raw at 20k, but the EMA gain has become small (`~2.985` FID).

## Loss trend

Final observed train point:

```text
step=20000
loss=0.4310128092765808
lr=5e-05
```

Windowed loss:

| steps | mean loss | median loss |
|---:|---:|---:|
| 1–2000 | 0.466969 | 0.401819 |
| 2001–4000 | 0.385737 | 0.385336 |
| 4001–6000 | 0.390017 | 0.389311 |
| 6001–8000 | 0.393939 | 0.393162 |
| 8001–10000 | 0.399331 | 0.397359 |
| 10001–12000 | 0.402546 | 0.399990 |
| 12001–14000 | 0.412226 | 0.406187 |
| 14001–16000 | 0.418143 | 0.412925 |
| 16001–18000 | 0.430386 | 0.422084 |
| 18001–20000 | 0.434180 | 0.426940 |

Interpretation: loss bottoms around `2k–4k` and then drifts upward. Since FID continues improving, this is not a simple divergence/collapse; however, the FID slope is too shallow for this to be considered a good lower-bound route.

## Lower-bound context

Previously recorded PAE reconstruction lower-bound:

```text
PAE reconstruction lower-bound 5K true Inception FID = 1.9952405483597886
handoff: handoff/PAE_RECONSTRUCTION_FID_LOWER_BOUND_5K_2026-05-30.md
HF prefix: b3_meanflow_realdata/pae_reconstruction_fid_lower_bound_5k_seed20260529
HF commit: 482dc1dc3d66828768716fb54fd4881c324f9648
```

Therefore, the current generated FID `~58.6` on this subset-real protocol is not bounded by PAE reconstruction quality. It is a model/flow/optimization/generalization gap.

## Decision / next recommendation

Do **not** spend a long run on this exact `100M / first-64k / b384 / lr5e-4 / equal_prob=0.75` recipe as the main lower-bound attempt.

Recommended next lower-bound probe:

1. Make the task easier and more diagnostic:
   - first `8k` contiguous subset, or class-balanced `5k–10k` subset.
   - Keep the real FID reference matched to that exact subset.
2. Reduce MeanFlow degeneracy pressure:
   - try `equal_prob=0.5` instead of `0.75`.
3. Use a less aggressive but still fast-fitting LR:
   - `lr=2e-4` or `3e-4`, cosine to `2e-5/3e-5`.
4. Keep the same true-FID protocol:
   - 32 NFE/sample steps,
   - EMA and raw,
   - 2K early anchors and 5K final anchor,
   - PAE image-space decode + true torchvision InceptionV3 FID,
   - label-source matched to the training subset.

Operationally, GPU was idle after this fitprobe and the HF export; no active training/eval process was found at the time of this handoff update.
