# PAE reconstruction FID lower-bound — 5K matched real-vs-reconstruction

更新时间：`2026-05-30T16:08:58Z`

## 结论

已完成 PAE reconstruction FID lower-bound，用来判断当前 PAE latent cache + PAE decoder 本身给 image-space FID 带来的下界/天花板约束。

**结果：5K matched ImageNet-256 real vs PAE reconstruction FID = `1.9952405483597886`。**

这说明当前 PAE cache/decoder 的重建保真度不是当前 B3 MeanFlow 生成 FID `~48` 的主要瓶颈；从重建下界看，image-space 单位数 FID 在 decoder/cache 口径上是技术可行的。后续主要问题仍是生成模型/训练 recipe/模型容量/训练步数，而不是 PAE 重建 ceiling。

## Eval 口径

| item | value |
|---|---|
| comparison | matched `real cropped ImageNet-256` vs `PAE decode(cached latent)` |
| sample count | `5000 real / 5000 reconstructed` |
| seed | `20260529` |
| real cache | `/workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors` |
| latent cache | `/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full` |
| latent key | `latents` |
| PAE config | `/workspace/PDM/external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml` |
| PAE ckpt | `/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt` |
| decode | `cuda` / `torch.bfloat16`, batch `16` |
| Inception | `torchvision_inception_v3_imagenet1k_v1_pool2048` |
| Inception device | `cuda`, batch `96` |
| elapsed | `126.454s` |

## Metrics

| metric | value |
|---|---:|
| PAE recon FID lower-bound | `1.9952405483597886` |
| FID raw | `1.9952405483597886` |
| MMD RBF | `-0.00013760608251600637` |
| MMD bandwidth^2 | `363.7931474007587` |
| KID poly3 | `-0.00012595911193669096` |
| feature dim | `2048` |

## Integrity checks

| check | value |
|---|---:|
| labels_match_real_vs_latent | `True` |
| global_indices_match_count | `5000` |
| real feature finite | `True` |
| reconstruction feature finite | `True` |

## Image statistics

| split | mean | std | channel_mean_rgb | channel_std_rgb |
|---|---:|---:|---|---|
| real_imagenet256 | `0.4518995582091894` | `0.2797693415742301` | `[0.48735521582503927, 0.4588731555985122, 0.40947030305126053]` | `[0.2790217641524746, 0.27094642090868054, 0.28362222970058454]` |
| pae_reconstruction | `0.45125950985179253` | `0.27802859199517127` | `[0.486859197613986, 0.45806139150227165, 0.4088579403062656]` | `[0.27725974143014703, 0.2694398101269704, 0.2816334885800515]` |

Note: `decode_and_compact_eval_step.py:image_summary` was patched to use float64 reductions for large image arrays. The metric path itself was unaffected; this only fixes large-N summary-stat precision.

## Local artifacts

```text
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/pae_reconstruction_fid_lower_bound_5k_seed20260529/inception_metrics.json
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/pae_reconstruction_fid_lower_bound_5k_seed20260529/inception_metrics.md
/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/pae_reconstruction_fid_lower_bound_5k_seed20260529/inception_features.npz
```

Script added/copied into private repo:

```text
experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/pae_reconstruction_inception_eval.py
```


## HF artifact upload

Uploaded to shared model artifact repo:

```text
https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/tree/main/b3_meanflow_realdata/pae_reconstruction_fid_lower_bound_5k_seed20260529
```

Commit:

```text
https://huggingface.co/LAXMAYDAY/pdm3-ht-model-artifacts/commit/482dc1dc3d66828768716fb54fd4881c324f9648
```

## Interpretation for current B3 line

Recent true 5K Inception FID anchors for b128/lr1e-4/cosine/NFE32 were:

| step | FID |
|---:|---:|
| 30k | `72.2560` |
| 37.5k | `64.7696` |
| 50k | `58.1837` |
| 75k | `52.6315` |
| 100k | `48.3435` |

Against the PAE reconstruction lower-bound `~1.995`, the current gap at 100k is about `46.35` FID. Therefore:

1. PAE reconstruction is not blocking single-digit FID by itself.
2. Continuing the current line may still improve, but the current slope does not justify assuming it will reach single digits merely by waiting.
3. Next useful comparisons should focus on model/training recipe: longer run with anchors, larger backbone, official hparams alignment, and/or alternative loss/conditioning improvements.
