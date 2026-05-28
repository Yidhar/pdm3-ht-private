
## 2026-05-27T18:46:03Z — Priority 1 + Priority 2C optimization launched

Decision:
- Choose **Priority 2C: ADM-cropped uint8 safetensors cache** over **2B: WebDataset JPEG tar**.
- Reason: local HF parquet already stores the JPEG bytes; WebDataset tar would still require JPEG decode + crop for every later PAE/FID/MMD pass. Cropped uint8 amortizes decode/crop once and can be reused by PAE latent cache, FID real-stat preprocessing, MMD, visualization, and preprocessing audits.

Smoke results:
- Cropped uint8 smoke8192: 8192 samples, avg 594.67 samples/s, output schema images uint8 [N,3,256,256] + labels + source_indices.
- PAE-from-cropped smoke8192: 8192 samples, avg 144.01 samples/s, peak CUDA allocated 23361 MB, latent schema compatible with existing B3 pipeline.

Engineering fixes before full run:
- Plain nohup jobs were reaped with the launcher process group; switched to **setsid** for durable background jobs.
- cgroup page cache reached memory.max and caused one OOM kill while writing huge shards; added best-effort **posix_fadvise(DONTNEED)** after large safetensors writes and parquet/crop reads.
- Created PAE follow watcher: waits until cropped cache exceeds current latent resume point by at least 131072 samples, then runs resumable PAE encoding passes; this overlaps CPU crop cache and GPU latent cache once enough cropped shards exist.

Current background jobs:
- Cropped full cache PID: \1624789, log: \/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized/logs/008_cropped_full_cache_setsid_fadvise_20260527T184257Z.log
- PAE follow PID: \1624791, log: \/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized/logs/009_pae_from_cropped_follow_full_setsid_20260527T184257Z.log

Machine-readable summary:
- /workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized/results/optimization_decision_and_launch_20260527T184603Z.json

## 2026-05-27T18:52:40Z — Optimization status / Priority 2C confirmed

Decision:
- Use **Priority 2C: ADM-cropped uint8 safetensors cache** rather than **2B: WebDataset JPEG tar**.
- Reason: local HF parquet is already a raw JPEG store; 2B would mostly repackage JPEG and still pay decode/crop later. 2C pays JPEG decode + ADM center crop once and is reusable by PAE full latent cache, FID real-stat pass, MMD, visualization, and preprocessing audit.

Runtime changes:
- Crop full-cache job remains running under \ with fadvise page-cache eviction.
- PAE follow watcher was relaunched with \ and \ so GPU PAE encoding starts earlier after cropped cache catches the existing 217088 latent samples, without too many model reloads.

Current machine-readable status:
- /workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-cropped256-cache-optimized/results/current_optimization_status_20260527T185236Z.json

Correction to previous note:
- The literal runtime terms were stripped by shell command-substitution in the note append only; no job command was affected.
- Crop full-cache job remains running under `setsid` with fadvise page-cache eviction.
- PAE follow watcher was relaunched with `PAE_FOLLOW_MIN_GAP=65536` and `PAE_FOLLOW_MAX_PASSES=32`.

## 2026-05-27 — VAE-agnostic framing update

User corrected the PDM-3-HT story: PAE must be treated as a latent backend / baseline tool, not as the innovation. The core proposed mechanism is HT + MeanFlow per-patch integration, validated over multiple latent backends.

Action item opened: build FLUX.2 dev VAE latent cache from the shared cropped uint8 ImageNet-256 cache, then later Qwen Image VAE cache. Report A/B/C outcomes honestly, including possible worse FID on Qwen/FLUX if those VAEs are not ImageNet-256 tuned.
