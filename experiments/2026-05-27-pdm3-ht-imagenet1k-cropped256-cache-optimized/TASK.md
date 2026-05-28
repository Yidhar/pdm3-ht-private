# Optimized ImageNet-1k 256x256 Cropped uint8 Cache + Fast PAE Encoding

Goal: remove the CPU/HF/PIL input bottleneck in the full ImageNet-1k -> PAE latent cache path.

Decision: use C (cropped uint8 tensor cache) rather than B (WebDataset JPEG tar), because local HF parquet already provides the raw JPEG archive, while cropped uint8 shards amortize JPEG decode/crop once and can be reused by PAE latent build, FID real-stat computation, MMD, visualization, and preprocessing audits.
