# Optimized cropped uint8 cache experiment

Artifacts:
- `scripts/build_cropped_uint8_cache.py`: pyarrow local parquet -> multiprocessing JPEG decode/ADM center crop -> uint8 safetensors shards.
- `scripts/build_pae_latents_from_cropped_cache.py`: cropped uint8 safetensors -> PAE_DINOv2L_d32 latent shards.
- `configs/optimized_cache.env`: paths/parameters.

Schema for cropped shards:
- `images`: uint8 `[N,3,256,256]`
- `labels`: int64 `[N]`
- `source_indices`: int64 `[N]`, original ImageNet train order index.

This C cache is preferred over WebDataset JPEG tar for this workflow because parquet is already the raw JPEG archive and C removes repeated JPEG decode/crop for PAE, FID real stats and MMD.
