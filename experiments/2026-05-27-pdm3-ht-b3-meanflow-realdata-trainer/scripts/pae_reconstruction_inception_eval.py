#!/usr/bin/env python3
"""PAE reconstruction image-space lower-bound eval.

Samples real cropped ImageNet-256 examples, loads the matching cached PAE
latents for the same global indices, decodes those latents through the restored
PAE decoder, and computes torchvision InceptionV3 pool-2048 FID/MMD/KID between:

  real cropped uint8 images  vs  PAE decode(PAE cached latents)

This is the practical image-space lower bound for any generator trained on the
same bf16 PAE latent cache and decoded with the same PAE decoder/eval recipe.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors import safe_open

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from decode_and_compact_eval_step import (  # noqa: E402
    REPO_ROOT,
    compact_metrics,
    decode_latents,
    dtype_from_name,
    image_summary,
    latent_summary,
    load_pae_model,
    make_grid,
    pick_device,
    save_images,
    save_json,
    tensor_to_uint8_images,
)
from decode_and_inception_eval_step import (  # noqa: E402
    compute_inception_metrics,
    inception_features,
    load_inception_feature_extractor,
    load_real_uint8_reference,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_latent_shards(latent_cache: Path, glob_pat: str) -> List[Path]:
    files = sorted([p for p in latent_cache.glob(glob_pat) if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No latent shard files matched {latent_cache / glob_pat}")
    return files


def read_latent_shard_len(path: Path, key: str) -> int:
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        if key not in sf.keys():
            raise KeyError(f"{path} missing key {key!r}; keys={list(sf.keys())}")
        if "labels" not in sf.keys():
            raise KeyError(f"{path} missing key 'labels'; keys={list(sf.keys())}")
        shape = sf.get_slice(key).get_shape()
        if len(shape) != 4 or shape[1:] != [32, 16, 16]:
            raise ValueError(f"{path} expected {key} shape [N,32,16,16], got {shape}")
        labels_shape = sf.get_slice("labels").get_shape()
        if labels_shape != [shape[0]]:
            raise ValueError(f"{path} labels shape {labels_shape} does not match {key} N={shape[0]}")
        return int(shape[0])


def load_latents_for_global_indices(
    latent_cache: Path,
    global_indices: Sequence[int],
    *,
    key: str = "latents",
    shard_glob: str = "latents_rank00_shard*.safetensors",
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Load cached PAE latents for sorted global dataset indices."""
    indices = np.asarray([int(i) for i in global_indices], dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("global_indices must be a non-empty 1D sequence")
    if np.any(indices[1:] < indices[:-1]):
        raise ValueError("global_indices must be sorted ascending")

    files = list_latent_shards(latent_cache, shard_glob)
    lengths = [read_latent_shard_len(p, key) for p in files]
    total = int(sum(lengths))
    if int(indices[-1]) >= total:
        raise ValueError(f"max requested index {int(indices[-1])} >= latent cache total {total}")

    latent_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    refs: List[Dict[str, Any]] = []
    cursor = 0
    pos = 0
    for shard_idx, (path, length) in enumerate(zip(files, lengths)):
        next_cursor = cursor + length
        local_indices: List[int] = []
        global_taken: List[int] = []
        while pos < len(indices) and cursor <= int(indices[pos]) < next_cursor:
            gi = int(indices[pos])
            local_indices.append(gi - cursor)
            global_taken.append(gi)
            pos += 1
        if local_indices:
            with safe_open(str(path), framework="pt", device="cpu") as sf:
                z = sf.get_slice(key)[local_indices].float().contiguous()
                y = sf.get_slice("labels")[local_indices].long().contiguous()
            latent_chunks.append(z)
            label_chunks.append(y)
            for local_i, global_i, label_i in zip(local_indices, global_taken, y.tolist()):
                refs.append(
                    {
                        "global_index": int(global_i),
                        "shard": path.name,
                        "shard_index": int(shard_idx),
                        "local_index": int(local_i),
                        "label": int(label_i),
                        "key": key,
                    }
                )
        cursor = next_cursor
        if pos >= len(indices):
            break
    if pos != len(indices):
        raise RuntimeError(f"Only collected {pos}/{len(indices)} requested latents")
    return torch.cat(latent_chunks, dim=0), torch.cat(label_chunks, dim=0), refs


def save_limited_images(
    arr: np.ndarray,
    labels: Optional[torch.Tensor],
    out_dir: Path,
    prefix: str,
    *,
    max_save: int,
) -> List[str]:
    if max_save <= 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        return []
    n = min(int(arr.shape[0]), int(max_save))
    labs = labels[:n] if labels is not None else None
    return save_images(arr[:n], labs, out_dir, prefix)


def write_markdown(path: Path, rec: Dict[str, Any]) -> None:
    m = rec["metrics"]
    lines = [
        "# PAE reconstruction Inception image-space lower-bound\n\n",
        f"- created_at_utc: `{rec['created_at_utc']}`\n",
        f"- status: `{rec['status']}`\n",
        f"- output_dir: `{rec['output_dir']}`\n",
        f"- real image cache: `{rec['real_image_cache_dir']}`\n",
        f"- latent cache: `{rec['latent_cache_dir']}`\n",
        f"- latent key: `{rec['latent_key']}`\n",
        f"- PAE config: `{rec['pae_config']}`\n",
        f"- PAE ckpt: `{rec['pae_ckpt']}`\n",
        f"- decode device/dtype: `{rec['decode_device']}` / `{rec['decode_dtype']}`\n",
        f"- Inception device: `{rec['inception_device']}`\n",
        f"- num_recon_metric: `{m['num_generated']}`; num_real_metric: `{m['num_real']}`\n",
        f"- seed: `{rec['seed']}`\n",
        "\n## Inception metrics\n\n",
        "| metric | value |\n|---|---:|\n",
        f"| PAE recon FID lower-bound | `{m['fid']}` |\n",
        f"| FID raw | `{m['fid_raw']}` |\n",
        f"| MMD RBF | `{m['inception_mmd_rbf']}` |\n",
        f"| MMD bandwidth^2 | `{m['inception_mmd_rbf_bandwidth2']}` |\n",
        f"| KID poly3 | `{m['inception_kid_poly3']}` |\n",
        f"| feature dim | `{m['feature_dim']}` |\n",
        "\n> This compares matched real cropped ImageNet-256 images against PAE-decoded cached latents for the same global indices. "
        "It is the practical image-space lower-bound for models trained on this bf16 PAE latent cache under the same decoder/eval recipe.\n\n",
        "## Image stats\n\n",
        "| split | shape | mean | std | min | max | finite |\n|---|---|---:|---:|---:|---:|---|\n",
        f"| real_imagenet256 | `{rec['real_image_summary']['shape']}` | `{rec['real_image_summary']['mean']}` | `{rec['real_image_summary']['std']}` | `{rec['real_image_summary']['min']}` | `{rec['real_image_summary']['max']}` | `{rec['real_image_summary']['finite']}` |\n",
        f"| pae_reconstruction | `{rec['reconstruction_image_summary']['shape']}` | `{rec['reconstruction_image_summary']['mean']}` | `{rec['reconstruction_image_summary']['std']}` | `{rec['reconstruction_image_summary']['min']}` | `{rec['reconstruction_image_summary']['max']}` | `{rec['reconstruction_image_summary']['finite']}` |\n",
        "\n## Integrity checks\n\n",
        "| check | value |\n|---|---:|\n",
        f"| labels_match_real_vs_latent | `{rec['labels_match_real_vs_latent']}` |\n",
        f"| global_indices_match_count | `{rec['global_indices_match_count']}` |\n",
        f"| elapsed_sec | `{rec['elapsed_sec']}` |\n",
    ]
    if rec.get("compact_metrics"):
        cm = rec["compact_metrics"]
        lines.extend(
            [
                "\n## Compact metrics carried for continuity\n\n",
                "| metric | value |\n|---|---:|\n",
                f"| compact_fid_rp{cm['feature_dim']} | `{cm['compact_fid_rp']}` |\n",
                f"| compact_mmd_rbf | `{cm['compact_mmd_rbf']}` |\n",
                f"| compact_kid_poly3 | `{cm['compact_kid_poly3']}` |\n",
            ]
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--seed", type=int, default=20260529)
    p.add_argument("--pae-config", type=Path, default=REPO_ROOT / "external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml")
    p.add_argument("--pae-ckpt", type=Path, default=REPO_ROOT / "data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt")
    p.add_argument("--real-image-cache", type=Path, default=REPO_ROOT / "data/cropped_uint8/imagenet1k_train_256_adm_safetensors")
    p.add_argument("--real-shard-glob", default="images_uint8_shard*.safetensors")
    p.add_argument("--latent-cache", type=Path, default=REPO_ROOT / "data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full")
    p.add_argument("--latent-shard-glob", default="latents_rank00_shard*.safetensors")
    p.add_argument("--latent-key", default="latents", choices=["latents", "latents_flip"])
    p.add_argument("--decode-device", default="cuda")
    p.add_argument("--decode-dtype", default="bf16")
    p.add_argument("--decode-batch-size", type=int, default=32)
    p.add_argument("--inception-device", default="cuda")
    p.add_argument("--inception-batch-size", type=int, default=128)
    p.add_argument("--torch-num-threads", type=int, default=32)
    p.add_argument("--max-save-images", type=int, default=0)
    p.add_argument("--grid-count", type=int, default=0)
    p.add_argument("--no-save-npz", action="store_true")
    p.add_argument("--skip-compact", action="store_true")
    p.add_argument("--no-pretrained-inception", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    started = time.perf_counter()
    out_dir = args.output_dir.resolve()
    metrics_path = out_dir / "inception_metrics.json"
    if metrics_path.exists() and not args.overwrite:
        print(f"Metrics already exist and --overwrite not set: {metrics_path}", flush=True)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))

    n = int(args.num_samples)
    if n < 2:
        raise ValueError("Need at least two samples for FID/KID/MMD")

    decode_device = pick_device(args.decode_device)
    decode_dtype = dtype_from_name(args.decode_dtype)
    if decode_device.type == "cpu" and decode_dtype != torch.float32:
        print(f"CPU decode requested with {decode_dtype}; using fp32", flush=True)
        decode_dtype = torch.float32
    inception_device = pick_device(args.inception_device)

    print(f"Loading {n} real ImageNet-256 references seed={args.seed} from {args.real_image_cache}", flush=True)
    real_arr, real_labels, real_source_indices, real_refs = load_real_uint8_reference(
        args.real_image_cache,
        n,
        int(args.seed),
        shard_glob=args.real_shard_glob,
    )
    global_indices = [int(r["global_index"]) for r in real_refs]

    print(f"Loading matching cached PAE latents key={args.latent_key} from {args.latent_cache}", flush=True)
    z, latent_labels, latent_refs = load_latents_for_global_indices(
        args.latent_cache,
        global_indices,
        key=args.latent_key,
        shard_glob=args.latent_shard_glob,
    )
    labels_match = bool(torch.equal(real_labels.cpu(), latent_labels.cpu()))
    if not labels_match:
        mism = (real_labels.cpu() != latent_labels.cpu()).nonzero(as_tuple=False).flatten()[:20].tolist()
        print(f"WARNING label mismatch real vs latent at first indices {mism}", flush=True)

    print(f"Loading PAE and decoding {n} cached latents on {decode_device}", flush=True)
    pae = load_pae_model(args.pae_config, args.pae_ckpt, decode_device, decode_dtype)
    recon_x = decode_latents(pae, z, decode_device, decode_dtype, int(args.decode_batch_size))
    del pae
    if decode_device.type == "cuda":
        torch.cuda.empty_cache()
    recon_arr = tensor_to_uint8_images(recon_x)

    image_root = out_dir / "images"
    real_dir = image_root / "real_imagenet256"
    recon_dir = image_root / "pae_reconstruction"
    real_paths = save_limited_images(real_arr, real_labels, real_dir, "real_imagenet256", max_save=int(args.max_save_images))
    recon_paths = save_limited_images(recon_arr, latent_labels, recon_dir, "pae_recon", max_save=int(args.max_save_images))

    grid_n = min(int(args.grid_count), n)
    real_grid = image_root / "real_imagenet256_grid.png"
    recon_grid = image_root / "pae_reconstruction_grid.png"
    if grid_n > 0:
        make_grid(real_arr[:grid_n], real_labels[:grid_n], real_grid, cols=8, annotate=True)
        make_grid(recon_arr[:grid_n], latent_labels[:grid_n], recon_grid, cols=8, annotate=True)

    npz_path: Optional[Path] = None
    if not args.no_save_npz:
        npz_path = out_dir / "pae_reconstruction_and_real_samples.npz"
        np.savez_compressed(
            npz_path,
            real_imagenet256=real_arr,
            pae_reconstruction=recon_arr,
            real_labels=real_labels.cpu().numpy(),
            latent_labels=latent_labels.cpu().numpy(),
            real_source_indices=(real_source_indices.cpu().numpy() if real_source_indices is not None else np.array([], dtype=np.int64)),
            global_indices=np.asarray(global_indices, dtype=np.int64),
        )

    print(f"Loading InceptionV3 feature extractor on {inception_device}", flush=True)
    inception = load_inception_feature_extractor(inception_device, no_pretrained=args.no_pretrained_inception)
    print("Extracting PAE reconstruction Inception features", flush=True)
    recon_feats = inception_features(recon_arr, model=inception, device=inception_device, batch_size=int(args.inception_batch_size))
    print("Extracting real ImageNet-256 Inception features", flush=True)
    real_feats = inception_features(real_arr, model=inception, device=inception_device, batch_size=int(args.inception_batch_size))
    del inception
    if inception_device.type == "cuda":
        torch.cuda.empty_cache()

    metrics = compute_inception_metrics(real_feats, recon_feats)
    compact: Optional[Dict[str, Any]] = None
    if not args.skip_compact:
        compact = compact_metrics(recon_arr, real_arr, 128, 32, int(args.seed))

    feature_npz = out_dir / "inception_features.npz"
    np.savez_compressed(
        feature_npz,
        pae_reconstruction=recon_feats.astype(np.float32),
        real_imagenet256=real_feats.astype(np.float32),
        labels=real_labels.cpu().numpy(),
        global_indices=np.asarray(global_indices, dtype=np.int64),
    )

    rec: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "status": "ok",
        "repo_root": str(REPO_ROOT),
        "output_dir": str(out_dir),
        "real_image_cache_dir": str(args.real_image_cache),
        "real_shard_glob": str(args.real_shard_glob),
        "latent_cache_dir": str(args.latent_cache),
        "latent_shard_glob": str(args.latent_shard_glob),
        "latent_key": str(args.latent_key),
        "pae_config": str(args.pae_config),
        "pae_ckpt": str(args.pae_ckpt),
        "decode_device": str(decode_device),
        "decode_dtype": str(decode_dtype),
        "decode_batch_size": int(args.decode_batch_size),
        "inception_device": str(inception_device),
        "inception_batch_size": int(args.inception_batch_size),
        "num_samples": n,
        "seed": int(args.seed),
        "max_save_images": int(args.max_save_images),
        "real_image_dir": str(real_dir),
        "reconstruction_image_dir": str(recon_dir),
        "real_grid_path": str(real_grid if grid_n > 0 else ""),
        "reconstruction_grid_path": str(recon_grid if grid_n > 0 else ""),
        "decoded_npz_path": str(npz_path) if npz_path is not None else None,
        "inception_feature_npz_path": str(feature_npz),
        "real_image_paths_first16": real_paths[:16],
        "reconstruction_image_paths_first16": recon_paths[:16],
        "real_refs_first32": real_refs[:32],
        "latent_refs_first32": latent_refs[:32],
        "labels_match_real_vs_latent": labels_match,
        "global_indices_match_count": int(len(global_indices)),
        "global_indices_first32": global_indices[:32],
        "latent_summary": latent_summary(z),
        "real_image_summary": image_summary(real_arr),
        "reconstruction_image_summary": image_summary(recon_arr),
        "real_feature_summary": {
            "shape": list(real_feats.shape),
            "finite": bool(np.isfinite(real_feats).all()),
            "mean": float(real_feats.mean()),
            "std": float(real_feats.std(ddof=0)),
        },
        "reconstruction_feature_summary": {
            "shape": list(recon_feats.shape),
            "finite": bool(np.isfinite(recon_feats).all()),
            "mean": float(recon_feats.mean()),
            "std": float(recon_feats.std(ddof=0)),
        },
        "metrics": metrics,
        "compact_metrics": compact,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    save_json(metrics_path, rec)
    write_markdown(out_dir / "inception_metrics.md", rec)
    print("PAE_RECON_INCEPTION_EVAL", json.dumps(rec, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
