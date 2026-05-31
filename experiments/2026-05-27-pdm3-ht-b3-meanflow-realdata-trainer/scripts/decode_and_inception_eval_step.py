#!/usr/bin/env python3
"""Decode PAE latent samples and compute real ImageNet-256 Inception metrics.

This helper is intended for the B3 MeanFlow real-data run's step-20000 gate.
It complements ``decode_and_compact_eval_step.py``:

* generated samples are decoded from PAE latents with the restored PAE decoder;
* real references are read directly from the cropped uint8 ImageNet-256 cache
  (ADM-style center crop/resize), not from decoded PAE latents;
* image-space metrics are computed on torchvision InceptionV3 pool features:
  FID, RBF-MMD, and polynomial KID.

Default devices are CPU-only so the active H100 training process is not
disturbed.  The default trainer eval currently emits 64 generated samples; with
that sample count, these metrics are useful as an early wiring/quality signal,
not as a publishable 50k FID.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open

# Reuse the already tested PAE loading/decoding/image helpers.
from decode_and_compact_eval_step import (  # noqa: E402
    REPO_ROOT,
    compact_metrics,
    decode_latents,
    dtype_from_name,
    image_summary,
    latent_summary,
    load_generated_latents,
    load_pae_model,
    make_grid,
    pick_device,
    save_images,
    save_json,
    tensor_to_uint8_images,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_real_image_shards(data_dir: Path, glob_pat: str) -> List[Path]:
    files = sorted([p for p in data_dir.glob(glob_pat) if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No real-image shard files matched {data_dir / glob_pat}")
    return files


def read_real_shard_len(path: Path) -> int:
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        if "images" not in sf.keys():
            raise KeyError(f"{path} missing key 'images'; keys={list(sf.keys())}")
        shape = sf.get_slice("images").get_shape()
        if len(shape) != 4 or shape[1:] != [3, 256, 256]:
            raise ValueError(f"{path} expected images [N,3,256,256], got {shape}")
        return int(shape[0])


def load_real_uint8_reference(
    data_dir: Path,
    n: int,
    seed: int,
    *,
    shard_glob: str = "images_uint8_shard*.safetensors",
    max_global_index_exclusive: Optional[int] = None,
) -> Tuple[np.ndarray, torch.Tensor, Optional[torch.Tensor], List[Dict[str, Any]]]:
    """Deterministically sample real cropped ImageNet-256 images from safetensors.

    Returns:
        arr: NHWC uint8 RGB images.
        labels: int64 class labels.
        source_indices: optional original ImageNet/cache indices if present.
        refs: lightweight provenance records for the sampled images.
    """
    files = list_real_image_shards(data_dir, shard_glob)
    lengths = [read_real_shard_len(p) for p in files]
    total_all = int(sum(lengths))
    if max_global_index_exclusive is not None and int(max_global_index_exclusive) > 0:
        total = min(total_all, int(max_global_index_exclusive))
    else:
        total = total_all
    if n <= 0:
        raise ValueError("n must be positive")
    if n > total:
        raise ValueError(
            f"Requested {n} real images but selected real-index pool only contains {total} "
            f"(cache_total={total_all}, max_global_index_exclusive={max_global_index_exclusive})"
        )

    rng = np.random.default_rng(seed)
    global_indices = np.sort(rng.choice(total, size=n, replace=False))

    image_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    source_chunks: List[torch.Tensor] = []
    refs: List[Dict[str, Any]] = []
    cursor = 0
    pos = 0
    for shard_idx, (path, length) in enumerate(zip(files, lengths)):
        next_cursor = cursor + length
        local_indices: List[int] = []
        global_taken: List[int] = []
        while pos < len(global_indices) and cursor <= int(global_indices[pos]) < next_cursor:
            gi = int(global_indices[pos])
            local_indices.append(gi - cursor)
            global_taken.append(gi)
            pos += 1
        if local_indices:
            with safe_open(str(path), framework="pt", device="cpu") as sf:
                images = sf.get_slice("images")[local_indices].contiguous()
                labels = sf.get_slice("labels")[local_indices].long().contiguous()
                if "source_indices" in sf.keys():
                    src = sf.get_slice("source_indices")[local_indices].long().contiguous()
                else:
                    src = torch.full((len(local_indices),), -1, dtype=torch.long)
            if images.dtype != torch.uint8:
                raise TypeError(f"{path} images expected uint8, got {images.dtype}")
            image_chunks.append(images)
            label_chunks.append(labels)
            source_chunks.append(src)
            for local_i, global_i, label_i, src_i in zip(local_indices, global_taken, labels.tolist(), src.tolist()):
                refs.append(
                    {
                        "global_index": int(global_i),
                        "shard": path.name,
                        "shard_index": int(shard_idx),
                        "local_index": int(local_i),
                        "label": int(label_i),
                        "source_index": int(src_i),
                    }
                )
        cursor = next_cursor
        if pos >= len(global_indices):
            break
    if pos != len(global_indices):
        raise RuntimeError(f"Only collected {pos}/{len(global_indices)} requested real images")

    images_chw = torch.cat(image_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    source_indices = torch.cat(source_chunks, dim=0) if source_chunks else None
    arr = images_chw.permute(0, 2, 3, 1).contiguous().numpy()
    return arr, labels, source_indices, refs


def load_inception_feature_extractor(device: torch.device, *, no_pretrained: bool = False) -> torch.nn.Module:
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = None if no_pretrained else Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def inception_features(
    arr: np.ndarray,
    *,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers_note: str = "single_process",
) -> np.ndarray:
    """Extract torchvision InceptionV3 pool/fc-input features from NHWC uint8 images."""
    del num_workers_note  # kept in signature for explicit metadata compatibility
    if arr.ndim != 4 or arr.shape[-1] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"expected NHWC uint8 RGB images, got shape={arr.shape} dtype={arr.dtype}")
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    feats: List[torch.Tensor] = []
    total = int(arr.shape[0])
    for start in range(0, total, batch_size):
        batch_np = arr[start : start + batch_size]
        x = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().float().div_(255.0)
        x = x.to(device=device, non_blocking=False)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False, antialias=True)
        x = (x - mean) / std
        y = model(x)
        if isinstance(y, tuple):
            y = y[0]
        y = y.float().flatten(1).detach().cpu()
        feats.append(y)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return torch.cat(feats, dim=0).numpy().astype(np.float64)


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def fid_empirical_lowrank(real: np.ndarray, gen: np.ndarray) -> Dict[str, float | str]:
    """Exact empirical Gaussian FID using an SVD trace-sqrt in sample space.

    For the common early-eval case N << 2048, this avoids a slow 2048x2048
    matrix square root while computing the same no-epsilon empirical covariance
    trace term.  Requires at least two samples per split.
    """
    real = np.asarray(real, dtype=np.float64)
    gen = np.asarray(gen, dtype=np.float64)
    if real.ndim != 2 or gen.ndim != 2 or real.shape[1] != gen.shape[1]:
        raise ValueError(f"feature shape mismatch: real={real.shape}, gen={gen.shape}")
    n, m = real.shape[0], gen.shape[0]
    if n < 2 or m < 2:
        raise ValueError("FID requires at least two samples per split")
    mu_r = real.mean(axis=0)
    mu_g = gen.mean(axis=0)
    xr = real - mu_r
    xg = gen - mu_g
    diff = mu_r - mu_g
    mean_term = float(diff.dot(diff))
    trace_r = float(np.sum(xr * xr) / (n - 1))
    trace_g = float(np.sum(xg * xg) / (m - 1))
    cross = (xr @ xg.T) / math.sqrt(float((n - 1) * (m - 1)))
    # Singular values of cross are sqrt(non-zero eigenvalues of C_real*C_gen).
    trace_sqrt = float(np.linalg.svd(cross, compute_uv=False).sum())
    raw = mean_term + trace_r + trace_g - 2.0 * trace_sqrt
    return {
        "fid": float(max(raw, 0.0)),
        "fid_raw": float(raw),
        "fid_mean_term": mean_term,
        "fid_trace_real": trace_r,
        "fid_trace_gen": trace_g,
        "fid_trace_sqrt_product": trace_sqrt,
        "fid_covariance_trace_method": "exact_empirical_lowrank_svd",
    }


def mmd_rbf_unbiased(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.concatenate([x, y], axis=0)
    d = squared_distances(z, z)
    tri = d[np.triu_indices_from(d, k=1)]
    bandwidth2 = float(np.median(tri[tri > 0])) if np.any(tri > 0) else 1.0
    if not np.isfinite(bandwidth2) or bandwidth2 <= 0:
        bandwidth2 = 1.0
    kxx = np.exp(-squared_distances(x, x) / (2.0 * bandwidth2))
    kyy = np.exp(-squared_distances(y, y) / (2.0 * bandwidth2))
    kxy = np.exp(-squared_distances(x, y) / (2.0 * bandwidth2))
    n, m = x.shape[0], y.shape[0]
    xx = (kxx.sum() - np.trace(kxx)) / max(n * (n - 1), 1)
    yy = (kyy.sum() - np.trace(kyy)) / max(m * (m - 1), 1)
    xy = kxy.mean()
    return float(xx + yy - 2.0 * xy), bandwidth2


def kid_poly3_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = float(x.shape[1])
    kxx = (x @ x.T / d + 1.0) ** 3
    kyy = (y @ y.T / d + 1.0) ** 3
    kxy = (x @ y.T / d + 1.0) ** 3
    n, m = x.shape[0], y.shape[0]
    xx = (kxx.sum() - np.trace(kxx)) / max(n * (n - 1), 1)
    yy = (kyy.sum() - np.trace(kyy)) / max(m * (m - 1), 1)
    xy = kxy.mean()
    return float(xx + yy - 2.0 * xy)


def compute_inception_metrics(real_feats: np.ndarray, gen_feats: np.ndarray) -> Dict[str, Any]:
    fid_rec = fid_empirical_lowrank(real_feats, gen_feats)
    mmd, bandwidth2 = mmd_rbf_unbiased(real_feats, gen_feats)
    kid = kid_poly3_unbiased(real_feats, gen_feats)
    return {
        "feature_method": "torchvision_inception_v3_imagenet1k_v1_pool2048",
        "feature_dim": int(real_feats.shape[1]),
        "num_real": int(real_feats.shape[0]),
        "num_generated": int(gen_feats.shape[0]),
        **fid_rec,
        "inception_mmd_rbf": float(mmd),
        "inception_mmd_rbf_bandwidth2": float(bandwidth2),
        "inception_kid_poly3": float(kid),
        "sample_count_note": (
            "Early/sample-count-limited diagnostic unless num_generated and num_real are large "
            "(e.g. 50k for publishable FID)."
        ),
    }


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
        "# Step Inception image-space eval\n\n",
        f"- created_at_utc: `{rec['created_at_utc']}`\n",
        f"- sample_latents: `{rec['sample_latents_path']}`\n",
        f"- output_dir: `{rec['output_dir']}`\n",
        f"- generated images: `{rec['generated_image_dir']}`\n",
        f"- real ImageNet-256 images: `{rec['real_image_dir']}`\n",
        f"- real image cache: `{rec['real_image_cache_dir']}`\n",
        f"- PAE decode device/dtype: `{rec['decode_device']}` / `{rec['decode_dtype']}`\n",
        f"- Inception device: `{rec['inception_device']}`\n",
        f"- num_generated_metric: `{m['num_generated']}`; num_real_metric: `{m['num_real']}`\n",
        "\n## Inception metrics\n\n",
        "| metric | value |\n|---|---:|\n",
        f"| FID | `{m['fid']}` |\n",
        f"| FID raw | `{m['fid_raw']}` |\n",
        f"| MMD RBF | `{m['inception_mmd_rbf']}` |\n",
        f"| MMD bandwidth^2 | `{m['inception_mmd_rbf_bandwidth2']}` |\n",
        f"| KID poly3 | `{m['inception_kid_poly3']}` |\n",
        f"| feature dim | `{m['feature_dim']}` |\n",
        "\n> Note: these are torchvision InceptionV3 pool-2048 image-space metrics against the real cropped uint8 ImageNet-256 cache. "
        "With the default 64 generated samples they are early/sample-count-limited diagnostics, not publishable 50k FID.\n\n",
        "## Image stats\n\n",
        "| split | shape | mean | std | min | max | finite |\n|---|---|---:|---:|---:|---:|---|\n",
        f"| generated | `{rec['generated_image_summary']['shape']}` | `{rec['generated_image_summary']['mean']}` | `{rec['generated_image_summary']['std']}` | `{rec['generated_image_summary']['min']}` | `{rec['generated_image_summary']['max']}` | `{rec['generated_image_summary']['finite']}` |\n",
        f"| real_imagenet256 | `{rec['real_image_summary']['shape']}` | `{rec['real_image_summary']['mean']}` | `{rec['real_image_summary']['std']}` | `{rec['real_image_summary']['min']}` | `{rec['real_image_summary']['max']}` | `{rec['real_image_summary']['finite']}` |\n",
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
    p.add_argument("--sample-latents", type=Path, required=True, help="sample_latents.safetensors from trainer eval")
    p.add_argument("--output-dir", type=Path, default=None, help="defaults to sample_latents parent / inception_eval")
    p.add_argument("--pae-config", type=Path, default=REPO_ROOT / "external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml")
    p.add_argument("--pae-ckpt", type=Path, default=REPO_ROOT / "data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt")
    p.add_argument(
        "--real-image-cache",
        type=Path,
        default=REPO_ROOT / "data/cropped_uint8/imagenet1k_train_256_adm_safetensors",
        help="cropped uint8 ImageNet-256 safetensors cache",
    )
    p.add_argument("--real-shard-glob", default="images_uint8_shard*.safetensors")
    p.add_argument("--num-real", type=int, default=0, help="0 means match generated metric sample count")
    p.add_argument("--real-seed", type=int, default=20260528)
    p.add_argument(
        "--real-max-samples",
        type=int,
        default=0,
        help=(
            "if >0, sample real references only from global cache indices [0, real_max_samples); "
            "useful for small-subset fit probes where generated labels/training data come from the same first-N subset"
        ),
    )
    p.add_argument("--max-gen-samples", type=int, default=0, help="optional cap for generated samples")
    p.add_argument("--decode-device", default="cpu", help="cpu/cuda/cuda:0/auto; default cpu to avoid H100 contention")
    p.add_argument("--decode-dtype", default="fp32", help="fp32/bf16/fp16; CPU coerces to fp32")
    p.add_argument("--decode-batch-size", type=int, default=2)
    p.add_argument("--inception-device", default="cpu", help="cpu/cuda/cuda:0/auto; default cpu")
    p.add_argument("--inception-batch-size", type=int, default=16)
    p.add_argument("--torch-num-threads", type=int, default=0, help="optional torch CPU thread cap")
    p.add_argument("--max-save-images", type=int, default=256, help="save at most this many PNGs per split")
    p.add_argument("--grid-count", type=int, default=64)
    p.add_argument("--no-save-npz", action="store_true")
    p.add_argument("--skip-compact", action="store_true", help="skip compact RP metrics carried for continuity")
    p.add_argument("--no-pretrained-inception", action="store_true", help="debug only; do not use pretrained Inception weights")
    args = p.parse_args()

    started = time.perf_counter()
    if args.torch_num_threads and args.torch_num_threads > 0:
        torch.set_num_threads(int(args.torch_num_threads))

    out_dir = args.output_dir or (args.sample_latents.parent / "inception_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    decode_device = pick_device(args.decode_device)
    decode_dtype = dtype_from_name(args.decode_dtype)
    if decode_device.type == "cpu" and decode_dtype != torch.float32:
        print(f"CPU decode requested with {decode_dtype}; using fp32", flush=True)
        decode_dtype = torch.float32

    inception_device = pick_device(args.inception_device)

    gen_z, gen_labels, gen_keys = load_generated_latents(args.sample_latents)
    if args.max_gen_samples and args.max_gen_samples > 0:
        gen_z = gen_z[: args.max_gen_samples]
        if gen_labels is not None:
            gen_labels = gen_labels[: args.max_gen_samples]
    n_gen = int(gen_z.shape[0])
    if n_gen < 2:
        raise ValueError("Need at least two generated samples for Inception FID/KID/MMD")
    n_real = int(args.num_real) if args.num_real and args.num_real > 0 else n_gen
    if n_real < 2:
        raise ValueError("Need at least two real samples for Inception FID/KID/MMD")

    print(f"Loading {n_real} real ImageNet-256 references from {args.real_image_cache}", flush=True)
    real_arr, real_labels, real_source_indices, real_refs = load_real_uint8_reference(
        args.real_image_cache,
        n_real,
        args.real_seed,
        shard_glob=args.real_shard_glob,
        max_global_index_exclusive=(int(args.real_max_samples) if int(args.real_max_samples) > 0 else None),
    )

    print(f"Loading PAE and decoding {n_gen} generated latents on {decode_device}", flush=True)
    pae = load_pae_model(args.pae_config, args.pae_ckpt, decode_device, decode_dtype)
    gen_x = decode_latents(pae, gen_z, decode_device, decode_dtype, args.decode_batch_size)
    del pae
    if decode_device.type == "cuda":
        torch.cuda.empty_cache()
    gen_arr = tensor_to_uint8_images(gen_x)

    image_root = out_dir / "images"
    gen_dir = image_root / "generated"
    real_dir = image_root / "real_imagenet256"
    gen_paths = save_limited_images(gen_arr, gen_labels, gen_dir, "gen", max_save=args.max_save_images)
    real_paths = save_limited_images(real_arr, real_labels, real_dir, "real_imagenet256", max_save=args.max_save_images)

    grid_n_gen = min(int(args.grid_count), int(gen_arr.shape[0]))
    grid_n_real = min(int(args.grid_count), int(real_arr.shape[0]))
    gen_grid = image_root / "generated_grid.png"
    real_grid = image_root / "real_imagenet256_grid.png"
    if grid_n_gen > 0:
        make_grid(gen_arr[:grid_n_gen], gen_labels[:grid_n_gen] if gen_labels is not None else None, gen_grid, cols=8, annotate=True)
    if grid_n_real > 0:
        make_grid(real_arr[:grid_n_real], real_labels[:grid_n_real], real_grid, cols=8, annotate=True)

    npz_path: Optional[Path] = None
    if not args.no_save_npz:
        npz_path = out_dir / "decoded_and_real_imagenet256_samples.npz"
        np.savez_compressed(
            npz_path,
            generated=gen_arr,
            real_imagenet256=real_arr,
            generated_labels=(gen_labels.cpu().numpy() if gen_labels is not None else np.array([], dtype=np.int64)),
            real_labels=real_labels.cpu().numpy(),
            real_source_indices=(real_source_indices.cpu().numpy() if real_source_indices is not None else np.array([], dtype=np.int64)),
        )

    print(f"Loading InceptionV3 feature extractor on {inception_device}", flush=True)
    inception = load_inception_feature_extractor(inception_device, no_pretrained=args.no_pretrained_inception)
    print("Extracting generated Inception features", flush=True)
    gen_feats = inception_features(gen_arr, model=inception, device=inception_device, batch_size=args.inception_batch_size)
    print("Extracting real ImageNet-256 Inception features", flush=True)
    real_feats = inception_features(real_arr, model=inception, device=inception_device, batch_size=args.inception_batch_size)
    del inception
    if inception_device.type == "cuda":
        torch.cuda.empty_cache()

    metrics = compute_inception_metrics(real_feats, gen_feats)
    compact: Optional[Dict[str, Any]] = None
    if not args.skip_compact:
        compact = compact_metrics(gen_arr[: min(n_gen, n_real)], real_arr[: min(n_gen, n_real)], 128, 32, args.real_seed)

    feature_npz = out_dir / "inception_features.npz"
    np.savez_compressed(
        feature_npz,
        generated=gen_feats.astype(np.float32),
        real_imagenet256=real_feats.astype(np.float32),
        generated_labels=(gen_labels.cpu().numpy() if gen_labels is not None else np.array([], dtype=np.int64)),
        real_labels=real_labels.cpu().numpy(),
    )

    rec: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "status": "ok",
        "repo_root": str(REPO_ROOT),
        "sample_latents_path": str(args.sample_latents),
        "sample_latents_keys": gen_keys,
        "output_dir": str(out_dir),
        "generated_image_dir": str(gen_dir),
        "real_image_dir": str(real_dir),
        "generated_grid_path": str(gen_grid if grid_n_gen > 0 else ""),
        "real_grid_path": str(real_grid if grid_n_real > 0 else ""),
        "decoded_npz_path": str(npz_path) if npz_path is not None else None,
        "inception_feature_npz_path": str(feature_npz),
        "pae_config": str(args.pae_config),
        "pae_ckpt": str(args.pae_ckpt),
        "real_image_cache_dir": str(args.real_image_cache),
        "real_shard_glob": str(args.real_shard_glob),
        "real_max_samples": int(args.real_max_samples),
        "decode_device": str(decode_device),
        "decode_dtype": str(decode_dtype),
        "decode_batch_size": int(args.decode_batch_size),
        "inception_device": str(inception_device),
        "inception_batch_size": int(args.inception_batch_size),
        "num_generated_total": int(gen_arr.shape[0]),
        "num_real_total": int(real_arr.shape[0]),
        "real_seed": int(args.real_seed),
        "max_save_images": int(args.max_save_images),
        "generated_image_paths_first16": gen_paths[:16],
        "real_image_paths_first16": real_paths[:16],
        "real_refs_first32": real_refs[:32],
        "generated_latent_summary": latent_summary(gen_z),
        "generated_image_summary": image_summary(gen_arr),
        "real_image_summary": image_summary(real_arr),
        "generated_feature_summary": {
            "shape": list(gen_feats.shape),
            "finite": bool(np.isfinite(gen_feats).all()),
            "mean": float(gen_feats.mean()),
            "std": float(gen_feats.std(ddof=0)),
        },
        "real_feature_summary": {
            "shape": list(real_feats.shape),
            "finite": bool(np.isfinite(real_feats).all()),
            "mean": float(real_feats.mean()),
            "std": float(real_feats.std(ddof=0)),
        },
        "metrics": metrics,
        "compact_metrics": compact,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    save_json(out_dir / "inception_metrics.json", rec)
    write_markdown(out_dir / "inception_metrics.md", rec)
    print("INCEPTION_EVAL", json.dumps(rec, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
