#!/usr/bin/env python3
"""Small PAE-vs-FLUX.2 VAE rFID ceiling diagnostic.

This script intentionally answers one narrow question:

    same N ImageNet-256 ADM crops
      -> PAE encode/decode      -> rFID(original, reconstruction)
      -> FLUX.2 encode/decode   -> rFID(original, reconstruction)

It uses the existing cropped uint8 safetensor cache so both VAEs see exactly the
same images.  Runtime outputs are experiment artifacts and should stay out of
Git (experiments/**/results is ignored).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("DISABLE_XFORMERS", "1")

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inception_metrics_lib import (  # noqa: E402
    build_inception,
    covariance_from_features,
    features_from_image_dir,
    frechet_distance,
    load_feature_bank,
    load_reference_stats,
    polynomial_mmd2_unbiased,
    resolve_device,
    write_json,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def dtype_from_name(name: str) -> torch.dtype:
    key = str(name).lower().strip()
    if key in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if key in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name!r}")


def images_to_uint8(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "zero1":
        y = x.clamp(0.0, 1.0).mul(255.0)
    elif mode == "minus1_1":
        y = x.clamp(-1.0, 1.0).add(1.0).mul(127.5)
    elif mode == "auto":
        lo = float(x.min().item())
        hi = float(x.max().item())
        if lo >= -0.05 and hi <= 1.05:
            y = x.clamp(0.0, 1.0).mul(255.0)
        else:
            y = x.clamp(-1.0, 1.0).add(1.0).mul(127.5)
    else:
        raise ValueError(f"Unsupported output mode: {mode!r}")
    return y.round().clamp(0, 255).to(torch.uint8)


def save_png_batch(images_u8: torch.Tensor, output_dir: Path, *, offset: int = 0) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if images_u8.ndim != 4 or images_u8.shape[1] != 3:
        raise RuntimeError(f"Expected [B,3,H,W] uint8 images, got {tuple(images_u8.shape)}")
    names: List[str] = []
    for i in range(int(images_u8.shape[0])):
        name = f"{offset + i:06d}.png"
        arr = images_u8[i].permute(1, 2, 0).contiguous().cpu().numpy()
        Image.fromarray(arr, mode="RGB").save(output_dir / name)
        names.append(name)
    return names


@dataclass
class CropShard:
    path: Path
    num_samples: int
    image_shape: List[int]
    has_labels: bool
    has_source_indices: bool


def list_crop_shards(cache_dir: Path, file_glob: str, image_key: str, label_key: str, source_index_key: str) -> List[CropShard]:
    files = sorted(cache_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No crop shards matching {file_glob!r} under {cache_dir}")
    out: List[CropShard] = []
    for p in files:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if image_key not in keys:
                raise KeyError(f"{p} missing image key={image_key!r}; keys={sorted(keys)}")
            shape = list(f.get_slice(image_key).get_shape())
            if len(shape) != 4 or shape[1] != 3:
                raise RuntimeError(f"Expected {image_key} [N,3,H,W] in {p}, got {shape}")
            out.append(
                CropShard(
                    path=p,
                    num_samples=int(shape[0]),
                    image_shape=shape,
                    has_labels=label_key in keys,
                    has_source_indices=source_index_key in keys,
                )
            )
    return out


def load_cropped_sample_tensor(
    *,
    cache_dir: Path,
    file_glob: str,
    image_key: str,
    label_key: str,
    source_index_key: str,
    start_index: int,
    max_images: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
    shards = list_crop_shards(cache_dir, file_glob, image_key, label_key, source_index_key)
    total = sum(s.num_samples for s in shards)
    if start_index < 0 or start_index >= total:
        raise RuntimeError(f"start_index={start_index} outside total={total}")
    target = min(int(max_images), total - int(start_index))
    images: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    source_indices: List[torch.Tensor] = []
    remaining_start = int(start_index)
    seen = 0
    first_record: Optional[Dict[str, Any]] = None
    last_record: Optional[Dict[str, Any]] = None
    for shard_idx, shard in enumerate(shards):
        shard_global_start = seen
        shard_global_end = seen + shard.num_samples
        seen = shard_global_end
        if shard_global_end <= start_index:
            continue
        if sum(int(x.shape[0]) for x in images) >= target:
            break
        local_start = max(0, remaining_start - shard_global_start)
        already = sum(int(x.shape[0]) for x in images)
        local_take = min(shard.num_samples - local_start, target - already)
        if local_take <= 0:
            continue
        local_end = local_start + local_take
        with safe_open(str(shard.path), framework="pt", device="cpu") as f:
            x = f.get_slice(image_key)[local_start:local_end].contiguous()
            images.append(x)
            y = f.get_slice(label_key)[local_start:local_end].long().contiguous() if shard.has_labels else None
            src = f.get_slice(source_index_key)[local_start:local_end].long().contiguous() if shard.has_source_indices else None
            if y is not None:
                labels.append(y)
            if src is not None:
                source_indices.append(src)
            rec = {
                "shard_index": int(shard_idx),
                "shard_path": str(shard.path),
                "local_start": int(local_start),
                "local_end_exclusive": int(local_end),
                "global_start": int(shard_global_start + local_start),
                "global_end_exclusive": int(shard_global_start + local_end),
            }
            if y is not None and y.numel():
                rec.update({"label_first": int(y[0].item()), "label_last": int(y[-1].item())})
            if src is not None and src.numel():
                rec.update({"source_index_first": int(src[0].item()), "source_index_last": int(src[-1].item())})
            if first_record is None:
                first_record = dict(rec)
            last_record = dict(rec)

    x_all = torch.cat(images, dim=0).contiguous()
    if int(x_all.shape[0]) != target:
        raise RuntimeError(f"Loaded {x_all.shape[0]} images but expected {target}")
    y_all = torch.cat(labels, dim=0).contiguous() if labels else None
    src_all = torch.cat(source_indices, dim=0).contiguous() if source_indices else None
    meta = {
        "crop_cache_dir": str(cache_dir),
        "crop_file_glob": file_glob,
        "crop_total": int(total),
        "crop_num_shards": int(len(shards)),
        "start_index": int(start_index),
        "max_images": int(max_images),
        "loaded_images": int(x_all.shape[0]),
        "image_shape": list(x_all.shape),
        "first_record": first_record,
        "last_record": last_record,
    }
    if y_all is not None:
        meta.update({"label_min": int(y_all.min().item()), "label_max": int(y_all.max().item())})
    return x_all, y_all, src_all, meta


class PixelDiffStats:
    def __init__(self) -> None:
        self.count = 0
        self.sumsq = 0.0
        self.sumabs = 0.0
        self.max_abs_uint8 = 0
        self.image_mse: List[float] = []
        self.image_mae: List[float] = []

    def update(self, original_u8: torch.Tensor, recon_u8: torch.Tensor) -> None:
        if original_u8.shape != recon_u8.shape:
            raise RuntimeError(f"Pixel shape mismatch: original={tuple(original_u8.shape)} recon={tuple(recon_u8.shape)}")
        diff_i = recon_u8.to(torch.int16) - original_u8.to(torch.int16)
        self.max_abs_uint8 = max(self.max_abs_uint8, int(diff_i.abs().max().item()))
        diff = diff_i.float().div(255.0)
        self.count += int(diff.numel())
        self.sumsq += float((diff.double() * diff.double()).sum().item())
        self.sumabs += float(diff.abs().double().sum().item())
        per = diff.flatten(1)
        self.image_mse.extend([float(v) for v in (per * per).mean(dim=1).cpu().tolist()])
        self.image_mae.extend([float(v) for v in per.abs().mean(dim=1).cpu().tolist()])

    def finalize(self) -> Dict[str, Any]:
        if self.count <= 0:
            return {"count": 0}
        mse = self.sumsq / float(self.count)
        mae = self.sumabs / float(self.count)
        im_mse = np.asarray(self.image_mse, dtype=np.float64)
        im_mae = np.asarray(self.image_mae, dtype=np.float64)
        im_psnr = np.where(im_mse <= 0, np.inf, 10.0 * np.log10(1.0 / im_mse))
        finite = np.isfinite(im_psnr)
        return {
            "count_pixels_times_channels": int(self.count),
            "mse_0_1": float(mse),
            "rmse_0_1": float(math.sqrt(max(0.0, mse))),
            "mae_0_1": float(mae),
            "psnr_db_global": float("inf") if mse <= 0 else float(10.0 * math.log10(1.0 / mse)),
            "max_abs_uint8": int(self.max_abs_uint8),
            "image_count": int(im_mse.shape[0]),
            "image_mse_0_1_mean": float(im_mse.mean()),
            "image_mse_0_1_median": float(np.median(im_mse)),
            "image_mae_0_1_mean": float(im_mae.mean()),
            "image_mae_0_1_median": float(np.median(im_mae)),
            "image_psnr_db_mean": float(im_psnr[finite].mean()) if finite.any() else float("inf"),
            "image_psnr_db_median": float(np.median(im_psnr[finite])) if finite.any() else float("inf"),
        }


class TensorStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = float("inf")
        self.max = float("-inf")
        self.shape_first: Optional[List[int]] = None
        self.shape_last: Optional[List[int]] = None

    def update(self, x: torch.Tensor) -> None:
        xf = x.detach().float().cpu()
        if self.shape_first is None:
            self.shape_first = list(xf.shape)
        self.shape_last = list(xf.shape)
        self.count += int(xf.numel())
        self.sum += float(xf.double().sum().item())
        self.sumsq += float((xf.double() * xf.double()).sum().item())
        self.min = min(self.min, float(xf.min().item()))
        self.max = max(self.max, float(xf.max().item()))

    def finalize(self) -> Dict[str, Any]:
        if self.count <= 0:
            return {"count": 0}
        mean = self.sum / float(self.count)
        var = max(0.0, self.sumsq / float(self.count) - mean * mean)
        return {
            "count": int(self.count),
            "mean": float(mean),
            "std": float(math.sqrt(var)),
            "rms": float(math.sqrt(max(0.0, self.sumsq / float(self.count)))),
            "min": float(self.min),
            "max": float(self.max),
            "absmax": float(max(abs(self.min), abs(self.max))),
            "shape_first": self.shape_first,
            "shape_last": self.shape_last,
        }


def import_pae_loader(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pdm_pae_latent_loader", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import PAE loader from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pdm_pae_latent_loader"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_pae(args: argparse.Namespace, device: torch.device, model_dtype: torch.dtype) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    loader = import_pae_loader(Path(args.pae_loader_script).resolve())
    t0 = time.perf_counter()
    model = loader.load_pae_model(Path(args.pae_config).resolve(), Path(args.pae_ckpt).resolve(), device, model_dtype)
    model.eval().requires_grad_(False)
    return model, {
        "kind": "PAE",
        "config": str(Path(args.pae_config).resolve()),
        "ckpt": str(Path(args.pae_ckpt).resolve()),
        "loader_script": str(Path(args.pae_loader_script).resolve()),
        "model_dtype": str(model_dtype),
        "load_elapsed_sec": float(time.perf_counter() - t0),
    }


def import_diffusers_class(class_name: str) -> Tuple[Any, str]:
    diffusers = importlib.import_module("diffusers")
    if not hasattr(diffusers, class_name):
        raise AttributeError(f"diffusers has no class {class_name!r}; version={getattr(diffusers, '__version__', '?')}")
    return getattr(diffusers, class_name), getattr(diffusers, "__version__", "?")


def load_flux_vae(args: argparse.Namespace, device: torch.device, model_dtype: torch.dtype) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    cls, version = import_diffusers_class(args.flux_vae_class)
    kwargs: Dict[str, Any] = {"torch_dtype": model_dtype}
    if args.flux_subfolder:
        kwargs["subfolder"] = args.flux_subfolder
    if args.flux_revision:
        kwargs["revision"] = args.flux_revision
    if args.local_files_only:
        kwargs["local_files_only"] = True
    t0 = time.perf_counter()
    vae = cls.from_pretrained(args.flux_repo_id, **kwargs)
    vae = vae.to(device=device, dtype=model_dtype).eval()
    vae.requires_grad_(False)
    try:
        cfg = dict(getattr(vae, "config", {}) or {})
    except Exception:
        cfg = {}
    return vae, {
        "kind": "FLUX.2 VAE",
        "repo_id": args.flux_repo_id,
        "subfolder": args.flux_subfolder,
        "revision": args.flux_revision,
        "vae_class": args.flux_vae_class,
        "diffusers_version": version,
        "model_dtype": str(model_dtype),
        "load_elapsed_sec": float(time.perf_counter() - t0),
        "config_scaling_factor": cfg.get("scaling_factor"),
        "config_shift_factor": cfg.get("shift_factor"),
        "config_latent_channels": cfg.get("latent_channels"),
        "config_sample_size": cfg.get("sample_size"),
    }


def postprocess_tensor4(x: torch.Tensor, name: str) -> torch.Tensor:
    if x.ndim == 5 and x.shape[2] == 1:
        x = x.squeeze(2)
    if x.ndim != 4:
        raise RuntimeError(f"Expected {name} as 4D [B,C,H,W], got {tuple(x.shape)}")
    return x


def reconstruct_pae(
    model: torch.nn.Module,
    images_u8: torch.Tensor,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    batch_size: int,
    output_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pix = PixelDiffStats()
    latent_stats = TensorStats()
    decoded_stats = TensorStats()
    started = time.perf_counter()
    use_autocast = device.type == "cuda" and model_dtype in {torch.bfloat16, torch.float16}
    for s in range(0, int(images_u8.shape[0]), int(batch_size)):
        e = min(int(images_u8.shape[0]), s + int(batch_size))
        x_u8 = images_u8[s:e].contiguous()
        x = x_u8.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=use_autocast):
                z = model.encode(x)
                rec = model.decode(z)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rec = postprocess_tensor4(rec, "PAE decoded image")
        rec_cpu = rec.detach().float().cpu()
        rec_u8 = images_to_uint8(rec_cpu, "zero1")
        save_png_batch(rec_u8, output_dir, offset=s)
        pix.update(x_u8.cpu(), rec_u8)
        latent_stats.update(z.detach().float().cpu())
        decoded_stats.update(rec_cpu)
        del x, z, rec, rec_cpu, rec_u8, x_u8
    return (
        {
            "elapsed_sec": float(time.perf_counter() - started),
            "samples_per_sec": float(images_u8.shape[0] / (time.perf_counter() - started)),
            "pixel_metrics": pix.finalize(),
        },
        {
            "latent_stats": latent_stats.finalize(),
            "decoded_float_stats": decoded_stats.finalize(),
        },
    )


def reconstruct_flux(
    vae: torch.nn.Module,
    images_u8: torch.Tensor,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    batch_size: int,
    output_dir: Path,
    latent_mode: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pix = PixelDiffStats()
    latent_stats = TensorStats()
    decoded_stats = TensorStats()
    started = time.perf_counter()
    for s in range(0, int(images_u8.shape[0]), int(batch_size)):
        e = min(int(images_u8.shape[0]), s + int(batch_size))
        x_u8 = images_u8[s:e].contiguous()
        x = x_u8.to(device=device, dtype=torch.float32, non_blocking=True).div_(127.5).sub_(1.0)
        if model_dtype != torch.float32:
            x = x.to(dtype=model_dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.inference_mode():
            enc = vae.encode(x)
            if latent_mode == "mode":
                z = enc.latent_dist.mode()
            elif latent_mode == "sample":
                z = enc.latent_dist.sample()
            else:
                raise ValueError(f"Unsupported latent_mode={latent_mode!r}")
            z = postprocess_tensor4(z, "FLUX latent")
            dec = vae.decode(z)
            rec = dec.sample if hasattr(dec, "sample") else dec[0]
            rec = postprocess_tensor4(rec, "FLUX decoded image")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rec_cpu = rec.detach().float().cpu()
        rec_u8 = images_to_uint8(rec_cpu, "minus1_1")
        save_png_batch(rec_u8, output_dir, offset=s)
        pix.update(x_u8.cpu(), rec_u8)
        latent_stats.update(z.detach().float().cpu())
        decoded_stats.update(rec_cpu)
        del x, z, rec, rec_cpu, rec_u8, x_u8
    return (
        {
            "elapsed_sec": float(time.perf_counter() - started),
            "samples_per_sec": float(images_u8.shape[0] / (time.perf_counter() - started)),
            "pixel_metrics": pix.finalize(),
        },
        {
            "latent_stats": latent_stats.finalize(),
            "decoded_float_stats": decoded_stats.finalize(),
        },
    )


def compute_pairwise_image_metrics(
    *,
    original_dir: Path,
    recon_dirs: Dict[str, Path],
    output_dir: Path,
    device: torch.device,
    dims: int,
    batch_size: int,
    num_workers: int,
    sqrt_device: torch.device,
    mmd_device: torch.device,
    mmd_chunk_size: int,
    full_ref_stats: str,
    full_ref_features: str,
    mmd_max_ref: int,
) -> Dict[str, Any]:
    started = time.perf_counter()
    model = build_inception(dims=int(dims), device=device)
    original_features, original_meta = features_from_image_dir(
        original_dir,
        model=model,
        device=device,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
    )
    orig_mu, orig_sigma = covariance_from_features(original_features)
    metrics: Dict[str, Any] = {
        "original": {
            "images_dir": str(original_dir),
            "num_images": int(original_features.shape[0]),
            "image_meta": original_meta,
            "feature_mean": float(original_features.mean()),
            "feature_std": float(original_features.std()),
            "sigma_trace": float(np.trace(orig_sigma)),
        },
        "reconstructions": {},
    }
    full_ref: Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any]]] = None
    if full_ref_stats:
        full_ref = load_reference_stats(Path(full_ref_stats).resolve())
    full_ref_bank: Optional[Tuple[np.ndarray, Dict[str, Any]]] = None
    if full_ref_features:
        full_ref_bank = load_feature_bank(Path(full_ref_features).resolve(), max_ref=int(mmd_max_ref) if mmd_max_ref else None)

    for name, recon_dir in recon_dirs.items():
        feats, meta = features_from_image_dir(
            recon_dir,
            model=model,
            device=device,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
        )
        mu, sigma = covariance_from_features(feats)
        rfid = frechet_distance(mu, sigma, orig_mu, orig_sigma, features1=feats, sqrt_device=sqrt_device)
        mmd2 = polynomial_mmd2_unbiased(feats, original_features, device=mmd_device, chunk_size=int(mmd_chunk_size))
        rec: Dict[str, Any] = {
            "images_dir": str(recon_dir),
            "num_images": int(feats.shape[0]),
            "image_meta": meta,
            "rfid_vs_original_100": float(rfid),
            "mmd2_vs_original_100": float(mmd2),
            "kid_x1000_vs_original_100": float(mmd2 * 1000.0),
            "feature_mean": float(feats.mean()),
            "feature_std": float(feats.std()),
            "sigma_trace": float(np.trace(sigma)),
        }
        if full_ref is not None:
            ref_mu, ref_sigma, ref_meta = full_ref
            fid_full = frechet_distance(mu, sigma, ref_mu, ref_sigma, features1=feats, sqrt_device=sqrt_device)
            rec.update(
                {
                    "fid_vs_full_imagenet_ref": float(fid_full),
                    "full_ref_meta": ref_meta,
                }
            )
        if full_ref_bank is not None:
            ref_feats, bank_meta = full_ref_bank
            mmd_full = polynomial_mmd2_unbiased(feats, ref_feats, device=mmd_device, chunk_size=int(mmd_chunk_size))
            rec.update(
                {
                    "mmd2_vs_full_ref_bank": float(mmd_full),
                    "kid_x1000_vs_full_ref_bank": float(mmd_full * 1000.0),
                    "full_ref_bank_meta": bank_meta,
                    "full_ref_features_used": int(ref_feats.shape[0]),
                }
            )
        metrics["reconstructions"][name] = rec

    metrics.update(
        {
            "status": "ok",
            "created_at_utc": utc_now(),
            "metric_scope": "same_100_image_reconstruction_fid_original_distribution_vs_recon_distribution",
            "dims": int(dims),
            "device": str(device),
            "sqrt_device": str(sqrt_device),
            "mmd_device": str(mmd_device),
            "elapsed_sec": float(time.perf_counter() - started),
        }
    )
    write_json(output_dir / "rfid_metrics.json", metrics)
    return metrics


def render_summary_md(summary: Dict[str, Any]) -> str:
    cmp = summary.get("comparison", {}) or {}
    pae = cmp.get("pae", {}) or {}
    flux = cmp.get("flux2", {}) or {}
    lines = [
        "# 100-image VAE rFID ceiling: PAE vs FLUX.2",
        "",
        f"- created_at_utc: `{summary.get('created_at_utc')}`",
        f"- status: `{summary.get('status')}`",
        f"- sample_count: `{summary.get('sample_count')}`",
        f"- source: `{summary.get('sample_meta', {}).get('crop_cache_dir')}`",
        f"- output_dir: `{summary.get('output_dir')}`",
        "",
        "## Primary same-image rFID",
        "",
        "| VAE | rFID vs same 100 originals ↓ | MMD2/KID ↓ | KID x1000 ↓ | PSNR ↑ | MAE ↓ |",
        "|---|---:|---:|---:|---:|---:|",
        f"| PAE DINOv2L d32 | `{pae.get('rfid_vs_original_100')}` | `{pae.get('mmd2_vs_original_100')}` | `{pae.get('kid_x1000_vs_original_100')}` | `{pae.get('pixel_metrics', {}).get('psnr_db_global')}` | `{pae.get('pixel_metrics', {}).get('mae_0_1')}` |",
        f"| FLUX.2 VAE | `{flux.get('rfid_vs_original_100')}` | `{flux.get('mmd2_vs_original_100')}` | `{flux.get('kid_x1000_vs_original_100')}` | `{flux.get('pixel_metrics', {}).get('psnr_db_global')}` | `{flux.get('pixel_metrics', {}).get('mae_0_1')}` |",
        "",
        "## Decision diagnostic",
        "",
        f"- flux_to_pae_rfid_ratio: `{cmp.get('flux_to_pae_rfid_ratio')}`",
        f"- flux_minus_pae_rfid: `{cmp.get('flux_minus_pae_rfid')}`",
        f"- automatic_label: `{cmp.get('automatic_label')}`",
        "",
        "Interpretation guardrail: this is a 100-image engineering ceiling check, not a publishable rFID. "
        "Use it to decide whether FLUX.2 reconstruction is obviously much worse than PAE on the same ImageNet-256 crops.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original_dir = output_dir / "original_png"
    pae_dir = output_dir / "pae_recon_png"
    flux_dir = output_dir / "flux2_recon_png"
    for d in (original_dir, pae_dir, flux_dir):
        d.mkdir(parents=True, exist_ok=True)
        # This script is intentionally a small deterministic smoke.  Clear old PNGs
        # in the target dirs to prevent stale images from contaminating rFID.
        for p in d.glob("*.png"):
            p.unlink()

    device = resolve_device(args.device)
    sqrt_device = device if str(args.sqrt_device).lower() == "auto" else resolve_device(args.sqrt_device)
    mmd_device = device if str(args.mmd_device).lower() == "auto" else resolve_device(args.mmd_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    run_config = {
        "created_at_utc": utc_now(),
        "argv": sys.argv,
        "args": vars(args),
        "output_dir": str(output_dir),
        "device": str(device),
        "sqrt_device": str(sqrt_device),
        "mmd_device": str(mmd_device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "pid": os.getpid(),
    }
    write_json(output_dir / "run_config.json", run_config)
    print("ARGS", json.dumps(run_config, indent=2, sort_keys=True), flush=True)

    images_u8, labels, source_indices, sample_meta = load_cropped_sample_tensor(
        cache_dir=Path(args.crop_cache_dir).resolve(),
        file_glob=args.crop_file_glob,
        image_key=args.image_key,
        label_key=args.label_key,
        source_index_key=args.source_index_key,
        start_index=int(args.start_index),
        max_images=int(args.max_images),
    )
    save_png_batch(images_u8, original_dir, offset=0)
    sample_meta["labels_head"] = labels[: min(20, int(labels.shape[0]))].tolist() if labels is not None else None
    sample_meta["source_indices_head"] = source_indices[: min(20, int(source_indices.shape[0]))].tolist() if source_indices is not None else None
    write_json(output_dir / "sample_meta.json", sample_meta)

    pae_dtype = dtype_from_name(args.pae_model_dtype)
    flux_dtype = dtype_from_name(args.flux_model_dtype)

    print("LOADING_PAE", flush=True)
    pae, pae_meta = load_pae(args, device, pae_dtype)
    print("RUNNING_PAE_RECON", flush=True)
    pae_runtime, pae_extra = reconstruct_pae(
        pae,
        images_u8,
        device=device,
        model_dtype=pae_dtype,
        batch_size=int(args.pae_batch_size),
        output_dir=pae_dir,
    )
    del pae
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("LOADING_FLUX2_VAE", flush=True)
    flux_vae, flux_meta = load_flux_vae(args, device, flux_dtype)
    print("RUNNING_FLUX2_RECON", flush=True)
    flux_runtime, flux_extra = reconstruct_flux(
        flux_vae,
        images_u8,
        device=device,
        model_dtype=flux_dtype,
        batch_size=int(args.flux_batch_size),
        output_dir=flux_dir,
        latent_mode=args.flux_latent_mode,
    )
    del flux_vae
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("COMPUTING_RFID", flush=True)
    metrics = compute_pairwise_image_metrics(
        original_dir=original_dir,
        recon_dirs={"pae": pae_dir, "flux2": flux_dir},
        output_dir=output_dir,
        device=device,
        dims=int(args.dims),
        batch_size=int(args.fid_batch_size),
        num_workers=int(args.fid_num_workers),
        sqrt_device=sqrt_device,
        mmd_device=mmd_device,
        mmd_chunk_size=int(args.mmd_chunk_size),
        full_ref_stats=args.full_ref_stats,
        full_ref_features=args.full_ref_features,
        mmd_max_ref=int(args.mmd_max_ref),
    )

    pae_rfid = float(metrics["reconstructions"]["pae"]["rfid_vs_original_100"])
    flux_rfid = float(metrics["reconstructions"]["flux2"]["rfid_vs_original_100"])
    ratio = float(flux_rfid / pae_rfid) if pae_rfid > 0 else float("inf")
    delta = float(flux_rfid - pae_rfid)
    if ratio >= float(args.much_worse_ratio) and delta >= float(args.much_worse_delta):
        label = "flux_rfid_much_worse_than_pae"
    else:
        label = "flux_rfid_not_much_worse_than_pae"

    comparison = {
        "pae": {
            **metrics["reconstructions"]["pae"],
            "pixel_metrics": pae_runtime["pixel_metrics"],
            "runtime": {k: v for k, v in pae_runtime.items() if k != "pixel_metrics"},
            **pae_extra,
        },
        "flux2": {
            **metrics["reconstructions"]["flux2"],
            "pixel_metrics": flux_runtime["pixel_metrics"],
            "runtime": {k: v for k, v in flux_runtime.items() if k != "pixel_metrics"},
            **flux_extra,
        },
        "flux_to_pae_rfid_ratio": ratio,
        "flux_minus_pae_rfid": delta,
        "much_worse_ratio_threshold": float(args.much_worse_ratio),
        "much_worse_delta_threshold": float(args.much_worse_delta),
        "automatic_label": label,
    }
    summary: Dict[str, Any] = {
        "status": "ok",
        "created_at_utc": utc_now(),
        "metric_scope": "100_image_same_input_reconstruction_fid_ceiling_pae_vs_flux2",
        "sample_count": int(images_u8.shape[0]),
        "sample_meta": sample_meta,
        "output_dir": str(output_dir),
        "original_dir": str(original_dir),
        "pae_recon_dir": str(pae_dir),
        "flux2_recon_dir": str(flux_dir),
        "pae_meta": pae_meta,
        "flux2_meta": flux_meta,
        "rfid_metrics": metrics,
        "comparison": comparison,
        "elapsed_sec": float(time.perf_counter() - started),
        "cuda_peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None,
        "artifact_policy": "Do not commit original/recon PNGs, logs, npz features, or runtime results by default.",
    }
    write_json(output_dir / "rfid_ceiling_summary.json", summary)
    (output_dir / "rfid_ceiling_summary.md").write_text(render_summary_md(summary), encoding="utf-8")
    print(
        "RFID_CEILING_SUMMARY",
        json.dumps(
            {
                "status": "ok",
                "sample_count": int(images_u8.shape[0]),
                "pae_rfid": pae_rfid,
                "flux2_rfid": flux_rfid,
                "flux_to_pae_rfid_ratio": ratio,
                "flux_minus_pae_rfid": delta,
                "automatic_label": label,
                "output_dir": str(output_dir),
                "elapsed_sec": summary["elapsed_sec"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crop-cache-dir", default="/workspace/PDM/data/cropped_uint8/imagenet1k_train_256_adm_safetensors")
    p.add_argument("--crop-file-glob", default="images_uint8_shard*.safetensors")
    p.add_argument("--image-key", default="images")
    p.add_argument("--label-key", default="labels")
    p.add_argument("--source-index-key", default="source_indices")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-images", type=int, default=100)
    p.add_argument("--output-dir", default="/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/vae_rfid_ceiling_100_pae_vs_flux2")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-tf32", action="store_true")

    p.add_argument("--pae-config", default="/workspace/PDM/external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml")
    p.add_argument("--pae-ckpt", default="/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt")
    p.add_argument("--pae-loader-script", default="/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/scripts/build_pae_latents_full_cache.py")
    p.add_argument("--pae-model-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--pae-batch-size", type=int, default=8)

    p.add_argument("--flux-repo-id", default="diffusers/FLUX.2-dev-bnb-4bit")
    p.add_argument("--flux-subfolder", default="vae")
    p.add_argument("--flux-revision", default="")
    p.add_argument("--flux-vae-class", default="AutoencoderKLFlux2")
    p.add_argument("--flux-model-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--flux-batch-size", type=int, default=8)
    p.add_argument("--flux-latent-mode", default="mode", choices=["mode", "sample"])
    p.add_argument("--local-files-only", action="store_true")

    p.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--fid-batch-size", type=int, default=64)
    p.add_argument("--fid-num-workers", type=int, default=2)
    p.add_argument("--sqrt-device", default="auto")
    p.add_argument("--mmd-device", default="auto")
    p.add_argument("--mmd-chunk-size", type=int, default=2048)
    p.add_argument("--full-ref-stats", default="/workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_stats.npz")
    p.add_argument("--full-ref-features", default="/workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/imagenet256_train_adm_inception2048_mmd8192_features.npz")
    p.add_argument("--mmd-max-ref", type=int, default=8192)

    p.add_argument("--much-worse-ratio", type=float, default=2.0)
    p.add_argument("--much-worse-delta", type=float, default=1.0)
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
        return 0 if summary.get("status") == "ok" else 1
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "error", "created_at_utc": utc_now(), "error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
