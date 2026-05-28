#!/usr/bin/env python3
"""FLUX.2 VAE reconstruction-ceiling diagnostic on real ImageNet-256 ADM crops.

This diagnostic answers a narrow question before continuing FLUX-latent B3 runs:

    real cropped uint8 image -> FLUX.2 VAE encode -> posterior.mode latent -> decode -> image metrics

It intentionally uses the same raw-latent convention as the FLUX.2 cache builder:

* input: cropped uint8 CHW ImageNet-256 safetensor shards, key ``images``;
* VAE input: float RGB mapped from [0,255] to [-1,1];
* latent: raw ``vae.encode(x).latent_dist.mode()`` without external scale/shift;
* reconstruction: raw ``vae.decode(z).sample`` converted back to uint8.

Outputs are experiment-local and should not be committed/uploaded by default:

* recon_png/*.png
* optional preview/input-pair PNGs
* latent_stats.json
* fid_mmd_metrics.json if reference stats are supplied
* reconstruction_summary.json / reconstruction_summary.md

The FID/MMD computed here is a sample-limited VAE reconstruction ceiling against the
real ImageNet-256 reference stats already built for this project, not a publishable
50k/1.28M benchmark unless run at that scale.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("DISABLE_XFORMERS", "1")

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def dtype_from_name(name: str) -> torch.dtype:
    key = str(name).lower().strip()
    if key in {"fp32", "float32", "torch.float32"}:
        return torch.float32
    if key in {"bf16", "bfloat16", "torch.bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half", "torch.float16"}:
        return torch.float16
    raise ValueError(f"Unsupported dtype name: {name!r}")


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but torch.cuda.is_available() is false: {device_arg}")
    return dev


def import_diffusers_class(class_name: str):
    diffusers = importlib.import_module("diffusers")
    if not hasattr(diffusers, class_name):
        raise AttributeError(
            f"diffusers has no class {class_name!r}; installed diffusers={getattr(diffusers, '__version__', '?')}"
        )
    return getattr(diffusers, class_name), getattr(diffusers, "__version__", "?")


def resolve_input_layout(args: argparse.Namespace) -> str:
    if args.input_layout != "auto":
        return args.input_layout
    # Keep Qwen-compatible knob here because this script may be reused for the next modern-VAE route.
    if "Qwen" in args.vae_class:
        return "video5d"
    return "image4d"


def prepare_vae_input(x4: torch.Tensor, input_layout: str) -> torch.Tensor:
    if input_layout == "image4d":
        return x4
    if input_layout == "video5d":
        return x4.unsqueeze(2)
    raise ValueError(f"Unsupported input_layout={input_layout!r}")


def postprocess_tensor4(x: torch.Tensor, *, name: str, squeeze_single_frame: bool) -> torch.Tensor:
    if x.ndim == 5 and squeeze_single_frame:
        if x.shape[2] != 1:
            raise RuntimeError(f"Cannot squeeze {name} frame dimension because shape is {tuple(x.shape)}")
        x = x.squeeze(2)
    if x.ndim != 4:
        raise RuntimeError(f"Expected {name} to be 4D [B,C,H,W], got shape {tuple(x.shape)}")
    return x


def posterior_to_latent(latent_dist: Any, latent_mode: str, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if latent_mode == "mode":
        return latent_dist.mode()
    if latent_mode == "sample":
        try:
            return latent_dist.sample(generator=generator)
        except TypeError:
            return latent_dist.sample()
    raise ValueError(f"Unsupported latent_mode={latent_mode!r}")


def load_vae(args: argparse.Namespace, device: torch.device, model_dtype: torch.dtype) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    cls, diffusers_version = import_diffusers_class(args.vae_class)
    kwargs: Dict[str, Any] = {"torch_dtype": model_dtype}
    if args.subfolder:
        kwargs["subfolder"] = args.subfolder
    if args.revision:
        kwargs["revision"] = args.revision
    if args.local_files_only:
        kwargs["local_files_only"] = True

    t0 = time.perf_counter()
    vae = cls.from_pretrained(args.repo_id, **kwargs)
    vae = vae.to(device=device, dtype=model_dtype).eval()
    vae.requires_grad_(False)
    try:
        config = dict(getattr(vae, "config", {}) or {})
    except Exception:
        config = {}

    meta = {
        "repo_id": args.repo_id,
        "subfolder": args.subfolder or "",
        "revision": args.revision or "",
        "vae_class": args.vae_class,
        "diffusers_version": diffusers_version,
        "model_dtype": str(model_dtype),
        "device": str(device),
        "load_elapsed_sec": time.perf_counter() - t0,
        "vae_config_scaling_factor": config.get("scaling_factor"),
        "vae_config_shift_factor": config.get("shift_factor"),
        "vae_config_latent_channels": config.get("latent_channels"),
        "vae_config_sample_size": config.get("sample_size"),
        "vae_config_json_head": json.dumps(config, sort_keys=True, default=str)[:16000],
    }
    return vae, meta


def images_to_uint8(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "minus1_1":
        y = x.clamp(-1.0, 1.0).add(1.0).mul(127.5)
    elif mode == "zero1":
        y = x.clamp(0.0, 1.0).mul(255.0)
    elif mode == "auto":
        lo = float(x.min().item())
        hi = float(x.max().item())
        if lo >= -0.05 and hi <= 1.05:
            y = x.clamp(0.0, 1.0).mul(255.0)
        else:
            y = x.clamp(-1.0, 1.0).add(1.0).mul(127.5)
    else:
        raise ValueError(f"Unsupported output_range={mode!r}")
    return y.round().clamp(0, 255).to(torch.uint8)


@dataclass
class CropShardInfo:
    path: Path
    num_samples: int
    image_shape: List[int]
    image_dtype: str
    has_labels: bool
    has_source_indices: bool


def list_crop_shards(cache_dir: Path, file_glob: str, image_key: str, label_key: str, source_index_key: str) -> List[CropShardInfo]:
    files = sorted(cache_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No crop shards matching {file_glob!r} under {cache_dir}")
    out: List[CropShardInfo] = []
    for p in files:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if image_key not in keys:
                raise KeyError(f"{p} missing image key={image_key!r}; keys={sorted(keys)}")
            sl = f.get_slice(image_key)
            shape = list(sl.get_shape())
            dtype = str(sl.get_dtype())
            if len(shape) != 4:
                raise RuntimeError(f"Expected {image_key} [N,3,H,W] in {p}, got {shape}")
            if shape[1] != 3:
                raise RuntimeError(f"Expected RGB CHW image key {image_key} with C=3 in {p}, got {shape}")
            out.append(
                CropShardInfo(
                    path=p,
                    num_samples=int(shape[0]),
                    image_shape=shape,
                    image_dtype=dtype,
                    has_labels=label_key in keys,
                    has_source_indices=source_index_key in keys,
                )
            )
    return out


def iter_crop_batches(
    shards: Sequence[CropShardInfo],
    *,
    image_key: str,
    label_key: str,
    source_index_key: str,
    start_index: int,
    max_images: Optional[int],
    batch_size: int,
) -> Iterator[Tuple[int, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]]:
    target_end = None if max_images is None else int(start_index) + int(max_images)
    global_start = 0
    for shard_idx, shard in enumerate(shards):
        shard_global_start = global_start
        shard_global_end = shard_global_start + shard.num_samples
        global_start = shard_global_end
        if shard_global_end <= start_index:
            continue
        if target_end is not None and shard_global_start >= target_end:
            break
        local_start = max(0, int(start_index) - shard_global_start)
        local_end = shard.num_samples if target_end is None else min(shard.num_samples, target_end - shard_global_start)
        if local_start >= local_end:
            continue
        with safe_open(str(shard.path), framework="pt", device="cpu") as f:
            images = f.get_slice(image_key)
            labels_slice = f.get_slice(label_key) if shard.has_labels else None
            src_slice = f.get_slice(source_index_key) if shard.has_source_indices else None
            for s in range(local_start, local_end, int(batch_size)):
                e = min(local_end, s + int(batch_size))
                x_u8 = images[s:e].contiguous()
                labels = labels_slice[s:e].long().contiguous() if labels_slice is not None else None
                src = src_slice[s:e].long().contiguous() if src_slice is not None else None
                batch_global_start = shard_global_start + s
                meta = {
                    "shard_index_in_scan": int(shard_idx),
                    "crop_path": str(shard.path),
                    "local_start": int(s),
                    "local_end_exclusive": int(e),
                    "global_start": int(batch_global_start),
                    "global_end_exclusive": int(shard_global_start + e),
                    "batch_samples": int(e - s),
                }
                if src is not None and src.numel() > 0:
                    meta.update({"source_index_first": int(src[0].item()), "source_index_last": int(src[-1].item())})
                yield batch_global_start, x_u8, labels, src, meta


class TensorRunningStats:
    def __init__(self, name: str, per_channel: bool = False) -> None:
        self.name = name
        self.per_channel = bool(per_channel)
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0
        self.min = math.inf
        self.max = -math.inf
        self.absmax = 0.0
        self.shape_first: Optional[List[int]] = None
        self.shape_last: Optional[List[int]] = None
        self.channel_count: Optional[int] = None
        self.channel_sum: Optional[np.ndarray] = None
        self.channel_sumsq: Optional[np.ndarray] = None
        self.channel_count_scalar = 0

    def update(self, x: torch.Tensor) -> None:
        xf = x.detach().float().cpu()
        if self.shape_first is None:
            self.shape_first = list(xf.shape)
        self.shape_last = list(xf.shape)
        self.count += int(xf.numel())
        self.sum += float(xf.sum(dtype=torch.float64).item())
        self.sumsq += float((xf.double() * xf.double()).sum().item())
        self.min = min(self.min, float(xf.min().item()))
        self.max = max(self.max, float(xf.max().item()))
        self.absmax = max(self.absmax, float(xf.abs().max().item()))
        if self.per_channel:
            if xf.ndim < 2:
                raise RuntimeError("per_channel stats require tensor with channel dimension")
            c = int(xf.shape[1])
            if self.channel_count is None:
                self.channel_count = c
                self.channel_sum = np.zeros((c,), dtype=np.float64)
                self.channel_sumsq = np.zeros((c,), dtype=np.float64)
            if c != self.channel_count:
                raise RuntimeError(f"Channel count changed for {self.name}: {c} vs {self.channel_count}")
            reduce_dims = tuple(i for i in range(xf.ndim) if i != 1)
            ch_sum = xf.double().sum(dim=reduce_dims).numpy()
            ch_sumsq = (xf.double() * xf.double()).sum(dim=reduce_dims).numpy()
            self.channel_sum += ch_sum  # type: ignore[operator]
            self.channel_sumsq += ch_sumsq  # type: ignore[operator]
            self.channel_count_scalar += int(xf.numel() // c)

    def finalize(self) -> Dict[str, Any]:
        if self.count <= 0:
            return {"name": self.name, "count": 0}
        mean = self.sum / float(self.count)
        var = max(0.0, self.sumsq / float(self.count) - mean * mean)
        out: Dict[str, Any] = {
            "name": self.name,
            "count": int(self.count),
            "mean": float(mean),
            "std": float(math.sqrt(var)),
            "min": float(self.min),
            "max": float(self.max),
            "absmax": float(self.absmax),
            "rms": float(math.sqrt(max(0.0, self.sumsq / float(self.count)))),
            "shape_first": self.shape_first,
            "shape_last": self.shape_last,
        }
        if self.per_channel and self.channel_sum is not None and self.channel_sumsq is not None:
            ch_mean = self.channel_sum / float(self.channel_count_scalar)
            ch_var = np.maximum(0.0, self.channel_sumsq / float(self.channel_count_scalar) - ch_mean * ch_mean)
            ch_std = np.sqrt(ch_var)
            # Full arrays are useful for scale debugging and still tiny (C=32 or 3).
            out.update(
                {
                    "per_channel_count": int(self.channel_count_scalar),
                    "per_channel_mean": [float(v) for v in ch_mean.tolist()],
                    "per_channel_std": [float(v) for v in ch_std.tolist()],
                    "per_channel_mean_summary": {
                        "min": float(ch_mean.min()),
                        "max": float(ch_mean.max()),
                        "mean": float(ch_mean.mean()),
                        "std": float(ch_mean.std()),
                    },
                    "per_channel_std_summary": {
                        "min": float(ch_std.min()),
                        "max": float(ch_std.max()),
                        "mean": float(ch_std.mean()),
                        "std": float(ch_std.std()),
                    },
                }
            )
        return out


class PixelDiffStats:
    def __init__(self) -> None:
        self.count = 0
        self.sumsq = 0.0
        self.sumabs = 0.0
        self.max_abs_u8 = 0
        self.image_mse: List[float] = []
        self.image_mae: List[float] = []

    def update(self, original_u8: torch.Tensor, recon_u8: torch.Tensor) -> None:
        if original_u8.shape != recon_u8.shape:
            raise RuntimeError(f"Pixel metric shape mismatch: original={tuple(original_u8.shape)} recon={tuple(recon_u8.shape)}")
        diff_u8 = recon_u8.to(torch.int16) - original_u8.to(torch.int16)
        self.max_abs_u8 = max(self.max_abs_u8, int(diff_u8.abs().max().item()))
        diff = diff_u8.float().div(255.0)
        self.count += int(diff.numel())
        self.sumsq += float((diff.double() * diff.double()).sum().item())
        self.sumabs += float(diff.abs().double().sum().item())
        per_image = diff.flatten(1)
        self.image_mse.extend([float(v) for v in (per_image * per_image).mean(dim=1).cpu().tolist()])
        self.image_mae.extend([float(v) for v in per_image.abs().mean(dim=1).cpu().tolist()])

    def finalize(self) -> Dict[str, Any]:
        if self.count <= 0:
            return {"count": 0}
        mse = self.sumsq / float(self.count)
        mae = self.sumabs / float(self.count)
        psnr = float("inf") if mse <= 0 else 10.0 * math.log10(1.0 / mse)
        im_mse = np.asarray(self.image_mse, dtype=np.float64)
        im_mae = np.asarray(self.image_mae, dtype=np.float64)
        im_psnr = np.where(im_mse <= 0, np.inf, 10.0 * np.log10(1.0 / im_mse))
        return {
            "count_pixels_times_channels": int(self.count),
            "mse_0_1": float(mse),
            "rmse_0_1": float(math.sqrt(mse)),
            "mae_0_1": float(mae),
            "psnr_db_global": float(psnr),
            "max_abs_uint8": int(self.max_abs_u8),
            "image_count": int(im_mse.shape[0]),
            "image_mse_0_1_mean": float(im_mse.mean()) if im_mse.size else None,
            "image_mse_0_1_median": float(np.median(im_mse)) if im_mse.size else None,
            "image_mse_0_1_min": float(im_mse.min()) if im_mse.size else None,
            "image_mse_0_1_max": float(im_mse.max()) if im_mse.size else None,
            "image_mae_0_1_mean": float(im_mae.mean()) if im_mae.size else None,
            "image_mae_0_1_median": float(np.median(im_mae)) if im_mae.size else None,
            "image_psnr_db_mean": float(np.mean(im_psnr[np.isfinite(im_psnr)])) if np.isfinite(im_psnr).any() else float("inf"),
            "image_psnr_db_median": float(np.median(im_psnr[np.isfinite(im_psnr)])) if np.isfinite(im_psnr).any() else float("inf"),
            "image_psnr_db_min": float(np.min(im_psnr[np.isfinite(im_psnr)])) if np.isfinite(im_psnr).any() else float("inf"),
            "image_psnr_db_max": float(np.max(im_psnr[np.isfinite(im_psnr)])) if np.isfinite(im_psnr).any() else float("inf"),
        }


def save_png_batch(images_u8: torch.Tensor, output_dir: Path, *, offset: int) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames: List[str] = []
    if images_u8.ndim != 4 or images_u8.shape[1] != 3:
        raise RuntimeError(f"Expected images_u8 [B,3,H,W], got {tuple(images_u8.shape)}")
    for i in range(int(images_u8.shape[0])):
        filename = f"{offset + i:06d}.png"
        arr = images_u8[i].permute(1, 2, 0).contiguous().numpy()
        Image.fromarray(arr, mode="RGB").save(output_dir / filename)
        filenames.append(filename)
    return filenames


def write_preview_grid(images_u8: torch.Tensor, output_path: Path, *, columns: int = 4) -> None:
    if images_u8.numel() == 0:
        return
    b, c, h, w = images_u8.shape
    if c != 3:
        raise RuntimeError(f"Expected RGB preview tensor, got {tuple(images_u8.shape)}")
    cols = max(1, min(int(columns), b))
    rows = (b + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))
    for i in range(b):
        arr = images_u8[i].permute(1, 2, 0).contiguous().numpy()
        canvas.paste(Image.fromarray(arr, mode="RGB"), ((i % cols) * w, (i // cols) * h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def write_pair_contact_sheet(pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]], output_path: Path, *, columns: int = 4) -> None:
    """Write a compact original/reconstruction contact sheet for human sanity check."""
    if not pairs:
        return
    h = int(pairs[0][0].shape[-2])
    w = int(pairs[0][0].shape[-1])
    tile_h = h * 2 + 18
    tile_w = w
    cols = max(1, min(int(columns), len(pairs)))
    rows = (len(pairs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), color=(24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for i, (orig, recon) in enumerate(pairs):
        x = (i % cols) * tile_w
        y = (i // cols) * tile_h
        orig_img = Image.fromarray(orig.permute(1, 2, 0).contiguous().numpy(), mode="RGB")
        rec_img = Image.fromarray(recon.permute(1, 2, 0).contiguous().numpy(), mode="RGB")
        canvas.paste(orig_img, (x, y))
        canvas.paste(rec_img, (x, y + h + 18))
        draw.text((x + 3, y + h + 2), "top: original / bottom: recon", fill=(230, 230, 230))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def compute_fid_mmd(args: argparse.Namespace, recon_dir: Path, device: torch.device, output_json: Path) -> Dict[str, Any]:
    if not args.ref_stats:
        return {"status": "skipped_no_ref_stats"}
    # Import experiment-local Inception helpers lazily so the encode/decode diagnostic remains usable without FID deps.
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from inception_metrics_lib import (  # noqa: WPS433
        build_inception,
        covariance_from_features,
        features_from_image_dir,
        frechet_distance,
        load_feature_bank,
        load_reference_stats,
        polynomial_mmd2_unbiased,
    )

    ref_stats_path = Path(args.ref_stats).resolve()
    ref_features_path = Path(args.ref_features).resolve() if args.ref_features else None
    t0 = time.perf_counter()
    model = build_inception(dims=int(args.dims), device=device)
    gen_features, image_meta = features_from_image_dir(
        recon_dir,
        model=model,
        device=device,
        batch_size=int(args.fid_batch_size),
        num_workers=int(args.fid_num_workers),
        max_images=None,
        recursive=False,
    )
    gen_mu, gen_sigma = covariance_from_features(gen_features)
    ref_mu, ref_sigma, ref_meta = load_reference_stats(ref_stats_path)
    sqrt_device = device if str(args.sqrt_device).lower() == "auto" else resolve_device(args.sqrt_device)
    fid = frechet_distance(gen_mu, gen_sigma, ref_mu, ref_sigma, features1=gen_features, sqrt_device=sqrt_device)
    record: Dict[str, Any] = {
        "status": "ok",
        "created_at_utc": utc_now(),
        "metric_scope": "flux2_vae_reconstruction_ceiling_sample_limited",
        "images_dir": str(recon_dir),
        "ref_stats": str(ref_stats_path),
        "num_images": int(gen_features.shape[0]),
        "dims": int(args.dims),
        "fid": float(fid),
        "FID": float(fid),
        "gen_feature_mean": float(gen_features.mean()),
        "gen_feature_std": float(gen_features.std()),
        "gen_sigma_trace": float(np.trace(gen_sigma)),
        "ref_count": int(ref_meta.get("count", -1)) if "count" in ref_meta else None,
        "ref_meta": ref_meta,
        "image_meta": image_meta,
        "device": str(device),
        "sqrt_device": str(sqrt_device),
        "batch_size": int(args.fid_batch_size),
        "allow_tf32": bool(args.allow_tf32),
    }
    if ref_features_path is not None:
        mmd_device = device if str(args.mmd_device).lower() == "auto" else resolve_device(args.mmd_device)
        ref_feats, bank_meta = load_feature_bank(ref_features_path, max_ref=int(args.mmd_max_ref) if args.mmd_max_ref else None)
        mmd2 = polynomial_mmd2_unbiased(
            gen_features,
            ref_feats,
            device=mmd_device,
            chunk_size=int(args.mmd_chunk_size),
        )
        record.update(
            {
                "ref_features": str(ref_features_path),
                "ref_features_used": int(ref_feats.shape[0]),
                "ref_features_meta": bank_meta,
                "mmd_kernel": "poly3_kid",
                "mmd2": float(mmd2),
                "mmd2_poly3": float(mmd2),
                "kid": float(mmd2),
                "kid_x1000": float(mmd2 * 1000.0),
                "mmd_device": str(mmd_device),
            }
        )
    else:
        record.update({"mmd2": None, "kid": None, "mmd_status": "skipped_no_ref_features"})
    record["elapsed_sec"] = time.perf_counter() - t0
    if device.type == "cuda":
        record["cuda_peak_memory_mb_at_fid_end"] = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    write_json(output_json, record)
    return record


def render_summary_md(summary: Dict[str, Any]) -> str:
    pix = summary.get("pixel_metrics", {}) or {}
    latent = summary.get("latent_stats", {}) or {}
    fid = summary.get("fid_mmd_metrics", {}) or {}
    lines = [
        "# FLUX.2 VAE Reconstruction Ceiling Diagnostic",
        "",
        f"- created_at_utc: `{summary.get('created_at_utc')}`",
        f"- status: `{summary.get('status')}`",
        f"- sample_count: `{summary.get('num_images')}`",
        f"- crop_cache_dir: `{summary.get('crop_cache_dir')}`",
        f"- output_dir: `{summary.get('output_dir')}`",
        f"- VAE: `{summary.get('vae_meta', {}).get('vae_class')}` from `{summary.get('vae_meta', {}).get('repo_id')}` / `{summary.get('vae_meta', {}).get('subfolder')}`",
        f"- model_dtype: `{summary.get('model_dtype')}`; device: `{summary.get('device')}`; latent_mode: `{summary.get('latent_mode')}`",
        "",
        "## Reconstruction pixel metrics",
        "",
        f"- mse_0_1: `{pix.get('mse_0_1')}`",
        f"- mae_0_1: `{pix.get('mae_0_1')}`",
        f"- psnr_db_global: `{pix.get('psnr_db_global')}`",
        f"- max_abs_uint8: `{pix.get('max_abs_uint8')}`",
        "",
        "## Latent stats",
        "",
        f"- first_shape: `{latent.get('shape_first')}`",
        f"- mean/std: `{latent.get('mean')}` / `{latent.get('std')}`",
        f"- min/max/absmax: `{latent.get('min')}` / `{latent.get('max')}` / `{latent.get('absmax')}`",
        f"- rms: `{latent.get('rms')}`",
        "",
        "## Image-space FID/MMD/KID against real ImageNet-256 reference",
        "",
        f"- metric_status: `{fid.get('status')}`",
        f"- FID: `{fid.get('fid')}`",
        f"- MMD2/KID: `{fid.get('mmd2')}`",
        f"- KID x1000: `{fid.get('kid_x1000')}`",
        f"- metric_num_images: `{fid.get('num_images')}`",
        "",
        "## Interpretation guardrail",
        "",
        "This is a sample-limited VAE reconstruction ceiling diagnostic. If this FID is already high, "
        "the FLUX.2 VAE/ImageNet-256 ADM-crop domain mismatch is likely a major part of the FLUX B3 short-run FID ceiling. "
        "If this FID is low while B3 samples are bad, the issue is more likely latent normalization, B3/MeanFlow config, or sampler/training retuning. "
        "Do not interpret this as a final verdict on FLUX/Qwen VAE quality for later T2I/general-image baselines.",
        "",
    ]
    return "\n".join(lines)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    crop_dir = Path(args.crop_cache_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    recon_dir = output_dir / "recon_png"
    input_preview_dir = output_dir / "input_preview_png"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_recon_png:
        recon_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model_dtype = dtype_from_name(args.model_dtype)
    input_layout = resolve_input_layout(args)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    shards = list_crop_shards(crop_dir, args.crop_file_glob, args.image_key, args.label_key, args.source_index_key)
    crop_total = sum(s.num_samples for s in shards)
    if int(args.start_index) < 0 or int(args.start_index) >= crop_total:
        raise RuntimeError(f"start_index={args.start_index} outside crop_total={crop_total}")
    target_total = crop_total - int(args.start_index)
    if args.max_images is not None:
        target_total = min(target_total, int(args.max_images))
    if target_total <= 0:
        raise RuntimeError("target_total <= 0")

    run_config = {
        "created_at_utc": utc_now(),
        "argv": sys.argv,
        "args": vars(args),
        "crop_cache_dir": str(crop_dir),
        "crop_file_glob": args.crop_file_glob,
        "crop_total": int(crop_total),
        "crop_num_shards": len(shards),
        "target_total": int(target_total),
        "output_dir": str(output_dir),
        "recon_dir": str(recon_dir),
        "device": str(device),
        "model_dtype": str(model_dtype),
        "input_layout_resolved": input_layout,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "pid": os.getpid(),
    }
    write_json(output_dir / "run_config.json", run_config)
    print("ARGS", json.dumps(run_config, indent=2, sort_keys=True), flush=True)

    vae, vae_meta = load_vae(args, device, model_dtype)
    generator: Optional[torch.Generator] = None
    if args.latent_mode == "sample":
        generator = torch.Generator(device=device)
        generator.manual_seed(int(args.sample_seed))

    latent_stats = TensorRunningStats("flux2_raw_posterior_latent", per_channel=True)
    decoded_float_stats = TensorRunningStats("decoded_float_before_uint8", per_channel=True)
    recon_u8_stats = TensorRunningStats("reconstruction_uint8", per_channel=True)
    input_u8_stats = TensorRunningStats("input_uint8", per_channel=True)
    pix_stats = PixelDiffStats()

    image_records: List[Dict[str, Any]] = []
    first_record: Optional[Dict[str, Any]] = None
    last_record: Optional[Dict[str, Any]] = None
    preview_recons: List[torch.Tensor] = []
    preview_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    batch_summaries: List[Dict[str, Any]] = []

    n_done = 0
    encode_decode_elapsed = 0.0
    io_elapsed = 0.0
    save_elapsed = 0.0
    last_log_t = time.perf_counter()

    for batch_idx, (batch_global_start, x_u8, labels, src, meta) in enumerate(
        iter_crop_batches(
            shards,
            image_key=args.image_key,
            label_key=args.label_key,
            source_index_key=args.source_index_key,
            start_index=int(args.start_index),
            max_images=int(args.max_images) if args.max_images is not None else None,
            batch_size=int(args.batch_size),
        ),
        start=1,
    ):
        t_batch0 = time.perf_counter()
        input_u8_stats.update(x_u8)
        x = x_u8.to(device=device, non_blocking=True).to(dtype=torch.float32).div_(127.5).sub_(1.0)
        if model_dtype != torch.float32:
            x = x.to(dtype=model_dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_compute0 = time.perf_counter()
        with torch.inference_mode():
            xin = prepare_vae_input(x, input_layout)
            enc = vae.encode(xin)
            z = posterior_to_latent(enc.latent_dist, args.latent_mode, generator=generator)
            z = postprocess_tensor4(z, name="latent", squeeze_single_frame=bool(args.squeeze_single_frame))
            dec = vae.decode(z)
            rec = dec.sample if hasattr(dec, "sample") else dec[0]
            rec = postprocess_tensor4(rec, name="decoded image", squeeze_single_frame=bool(args.squeeze_single_frame))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_compute1 = time.perf_counter()
        encode_decode_elapsed += t_compute1 - t_compute0

        z_cpu = z.detach().float().cpu()
        rec_float_cpu = rec.detach().float().cpu()
        rec_u8 = images_to_uint8(rec_float_cpu, args.output_range).cpu()
        x_u8 = x_u8.cpu()

        latent_stats.update(z_cpu)
        decoded_float_stats.update(rec_float_cpu)
        recon_u8_stats.update(rec_u8)
        pix_stats.update(x_u8, rec_u8)

        saved_files: List[str] = []
        if args.save_recon_png:
            t_save0 = time.perf_counter()
            saved_files = save_png_batch(rec_u8, recon_dir, offset=batch_global_start)
            save_elapsed += time.perf_counter() - t_save0
            for i, fn in enumerate(saved_files):
                rec_record: Dict[str, Any] = {
                    "global_index": int(batch_global_start + i),
                    "filename": fn,
                }
                if labels is not None:
                    rec_record["label"] = int(labels[i].item())
                if src is not None:
                    rec_record["source_index"] = int(src[i].item())
                if first_record is None:
                    first_record = dict(rec_record)
                last_record = dict(rec_record)
                if args.manifest_mode == "full":
                    image_records.append(rec_record)

        if args.save_input_preview and n_done < int(args.preview_max_images):
            input_preview_dir.mkdir(parents=True, exist_ok=True)
            take = min(int(args.preview_max_images) - n_done, int(x_u8.shape[0]))
            if take > 0:
                save_png_batch(x_u8[:take], input_preview_dir, offset=batch_global_start)

        if len(preview_recons) < int(args.preview_max_images):
            take = min(int(args.preview_max_images) - len(preview_recons), int(rec_u8.shape[0]))
            for i in range(take):
                preview_recons.append(rec_u8[i].clone())
                preview_pairs.append((x_u8[i].clone(), rec_u8[i].clone()))

        batch_n = int(x_u8.shape[0])
        n_done += batch_n
        batch_elapsed = time.perf_counter() - t_batch0
        io_elapsed += max(0.0, batch_elapsed - (t_compute1 - t_compute0))
        batch_summary = {
            "batch_index": int(batch_idx),
            "global_start": int(batch_global_start),
            "samples": int(batch_n),
            "batch_elapsed_sec": float(batch_elapsed),
            "encode_decode_sec": float(t_compute1 - t_compute0),
            "samples_per_sec_encode_decode": float(batch_n / (t_compute1 - t_compute0)) if (t_compute1 - t_compute0) > 0 else None,
            "latent_shape": list(z_cpu.shape),
            "decoded_float_min": float(rec_float_cpu.min().item()),
            "decoded_float_max": float(rec_float_cpu.max().item()),
            "latent_mean": float(z_cpu.mean().item()),
            "latent_std": float(z_cpu.std(unbiased=False).item()),
            "batch_meta": meta,
        }
        if len(batch_summaries) < int(args.keep_batch_summaries):
            batch_summaries.append(batch_summary)

        now = time.perf_counter()
        if (now - last_log_t) >= float(args.log_interval_sec) or n_done >= target_total or batch_idx == 1:
            elapsed = now - started
            avg = n_done / elapsed if elapsed > 0 else None
            print(
                "RECON_PROGRESS",
                json.dumps(
                    {
                        "batch": batch_idx,
                        "done": n_done,
                        "target_total": target_total,
                        "avg_samples_per_sec": avg,
                        "last_batch_samples_per_sec_encode_decode": batch_summary["samples_per_sec_encode_decode"],
                        "latent_shape": batch_summary["latent_shape"],
                        "cuda_peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            last_log_t = now

        del x, z, rec, z_cpu, rec_float_cpu, rec_u8, x_u8
        if device.type == "cuda" and bool(args.empty_cache_each_batch):
            torch.cuda.empty_cache()

    if n_done != target_total:
        raise RuntimeError(f"Processed {n_done} images but expected target_total={target_total}")

    if args.save_grid and preview_recons:
        write_preview_grid(torch.stack(preview_recons, dim=0), output_dir / "recon_preview_grid.png", columns=int(args.preview_columns))
    if args.save_pair_contact_sheet and preview_pairs:
        write_pair_contact_sheet(preview_pairs, output_dir / "reconstruction_pairs_contact_sheet.png", columns=int(args.preview_columns))

    latent_record = latent_stats.finalize()
    decoded_float_record = decoded_float_stats.finalize()
    recon_u8_record = recon_u8_stats.finalize()
    input_u8_record = input_u8_stats.finalize()
    pixel_record = pix_stats.finalize()
    write_json(output_dir / "latent_stats.json", latent_record)

    # Free VAE before Inception evaluation so FID peak memory is not inflated by a loaded decoder.
    del vae
    if device.type == "cuda":
        torch.cuda.empty_cache()

    fid_record: Dict[str, Any] = {"status": "skipped"}
    if args.compute_fid:
        if not args.save_recon_png:
            raise RuntimeError("--compute-fid requires --save-recon-png")
        fid_record = compute_fid_mmd(args, recon_dir, device, output_dir / "fid_mmd_metrics.json")

    summary: Dict[str, Any] = {
        "status": "ok",
        "created_at_utc": utc_now(),
        "metric_scope": "flux2_vae_reconstruction_ceiling_sample_limited",
        "num_images": int(n_done),
        "crop_cache_dir": str(crop_dir),
        "crop_total": int(crop_total),
        "crop_num_shards": len(shards),
        "start_index": int(args.start_index),
        "max_images": int(args.max_images) if args.max_images is not None else None,
        "output_dir": str(output_dir),
        "recon_dir": str(recon_dir) if args.save_recon_png else None,
        "device": str(device),
        "model_dtype": str(model_dtype),
        "latent_mode": args.latent_mode,
        "input_preprocess": "uint8 [0,255] -> float [-1,1] after ADM center crop",
        "latent_convention": "raw vae.encode(x).latent_dist.mode/sample without external scale/shift",
        "decode_convention": "raw vae.decode(z).sample, then output_range conversion",
        "output_range": args.output_range,
        "allow_tf32": bool(args.allow_tf32),
        "batch_size": int(args.batch_size),
        "vae_meta": vae_meta,
        "pixel_metrics": pixel_record,
        "latent_stats": latent_record,
        "decoded_float_stats": decoded_float_record,
        "input_uint8_stats": input_u8_record,
        "reconstruction_uint8_stats": recon_u8_record,
        "fid_mmd_metrics": fid_record,
        "batch_summaries_head": batch_summaries,
        "manifest_mode": args.manifest_mode,
        "images_manifest_count": len(image_records),
        "images_manifest_omitted": args.manifest_mode != "full",
        "first_image_record": first_record,
        "last_image_record": last_record,
        "encode_decode_elapsed_sec": float(encode_decode_elapsed),
        "encode_decode_samples_per_sec": float(n_done / encode_decode_elapsed) if encode_decode_elapsed > 0 else None,
        "save_png_elapsed_sec": float(save_elapsed),
        "elapsed_sec": float(time.perf_counter() - started),
        "cuda_peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None,
        "artifact_policy": "Do not commit/upload recon_png, input previews, logs, npz stats, checkpoints, or safetensor caches by default.",
    }
    if args.manifest_mode == "full":
        summary["images_manifest"] = image_records
    write_json(output_dir / "reconstruction_summary.json", summary)
    (output_dir / "reconstruction_summary.md").write_text(render_summary_md(summary), encoding="utf-8")
    print("RECON_SUMMARY", json.dumps({k: summary.get(k) for k in ["status", "num_images", "output_dir", "encode_decode_samples_per_sec", "elapsed_sec", "cuda_peak_memory_mb"]}, sort_keys=True), flush=True)
    if fid_record:
        print("RECON_FID_MMD", json.dumps({k: fid_record.get(k) for k in ["status", "num_images", "fid", "mmd2", "kid_x1000", "elapsed_sec"]}, sort_keys=True), flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crop-cache-dir", required=True)
    p.add_argument("--crop-file-glob", default="images_uint8_shard*.safetensors")
    p.add_argument("--image-key", default="images")
    p.add_argument("--label-key", default="labels")
    p.add_argument("--source-index-key", default="source_indices")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-images", type=int, default=1024)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--repo-id", default="diffusers/FLUX.2-dev-bnb-4bit")
    p.add_argument("--subfolder", default="vae")
    p.add_argument("--revision", default="")
    p.add_argument("--vae-class", default="AutoencoderKLFlux2")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--input-layout", default="auto", choices=["auto", "image4d", "video5d"])
    p.add_argument("--squeeze-single-frame", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--latent-mode", default="mode", choices=["mode", "sample"])
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--output-range", default="minus1_1", choices=["minus1_1", "zero1", "auto"])
    p.add_argument("--allow-tf32", action="store_true", help="allow TF32 for VAE/FID CUDA ops; default keeps strict fp32")
    p.add_argument("--save-recon-png", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-input-preview", action="store_true")
    p.add_argument("--save-grid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-pair-contact-sheet", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--preview-max-images", type=int, default=16)
    p.add_argument("--preview-columns", type=int, default=4)
    p.add_argument("--manifest-mode", default="first_last", choices=["full", "first_last", "none"])
    p.add_argument("--keep-batch-summaries", type=int, default=8)
    p.add_argument("--empty-cache-each-batch", action="store_true")
    p.add_argument("--log-interval-sec", type=float, default=15.0)

    p.add_argument("--compute-fid", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ref-stats", default="")
    p.add_argument("--ref-features", default="")
    p.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--fid-batch-size", type=int, default=64)
    p.add_argument("--fid-num-workers", type=int, default=2)
    p.add_argument("--sqrt-device", default="auto")
    p.add_argument("--mmd-max-ref", type=int, default=8192)
    p.add_argument("--mmd-device", default="auto")
    p.add_argument("--mmd-chunk-size", type=int, default=2048)
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = run(args)
        return 0 if summary.get("status") == "ok" else 1
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "error", "error_type": type(exc).__name__, "error": str(exc), "created_at_utc": utc_now()}
        print(json.dumps(payload, sort_keys=True), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
