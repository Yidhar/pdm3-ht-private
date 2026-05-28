#!/usr/bin/env python3
"""Decode FLUX.2 AutoencoderKLFlux2 latent safetensors into image files.

This is intentionally experiment-local and mirrors the FLUX.2 cache builder:

- cached/trained latent shape: [B, 32, 32, 32]
- cached latent convention: raw ``vae.encode(x).latent_dist.mode()`` without an
  external Stable-Diffusion-style scaling/shift transform
- decode convention here: raw ``vae.decode(z).sample`` by default

The script supports the B3 trainer's eval output file
``sample_latents.safetensors`` with keys ``samples`` and ``labels``, and the
cache shard schema with keys ``latents`` / ``latents_flip`` / ``labels``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("DISABLE_XFORMERS", "1")

import torch
from PIL import Image
from safetensors import safe_open

FLUX_IMAGENET256_RAW_REFERENCE_STATS: Dict[str, float] = {
    # Measured on the completed FLUX.2 ImageNet-256 latent cache with
    # compute_latent_cache_stats.py, linspace 32 shards / 130,191 latents.
    # These are diagnostic references only; they are not used to modify decode.
    "reference_raw_flux_mean": -0.009054178792269978,
    "reference_raw_flux_std": 1.7140430386576422,
    "reference_raw_flux_rms": 1.7140669521708671,
}


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
    raise ValueError(f"Unsupported dtype name: {name!r}")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    return dev


def import_diffusers_class(class_name: str):
    diffusers = importlib.import_module("diffusers")
    if not hasattr(diffusers, class_name):
        raise AttributeError(
            f"diffusers has no class {class_name!r}; installed diffusers={getattr(diffusers, '__version__', '?')}"
        )
    return getattr(diffusers, class_name), getattr(diffusers, "__version__", "?")


def load_vae(args: argparse.Namespace, device: torch.device) -> Tuple[torch.nn.Module, str, Dict[str, Any]]:
    cls, diffusers_version = import_diffusers_class(args.vae_class)
    model_dtype = dtype_from_name(args.model_dtype)
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
    }
    return vae, diffusers_version, meta


def choose_tensor_key(path: Path, requested: str) -> str:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        keys = list(f.keys())
    if requested != "auto":
        if requested not in keys:
            raise KeyError(f"requested tensor key {requested!r} missing from {path}; keys={keys}")
        return requested
    for key in ("samples", "latents", "latents_flip"):
        if key in keys:
            return key
    raise KeyError(f"Could not auto-select latent tensor key from {path}; keys={keys}")


def read_latent_file(
    path: Path,
    *,
    tensor_key: str,
    max_images: Optional[int],
    start_index: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, Any]]:
    chosen = choose_tensor_key(path, tensor_key)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        z_slice = f.get_slice(chosen)
        z_shape = list(z_slice.get_shape())
        if len(z_shape) != 4:
            raise RuntimeError(f"Expected 4D latent tensor [N,C,H,W], got {z_shape} from key={chosen}")
        n_total = int(z_shape[0])
        if start_index < 0 or start_index >= n_total:
            raise IndexError(f"start_index={start_index} outside latent N={n_total}")
        n = n_total - int(start_index)
        if max_images is not None:
            n = min(n, int(max_images))
        z = f.get_slice(chosen)[start_index : start_index + n].contiguous()
        labels = None
        if "labels" in f.keys():
            labels = f.get_slice("labels")[start_index : start_index + n].long().contiguous()
    meta = {
        "latents_path": str(path),
        "tensor_key": chosen,
        "source_total_samples": n_total,
        "start_index": int(start_index),
        "decoded_requested_samples": int(n),
        "latent_shape_full": z_shape,
        "latent_shape_selected": list(z.shape),
        "latent_dtype": str(z.dtype),
        "latent_mean": float(z.float().mean().item()),
        "latent_std": float(z.float().std(unbiased=False).item()),
        "latent_rms": float(z.float().square().mean().sqrt().item()),
        "latent_min": float(z.float().min().item()),
        "latent_max": float(z.float().max().item()),
        "latent_absmax": float(z.float().abs().max().item()),
        "has_labels": labels is not None,
    }
    if labels is not None and labels.numel() > 0:
        meta.update(
            {
                "label_min": int(labels.min().item()),
                "label_max": int(labels.max().item()),
                "label_unique_count": int(torch.unique(labels).numel()),
            }
        )
    return z, labels, meta


def affine_latent_stats(latent_meta: Dict[str, Any], *, scale: float, shift: float) -> Dict[str, Any]:
    """Return sample-space and pre-decode affine-space latent diagnostics.

    The B3 trainer writes sampled tensors in its training/sampler space.  The
    FLUX decoder may then apply an affine scalar transform
    ``z_decode = z_sample * scale + shift``.  Keeping both spaces explicit makes
    scale-policy mistakes visible in eval records.
    """
    sample_mean = float(latent_meta["latent_mean"])
    sample_std = float(latent_meta["latent_std"])
    sample_rms = float(latent_meta["latent_rms"])
    sample_min = float(latent_meta["latent_min"])
    sample_max = float(latent_meta["latent_max"])
    sample_absmax = float(latent_meta["latent_absmax"])

    scale = float(scale)
    shift = float(shift)
    decode_mean = sample_mean * scale + shift
    decode_std = sample_std * abs(scale)
    # E[(aX+b)^2] = a^2 E[X^2] + 2ab E[X] + b^2
    decode_rms2 = (scale * scale) * (sample_rms * sample_rms) + 2.0 * scale * shift * sample_mean + shift * shift
    decode_rms = float(max(decode_rms2, 0.0) ** 0.5)
    y0 = sample_min * scale + shift
    y1 = sample_max * scale + shift
    decode_min = float(min(y0, y1))
    decode_max = float(max(y0, y1))
    decode_absmax = float(max(abs(decode_min), abs(decode_max)))

    ref = FLUX_IMAGENET256_RAW_REFERENCE_STATS
    ref_std = float(ref["reference_raw_flux_std"])
    ref_rms = float(ref["reference_raw_flux_rms"])
    out: Dict[str, Any] = {
        "sample_space_mean": sample_mean,
        "sample_space_std": sample_std,
        "sample_space_rms": sample_rms,
        "sample_space_min": sample_min,
        "sample_space_max": sample_max,
        "sample_space_absmax": sample_absmax,
        "pre_decode_scale": scale,
        "pre_decode_shift": shift,
        "decode_space_mean": decode_mean,
        "decode_space_std": decode_std,
        "decode_space_rms": decode_rms,
        "decode_space_min": decode_min,
        "decode_space_max": decode_max,
        "decode_space_absmax": decode_absmax,
        **ref,
        "reference_raw_flux_stats_source": "compute_latent_cache_stats.py linspace32 shards / 130191 ImageNet-256 train latents",
        "sample_space_std_over_reference_raw_flux_std": sample_std / ref_std if ref_std else None,
        "decode_space_std_over_reference_raw_flux_std": decode_std / ref_std if ref_std else None,
        "sample_space_rms_over_reference_raw_flux_rms": sample_rms / ref_rms if ref_rms else None,
        "decode_space_rms_over_reference_raw_flux_rms": decode_rms / ref_rms if ref_rms else None,
    }
    return out


def decode_batch(
    vae: torch.nn.Module,
    z_cpu: torch.Tensor,
    *,
    device: torch.device,
    model_dtype: torch.dtype,
    pre_decode_scale: float,
    pre_decode_shift: float,
) -> torch.Tensor:
    z = z_cpu.to(device=device, dtype=model_dtype, non_blocking=True)
    # Default is identity: cache/trainer latents are raw VAE posterior latents.
    if pre_decode_scale != 1.0 or pre_decode_shift != 0.0:
        z = z.float().mul(float(pre_decode_scale)).add(float(pre_decode_shift)).to(dtype=model_dtype)
    with torch.inference_mode():
        out = vae.decode(z)
        x = out.sample if hasattr(out, "sample") else out[0]
    if x.ndim == 5:
        if x.shape[2] != 1:
            raise RuntimeError(f"Cannot squeeze decoded video frame dimension from shape {tuple(x.shape)}")
        x = x.squeeze(2)
    if x.ndim != 4:
        raise RuntimeError(f"Expected decoded image tensor [B,C,H,W], got shape {tuple(x.shape)}")
    if x.shape[1] != 3:
        raise RuntimeError(f"Expected decoded RGB channels C=3, got shape {tuple(x.shape)}")
    return x.detach().float().cpu()


def images_to_uint8(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "minus1_1":
        y = x.clamp(-1.0, 1.0).add(1.0).mul(127.5)
    elif mode == "zero1":
        y = x.clamp(0.0, 1.0).mul(255.0)
    elif mode == "auto":
        # FLUX.2 VAE decode is expected to output approximately [-1, 1].
        lo = float(x.min().item())
        hi = float(x.max().item())
        if lo >= -0.05 and hi <= 1.05:
            y = x.clamp(0.0, 1.0).mul(255.0)
        else:
            y = x.clamp(-1.0, 1.0).add(1.0).mul(127.5)
    else:
        raise ValueError(f"Unsupported output_range={mode!r}")
    return y.round().clamp(0, 255).to(torch.uint8)


def save_pngs(images_u8: torch.Tensor, labels: Optional[torch.Tensor], output_dir: Path, *, offset: int) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if images_u8.ndim != 4:
        raise RuntimeError(f"images_u8 must be [B,3,H,W], got {tuple(images_u8.shape)}")
    records: List[Dict[str, Any]] = []
    for i in range(images_u8.shape[0]):
        idx = offset + i
        arr = images_u8[i].permute(1, 2, 0).contiguous().numpy()
        filename = f"{idx:06d}.png"
        Image.fromarray(arr, mode="RGB").save(output_dir / filename)
        rec: Dict[str, Any] = {"index": int(idx), "filename": filename}
        if labels is not None:
            rec["label"] = int(labels[i].item())
        records.append(rec)
    return records


def write_preview_grid(images_u8: torch.Tensor, output_path: Path, *, columns: int = 4) -> None:
    if images_u8.numel() == 0:
        return
    b, c, h, w = images_u8.shape
    cols = max(1, min(int(columns), b))
    rows = (b + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rows * h), color=(0, 0, 0))
    for i in range(b):
        arr = images_u8[i].permute(1, 2, 0).contiguous().numpy()
        img = Image.fromarray(arr, mode="RGB")
        canvas.paste(img, ((i % cols) * w, (i // cols) * h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def decode_latent_file(args: argparse.Namespace) -> Dict[str, Any]:
    latents_path = Path(args.latents).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not latents_path.exists():
        raise FileNotFoundError(latents_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    model_dtype = dtype_from_name(args.model_dtype)
    if device.type == "cuda":
        # On this container's PyTorch build, reset_peak_memory_stats(cuda:0)
        # can throw "Invalid device argument" before the CUDA context is
        # selected/initialized. Selecting the device first makes the memory
        # accounting path reliable without changing compute behavior.
        torch.cuda.set_device(device)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except RuntimeError:
            torch.cuda.reset_peak_memory_stats(device.index if device.index is not None else None)
    started = time.perf_counter()

    z_all, labels, latent_meta = read_latent_file(
        latents_path,
        tensor_key=args.tensor_key,
        max_images=args.max_images,
        start_index=args.start_index,
    )
    latent_space_stats = affine_latent_stats(
        latent_meta,
        scale=float(args.pre_decode_scale),
        shift=float(args.pre_decode_shift),
    )
    vae, diffusers_version, vae_meta = load_vae(args, device)

    manifest_mode = str(getattr(args, "manifest_mode", "full")).strip().lower()
    if manifest_mode not in {"full", "first_last", "none"}:
        raise ValueError(f"Unsupported manifest_mode={manifest_mode!r}; expected full, first_last, or none")

    image_records: List[Dict[str, Any]] = []
    first_image: Optional[Dict[str, Any]] = None
    last_image: Optional[Dict[str, Any]] = None
    num_saved_images = 0
    decoded_float_stats: List[Dict[str, Any]] = []
    all_preview: List[torch.Tensor] = []
    decode_started = time.perf_counter()
    for s in range(0, int(z_all.shape[0]), int(args.batch_size)):
        e = min(int(z_all.shape[0]), s + int(args.batch_size))
        batch = z_all[s:e]
        batch_labels = labels[s:e] if labels is not None else None
        x = decode_batch(
            vae,
            batch,
            device=device,
            model_dtype=model_dtype,
            pre_decode_scale=float(args.pre_decode_scale),
            pre_decode_shift=float(args.pre_decode_shift),
        )
        decoded_float_stats.append(
            {
                "batch_start": int(s),
                "batch_end": int(e),
                "decoded_shape": list(x.shape),
                "decoded_min": float(x.min().item()),
                "decoded_max": float(x.max().item()),
                "decoded_mean": float(x.mean().item()),
                "decoded_std": float(x.std(unbiased=False).item()),
            }
        )
        u8 = images_to_uint8(x, args.output_range)
        batch_records = save_pngs(u8, batch_labels, output_dir, offset=s + int(args.start_index))
        if batch_records:
            if first_image is None:
                first_image = batch_records[0]
            last_image = batch_records[-1]
            num_saved_images += len(batch_records)
            if manifest_mode == "full":
                image_records.extend(batch_records)
        if len(all_preview) < int(args.preview_max_images):
            take = min(int(args.preview_max_images) - len(all_preview), int(u8.shape[0]))
            all_preview.extend([u8[i].clone() for i in range(take)])
        del x, u8, batch
        if device.type == "cuda":
            torch.cuda.empty_cache()
    decode_elapsed = time.perf_counter() - decode_started

    if args.save_grid and all_preview:
        preview = torch.stack(all_preview, dim=0)
        write_preview_grid(preview, output_dir / "preview_grid.png", columns=int(args.preview_columns))

    summary: Dict[str, Any] = {
        "status": "ok",
        "created_at_utc": utc_now(),
        "latents_path": str(latents_path),
        "output_dir": str(output_dir),
        "num_images": int(num_saved_images),
        "batch_size": int(args.batch_size),
        "output_format": "png",
        "output_range": args.output_range,
        "pre_decode_scale": float(args.pre_decode_scale),
        "pre_decode_shift": float(args.pre_decode_shift),
        **latent_space_stats,
        "latent_space_stats": latent_space_stats,
        "latent_meta": latent_meta,
        "vae_meta": vae_meta,
        "decoded_float_stats": decoded_float_stats,
        "images_manifest_mode": manifest_mode,
        "images_manifest_count": len(image_records),
        "images_manifest_omitted": manifest_mode != "full",
        "decode_elapsed_sec": decode_elapsed,
        "elapsed_sec": time.perf_counter() - started,
        "cuda_peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None,
    }
    if manifest_mode == "full":
        summary["images_manifest"] = image_records
    if first_image is not None:
        summary["first_image"] = first_image
    if last_image is not None:
        summary["last_image"] = last_image
    (output_dir / "decode_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents", required=True, help="sample_latents.safetensors or latent shard path")
    p.add_argument("--output-dir", required=True, help="directory where PNGs and decode_summary.json are written")
    p.add_argument("--tensor-key", default="auto", help="auto, samples, latents, or latents_flip")
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--model-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--repo-id", default="diffusers/FLUX.2-dev-bnb-4bit")
    p.add_argument("--subfolder", default="vae")
    p.add_argument("--revision", default="")
    p.add_argument("--vae-class", default="AutoencoderKLFlux2")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--output-range", default="minus1_1", choices=["minus1_1", "zero1", "auto"])
    p.add_argument("--pre-decode-scale", type=float, default=1.0, help="identity by default; cache/trainer latents are raw VAE latents")
    p.add_argument("--pre-decode-shift", type=float, default=0.0, help="identity by default; cache/trainer latents are raw VAE latents")
    p.add_argument("--save-grid", action="store_true")
    p.add_argument("--manifest-mode", default="full", choices=["full", "first_last", "none"], help="decode_summary image manifest policy; use first_last/none for large evals")
    p.add_argument("--preview-max-images", type=int, default=16)
    p.add_argument("--preview-columns", type=int, default=4)
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        summary = decode_latent_file(args)
        # Single-line JSON for trainer/eval logs.
        print(json.dumps({k: summary.get(k) for k in ["status", "num_images", "output_dir", "elapsed_sec", "cuda_peak_memory_mb"]}, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        payload = {"status": "error", "error_type": type(exc).__name__, "error": str(exc), "created_at_utc": utc_now()}
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
