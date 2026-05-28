#!/usr/bin/env python3
"""Direct-feature 50K VAE reconstruction rFID protocol: PAE vs FLUX.2.

Paper-scale reconstruction rFID does not require saving 50K original + 50K
reconstruction PNG intermediates.  This script keeps the protocol equivalent to
an image-file evaluator by quantizing every original/reconstruction to uint8 RGB
[0,255], then feeding those tensors directly to the same LightningDiT/pytorch-fid
Inception wrapper used by the PNG-based diagnostic.

Primary protocol:

    ImageNet validation 50K ADM center crops @ 256
      -> original uint8 features
      -> PAE encode/decode -> uint8 recon features -> rFID(original, PAE recon)
      -> FLUX.2 encode/decode -> uint8 recon features -> rFID(original, FLUX recon)

No MMD/KID is computed here; the requested paper-standard reconstruction audit is
rFID. Runtime outputs are experiment artifacts and should stay out of Git.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("DISABLE_XFORMERS", "1")

import numpy as np
import torch
from safetensors import safe_open

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from inception_metrics_lib import (  # noqa: E402
    RunningMoments,
    build_inception,
    frechet_distance,
    inception_forward,
    resolve_device,
    write_json,
)
from run_vae_rfid_ceiling_100 import (  # noqa: E402
    PixelDiffStats,
    TensorStats,
    dtype_from_name,
    images_to_uint8,
    list_crop_shards,
    load_flux_vae,
    load_pae,
    postprocess_tensor4,
    render_summary_md,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CropBatch:
    global_start: int
    images_u8: torch.Tensor
    labels: Optional[torch.Tensor]
    source_indices: Optional[torch.Tensor]
    shard_path: Path
    local_start: int
    local_end: int


def crop_cache_meta(
    *,
    cache_dir: Path,
    file_glob: str,
    image_key: str,
    label_key: str,
    source_index_key: str,
    start_index: int,
    max_images: int,
) -> Dict[str, Any]:
    shards = list_crop_shards(cache_dir, file_glob, image_key, label_key, source_index_key)
    total = sum(s.num_samples for s in shards)
    if start_index < 0 or start_index >= total:
        raise RuntimeError(f"start_index={start_index} outside total={total}")
    loaded = min(int(max_images), total - int(start_index))
    return {
        "crop_cache_dir": str(cache_dir),
        "crop_file_glob": file_glob,
        "crop_total": int(total),
        "crop_num_shards": int(len(shards)),
        "start_index": int(start_index),
        "max_images": int(max_images),
        "loaded_images": int(loaded),
        "first_shard": str(shards[0].path),
        "last_shard": str(shards[-1].path),
        "image_shape_per_shard_first": shards[0].image_shape,
        "image_shape_per_shard_last": shards[-1].image_shape,
    }


def iter_crop_batches(
    *,
    cache_dir: Path,
    file_glob: str,
    image_key: str,
    label_key: str,
    source_index_key: str,
    start_index: int,
    max_images: int,
    batch_size: int,
) -> Iterator[CropBatch]:
    shards = list_crop_shards(cache_dir, file_glob, image_key, label_key, source_index_key)
    total = sum(s.num_samples for s in shards)
    if start_index < 0 or start_index >= total:
        raise RuntimeError(f"start_index={start_index} outside total={total}")
    target = min(int(max_images), total - int(start_index))
    emitted = 0
    seen = 0
    for shard in shards:
        shard_global_start = seen
        shard_global_end = seen + shard.num_samples
        seen = shard_global_end
        if shard_global_end <= int(start_index):
            continue
        if emitted >= target:
            break
        local_start_all = max(0, int(start_index) - shard_global_start)
        local_available = shard.num_samples - local_start_all
        take_from_shard = min(local_available, target - emitted)
        if take_from_shard <= 0:
            continue
        with safe_open(str(shard.path), framework="pt", device="cpu") as f:
            img_slice = f.get_slice(image_key)
            lab_slice = f.get_slice(label_key) if shard.has_labels else None
            src_slice = f.get_slice(source_index_key) if shard.has_source_indices else None
            for rel in range(0, take_from_shard, int(batch_size)):
                local_start = local_start_all + rel
                local_end = local_start_all + min(take_from_shard, rel + int(batch_size))
                x = img_slice[local_start:local_end].contiguous()
                y = lab_slice[local_start:local_end].long().contiguous() if lab_slice is not None else None
                src = src_slice[local_start:local_end].long().contiguous() if src_slice is not None else None
                b = int(x.shape[0])
                yield CropBatch(
                    global_start=shard_global_start + local_start,
                    images_u8=x,
                    labels=y,
                    source_indices=src,
                    shard_path=shard.path,
                    local_start=local_start,
                    local_end=local_end,
                )
                emitted += b
                if emitted >= target:
                    break


def update_sample_heads(meta: Dict[str, Any], batch: CropBatch) -> None:
    if "first_record" not in meta:
        rec = {
            "shard_path": str(batch.shard_path),
            "local_start": int(batch.local_start),
            "local_end_exclusive": int(batch.local_end),
            "global_start": int(batch.global_start),
            "global_end_exclusive": int(batch.global_start + int(batch.images_u8.shape[0])),
        }
        if batch.labels is not None and batch.labels.numel():
            rec.update({"label_first": int(batch.labels[0].item()), "label_last": int(batch.labels[-1].item())})
            meta["labels_head"] = batch.labels[: min(20, int(batch.labels.shape[0]))].tolist()
        if batch.source_indices is not None and batch.source_indices.numel():
            rec.update({"source_index_first": int(batch.source_indices[0].item()), "source_index_last": int(batch.source_indices[-1].item())})
            meta["source_indices_head"] = batch.source_indices[: min(20, int(batch.source_indices.shape[0]))].tolist()
        meta["first_record"] = rec
    last = {
        "shard_path": str(batch.shard_path),
        "local_start": int(batch.local_start),
        "local_end_exclusive": int(batch.local_end),
        "global_start": int(batch.global_start),
        "global_end_exclusive": int(batch.global_start + int(batch.images_u8.shape[0])),
    }
    if batch.labels is not None and batch.labels.numel():
        last.update({"label_first": int(batch.labels[0].item()), "label_last": int(batch.labels[-1].item())})
    if batch.source_indices is not None and batch.source_indices.numel():
        last.update({"source_index_first": int(batch.source_indices[0].item()), "source_index_last": int(batch.source_indices[-1].item())})
    meta["last_record"] = last


def features_from_uint8_chw(x_u8: torch.Tensor, *, model: torch.nn.Module, device: torch.device) -> np.ndarray:
    x01 = x_u8.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
    feats = inception_forward(model, x01)
    return feats.detach().cpu().numpy().astype(np.float32, copy=False)


def compute_original_moments(args: argparse.Namespace, model: torch.nn.Module, device: torch.device, sample_meta: Dict[str, Any]) -> Dict[str, Any]:
    moments = RunningMoments(dims=int(args.dims))
    started = time.perf_counter()
    total = int(sample_meta["loaded_images"])
    for batch_idx, batch in enumerate(
        iter_crop_batches(
            cache_dir=Path(args.crop_cache_dir).resolve(),
            file_glob=args.crop_file_glob,
            image_key=args.image_key,
            label_key=args.label_key,
            source_index_key=args.source_index_key,
            start_index=int(args.start_index),
            max_images=int(args.max_images),
            batch_size=int(args.feature_batch_size),
        )
    ):
        done_before = moments.count
        if int(args.log_every_batches) > 0 and (batch_idx % int(args.log_every_batches) == 0):
            elapsed = time.perf_counter() - started
            print(
                f"ORIGINAL_FEATURES_PROGRESS batch={batch_idx} images={done_before}/{total} "
                f"elapsed_sec={elapsed:.1f} samples_per_sec={done_before / max(elapsed, 1e-9):.3f}",
                flush=True,
            )
        update_sample_heads(sample_meta, batch)
        moments.update(features_from_uint8_chw(batch.images_u8, model=model, device=device))
    mu, sigma = moments.finalize()
    elapsed = time.perf_counter() - started
    return {
        "num_images": int(moments.count),
        "mu": mu,
        "sigma": sigma,
        "feature_mean": float(mu.mean()),
        "feature_std_proxy_sqrt_trace_over_dims": float(math.sqrt(max(0.0, float(np.trace(sigma)) / float(args.dims)))),
        "sigma_trace": float(np.trace(sigma)),
        "elapsed_sec": float(elapsed),
        "samples_per_sec": float(moments.count / max(elapsed, 1e-9)),
    }


def reconstruct_pae_direct(
    args: argparse.Namespace,
    pae: torch.nn.Module,
    inception: torch.nn.Module,
    device: torch.device,
    model_dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    moments = RunningMoments(dims=int(args.dims))
    pix = PixelDiffStats()
    latent_stats = TensorStats()
    decoded_stats = TensorStats()
    started = time.perf_counter()
    total = int(args.max_images)
    use_autocast = device.type == "cuda" and model_dtype in {torch.bfloat16, torch.float16}
    for batch_idx, batch in enumerate(
        iter_crop_batches(
            cache_dir=Path(args.crop_cache_dir).resolve(),
            file_glob=args.crop_file_glob,
            image_key=args.image_key,
            label_key=args.label_key,
            source_index_key=args.source_index_key,
            start_index=int(args.start_index),
            max_images=int(args.max_images),
            batch_size=int(args.pae_batch_size),
        )
    ):
        s_done = moments.count
        if int(args.log_every_batches) > 0 and (batch_idx % int(args.log_every_batches) == 0):
            elapsed = time.perf_counter() - started
            print(
                f"PAE_DIRECT_PROGRESS batch={batch_idx} images={s_done}/{total} "
                f"elapsed_sec={elapsed:.1f} samples_per_sec={s_done / max(elapsed, 1e-9):.3f}",
                flush=True,
            )
        x_u8 = batch.images_u8.contiguous()
        x = x_u8.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=model_dtype, enabled=use_autocast):
                z = pae.encode(x)
                rec = pae.decode(z)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rec = postprocess_tensor4(rec, "PAE decoded image")
        rec_float = rec.detach().float()
        rec_u8 = images_to_uint8(rec_float, "zero1")
        feats = features_from_uint8_chw(rec_u8, model=inception, device=device)
        moments.update(feats)
        rec_u8_cpu = rec_u8.detach().cpu()
        pix.update(x_u8.cpu(), rec_u8_cpu)
        latent_stats.update(z.detach().float().cpu())
        decoded_stats.update(rec_float.cpu())
        del x, z, rec, rec_float, rec_u8, rec_u8_cpu, feats, x_u8
    mu, sigma = moments.finalize()
    elapsed = time.perf_counter() - started
    runtime = {
        "elapsed_sec": float(elapsed),
        "samples_per_sec": float(moments.count / max(elapsed, 1e-9)),
        "pixel_metrics": pix.finalize(),
    }
    extra = {
        "latent_stats": latent_stats.finalize(),
        "decoded_float_stats": decoded_stats.finalize(),
    }
    return runtime, extra, mu, sigma


def reconstruct_flux_direct(
    args: argparse.Namespace,
    vae: torch.nn.Module,
    inception: torch.nn.Module,
    device: torch.device,
    model_dtype: torch.dtype,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray, np.ndarray]:
    moments = RunningMoments(dims=int(args.dims))
    pix = PixelDiffStats()
    latent_stats = TensorStats()
    decoded_stats = TensorStats()
    started = time.perf_counter()
    total = int(args.max_images)
    for batch_idx, batch in enumerate(
        iter_crop_batches(
            cache_dir=Path(args.crop_cache_dir).resolve(),
            file_glob=args.crop_file_glob,
            image_key=args.image_key,
            label_key=args.label_key,
            source_index_key=args.source_index_key,
            start_index=int(args.start_index),
            max_images=int(args.max_images),
            batch_size=int(args.flux_batch_size),
        )
    ):
        s_done = moments.count
        if int(args.log_every_batches) > 0 and (batch_idx % int(args.log_every_batches) == 0):
            elapsed = time.perf_counter() - started
            print(
                f"FLUX2_DIRECT_PROGRESS batch={batch_idx} images={s_done}/{total} "
                f"elapsed_sec={elapsed:.1f} samples_per_sec={s_done / max(elapsed, 1e-9):.3f}",
                flush=True,
            )
        x_u8 = batch.images_u8.contiguous()
        x = x_u8.to(device=device, dtype=torch.float32, non_blocking=True).div_(127.5).sub_(1.0)
        if model_dtype != torch.float32:
            x = x.to(dtype=model_dtype)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.inference_mode():
            enc = vae.encode(x)
            if args.flux_latent_mode == "mode":
                z = enc.latent_dist.mode()
            elif args.flux_latent_mode == "sample":
                z = enc.latent_dist.sample()
            else:
                raise ValueError(f"Unsupported latent mode {args.flux_latent_mode!r}")
            z = postprocess_tensor4(z, "FLUX latent")
            dec = vae.decode(z)
            rec = dec.sample if hasattr(dec, "sample") else dec[0]
            rec = postprocess_tensor4(rec, "FLUX decoded image")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        rec_float = rec.detach().float()
        rec_u8 = images_to_uint8(rec_float, "minus1_1")
        feats = features_from_uint8_chw(rec_u8, model=inception, device=device)
        moments.update(feats)
        rec_u8_cpu = rec_u8.detach().cpu()
        pix.update(x_u8.cpu(), rec_u8_cpu)
        latent_stats.update(z.detach().float().cpu())
        decoded_stats.update(rec_float.cpu())
        del x, z, rec, rec_float, rec_u8, rec_u8_cpu, feats, x_u8
    mu, sigma = moments.finalize()
    elapsed = time.perf_counter() - started
    runtime = {
        "elapsed_sec": float(elapsed),
        "samples_per_sec": float(moments.count / max(elapsed, 1e-9)),
        "pixel_metrics": pix.finalize(),
    }
    extra = {
        "latent_stats": latent_stats.finalize(),
        "decoded_float_stats": decoded_stats.finalize(),
    }
    return runtime, extra, mu, sigma


def matrix_record(mu: np.ndarray, sigma: np.ndarray, *, num_images: int, name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "num_images": int(num_images),
        "feature_mean": float(mu.mean()),
        "feature_std_proxy_sqrt_trace_over_dims": float(math.sqrt(max(0.0, float(np.trace(sigma)) / float(mu.shape[0])))),
        "sigma_trace": float(np.trace(sigma)),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    sqrt_device = device if str(args.sqrt_device).lower() == "auto" else resolve_device(args.sqrt_device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    sample_meta = crop_cache_meta(
        cache_dir=Path(args.crop_cache_dir).resolve(),
        file_glob=args.crop_file_glob,
        image_key=args.image_key,
        label_key=args.label_key,
        source_index_key=args.source_index_key,
        start_index=int(args.start_index),
        max_images=int(args.max_images),
    )

    run_config = {
        "created_at_utc": utc_now(),
        "argv": sys.argv,
        "args": vars(args),
        "output_dir": str(output_dir),
        "device": str(device),
        "sqrt_device": str(sqrt_device),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "pid": os.getpid(),
        "direct_feature_protocol": True,
        "sample_meta_initial": sample_meta,
    }
    write_json(output_dir / "run_config.json", run_config)
    print("ARGS", json.dumps(run_config, indent=2, sort_keys=True), flush=True)

    # Load both VAEs before building LightningDiT Inception.  The external
    # Inception helper installs a tiny scipy placeholder when scipy is absent;
    # loading PAE/diffusers first avoids downstream optional-dependency probes
    # seeing that placeholder.
    pae_dtype = dtype_from_name(args.pae_model_dtype)
    flux_dtype = dtype_from_name(args.flux_model_dtype)

    print("LOADING_PAE", flush=True)
    pae, pae_meta = load_pae(args, device, pae_dtype)
    print("LOADING_FLUX2_VAE", flush=True)
    flux_vae, flux_meta = load_flux_vae(args, device, flux_dtype)

    print("LOADING_INCEPTION", flush=True)
    inception = build_inception(dims=int(args.dims), device=device)

    print("COMPUTING_ORIGINAL_FEATURES_DIRECT", flush=True)
    orig = compute_original_moments(args, inception, device, sample_meta)
    orig_mu, orig_sigma = orig.pop("mu"), orig.pop("sigma")
    write_json(output_dir / "sample_meta.json", sample_meta)
    write_json(output_dir / "original_feature_stats.json", {**orig, "sample_meta": sample_meta})

    print("RUNNING_PAE_RECON_DIRECT", flush=True)
    pae_runtime, pae_extra, pae_mu, pae_sigma = reconstruct_pae_direct(args, pae, inception, device, pae_dtype)
    del pae
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("RUNNING_FLUX2_RECON_DIRECT", flush=True)
    flux_runtime, flux_extra, flux_mu, flux_sigma = reconstruct_flux_direct(args, flux_vae, inception, device, flux_dtype)
    del flux_vae
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("COMPUTING_FRECHET_DIRECT", flush=True)
    pae_rfid = frechet_distance(pae_mu, pae_sigma, orig_mu, orig_sigma, sqrt_device=sqrt_device)
    flux_rfid = frechet_distance(flux_mu, flux_sigma, orig_mu, orig_sigma, sqrt_device=sqrt_device)
    ratio = float(flux_rfid / pae_rfid) if pae_rfid > 0 else float("inf")
    delta = float(flux_rfid - pae_rfid)
    if ratio >= float(args.much_worse_ratio) and delta >= float(args.much_worse_delta):
        label = "flux_rfid_much_worse_than_pae"
    else:
        label = "flux_rfid_not_much_worse_than_pae"

    metrics = {
        "status": "ok",
        "created_at_utc": utc_now(),
        "metric_scope": f"same_{int(sample_meta['loaded_images'])}_image_reconstruction_fid_original_distribution_vs_recon_distribution",
        "protocol": "imagenet_val50k_adm256_direct_uint8_inception_features_no_png",
        "dims": int(args.dims),
        "device": str(device),
        "sqrt_device": str(sqrt_device),
        "skip_mmd": True,
        "original": matrix_record(orig_mu, orig_sigma, num_images=int(sample_meta["loaded_images"]), name="original"),
        "reconstructions": {
            "pae": {
                **matrix_record(pae_mu, pae_sigma, num_images=int(sample_meta["loaded_images"]), name="pae"),
                "rfid_vs_original": float(pae_rfid),
                "rfid_vs_original_100": float(pae_rfid),
                "mmd2_vs_original": None,
                "kid_x1000_vs_original": None,
                "mmd2_vs_original_100": None,
                "kid_x1000_vs_original_100": None,
                "mmd_skipped": True,
                "mmd_skip_reason": "direct 50K reconstruction protocol primary metric is rFID",
            },
            "flux2": {
                **matrix_record(flux_mu, flux_sigma, num_images=int(sample_meta["loaded_images"]), name="flux2"),
                "rfid_vs_original": float(flux_rfid),
                "rfid_vs_original_100": float(flux_rfid),
                "mmd2_vs_original": None,
                "kid_x1000_vs_original": None,
                "mmd2_vs_original_100": None,
                "kid_x1000_vs_original_100": None,
                "mmd_skipped": True,
                "mmd_skip_reason": "direct 50K reconstruction protocol primary metric is rFID",
            },
        },
        "elapsed_sec": float(time.perf_counter() - started),
        "original_feature_runtime": orig,
    }
    write_json(output_dir / "rfid_metrics.json", metrics)

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
        "metric_scope": f"{int(sample_meta['loaded_images'])}_image_same_input_reconstruction_fid_ceiling_pae_vs_flux2",
        "protocol": "imagenet_val50k_adm256_direct_uint8_inception_features_no_png",
        "protocol_equivalence_note": "Each reconstruction is quantized to uint8 RGB before Inception, matching the PNG-based path without lossless file IO.",
        "sample_count": int(sample_meta["loaded_images"]),
        "sample_meta": sample_meta,
        "output_dir": str(output_dir),
        "pae_meta": pae_meta,
        "flux2_meta": flux_meta,
        "rfid_metrics": metrics,
        "comparison": comparison,
        "elapsed_sec": float(time.perf_counter() - started),
        "cuda_peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None,
        "artifact_policy": "Do not commit logs, runtime results, or large feature/stat artifacts by default.",
    }
    write_json(output_dir / "rfid_ceiling_summary.json", summary)
    (output_dir / "rfid_ceiling_summary.md").write_text(render_summary_md(summary), encoding="utf-8")
    print(
        "RFID_CEILING_SUMMARY",
        json.dumps(
            {
                "status": "ok",
                "sample_count": int(sample_meta["loaded_images"]),
                "pae_rfid": float(pae_rfid),
                "flux2_rfid": float(flux_rfid),
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
    p.add_argument("--crop-cache-dir", default="/workspace/PDM/data/cropped_uint8/imagenet1k_validation_256_adm_safetensors")
    p.add_argument("--crop-file-glob", default="images_uint8_shard*.safetensors")
    p.add_argument("--image-key", default="images")
    p.add_argument("--label-key", default="labels")
    p.add_argument("--source-index-key", default="source_indices")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--max-images", type=int, default=50000)
    p.add_argument("--output-dir", default="/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/vae_rfid_protocol_val50k_pae_vs_flux2_direct")
    p.add_argument("--device", default="auto")
    p.add_argument("--allow-tf32", action="store_true")

    p.add_argument("--pae-config", default="/workspace/PDM/external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml")
    p.add_argument("--pae-ckpt", default="/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt")
    p.add_argument("--pae-loader-script", default="/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/scripts/build_pae_latents_full_cache.py")
    p.add_argument("--pae-model-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--pae-batch-size", type=int, default=64)

    p.add_argument("--flux-repo-id", default="diffusers/FLUX.2-dev-bnb-4bit")
    p.add_argument("--flux-subfolder", default="vae")
    p.add_argument("--flux-revision", default="")
    p.add_argument("--flux-vae-class", default="AutoencoderKLFlux2")
    p.add_argument("--flux-model-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--flux-batch-size", type=int, default=64)
    p.add_argument("--flux-latent-mode", default="mode", choices=["mode", "sample"])
    p.add_argument("--local-files-only", action="store_true")

    p.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--feature-batch-size", type=int, default=256)
    p.add_argument("--sqrt-device", default="auto")
    p.add_argument("--log-every-batches", type=int, default=50)

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
