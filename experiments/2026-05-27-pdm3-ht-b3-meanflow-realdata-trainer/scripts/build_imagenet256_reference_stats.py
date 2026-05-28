#!/usr/bin/env python3
"""Build real ImageNet-256 Inception reference stats from cropped uint8 shards."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from inception_metrics_lib import (
    RunningMoments,
    build_inception,
    iter_safetensor_image_batches,
    resolve_device,
    safetensor_image_shards,
    save_npz_atomic,
    shard_num_images,
    utc_now,
    write_json,
    inception_forward,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True, help="cropped uint8 safetensors cache dir")
    p.add_argument("--file-glob", default="images_uint8_shard*.safetensors")
    p.add_argument("--image-key", default="images")
    p.add_argument("--output-stats", required=True, help="output .npz with mu/sigma/count")
    p.add_argument("--output-summary", default="", help="optional summary json path")
    p.add_argument("--output-mmd-features", default="", help="optional .npz with deterministic real feature bank")
    p.add_argument("--mmd-ref-samples", type=int, default=8192)
    p.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-shards", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--progress-every-batches", type=int, default=20)
    p.add_argument("--allow-tf32", action="store_true", help="allow TF32 in CUDA matmul/conv; default is strict fp32")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    cache_dir = Path(args.cache_dir).resolve()
    output_stats = Path(args.output_stats).resolve()
    output_summary = Path(args.output_summary).resolve() if args.output_summary else output_stats.with_suffix(".summary.json")
    output_mmd = Path(args.output_mmd_features).resolve() if args.output_mmd_features else None
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    shards = safetensor_image_shards(cache_dir, args.file_glob)
    if args.max_shards is not None:
        shards = shards[: int(args.max_shards)]
    shard_counts = [shard_num_images(p, args.image_key) for p in shards]
    planned_total = int(sum(shard_counts))
    if args.max_samples is not None:
        planned_total = min(planned_total, int(args.max_samples))
    if planned_total < 2:
        raise RuntimeError(f"Need at least 2 reference samples, planned_total={planned_total}")

    mmd_target = int(args.mmd_ref_samples or 0)
    selected_indices: np.ndarray
    if output_mmd is not None and mmd_target > 0:
        mmd_target = min(mmd_target, planned_total)
        selected_indices = np.linspace(0, planned_total - 1, num=mmd_target, dtype=np.int64)
    else:
        selected_indices = np.zeros((0,), dtype=np.int64)
    sel_ptr = 0
    selected_features: List[np.ndarray] = []
    selected_global_indices: List[int] = []

    model = build_inception(dims=int(args.dims), device=device)
    stats = RunningMoments(dims=int(args.dims))

    batch_count = 0
    last_print = time.perf_counter()
    for global_start, images_u8, shard in iter_safetensor_image_batches(
        shards,
        image_key=args.image_key,
        batch_size=int(args.batch_size),
        max_samples=args.max_samples,
    ):
        bsz = int(images_u8.shape[0])
        x = images_u8.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
        feats = inception_forward(model, x).detach().cpu().numpy().astype(np.float32, copy=False)
        del x, images_u8
        stats.update(feats)

        batch_end = global_start + bsz
        while sel_ptr < int(selected_indices.shape[0]) and int(selected_indices[sel_ptr]) < batch_end:
            idx = int(selected_indices[sel_ptr])
            if idx >= global_start:
                selected_features.append(feats[idx - global_start : idx - global_start + 1].copy())
                selected_global_indices.append(idx)
            sel_ptr += 1

        batch_count += 1
        if int(args.progress_every_batches) > 0 and batch_count % int(args.progress_every_batches) == 0:
            now = time.perf_counter()
            rate = stats.count / max(now - started, 1e-9)
            eta = (planned_total - stats.count) / max(rate, 1e-9)
            payload = {
                "event": "progress",
                "time_utc": utc_now(),
                "count": stats.count,
                "planned_total": planned_total,
                "percent": round(100.0 * stats.count / planned_total, 3),
                "samples_per_sec": round(rate, 3),
                "eta_sec": round(eta, 1),
                "last_shard": str(shard.name),
                "batch_count": batch_count,
                "interval_sec": round(now - last_print, 3),
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
            last_print = now
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mu, sigma = stats.finalize()
    save_npz_atomic(
        output_stats,
        mu=mu,
        sigma=sigma,
        count=np.array(stats.count, dtype=np.int64),
        dims=np.array(int(args.dims), dtype=np.int64),
        image_size=np.array(256, dtype=np.int64),
        created_unix=np.array(time.time(), dtype=np.float64),
    )

    mmd_written = None
    if output_mmd is not None and selected_features:
        bank = np.concatenate(selected_features, axis=0).astype(np.float32, copy=False)
        save_npz_atomic(
            output_mmd,
            features=bank,
            global_indices=np.asarray(selected_global_indices, dtype=np.int64),
            dims=np.array(int(args.dims), dtype=np.int64),
            reference_count=np.array(stats.count, dtype=np.int64),
            created_unix=np.array(time.time(), dtype=np.float64),
        )
        mmd_written = {
            "path": str(output_mmd),
            "num_features": int(bank.shape[0]),
            "dims": int(bank.shape[1]),
            "global_index_first": int(selected_global_indices[0]),
            "global_index_last": int(selected_global_indices[-1]),
        }

    summary: Dict[str, Any] = {
        "status": "ok",
        "created_at_utc": utc_now(),
        "cache_dir": str(cache_dir),
        "file_glob": args.file_glob,
        "image_key": args.image_key,
        "num_shards": len(shards),
        "planned_total": planned_total,
        "count": int(stats.count),
        "dims": int(args.dims),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "allow_tf32": bool(args.allow_tf32),
        "max_shards": args.max_shards,
        "max_samples": args.max_samples,
        "output_stats": str(output_stats),
        "output_mmd_features": mmd_written,
        "elapsed_sec": time.perf_counter() - started,
        "samples_per_sec": stats.count / max(time.perf_counter() - started, 1e-9),
        "mu_mean": float(mu.mean()),
        "mu_std": float(mu.std()),
        "sigma_trace": float(np.trace(sigma)),
        "first_shard": str(shards[0]),
        "last_shard": str(shards[-1]),
    }
    if device.type == "cuda":
        summary["cuda_peak_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    write_json(output_summary, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
