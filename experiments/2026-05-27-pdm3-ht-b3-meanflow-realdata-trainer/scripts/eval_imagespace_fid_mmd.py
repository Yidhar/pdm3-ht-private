#!/usr/bin/env python3
"""Compute real FID and polynomial-MMD/KID for an image directory."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from inception_metrics_lib import (
    build_inception,
    covariance_from_features,
    features_from_image_dir,
    frechet_distance,
    load_feature_bank,
    load_reference_stats,
    polynomial_mmd2_unbiased,
    resolve_device,
    utc_now,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images-dir", required=True)
    p.add_argument("--ref-stats", required=True, help="ImageNet-256 reference stats .npz")
    p.add_argument("--ref-features", default="", help="optional real feature bank .npz for MMD/KID")
    p.add_argument("--output-json", default="")
    p.add_argument("--dims", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--device", default="auto")
    p.add_argument("--sqrt-device", default="auto", help="device for Frechet sqrt/eig; auto follows --device")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--mmd-max-ref", type=int, default=8192)
    p.add_argument("--mmd-device", default="auto")
    p.add_argument("--mmd-chunk-size", type=int, default=2048)
    p.add_argument("--save-features", default="", help="optional .npz path for generated Inception features")
    p.add_argument("--allow-tf32", action="store_true", help="allow TF32 in CUDA matmul/conv; default is strict fp32")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    images_dir = Path(args.images_dir).resolve()
    ref_stats_path = Path(args.ref_stats).resolve()
    ref_features_path = Path(args.ref_features).resolve() if args.ref_features else None
    output_json = Path(args.output_json).resolve() if args.output_json else images_dir / "fid_mmd_metrics.json"
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)
    sqrt_device = device if str(args.sqrt_device).lower() == "auto" else resolve_device(args.sqrt_device)
    mmd_device = device if str(args.mmd_device).lower() == "auto" else resolve_device(args.mmd_device)
    torch.set_float32_matmul_precision("high" if args.allow_tf32 else "highest")

    model = build_inception(dims=int(args.dims), device=device)
    gen_features, image_meta = features_from_image_dir(
        images_dir,
        model=model,
        device=device,
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        max_images=args.max_images,
        recursive=bool(args.recursive),
    )
    gen_mu, gen_sigma = covariance_from_features(gen_features)
    ref_mu, ref_sigma, ref_meta = load_reference_stats(ref_stats_path)
    fid = frechet_distance(
        gen_mu,
        gen_sigma,
        ref_mu,
        ref_sigma,
        features1=gen_features,
        sqrt_device=sqrt_device,
    )

    record: Dict[str, Any] = {
        "status": "ok",
        "created_at_utc": utc_now(),
        "images_dir": str(images_dir),
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
        "batch_size": int(args.batch_size),
        "allow_tf32": bool(args.allow_tf32),
    }

    if ref_features_path is not None:
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

    if args.save_features:
        save_path = Path(args.save_features).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_path, features=gen_features.astype(np.float32), image_count=np.array(gen_features.shape[0], dtype=np.int64))
        record["generated_features_path"] = str(save_path)

    record["elapsed_sec"] = time.perf_counter() - started
    if device.type == "cuda":
        record["cuda_peak_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / 1024**2)
    write_json(output_json, record)
    print(json.dumps(record, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
