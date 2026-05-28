#!/usr/bin/env python3
"""Fast PAE latent builder from pre-cropped uint8 safetensors cache.

Reads cropped cache shards produced by build_cropped_uint8_cache.py and writes the
same latent shard schema as build_pae_latents_full_cache.py:
  - latents: [N,32,16,16]
  - latents_flip: [N,32,16,16]
  - labels: [N]

This removes JPEG decode/PIL crop from the GPU encoding loop.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from safetensors import safe_open

DEFAULT_BASE_SCRIPT_DIR = Path("/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/scripts")
if str(DEFAULT_BASE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DEFAULT_BASE_SCRIPT_DIR))

from build_pae_latents_full_cache import (  # noqa: E402
    PendingShardBuffer,
    append_jsonl,
    cuda_mem,
    cuda_reset_peak,
    dtype_from_name,
    encode_pair_from_u8_batch,
    fadvise_dontneed,
    load_pae_model,
    save_shard_atomic,
    scan_existing_shards,
    seconds_to_hms,
    write_json_atomic,
)

_SHARD_RE = re.compile(r"images_uint8_shard(\d+)\.safetensors$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CropShard:
    shard_index: int
    path: str
    num_samples: int
    source_index_first: int
    source_index_last: int
    image_shape: List[int]
    image_dtype: str
    labels_min: int
    labels_max: int
    bytes: int


def parse_crop_shard_index(path: Path) -> int:
    m = _SHARD_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse crop shard index: {path.name}")
    return int(m.group(1))


def scan_crop_shards(crop_dir: Path, strict_contiguous: bool = True) -> List[CropShard]:
    shards: List[CropShard] = []
    for p in sorted(crop_dir.glob("images_uint8_shard*.safetensors")):
        idx = parse_crop_shard_index(p)
        with safe_open(str(p), framework="pt", device="cpu") as f:
            img_slice = f.get_slice("images")
            img_shape = list(img_slice.get_shape())
            img_dtype = str(img_slice.get_dtype())
            labels = f.get_tensor("labels")
            source_indices = f.get_tensor("source_indices")
        n = int(labels.numel())
        shards.append(
            CropShard(
                shard_index=idx,
                path=str(p),
                num_samples=n,
                source_index_first=int(source_indices[0].item()),
                source_index_last=int(source_indices[-1].item()),
                image_shape=img_shape,
                image_dtype=img_dtype,
                labels_min=int(labels.min().item()),
                labels_max=int(labels.max().item()),
                bytes=int(p.stat().st_size),
            )
        )
    shards.sort(key=lambda s: s.shard_index)
    if strict_contiguous:
        expected_source = 0
        for expected_idx, s in enumerate(shards):
            if s.shard_index != expected_idx:
                raise RuntimeError(f"Non-contiguous crop shard index: expected {expected_idx}, got {s.shard_index}")
            if s.source_index_first != expected_source:
                raise RuntimeError(f"Non-contiguous source index: expected {expected_source}, got {s.source_index_first} at {s.path}")
            expected_source += s.num_samples
            if s.source_index_last != expected_source - 1:
                raise RuntimeError(f"Bad source_index_last in {s.path}: expected {expected_source - 1}, got {s.source_index_last}")
    return shards


def iter_cropped_batches(
    crop_shards: Sequence[CropShard],
    *,
    start_index: int,
    target_total: int,
    batch_size: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    for shard in crop_shards:
        if shard.source_index_last < start_index:
            continue
        if shard.source_index_first >= target_total:
            break
        local_start = max(0, start_index - shard.source_index_first)
        local_end = min(shard.num_samples, target_total - shard.source_index_first)
        if local_start >= local_end:
            continue
        with safe_open(shard.path, framework="pt", device="cpu") as f:
            images = f.get_tensor("images")
            labels = f.get_tensor("labels")
            source_indices = f.get_tensor("source_indices")
        # Validate local source order for this segment.
        if int(source_indices[local_start].item()) != shard.source_index_first + local_start:
            raise RuntimeError(f"Bad source index in {shard.path} at local_start={local_start}")
        for s in range(local_start, local_end, batch_size):
            e = min(local_end, s + batch_size)
            x = images[s:e].contiguous()
            y = labels[s:e].to(dtype=torch.long).contiguous()
            meta = {
                "crop_shard_index": shard.shard_index,
                "crop_path": shard.path,
                "source_index_first": int(source_indices[s].item()),
                "source_index_last": int(source_indices[e - 1].item()),
                "batch_samples": int(e - s),
            }
            yield x, y, meta
        del images, labels, source_indices
        fadvise_dontneed(Path(shard.path))


def clear_latent_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pat in ("latents_rank00_shard*.safetensors", ".latents_rank00_shard*.tmp.*", "manifest.jsonl", "progress.jsonl", "progress.json", "build_summary.json", "run_config.json"):
        for p in output_dir.glob(pat):
            if p.is_file():
                p.unlink()


def build_latents(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    crop_dir = Path(args.crop_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        print(f"OVERWRITE enabled: clearing latent output artifacts in {output_dir}", flush=True)
        clear_latent_output_dir(output_dir)

    crop_shards = scan_crop_shards(crop_dir, strict_contiguous=True)
    crop_total = sum(s.num_samples for s in crop_shards)
    if crop_total <= 0:
        raise RuntimeError(f"No cropped samples found in {crop_dir}")
    target_total = min(args.max_samples, crop_total) if args.max_samples is not None else crop_total

    existing = scan_existing_shards(output_dir, strict_contiguous=True)
    completed_at_start = sum(r.num_samples for r in existing)
    next_shard_index = (existing[-1].shard_index + 1) if existing else 0
    if existing and not args.resume:
        raise RuntimeError(f"Found {len(existing)} existing latent shards in {output_dir}; pass --resume or --overwrite")
    if completed_at_start > target_total:
        raise RuntimeError(f"Existing latent cache has {completed_at_start} samples > target_total={target_total}")
    if target_total > crop_total:
        raise RuntimeError(f"Requested target_total={target_total} but crop cache only has {crop_total}")

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available; pass --allow-cpu only for debugging.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = dtype_from_name(args.model_dtype)
    save_dtype = dtype_from_name(args.save_dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        cuda_reset_peak(device)

    run_config = {
        "created_utc": utc_now(),
        "argv": sys.argv,
        "args": vars(args),
        "crop_cache_dir": str(crop_dir),
        "crop_total": crop_total,
        "crop_num_shards": len(crop_shards),
        "target_total": target_total,
        "existing_latent_shards": len(existing),
        "completed_at_start": completed_at_start,
        "next_shard_index": next_shard_index,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "pid": os.getpid(),
    }
    write_json_atomic(output_dir / "run_config.json", run_config)
    print("ARGS", json.dumps(run_config, indent=2, sort_keys=True), flush=True)

    if completed_at_start == target_total:
        summary = {
            "ok": True,
            "status": "already_complete",
            "output_dir": str(output_dir),
            "crop_cache_dir": str(crop_dir),
            "target_total": target_total,
            "encoded_total": completed_at_start,
            "new_encoded": 0,
            "num_shards": len(existing),
            "finished_utc": utc_now(),
        }
        write_json_atomic(output_dir / "progress.json", summary)
        write_json_atomic(output_dir / "build_summary.json", summary)
        print("BUILD_SUMMARY", json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    model = load_pae_model(Path(args.pae_config), Path(args.pae_ckpt), device, model_dtype)

    metadata = {
        "schema": "pae_latent_from_cropped_uint8_v1",
        "crop_cache_dir": str(crop_dir),
        "crop_schema": "imagenet_cropped_uint8_v1",
        "source_loader": "cropped_uint8_safetensors",
        "model_dtype": str(model_dtype),
        "save_dtype": str(save_dtype),
        "image_size": str(args.image_size),
    }

    pending = PendingShardBuffer()
    new_records: List[Any] = []
    start_t = time.time()
    last_log_t = start_t
    last_log_encoded = 0
    encoded_total = completed_at_start
    new_encoded = 0
    batch_index = 0
    h2d_time_total = 0.0
    encode_time_total = 0.0
    d2h_time_total = 0.0
    load_crop_time_total = 0.0
    save_time_total = 0.0

    def write_progress(event: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        nonlocal last_log_t, last_log_encoded
        now = time.time()
        elapsed = now - start_t
        interval_elapsed = now - last_log_t
        interval_samples = new_encoded - last_log_encoded
        remaining = max(0, target_total - encoded_total)
        avg = new_encoded / elapsed if elapsed > 0 else None
        eta = remaining / avg if avg and avg > 0 else None
        payload: Dict[str, Any] = {
            "event": event,
            "status": "running",
            "time_utc": utc_now(),
            "pid": os.getpid(),
            "output_dir": str(output_dir),
            "crop_cache_dir": str(crop_dir),
            "source_loader": "cropped_uint8_safetensors",
            "crop_total": crop_total,
            "crop_num_shards": len(crop_shards),
            "target_total": target_total,
            "completed_at_start": completed_at_start,
            "encoded_total": encoded_total,
            "new_encoded": new_encoded,
            "remaining_to_target": remaining,
            "elapsed_sec": elapsed,
            "avg_new_samples_per_sec": avg,
            "interval_samples_per_sec": interval_samples / interval_elapsed if interval_elapsed > 0 else None,
            "eta_sec": eta,
            "eta_hms": seconds_to_hms(eta),
            "batch_index": batch_index,
            "batch_size": args.batch_size,
            "num_existing_shards": len(existing),
            "num_new_shards": len(new_records),
            "next_shard_index": next_shard_index + len(new_records),
            "pending_shard_samples": pending.n,
            "latent_shard_size": args.latent_shard_size,
            "h2d_time_total_sec": h2d_time_total,
            "encode_time_total_sec": encode_time_total,
            "d2h_time_total_sec": d2h_time_total,
            "load_crop_time_total_sec": load_crop_time_total,
            "save_time_total_sec": save_time_total,
            **cuda_mem(device),
        }
        if extra:
            payload.update(extra)
        write_json_atomic(output_dir / "progress.json", payload)
        append_jsonl(output_dir / "progress.jsonl", payload)
        last_log_t = now
        last_log_encoded = new_encoded
        return payload

    write_progress("start")
    try:
        batch_iter = iter_cropped_batches(crop_shards, start_index=completed_at_start, target_total=target_total, batch_size=args.batch_size)
        for x_u8, y, batch_meta in batch_iter:
            if encoded_total >= target_total:
                break
            batch_index += 1
            t_batch = time.time()
            # x_u8 loading is already included in iterator; this measures encode call only.
            z, zf, timing = encode_pair_from_u8_batch(model, x_u8, device, model_dtype, save_dtype)
            batch_elapsed = time.time() - t_batch
            h2d_time_total += timing["h2d_sec"]
            encode_time_total += timing["encode_sec"]
            d2h_time_total += timing["d2h_sec"]

            pending.add(z, zf, y)
            n_batch = int(y.shape[0])
            encoded_total += n_batch
            new_encoded += n_batch

            saved_now: List[Dict[str, Any]] = []
            while pending.n >= args.latent_shard_size:
                shard_z, shard_zf, shard_y = pending.pop(args.latent_shard_size)
                shard_idx = next_shard_index + len(new_records)
                t_save = time.time()
                rec = save_shard_atomic(output_dir, shard_idx, shard_z, shard_zf, shard_y, metadata, args.sha256)
                save_time_total += time.time() - t_save
                new_records.append(rec)
                append_jsonl(output_dir / "manifest.jsonl", asdict(rec))
                saved_now.append(asdict(rec))
                del shard_z, shard_zf, shard_y

            progress = write_progress(
                "batch",
                {
                    "last_batch_samples": n_batch,
                    "last_batch_elapsed_sec": batch_elapsed,
                    "last_batch_samples_per_sec": n_batch / batch_elapsed if batch_elapsed > 0 else None,
                    "last_batch_timing": timing,
                    "last_crop_batch_meta": batch_meta,
                    "saved_shards_this_batch": saved_now,
                },
            )
            print(
                "PAE_CROP_PROGRESS",
                json.dumps(
                    {k: progress[k] for k in ["encoded_total", "target_total", "new_encoded", "avg_new_samples_per_sec", "interval_samples_per_sec", "eta_hms", "pending_shard_samples", "num_new_shards", "cuda_peak_allocated_mb"]},
                    sort_keys=True,
                ),
                flush=True,
            )

        if pending.n > 0:
            shard_z, shard_zf, shard_y = pending.pop(pending.n)
            shard_idx = next_shard_index + len(new_records)
            t_save = time.time()
            rec = save_shard_atomic(output_dir, shard_idx, shard_z, shard_zf, shard_y, metadata, args.sha256)
            save_time_total += time.time() - t_save
            new_records.append(rec)
            append_jsonl(output_dir / "manifest.jsonl", asdict(rec))
            del shard_z, shard_zf, shard_y

        all_records = scan_existing_shards(output_dir, strict_contiguous=True)
        total_final = sum(r.num_samples for r in all_records)
        summary = {
            "ok": total_final == target_total,
            "status": "complete" if total_final == target_total else "partial",
            "output_dir": str(output_dir),
            "crop_cache_dir": str(crop_dir),
            "target_total": target_total,
            "encoded_total": total_final,
            "new_encoded": new_encoded,
            "num_existing_shards_at_start": len(existing),
            "num_new_shards": len(new_records),
            "num_shards_total": len(all_records),
            "elapsed_sec": time.time() - start_t,
            "avg_new_samples_per_sec": new_encoded / (time.time() - start_t) if (time.time() - start_t) > 0 else None,
            "h2d_time_total_sec": h2d_time_total,
            "encode_time_total_sec": encode_time_total,
            "d2h_time_total_sec": d2h_time_total,
            "save_time_total_sec": save_time_total,
            "finished_utc": utc_now(),
            "shards_tail": [asdict(r) for r in all_records[-5:]],
            **cuda_mem(device),
        }
        write_json_atomic(output_dir / "progress.json", summary)
        write_json_atomic(output_dir / "build_summary.json", summary)
        print("BUILD_SUMMARY", json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary
    except Exception as exc:
        payload = write_progress("error", {"error": repr(exc)})
        print("ERROR_PROGRESS", json.dumps(payload, indent=2, sort_keys=True), flush=True)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crop-cache-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pae-config", required=True)
    p.add_argument("--pae-ckpt", required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--latent-shard-size", type=int, default=4096)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dtype", default="bf16")
    p.add_argument("--save-dtype", default="bf16")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sha256", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_latents(args)
    if not summary.get("ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
