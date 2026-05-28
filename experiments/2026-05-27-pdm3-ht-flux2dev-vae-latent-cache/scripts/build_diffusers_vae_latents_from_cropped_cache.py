#!/usr/bin/env python3
"""Build ImageNet latent shards from cropped uint8 safetensors using a diffusers VAE.

Primary target for this experiment:
  - diffusers.AutoencoderKLFlux2 from diffusers/FLUX.2-dev-bnb-4bit, subfolder=vae

The script intentionally mirrors the PAE latent cache schema:
  - latents:      [N,C,H,W] bf16/fp32/fp16
  - latents_flip: [N,C,H,W]
  - labels:       [N] int64

Input is the reusable ADM-cropped uint8 safetensors cache:
  - images:         uint8 [N,3,256,256]
  - labels:         int64 [N]
  - source_indices: int64 [N]

Default latent value is deterministic posterior.mode(), not posterior.sample(), to keep
cache reproducible and comparable to deterministic PAE encodes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from safetensors import safe_open

# Reuse battle-tested shard/progress helpers from the existing PAE cache builder.
DEFAULT_COMMON_SCRIPT_DIR = Path("/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-full-cache/scripts")
if str(DEFAULT_COMMON_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(DEFAULT_COMMON_SCRIPT_DIR))

from build_pae_latents_full_cache import (  # noqa: E402
    PendingShardBuffer,
    append_jsonl,
    cuda_mem,
    cuda_reset_peak,
    cuda_sync,
    dtype_from_name,
    fadvise_dontneed,
    save_shard_atomic,
    scan_existing_shards,
    seconds_to_hms,
    write_json_atomic,
)

_CROP_SHARD_RE = re.compile(r"images_uint8_shard(\d+)\.safetensors$")


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
    m = _CROP_SHARD_RE.search(path.name)
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
        for expected_idx, shard in enumerate(shards):
            if shard.shard_index != expected_idx:
                raise RuntimeError(
                    f"Non-contiguous crop shard index: expected {expected_idx}, got {shard.shard_index} at {shard.path}"
                )
            if shard.source_index_first != expected_source:
                raise RuntimeError(
                    f"Non-contiguous source index: expected {expected_source}, got {shard.source_index_first} at {shard.path}"
                )
            expected_source += shard.num_samples
            if shard.source_index_last != expected_source - 1:
                raise RuntimeError(
                    f"Bad source_index_last in {shard.path}: expected {expected_source - 1}, got {shard.source_index_last}"
                )
    return shards


def iter_cropped_batches(
    crop_shards: Sequence[CropShard],
    *,
    start_index: int,
    target_total: int,
    batch_size: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    """Yield contiguous uint8 image/label batches from crop shards."""
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


def clear_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = (
        "latents_rank00_shard*.safetensors",
        ".latents_rank00_shard*.tmp.*",
        "manifest.jsonl",
        "progress.jsonl",
        "progress.json",
        "build_summary.json",
        "run_config.json",
    )
    for pat in patterns:
        for p in output_dir.glob(pat):
            if p.is_file():
                p.unlink()


def import_diffusers_class(class_name: str):
    diffusers = importlib.import_module("diffusers")
    if not hasattr(diffusers, class_name):
        raise AttributeError(f"diffusers has no class {class_name!r}; installed diffusers={getattr(diffusers, '__version__', '?')}")
    return getattr(diffusers, class_name), getattr(diffusers, "__version__", "?")


def resolve_input_layout(args: argparse.Namespace) -> str:
    if args.input_layout != "auto":
        return args.input_layout
    # Qwen Image VAE implementation in diffusers uses a 5D video-style tensor [B,C,F,H,W].
    if "Qwen" in args.vae_class:
        return "video5d"
    return "image4d"


def load_diffusers_vae(args: argparse.Namespace, device: torch.device, model_dtype: torch.dtype):
    cls, diffusers_version = import_diffusers_class(args.vae_class)
    kwargs: Dict[str, Any] = {}
    if args.subfolder:
        kwargs["subfolder"] = args.subfolder
    if args.revision:
        kwargs["revision"] = args.revision
    if args.local_files_only:
        kwargs["local_files_only"] = True
    # For VAE cache quality, fp32 is the default. Non-fp32 is still supported for engineering tests.
    kwargs["torch_dtype"] = model_dtype
    print(
        "Loading diffusers VAE",
        json.dumps(
            {
                "repo_id": args.repo_id,
                "subfolder": args.subfolder,
                "vae_class": args.vae_class,
                "model_dtype": str(model_dtype),
                "device": str(device),
                "diffusers_version": diffusers_version,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    vae = cls.from_pretrained(args.repo_id, **kwargs)
    vae = vae.to(device=device, dtype=model_dtype).eval()
    vae.requires_grad_(False)
    return vae, diffusers_version


def prepare_vae_input(x4: torch.Tensor, input_layout: str) -> torch.Tensor:
    if input_layout == "image4d":
        return x4
    if input_layout == "video5d":
        return x4.unsqueeze(2)  # [B,C,H,W] -> [B,C,F=1,H,W]
    raise ValueError(f"Unsupported input_layout={input_layout!r}")


def postprocess_latent(z: torch.Tensor, *, squeeze_single_frame: bool) -> torch.Tensor:
    if z.ndim == 5 and squeeze_single_frame:
        if z.shape[2] != 1:
            raise RuntimeError(f"Cannot squeeze latent frame dim because shape is {tuple(z.shape)}")
        z = z.squeeze(2)
    if z.ndim != 4:
        raise RuntimeError(f"Expected cached latent to be 4D [B,C,H,W], got shape {tuple(z.shape)}")
    return z


def posterior_to_latent(latent_dist: Any, latent_mode: str, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if latent_mode == "mode":
        return latent_dist.mode()
    if latent_mode == "sample":
        try:
            return latent_dist.sample(generator=generator)
        except TypeError:
            return latent_dist.sample()
    raise ValueError(f"Unsupported latent_mode={latent_mode!r}")


def encode_one(
    vae: torch.nn.Module,
    x4: torch.Tensor,
    *,
    input_layout: str,
    latent_mode: str,
    squeeze_single_frame: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    xin = prepare_vae_input(x4, input_layout)
    out = vae.encode(xin)
    z = posterior_to_latent(out.latent_dist, latent_mode, generator=generator)
    z = postprocess_latent(z, squeeze_single_frame=squeeze_single_frame)
    return z


def encode_pair_from_u8_batch(
    vae: torch.nn.Module,
    x_u8: torch.Tensor,
    device: torch.device,
    model_dtype: torch.dtype,
    save_dtype: torch.dtype,
    *,
    input_layout: str,
    latent_mode: str,
    squeeze_single_frame: bool,
    generator: Optional[torch.Generator],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Move uint8 batch once, map [0,255] -> [-1,1], encode original and horizontal flip."""
    if device.type == "cuda":
        cuda_sync(device)
    t0 = time.time()
    x = x_u8.to(device=device, non_blocking=True).to(dtype=torch.float32).div_(127.5).sub_(1.0)
    if model_dtype != torch.float32:
        x = x.to(dtype=model_dtype)
    if device.type == "cuda":
        cuda_sync(device)
    t_h2d = time.time() - t0

    if device.type == "cuda":
        cuda_sync(device)
    t1 = time.time()
    with torch.inference_mode():
        z = encode_one(
            vae,
            x,
            input_layout=input_layout,
            latent_mode=latent_mode,
            squeeze_single_frame=squeeze_single_frame,
            generator=generator,
        )
        z_shape = list(z.shape)
        z_dtype = str(z.dtype)
        z_mean = float(z.detach().float().mean().item())
        z_std = float(z.detach().float().std(unbiased=False).item())
        # Copy original latent to CPU before flipped encode to reduce peak GPU memory.
        z_cpu = z.detach().to(dtype=save_dtype, device="cpu").contiguous()
        del z

        xf = torch.flip(x, dims=[-1])
        zf = encode_one(
            vae,
            xf,
            input_layout=input_layout,
            latent_mode=latent_mode,
            squeeze_single_frame=squeeze_single_frame,
            generator=generator,
        )
        zf_shape = list(zf.shape)
        if zf_shape != z_shape:
            raise RuntimeError(f"Flip latent shape mismatch: original={z_shape}, flip={zf_shape}")
        zf_mean = float(zf.detach().float().mean().item())
        zf_std = float(zf.detach().float().std(unbiased=False).item())
        zf_cpu = zf.detach().to(dtype=save_dtype, device="cpu").contiguous()
        del zf, xf
    if device.type == "cuda":
        cuda_sync(device)
    t_encode = time.time() - t1

    # z_cpu/zf_cpu copies are synchronized above. Keep timing field for schema comparability.
    t_d2h = 0.0
    del x
    return z_cpu, zf_cpu, {
        "h2d_sec": t_h2d,
        "encode_sec": t_encode,
        "d2h_sec": t_d2h,
        "latent_shape": z_shape,
        "latent_dtype_before_save": z_dtype,
        "latent_mean": z_mean,
        "latent_std": z_std,
        "latent_flip_mean": zf_mean,
        "latent_flip_std": zf_std,
    }


def build_latents(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    crop_dir = Path(args.crop_cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        print(f"OVERWRITE enabled: clearing output artifacts in {output_dir}", flush=True)
        clear_output_dir(output_dir)

    crop_shards = scan_crop_shards(crop_dir, strict_contiguous=not args.allow_noncontiguous_crop)
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

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available; pass --allow-cpu only for debugging.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = dtype_from_name(args.model_dtype)
    save_dtype = dtype_from_name(args.save_dtype)
    input_layout = resolve_input_layout(args)
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
        "output_dir": str(output_dir),
        "existing_latent_shards": len(existing),
        "completed_at_start": completed_at_start,
        "next_shard_index": next_shard_index,
        "input_layout_resolved": input_layout,
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

    vae, diffusers_version = load_diffusers_vae(args, device, model_dtype)
    vae_config = {}
    try:
        vae_config = dict(getattr(vae, "config", {}) or {})
    except Exception:
        vae_config = {}

    generator: Optional[torch.Generator] = None
    if args.latent_mode == "sample":
        generator = torch.Generator(device=device)
        generator.manual_seed(args.sample_seed)

    metadata = {
        "schema": "diffusers_vae_latent_from_cropped_uint8_v1",
        "crop_cache_dir": str(crop_dir),
        "crop_schema": "imagenet_cropped_uint8_v1",
        "source_loader": "cropped_uint8_safetensors",
        "vae_backend": args.backend_name,
        "repo_id": args.repo_id,
        "subfolder": args.subfolder or "",
        "revision": args.revision or "",
        "diffusers_class": args.vae_class,
        "diffusers_version": str(diffusers_version),
        "latent_mode": args.latent_mode,
        "input_layout": input_layout,
        "input_preprocess": "uint8 [0,255] -> float [-1,1] after ADM center crop",
        "model_dtype": str(model_dtype),
        "save_dtype": str(save_dtype),
        "image_size": str(args.image_size),
        "squeeze_single_frame": str(bool(args.squeeze_single_frame)),
        "vae_config_json": json.dumps(vae_config, sort_keys=True, default=str)[:16000],
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
    save_time_total = 0.0
    first_latent_shape: Optional[List[int]] = None
    last_batch_timing: Optional[Dict[str, Any]] = None

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
            "vae_backend": args.backend_name,
            "repo_id": args.repo_id,
            "diffusers_class": args.vae_class,
            "latent_mode": args.latent_mode,
            "input_layout": input_layout,
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
            "first_latent_shape": first_latent_shape,
            "h2d_time_total_sec": h2d_time_total,
            "encode_time_total_sec": encode_time_total,
            "d2h_time_total_sec": d2h_time_total,
            "save_time_total_sec": save_time_total,
            "last_batch_timing": last_batch_timing,
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
        batch_iter = iter_cropped_batches(
            crop_shards,
            start_index=completed_at_start,
            target_total=target_total,
            batch_size=args.batch_size,
        )
        for x_u8, y, batch_meta in batch_iter:
            if encoded_total >= target_total:
                break
            batch_index += 1
            t_batch = time.time()
            z, zf, timing = encode_pair_from_u8_batch(
                vae,
                x_u8,
                device,
                model_dtype,
                save_dtype,
                input_layout=input_layout,
                latent_mode=args.latent_mode,
                squeeze_single_frame=args.squeeze_single_frame,
                generator=generator,
            )
            batch_elapsed = time.time() - t_batch
            if first_latent_shape is None:
                first_latent_shape = list(z.shape)
                if args.expected_latent_channels is not None and first_latent_shape[1] != args.expected_latent_channels:
                    raise RuntimeError(
                        f"Expected latent channels {args.expected_latent_channels}, got shape {first_latent_shape}"
                    )
                if args.expected_latent_hw:
                    h, w = map(int, args.expected_latent_hw.lower().split("x"))
                    if first_latent_shape[-2:] != [h, w]:
                        raise RuntimeError(f"Expected latent HW {[h,w]}, got shape {first_latent_shape}")
            h2d_time_total += float(timing["h2d_sec"])
            encode_time_total += float(timing["encode_sec"])
            d2h_time_total += float(timing["d2h_sec"])
            last_batch_timing = timing

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
                    "last_crop_batch_meta": batch_meta,
                    "saved_shards_this_batch": saved_now,
                },
            )
            print(
                "DIFFUSERS_VAE_PROGRESS",
                json.dumps(
                    {
                        k: progress.get(k)
                        for k in [
                            "encoded_total",
                            "target_total",
                            "new_encoded",
                            "avg_new_samples_per_sec",
                            "interval_samples_per_sec",
                            "eta_hms",
                            "pending_shard_samples",
                            "num_new_shards",
                            "first_latent_shape",
                            "cuda_peak_allocated_mb",
                        ]
                    },
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
            "vae_backend": args.backend_name,
            "repo_id": args.repo_id,
            "diffusers_class": args.vae_class,
            "latent_mode": args.latent_mode,
            "input_layout": input_layout,
            "first_latent_shape": first_latent_shape,
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
            "last_batch_timing": last_batch_timing,
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
    p.add_argument("--repo-id", required=True)
    p.add_argument("--subfolder", default="vae")
    p.add_argument("--revision", default=None)
    p.add_argument("--vae-class", required=True, help="diffusers class name, e.g. AutoencoderKLFlux2")
    p.add_argument("--backend-name", default="diffusers_vae")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--latent-shard-size", type=int, default=4096)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dtype", default="fp32")
    p.add_argument("--save-dtype", default="bf16")
    p.add_argument("--latent-mode", choices=("mode", "sample"), default="mode")
    p.add_argument("--sample-seed", type=int, default=1234)
    p.add_argument("--input-layout", choices=("auto", "image4d", "video5d"), default="auto")
    p.add_argument("--squeeze-single-frame", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--expected-latent-channels", type=int, default=None)
    p.add_argument("--expected-latent-hw", default=None, help="Optional HxW check, e.g. 32x32")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sha256", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--allow-noncontiguous-crop", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_latents(args)
    if not summary.get("ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
