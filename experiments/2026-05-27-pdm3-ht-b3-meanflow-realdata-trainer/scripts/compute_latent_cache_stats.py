#!/usr/bin/env python3
"""Streaming mean/std scan for safetensors latent caches.

This is intentionally independent from the trainer's on-demand 10k-sample
``latents_stats.pt`` helper.  It is used for route diagnostics where we need to
know the actual latent scale in a large cache without materializing the cache in
RAM.

Expected shard schema:

* ``latents``:      [N, C, H, W]
* ``latents_flip``: [N, C, H, W] (optional)
* ``labels``:       [N]

Outputs:

* JSON summary with global and per-channel mean/std/min/max/RMS.
* Optional PyTorch stats file compatible with
  ``ShardIndexedLatentDataset(latent_norm=True, latent_stats_path=...)``.  When
  multiple keys are requested, the PT file uses the merged stats across all
  requested keys.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from safetensors import safe_open


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunningStats:
    """Float64 streaming stats for [N,C,H,W] tensors."""

    name: str
    count: int = 0
    sample_count: int = 0
    sum_: float = 0.0
    sumsq: float = 0.0
    min_: float = math.inf
    max_: float = -math.inf
    per_channel_sum: Optional[torch.Tensor] = None
    per_channel_sumsq: Optional[torch.Tensor] = None
    per_channel_count: int = 0
    shape_first: Optional[List[int]] = None
    shape_last: Optional[List[int]] = None

    def update(self, x: torch.Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"{self.name}: expected [N,C,H,W], got {list(x.shape)}")
        if self.shape_first is None:
            self.shape_first = list(x.shape)
        self.shape_last = list(x.shape)

        # Convert one chunk at a time; keep reductions in float64 to avoid the
        # accumulated full-cache stats being dominated by fp32 summation error.
        xd = x.to(dtype=torch.float64)
        self.count += int(xd.numel())
        self.sample_count += int(xd.shape[0])
        self.sum_ += float(xd.sum().item())
        self.sumsq += float((xd * xd).sum().item())
        self.min_ = min(self.min_, float(xd.min().item()))
        self.max_ = max(self.max_, float(xd.max().item()))

        c = int(xd.shape[1])
        if self.per_channel_sum is None:
            self.per_channel_sum = torch.zeros(c, dtype=torch.float64)
            self.per_channel_sumsq = torch.zeros(c, dtype=torch.float64)
        assert self.per_channel_sumsq is not None
        self.per_channel_sum += xd.sum(dim=(0, 2, 3)).cpu()
        self.per_channel_sumsq += (xd * xd).sum(dim=(0, 2, 3)).cpu()
        self.per_channel_count += int(xd.shape[0] * xd.shape[2] * xd.shape[3])

    def merge_from(self, other: "RunningStats") -> None:
        if other.count == 0:
            return
        if self.count == 0:
            self.shape_first = other.shape_first
        self.shape_last = other.shape_last
        self.count += other.count
        self.sample_count += other.sample_count
        self.sum_ += other.sum_
        self.sumsq += other.sumsq
        self.min_ = min(self.min_, other.min_)
        self.max_ = max(self.max_, other.max_)
        if other.per_channel_sum is not None:
            if self.per_channel_sum is None:
                self.per_channel_sum = torch.zeros_like(other.per_channel_sum)
                self.per_channel_sumsq = torch.zeros_like(other.per_channel_sumsq)
            assert self.per_channel_sumsq is not None and other.per_channel_sumsq is not None
            self.per_channel_sum += other.per_channel_sum
            self.per_channel_sumsq += other.per_channel_sumsq
            self.per_channel_count += other.per_channel_count

    def as_dict(self) -> Dict[str, Any]:
        if self.count <= 0:
            return {"name": self.name, "count": 0, "sample_count": 0}
        mean = self.sum_ / self.count
        var = max(0.0, self.sumsq / self.count - mean * mean)
        std = math.sqrt(var)
        rms = math.sqrt(max(0.0, self.sumsq / self.count))
        assert self.per_channel_sum is not None and self.per_channel_sumsq is not None
        ch_mean = (self.per_channel_sum / self.per_channel_count).numpy()
        ch_var = self.per_channel_sumsq.numpy() / self.per_channel_count - ch_mean * ch_mean
        ch_std = np.sqrt(np.maximum(ch_var, 0.0))
        return {
            "name": self.name,
            "sample_count": int(self.sample_count),
            "count": int(self.count),
            "shape_first": self.shape_first,
            "shape_last": self.shape_last,
            "mean": float(mean),
            "std": float(std),
            "rms": float(rms),
            "min": float(self.min_),
            "max": float(self.max_),
            "absmax": float(max(abs(self.min_), abs(self.max_))),
            "per_channel_count": int(self.per_channel_count),
            "per_channel_mean": [float(v) for v in ch_mean.tolist()],
            "per_channel_std": [float(v) for v in ch_std.tolist()],
            "per_channel_mean_summary": summary(ch_mean),
            "per_channel_std_summary": summary(ch_std),
        }

    def trainer_stats(self) -> Dict[str, torch.Tensor]:
        d = self.as_dict()
        mean = torch.tensor(d["per_channel_mean"], dtype=torch.float32).view(1, -1, 1, 1)
        std = torch.tensor(d["per_channel_std"], dtype=torch.float32).view(1, -1, 1, 1)
        return {
            "mean": mean,
            "std": std,
            "num_samples": int(self.sample_count),
            "created_at_utc": utc_now(),
        }


def summary(arr: np.ndarray) -> Dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def parse_keys(s: str) -> List[str]:
    keys = [x.strip() for x in s.split(",") if x.strip()]
    if not keys:
        raise argparse.ArgumentTypeError("at least one key is required")
    return keys


def parse_shard_indices(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [p.strip() for p in part.split(":")]
            if len(pieces) not in {2, 3}:
                raise argparse.ArgumentTypeError(f"bad shard slice {part!r}")
            start = int(pieces[0])
            stop = int(pieces[1])
            step = int(pieces[2]) if len(pieces) == 3 and pieces[2] else 1
            out.extend(range(start, stop, step))
        else:
            out.append(int(part))
    return out


def iter_shards(
    cache_dir: Path,
    file_glob: str,
    max_shards: Optional[int],
    shard_indices: Optional[List[int]],
    linspace_shards: Optional[int],
) -> List[Path]:
    files = sorted(cache_dir.glob(file_glob))
    if shard_indices is not None:
        files = [files[i] for i in shard_indices]
    if linspace_shards is not None:
        n = min(int(linspace_shards), len(files))
        if n > 0:
            if n == 1:
                idx = [0]
            else:
                idx = sorted(set(int(round(x)) for x in np.linspace(0, len(files) - 1, num=n)))
            files = [files[i] for i in idx]
    if max_shards is not None:
        files = files[: int(max_shards)]
    if not files:
        raise FileNotFoundError(f"no files matching {file_glob!r} in {cache_dir}")
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", required=True, type=Path)
    ap.add_argument("--file-glob", default="*.safetensors")
    ap.add_argument("--keys", type=parse_keys, default=parse_keys("latents"))
    ap.add_argument("--shard-indices", type=parse_shard_indices, default=None, help="comma list/slices, e.g. 0,10,20 or 0:100:10")
    ap.add_argument("--linspace-shards", type=int, default=None, help="scan this many shards evenly spaced across the matched file list")
    ap.add_argument("--max-shards", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None, help="stop after this many samples per key")
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--output-json", required=True, type=Path)
    ap.add_argument("--output-trainer-stats-pt", default=None, type=Path)
    ap.add_argument("--progress-every", type=int, default=8)
    args = ap.parse_args()

    cache_dir = args.cache_dir.resolve()
    files = iter_shards(cache_dir, args.file_glob, args.max_shards, args.shard_indices, args.linspace_shards)
    keys: List[str] = args.keys
    per_key = {key: RunningStats(key) for key in keys}
    merged = RunningStats("merged_" + "_".join(keys))
    started = time.perf_counter()
    processed_shards = 0

    for shard_i, path in enumerate(files):
        with safe_open(str(path), framework="pt", device="cpu") as f:
            present = set(f.keys())
            for key in keys:
                if key not in present:
                    raise KeyError(f"{path} missing key {key!r}; present={sorted(present)}")
            for key in keys:
                shape = list(f.get_slice(key).get_shape())
                n = int(shape[0])
                already = per_key[key].sample_count
                if args.max_samples is not None and already >= int(args.max_samples):
                    continue
                take_n = n
                if args.max_samples is not None:
                    take_n = min(take_n, int(args.max_samples) - already)
                for start in range(0, take_n, int(args.chunk_size)):
                    end = min(take_n, start + int(args.chunk_size))
                    x = f.get_slice(key)[start:end]
                    per_key[key].update(x)
                    del x
        processed_shards += 1
        if args.progress_every > 0 and (
            processed_shards == 1 or processed_shards % int(args.progress_every) == 0 or processed_shards == len(files)
        ):
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "processed_shards": processed_shards,
                        "total_shards_planned": len(files),
                        "samples_per_key": {k: v.sample_count for k, v in per_key.items()},
                        "elapsed_sec": elapsed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    for stats in per_key.values():
        merged.merge_from(stats)

    elapsed = time.perf_counter() - started
    record: Dict[str, Any] = {
        "status": "ok",
        "created_at_utc": utc_now(),
        "cache_dir": str(cache_dir),
        "file_glob": args.file_glob,
        "keys": keys,
        "shard_indices": args.shard_indices,
        "linspace_shards": args.linspace_shards,
        "max_shards": args.max_shards,
        "max_samples": args.max_samples,
        "chunk_size": int(args.chunk_size),
        "num_shards_planned": len(files),
        "num_shards_processed": processed_shards,
        "elapsed_sec": elapsed,
        "stats_by_key": {key: stats.as_dict() for key, stats in per_key.items()},
        "merged_stats": merged.as_dict(),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.output_trainer_stats_pt is not None:
        args.output_trainer_stats_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(merged.trainer_stats(), args.output_trainer_stats_pt)
        record["output_trainer_stats_pt"] = str(args.output_trainer_stats_pt)
        args.output_json.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
