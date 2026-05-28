#!/usr/bin/env python3
"""Fast verifier for PAE latent cache directories."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from safetensors import safe_open

REPO_ROOT = Path("/workspace/PDM")
PAE_ROOT = REPO_ROOT / "external" / "PAE" / "pae_with_generator"
sys.path.insert(0, str(PAE_ROOT))
from dataset.img_latent_dataset import ImgLatentDataset  # noqa: E402

_SHARD_RE = re.compile(r"shard(\d+)\.safetensors$")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def parse_idx(path: Path) -> int:
    m = _SHARD_RE.search(path.name)
    return int(m.group(1)) if m else -1


def scan(data_dir: Path, sha256: bool = False) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in sorted(data_dir.glob("*.safetensors"), key=parse_idx):
        with safe_open(str(p), framework="pt", device="cpu") as f:
            item: Dict[str, Any] = {"file": str(p), "shard_index": parse_idx(p), "keys": {}, "metadata": dict(f.metadata() or {})}
            for k in ("latents", "latents_flip", "labels"):
                sl = f.get_slice(k)
                item["keys"][k] = {"shape": list(sl.get_shape()), "dtype": str(sl.get_dtype())}
            labels = f.get_tensor("labels")
            item["num_samples"] = int(labels.shape[0])
            item["labels_min"] = int(labels.min().item())
            item["labels_max"] = int(labels.max().item())
        item["bytes"] = int(p.stat().st_size)
        if sha256:
            item["sha256"] = sha256_file(p)
        out.append(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--expected-total", type=int, default=None)
    ap.add_argument("--expected-latent-shape", default="32,16,16")
    ap.add_argument("--latent-norm", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--sample-items", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sha256", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    t0 = time.time()
    data_dir = Path(args.data_dir)
    shards = scan(data_dir, sha256=args.sha256)
    total = sum(s["num_samples"] for s in shards)
    expected_chw = [int(x) for x in args.expected_latent_shape.split(",")]

    contiguous = all(s["shard_index"] == i for i, s in enumerate(shards))
    shape_ok = True
    dtype_ok = True
    for s in shards:
        lat_shape = s["keys"]["latents"]["shape"]
        flip_shape = s["keys"]["latents_flip"]["shape"]
        labels_shape = s["keys"]["labels"]["shape"]
        n = s["num_samples"]
        shape_ok = shape_ok and lat_shape == flip_shape and lat_shape[0] == n and labels_shape == [n] and lat_shape[1:] == expected_chw
        dtype_ok = dtype_ok and s["keys"]["labels"]["dtype"] in ("I64", "torch.int64", "int64")

    sample_report: Dict[str, Any] = {}
    if args.sample_items and shards:
        ds = ImgLatentDataset(str(data_dir), latent_norm=args.latent_norm)
        indices = sorted(set([0, max(0, len(ds) // 2), max(0, len(ds) - 1)]))
        samples = []
        for idx in indices:
            x, y = ds[idx]
            samples.append({"idx": int(idx), "latent_shape": list(x.shape), "latent_dtype": str(x.dtype), "label": int(y.item())})
        sample_report = {"dataset_len": len(ds), "latent_norm": args.latent_norm, "samples": samples}

    ok = bool(shards) and contiguous and shape_ok and dtype_ok and (args.expected_total is None or total == args.expected_total)
    out = {
        "ok": ok,
        "data_dir": str(data_dir),
        "num_files": len(shards),
        "total_samples": total,
        "expected_total": args.expected_total,
        "contiguous_shards": contiguous,
        "shape_ok": shape_ok,
        "dtype_ok": dtype_ok,
        "total_bytes": sum(s["bytes"] for s in shards),
        "elapsed_sec": time.time() - t0,
        "sample_report": sample_report,
        "shards": shards,
    }
    text = json.dumps(out, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
