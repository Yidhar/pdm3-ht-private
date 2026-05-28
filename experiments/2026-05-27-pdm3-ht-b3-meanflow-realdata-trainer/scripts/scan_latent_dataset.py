#!/usr/bin/env python3
"""Scan a B3 real-data trainer config's latent dataset without building model/training."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_b3_meanflow_realdata import ShardIndexedLatentDataset, now_utc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--label-preview-count", type=int, default=1024)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    data = cfg["data"]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ds = ShardIndexedLatentDataset(
        data["data_path"],
        file_glob=data.get("file_glob", "*.safetensors"),
        flip_prob=float(data.get("flip_prob", 0.5)),
        latent_norm=False,
        latent_multiplier=float(data.get("latent_multiplier", 1.0)),
        max_shards=data.get("max_shards"),
        skip_unreadable=bool(data.get("skip_unreadable", True)),
    )
    full_snapshot = ds.summary(include_shards=len(ds.shards))
    summary = ds.summary(include_shards=8)
    summary.update(
        {
            "created_at_utc": now_utc(),
            "scan_elapsed_sec": time.perf_counter() - started,
            "expected_total_from_config": data.get("expected_total"),
            "ready_for_full_train": ds.total == data.get("expected_total"),
            "label_preview": ds.label_preview(args.label_preview_count),
        }
    )
    (out / "dataset_snapshot.json").write_text(json.dumps(full_snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "scan_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
