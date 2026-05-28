#!/usr/bin/env python3
"""Check that local ImageNet parquet stream order matches already-produced latent shards."""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from safetensors import safe_open


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def shard_files(latent_dir: Path) -> List[Path]:
    return sorted(latent_dir.glob("latents_rank00_shard*.safetensors"))


def load_cache_labels(latent_dir: Path, head_n: int, tail_n: int) -> Dict[str, Any]:
    files = shard_files(latent_dir)
    total = 0
    head: List[int] = []
    tail: List[int] = []
    per_shard = []
    for p in files:
        with safe_open(str(p), framework="pt", device="cpu") as f:
            labels = f.get_tensor("labels").to("cpu")
        labels_list = [int(x) for x in labels.tolist()]
        n = len(labels_list)
        per_shard.append({"path": str(p), "name": p.name, "num_samples": n})
        total += n
        if len(head) < head_n:
            head.extend(labels_list[: max(0, head_n - len(head))])
        tail = (tail + labels_list)[-tail_n:]
    return {
        "num_shards": len(files),
        "durable_samples": total,
        "first_shard": files[0].name if files else None,
        "last_shard": files[-1].name if files else None,
        "head_labels": head[:head_n],
        "tail_labels": tail[-tail_n:],
        "per_shard_first3": per_shard[:3],
        "per_shard_last3": per_shard[-3:],
    }


def load_parquet_labels(data_files_glob: str, split: str, start: int, count: int) -> List[int]:
    """Read only the `label` column from local parquet files.

    Avoid datasets.Image decoding here: this audit should be an order check, not
    another expensive JPEG pipeline.
    """
    import pyarrow.parquet as pq

    files = sorted(glob.glob(data_files_glob))
    if not files:
        raise FileNotFoundError(f"No files match {data_files_glob}")
    labels: List[int] = []
    global_pos = 0
    remaining_start = int(start)
    need = int(count)
    if need <= 0:
        return labels
    for file_path in files:
        pf = pq.ParquetFile(file_path)
        n_rows = pf.metadata.num_rows
        if global_pos + n_rows <= remaining_start:
            global_pos += n_rows
            continue
        local_start = max(0, remaining_start - global_pos)
        local_end = min(n_rows, local_start + need - len(labels))
        if local_end > local_start:
            table = pf.read(columns=["label"])
            col = table.column("label").to_pylist()
            labels.extend(int(x) for x in col[local_start:local_end])
        global_pos += n_rows
        if len(labels) >= need:
            break
    return labels[:need]


def parquet_row_count(data_files_glob: str) -> int:
    import pyarrow.parquet as pq

    files = sorted(glob.glob(data_files_glob))
    return sum(pq.ParquetFile(p).metadata.num_rows for p in files)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latent-dir", required=True)
    p.add_argument("--data-files-glob", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--head-n", type=int, default=64)
    p.add_argument("--tail-n", type=int, default=64)
    p.add_argument("--next-n", type=int, default=16)
    p.add_argument("--output-json", required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()
    cache = load_cache_labels(Path(args.latent_dir), args.head_n, args.tail_n)
    durable = int(cache["durable_samples"])
    parquet_files = sorted(glob.glob(args.data_files_glob))
    total_parquet_rows = parquet_row_count(args.data_files_glob)
    parquet_head = load_parquet_labels(args.data_files_glob, args.split, 0, args.head_n)
    tail_start = max(0, durable - args.tail_n)
    parquet_tail = load_parquet_labels(args.data_files_glob, args.split, tail_start, min(args.tail_n, durable)) if durable > 0 else []
    parquet_next = load_parquet_labels(args.data_files_glob, args.split, durable, args.next_n)
    head_match = cache["head_labels"] == parquet_head[: len(cache["head_labels"])]
    tail_match = cache["tail_labels"] == parquet_tail[-len(cache["tail_labels"]):] if cache["tail_labels"] else True
    payload = {
        "ok": bool(head_match and tail_match and durable > 0),
        "time_utc": utc_now(),
        "elapsed_sec": time.time() - t0,
        "latent_dir": args.latent_dir,
        "data_files_glob": args.data_files_glob,
        "parquet_files_count": len(parquet_files),
        "parquet_total_rows": total_parquet_rows,
        "parquet_files_first3": parquet_files[:3],
        "parquet_files_last3": parquet_files[-3:],
        "cache": {k: v for k, v in cache.items() if k not in ["head_labels", "tail_labels"]},
        "head_match": head_match,
        "tail_match_at_durable_boundary": tail_match,
        "durable_samples": durable,
        "tail_start": tail_start,
        "cache_head_labels": cache["head_labels"],
        "parquet_head_labels": parquet_head,
        "cache_tail_labels": cache["tail_labels"],
        "parquet_tail_labels": parquet_tail,
        "parquet_next_labels_after_durable": parquet_next,
    }
    write_json_atomic(Path(args.output_json), payload)
    print("LOCAL_PARQUET_ORDER_AUDIT", json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0 if payload["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
