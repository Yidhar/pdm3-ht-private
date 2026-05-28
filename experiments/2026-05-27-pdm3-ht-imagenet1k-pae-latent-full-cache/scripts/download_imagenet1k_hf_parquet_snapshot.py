#!/usr/bin/env python3
"""Mirror Hugging Face ImageNet-1k parquet shards to local disk before PAE encoding.

This is the raw-first path: keep the original HF parquet files (JPEG bytes + labels)
locally, then run build_pae_latents_full_cache.py with --data-files-glob so the GPU
encoder is not coupled to WAN streaming.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def resolve_hf_token(token_env: str) -> Tuple[Optional[str], str]:
    token = os.environ.get(token_env) or None
    if token:
        return token, "env"
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:
        token = None
    if token:
        return token, "hf_cache"
    return None, "none"


def split_patterns(split: str) -> List[str]:
    return [f"data/{split}-*.parquet", "README.md", "classes.py"]


def list_expected(repo_id: str, split: str, token: Optional[str]) -> Dict[str, Any]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    wanted = f"data/{split}-*.parquet"
    files = []
    total = 0
    for item in api.list_repo_tree(repo_id, repo_type="dataset", path_in_repo="data", recursive=True):
        path = getattr(item, "path", "")
        if fnmatch.fnmatch(path, wanted):
            size = int(getattr(item, "size", 0) or 0)
            files.append({"path": path, "size": size})
            total += size
    files.sort(key=lambda x: x["path"])
    return {"split": split, "pattern": wanted, "num_files": len(files), "total_bytes": total, "files": files}


def scan_local(local_dir: Path, split: str) -> Dict[str, Any]:
    files = sorted((local_dir / "data").glob(f"{split}-*.parquet")) if (local_dir / "data").exists() else []
    rows = []
    total = 0
    for p in files:
        size = p.stat().st_size
        total += size
        rows.append({"path": str(p), "name": p.name, "size": size})
    return {"local_dir": str(local_dir), "split": split, "num_files": len(files), "total_bytes": total, "files": rows}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="ILSVRC/imagenet-1k")
    p.add_argument("--split", default="train")
    p.add_argument("--local-dir", default="/workspace/PDM/data/raw/imagenet1k_hf_parquet")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument("--dry-run", action="store_true", help="Only list expected files/sizes; do not download.")
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--manifest", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    local_dir = Path(args.local_dir)
    manifest_path = Path(args.manifest) if args.manifest else local_dir / f"{args.split}_snapshot_manifest.json"
    token, token_source = resolve_hf_token(args.hf_token_env)
    started = time.time()

    print(
        json.dumps(
            {
                "event": "start",
                "time_utc": utc_now(),
                "repo_id": args.repo_id,
                "split": args.split,
                "local_dir": str(local_dir),
                "max_workers": args.max_workers,
                "token_source": token_source,
                "token_set": bool(token),
                "dry_run": args.dry_run,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    expected = list_expected(args.repo_id, args.split, token)
    local_before = scan_local(local_dir, args.split)
    plan = {
        "event": "plan",
        "time_utc": utc_now(),
        "repo_id": args.repo_id,
        "split": args.split,
        "allow_patterns": split_patterns(args.split),
        "token_source": token_source,
        "token_set": bool(token),
        "expected": {k: v for k, v in expected.items() if k != "files"},
        "local_before": {k: v for k, v in local_before.items() if k != "files"},
        "expected_files_first5": expected["files"][:5],
        "expected_files_last5": expected["files"][-5:],
    }
    write_json_atomic(manifest_path, plan)
    print("DOWNLOAD_PLAN", json.dumps(plan, indent=2, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    from huggingface_hub import snapshot_download

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=split_patterns(args.split),
        local_dir=str(local_dir),
        token=token,
        max_workers=args.max_workers,
        force_download=args.force_download,
        local_files_only=args.local_files_only,
    )
    elapsed = time.time() - started
    local_after = scan_local(local_dir, args.split)
    ok = expected["num_files"] == local_after["num_files"] and expected["total_bytes"] == local_after["total_bytes"]
    summary = {
        "event": "finished",
        "ok": ok,
        "time_utc": utc_now(),
        "repo_id": args.repo_id,
        "split": args.split,
        "snapshot_path": snapshot_path,
        "local_dir": str(local_dir),
        "max_workers": args.max_workers,
        "token_source": token_source,
        "token_set": bool(token),
        "elapsed_sec": elapsed,
        "downloaded_or_verified_bytes_per_sec": local_after["total_bytes"] / elapsed if elapsed > 0 else None,
        "expected": {k: v for k, v in expected.items() if k != "files"},
        "local_after": {k: v for k, v in local_after.items() if k != "files"},
        "local_files_first5": local_after["files"][:5],
        "local_files_last5": local_after["files"][-5:],
    }
    write_json_atomic(manifest_path, summary)
    print("DOWNLOAD_SUMMARY", json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
