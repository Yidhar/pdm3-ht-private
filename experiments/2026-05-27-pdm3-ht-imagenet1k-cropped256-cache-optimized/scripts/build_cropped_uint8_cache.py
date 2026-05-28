#!/usr/bin/env python3
"""Build a reusable cropped uint8 ImageNet cache from local HF parquet.

Input:  local parquet files with schema image{bytes,path}, label.
Output: safetensors shards with:
  - images: uint8 [N,3,H,W]
  - labels: int64 [N]
  - source_indices: int64 [N]

This intentionally avoids HuggingFace datasets' sample-by-sample PIL path during
PAE encoding. JPEG decode + ADM center crop are amortized once and the resulting
uint8 cache can be reused for PAE latent construction, FID real stats, MMD, and
visualization.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import glob
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Avoid multiplicative thread oversubscription inside many decode processes.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageFile
from safetensors import safe_open
from safetensors.torch import save_file

ImageFile.LOAD_TRUNCATED_IMAGES = True


_SHARD_RE = re.compile(r"images_uint8_shard(\d+)\.safetensors$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def seconds_to_hms(sec: Optional[float]) -> Optional[str]:
    if sec is None or sec < 0 or not np.isfinite(sec):
        return None
    s = int(sec)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")
        f.flush()


def sha256_file(path: Path, chunk_size: int = 32 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fadvise_dontneed(path: Path) -> None:
    """Best-effort page-cache eviction for large clean files.

    The container cgroup counts page cache against memory.max.  Without this,
    repeated ~805MB safetensors writes and parquet reads can accumulate clean
    cache and trigger a cgroup OOM even when host RAM is plentiful.
    """
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        return
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except Exception:
        return


@dataclass
class CropShardRecord:
    shard_index: int
    path: str
    num_samples: int
    image_shape: List[int]
    image_dtype: str
    labels_min: int
    labels_max: int
    source_index_first: int
    source_index_last: int
    bytes: int
    save_seconds: float
    sha256: Optional[str]
    verify_ok: bool
    created_utc: str


def shard_path_for(output_dir: Path, shard_index: int) -> Path:
    return output_dir / f"images_uint8_shard{shard_index:06d}.safetensors"


def parse_shard_index(path: Path) -> int:
    m = _SHARD_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse shard index from {path.name}")
    return int(m.group(1))


def verify_crop_safetensors(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "ok": False, "keys": {}, "metadata": {}}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        try:
            info["metadata"] = dict(f.metadata() or {})
        except Exception:
            info["metadata"] = {}
        for key in ("images", "labels", "source_indices"):
            sl = f.get_slice(key)
            info["keys"][key] = {"shape": list(sl.get_shape()), "dtype": str(sl.get_dtype())}
    n = int(info["keys"]["labels"]["shape"][0])
    ok = n > 0
    ok = ok and info["keys"]["source_indices"]["shape"] == [n]
    ok = ok and info["keys"]["images"]["shape"][0] == n
    ok = ok and info["keys"]["images"]["dtype"] in ("U8", "UINT8", "torch.uint8", "uint8")
    info["ok"] = bool(ok)
    info["num_samples"] = n
    return info


def scan_existing_crop_shards(output_dir: Path, strict_contiguous: bool = True) -> List[CropShardRecord]:
    records: List[CropShardRecord] = []
    for path in sorted(output_dir.glob("images_uint8_shard*.safetensors")):
        idx = parse_shard_index(path)
        info = verify_crop_safetensors(path)
        if not info["ok"]:
            raise RuntimeError(f"Existing crop shard failed verification: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            labels = f.get_tensor("labels")
            source_indices = f.get_tensor("source_indices")
            img_shape = list(f.get_slice("images").get_shape())
            img_dtype = str(f.get_slice("images").get_dtype())
            meta = dict(f.metadata() or {})
        records.append(
            CropShardRecord(
                shard_index=idx,
                path=str(path),
                num_samples=int(labels.numel()),
                image_shape=img_shape,
                image_dtype=img_dtype,
                labels_min=int(labels.min().item()),
                labels_max=int(labels.max().item()),
                source_index_first=int(source_indices[0].item()),
                source_index_last=int(source_indices[-1].item()),
                bytes=int(path.stat().st_size),
                save_seconds=0.0,
                sha256=None,
                verify_ok=True,
                created_utc=str(meta.get("created_utc", "")),
            )
        )
    records.sort(key=lambda r: r.shard_index)
    if strict_contiguous:
        expected_source = 0
        for expected_shard, rec in enumerate(records):
            if rec.shard_index != expected_shard:
                raise RuntimeError(
                    f"Non-contiguous crop shard indices in {output_dir}: expected {expected_shard}, got {rec.shard_index}"
                )
            if rec.source_index_first != expected_source:
                raise RuntimeError(
                    f"Non-contiguous source indices in {path}: expected first {expected_source}, got {rec.source_index_first}"
                )
            expected_source += rec.num_samples
            if rec.source_index_last != expected_source - 1:
                raise RuntimeError(
                    f"Bad source_index_last in {path}: expected {expected_source - 1}, got {rec.source_index_last}"
                )
    return records


def clear_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pat in ("images_uint8_shard*.safetensors", ".images_uint8_shard*.tmp.*", "manifest.jsonl", "progress.jsonl", "progress.json", "build_summary.json", "run_config.json"):
        for p in output_dir.glob(pat):
            if p.is_file():
                p.unlink()


def adm_center_crop_uint8_chw_from_bytes(image_bytes: bytes, image_size: int) -> np.ndarray:
    with Image.open(io.BytesIO(image_bytes)) as pil_image:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        else:
            # Force JPEG decode while the file handle is alive.
            pil_image.load()

        while min(*pil_image.size) >= 2 * image_size:
            pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX)

        scale = image_size / min(*pil_image.size)
        pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC)
        arr = np.array(pil_image, dtype=np.uint8, copy=True)

    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    arr = np.ascontiguousarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size, :])
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def _worker_init() -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = "1"
    ImageFile.LOAD_TRUNCATED_IMAGES = True


# record tuple: (source_index, jpeg_bytes, label, path)
def decode_chunk(task: Tuple[int, List[Tuple[int, bytes, int, str]], int]) -> Dict[str, Any]:
    chunk_index, records, image_size = task
    t0 = time.time()
    n = len(records)
    images = np.empty((n, 3, image_size, image_size), dtype=np.uint8)
    labels = np.empty((n,), dtype=np.int64)
    source_indices = np.empty((n,), dtype=np.int64)
    input_bytes = 0
    paths_first_last: List[str] = []
    for j, (source_index, jpeg_bytes, label, path) in enumerate(records):
        images[j] = adm_center_crop_uint8_chw_from_bytes(jpeg_bytes, image_size)
        labels[j] = label
        source_indices[j] = source_index
        input_bytes += len(jpeg_bytes)
        if j == 0 or j == n - 1:
            paths_first_last.append(path)
    return {
        "chunk_index": chunk_index,
        "images": images,
        "labels": labels,
        "source_indices": source_indices,
        "num_samples": n,
        "source_index_first": int(source_indices[0]),
        "source_index_last": int(source_indices[-1]),
        "input_bytes": int(input_bytes),
        "decode_seconds_worker": time.time() - t0,
        "paths_first_last": paths_first_last,
    }


def list_parquet_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet files matched: {pattern}")
    return files


def parquet_total_rows(files: Sequence[str]) -> int:
    return int(sum(pq.ParquetFile(p).metadata.num_rows for p in files))


def iter_record_chunks(
    files: Sequence[str],
    *,
    start_index: int,
    max_samples: int,
    chunk_size: int,
    read_batch_rows: int,
) -> Iterator[Tuple[int, List[Tuple[int, bytes, int, str]], int]]:
    emitted = 0
    global_index = 0
    chunk_index = 0
    chunk: List[Tuple[int, bytes, int, str]] = []
    for file_path in files:
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(batch_size=read_batch_rows, columns=["image", "label"]):
            rows = batch.num_rows
            if global_index + rows <= start_index:
                global_index += rows
                continue
            image_arr = batch.column(batch.schema.get_field_index("image"))
            label_arr = batch.column(batch.schema.get_field_index("label"))
            bytes_arr = image_arr.field("bytes")
            path_arr = image_arr.field("path")
            local_start = max(0, start_index - global_index)
            for i in range(local_start, rows):
                if emitted >= max_samples:
                    if chunk:
                        yield (chunk_index, chunk, 0)  # image_size is filled by caller before submit
                    fadvise_dontneed(Path(file_path))
                    return
                b = bytes_arr[i].as_py()
                if b is None:
                    raise RuntimeError(f"Missing image bytes at global source index {global_index + i}")
                path = path_arr[i].as_py() or ""
                label = int(label_arr[i].as_py())
                chunk.append((global_index + i, b, label, path))
                emitted += 1
                if len(chunk) >= chunk_size:
                    yield (chunk_index, chunk, 0)  # image_size is filled by caller before submit
                    chunk_index += 1
                    chunk = []
            global_index += rows
        fadvise_dontneed(Path(file_path))
    if chunk:
        yield (chunk_index, chunk, 0)


def ordered_decode_results(
    chunk_iter: Iterator[Tuple[int, List[Tuple[int, bytes, int, str]], int]],
    *,
    image_size: int,
    num_workers: int,
    max_inflight_chunks: int,
) -> Iterator[Dict[str, Any]]:
    max_inflight_chunks = max(1, int(max_inflight_chunks))
    next_emit = 0
    exhausted = False
    pending: Dict[futures.Future, int] = {}
    buffer: Dict[int, Dict[str, Any]] = {}

    def submit_until(executor: futures.ProcessPoolExecutor) -> None:
        nonlocal exhausted
        while not exhausted and (len(pending) + len(buffer)) < max_inflight_chunks:
            try:
                chunk_index, records, _ = next(chunk_iter)
            except StopIteration:
                exhausted = True
                return
            fut = executor.submit(decode_chunk, (chunk_index, records, image_size))
            pending[fut] = chunk_index

    with futures.ProcessPoolExecutor(max_workers=num_workers, initializer=_worker_init) as executor:
        submit_until(executor)
        while pending or buffer or not exhausted:
            while next_emit in buffer:
                yield buffer.pop(next_emit)
                next_emit += 1
                submit_until(executor)
            if not pending:
                if exhausted:
                    break
                submit_until(executor)
                continue
            done, _ = futures.wait(list(pending.keys()), return_when=futures.FIRST_COMPLETED)
            for fut in done:
                idx = pending.pop(fut)
                res = fut.result()
                if int(res["chunk_index"]) != idx:
                    raise RuntimeError(f"Decode chunk index mismatch: future {idx}, result {res['chunk_index']}")
                buffer[idx] = res
            submit_until(executor)


def pop_pending_arrays(
    pending_images: Deque[np.ndarray],
    pending_labels: Deque[np.ndarray],
    pending_indices: Deque[np.ndarray],
    n: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    imgs: List[np.ndarray] = []
    labs: List[np.ndarray] = []
    inds: List[np.ndarray] = []
    remain = n
    while remain > 0:
        if not pending_images:
            raise RuntimeError("Pending crop buffer underflow")
        cur_n = int(pending_labels[0].shape[0])
        if cur_n <= remain:
            imgs.append(pending_images.popleft())
            labs.append(pending_labels.popleft())
            inds.append(pending_indices.popleft())
            remain -= cur_n
        else:
            imgs.append(np.ascontiguousarray(pending_images[0][:remain]))
            labs.append(np.ascontiguousarray(pending_labels[0][:remain]))
            inds.append(np.ascontiguousarray(pending_indices[0][:remain]))
            pending_images[0] = np.ascontiguousarray(pending_images[0][remain:])
            pending_labels[0] = np.ascontiguousarray(pending_labels[0][remain:])
            pending_indices[0] = np.ascontiguousarray(pending_indices[0][remain:])
            remain = 0
    return np.ascontiguousarray(np.concatenate(imgs, axis=0)), np.ascontiguousarray(np.concatenate(labs, axis=0)), np.ascontiguousarray(np.concatenate(inds, axis=0))


def save_crop_shard_atomic(
    output_dir: Path,
    shard_index: int,
    images: np.ndarray,
    labels: np.ndarray,
    source_indices: np.ndarray,
    metadata: Dict[str, str],
    compute_sha256: bool,
) -> CropShardRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    assert images.dtype == np.uint8 and images.ndim == 4, (images.dtype, images.shape)
    assert labels.shape == source_indices.shape == (images.shape[0],), (images.shape, labels.shape, source_indices.shape)
    final_path = shard_path_for(output_dir, shard_index)
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing crop shard: {final_path}")
    tmp_path = output_dir / f".{final_path.name}.tmp.{os.getpid()}"
    if tmp_path.exists():
        tmp_path.unlink()

    meta = dict(metadata)
    meta.update(
        {
            "shard_index": str(shard_index),
            "num_samples": str(int(images.shape[0])),
            "image_dtype": "uint8",
            "image_shape": str(tuple(images.shape)),
            "labels_min": str(int(labels.min())),
            "labels_max": str(int(labels.max())),
            "source_index_first": str(int(source_indices[0])),
            "source_index_last": str(int(source_indices[-1])),
            "created_utc": utc_now(),
        }
    )
    save_dict = {
        "images": torch.from_numpy(images),
        "labels": torch.from_numpy(labels.astype(np.int64, copy=False)),
        "source_indices": torch.from_numpy(source_indices.astype(np.int64, copy=False)),
    }
    t0 = time.time()
    save_file(save_dict, str(tmp_path), metadata=meta)
    verify_tmp = verify_crop_safetensors(tmp_path)
    if not verify_tmp["ok"]:
        raise RuntimeError(f"Temporary crop safetensors verification failed: {json.dumps(verify_tmp, indent=2)}")
    os.replace(tmp_path, final_path)
    verify = verify_crop_safetensors(final_path)
    if not verify["ok"]:
        raise RuntimeError(f"Published crop safetensors verification failed: {json.dumps(verify, indent=2)}")
    digest = sha256_file(final_path) if compute_sha256 else None
    fadvise_dontneed(final_path)
    save_seconds = time.time() - t0
    rec = CropShardRecord(
        shard_index=shard_index,
        path=str(final_path),
        num_samples=int(images.shape[0]),
        image_shape=list(images.shape),
        image_dtype="uint8",
        labels_min=int(labels.min()),
        labels_max=int(labels.max()),
        source_index_first=int(source_indices[0]),
        source_index_last=int(source_indices[-1]),
        bytes=int(final_path.stat().st_size),
        save_seconds=save_seconds,
        sha256=digest,
        verify_ok=bool(verify["ok"]),
        created_utc=utc_now(),
    )
    print("SAVED_CROP_SHARD", json.dumps(asdict(rec), sort_keys=True), flush=True)
    return rec


def build_cache(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        print(f"OVERWRITE enabled: clearing crop cache artifacts in {output_dir}", flush=True)
        clear_output_dir(output_dir)

    files = list_parquet_files(args.data_files_glob)
    total_rows = parquet_total_rows(files)
    target_total = min(int(args.max_samples), total_rows) if args.max_samples is not None else total_rows

    existing = scan_existing_crop_shards(output_dir, strict_contiguous=True)
    completed_at_start = sum(r.num_samples for r in existing)
    next_shard_index = (existing[-1].shard_index + 1) if existing else 0
    if existing and not args.resume:
        raise RuntimeError(f"Found {len(existing)} existing crop shards in {output_dir}; pass --resume or --overwrite.")
    if completed_at_start > target_total:
        raise RuntimeError(f"Existing crop cache has {completed_at_start} samples > target_total={target_total}")

    run_config = {
        "created_utc": utc_now(),
        "argv": sys.argv,
        "args": vars(args),
        "data_files_count": len(files),
        "data_files_first": files[0],
        "data_files_last": files[-1],
        "parquet_total_rows": total_rows,
        "target_total": target_total,
        "existing_shards": len(existing),
        "completed_at_start": completed_at_start,
        "next_shard_index": next_shard_index,
        "torch": torch.__version__,
        "pid": os.getpid(),
    }
    write_json_atomic(output_dir / "run_config.json", run_config)
    print("ARGS", json.dumps(run_config, indent=2, sort_keys=True), flush=True)

    if completed_at_start == target_total:
        summary = {
            "ok": True,
            "status": "already_complete",
            "output_dir": str(output_dir),
            "target_total": target_total,
            "cropped_total": completed_at_start,
            "new_cropped": 0,
            "num_shards_total": len(existing),
            "finished_utc": utc_now(),
        }
        write_json_atomic(output_dir / "progress.json", summary)
        write_json_atomic(output_dir / "build_summary.json", summary)
        print("BUILD_SUMMARY", json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    metadata = {
        "schema": "imagenet_cropped_uint8_v1",
        "source": "hf_imagenet1k_local_parquet",
        "data_files_glob": args.data_files_glob,
        "image_size": str(args.image_size),
        "crop": "adm_center_crop",
        "layout": "NCHW",
        "dtype": "uint8",
    }

    pending_images: Deque[np.ndarray] = deque()
    pending_labels: Deque[np.ndarray] = deque()
    pending_indices: Deque[np.ndarray] = deque()
    pending_n = 0
    new_records: List[CropShardRecord] = []
    start_t = time.time()
    last_log_t = start_t
    last_log_samples = 0
    cropped_total = completed_at_start
    new_cropped = 0
    input_bytes_total = 0
    worker_decode_seconds_total = 0.0
    save_seconds_total = 0.0
    chunk_count = 0

    def write_progress(event: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        nonlocal last_log_t, last_log_samples
        now = time.time()
        elapsed = now - start_t
        interval_elapsed = now - last_log_t
        interval_samples = new_cropped - last_log_samples
        remaining = max(0, target_total - cropped_total)
        avg = new_cropped / elapsed if elapsed > 0 else None
        eta = remaining / avg if avg and avg > 0 else None
        payload: Dict[str, Any] = {
            "event": event,
            "status": "running",
            "time_utc": utc_now(),
            "pid": os.getpid(),
            "output_dir": str(output_dir),
            "source_loader": "pyarrow_local_parquet_multiprocessing",
            "data_files_count": len(files),
            "target_total": target_total,
            "parquet_total_rows": total_rows,
            "completed_at_start": completed_at_start,
            "cropped_total": cropped_total,
            "new_cropped": new_cropped,
            "remaining_to_target": remaining,
            "elapsed_sec": elapsed,
            "avg_new_samples_per_sec": avg,
            "interval_samples_per_sec": interval_samples / interval_elapsed if interval_elapsed > 0 else None,
            "eta_sec": eta,
            "eta_hms": seconds_to_hms(eta),
            "num_existing_shards": len(existing),
            "num_new_shards": len(new_records),
            "next_shard_index": next_shard_index + len(new_records),
            "pending_shard_samples": pending_n,
            "shard_size": args.shard_size,
            "chunk_size": args.chunk_size,
            "read_batch_rows": args.read_batch_rows,
            "num_workers": args.num_workers,
            "prefetch_chunks": args.prefetch_chunks,
            "chunk_count": chunk_count,
            "input_bytes_total": input_bytes_total,
            "worker_decode_seconds_total": worker_decode_seconds_total,
            "save_seconds_total": save_seconds_total,
        }
        if extra:
            payload.update(extra)
        write_json_atomic(output_dir / "progress.json", payload)
        append_jsonl(output_dir / "progress.jsonl", payload)
        last_log_t = now
        last_log_samples = new_cropped
        return payload

    write_progress("start")

    remaining_to_emit = target_total - completed_at_start
    raw_chunks = iter_record_chunks(
        files,
        start_index=completed_at_start,
        max_samples=remaining_to_emit,
        chunk_size=args.chunk_size,
        read_batch_rows=args.read_batch_rows,
    )
    try:
        for res in ordered_decode_results(
            raw_chunks,
            image_size=args.image_size,
            num_workers=args.num_workers,
            max_inflight_chunks=args.prefetch_chunks,
        ):
            chunk_count += 1
            imgs = res["images"]
            labs = res["labels"]
            inds = res["source_indices"]
            n = int(labs.shape[0])
            if int(inds[0]) != cropped_total:
                raise RuntimeError(f"Source order mismatch: expected next source index {cropped_total}, got {int(inds[0])}")
            pending_images.append(imgs)
            pending_labels.append(labs)
            pending_indices.append(inds)
            pending_n += n
            cropped_total += n
            new_cropped += n
            input_bytes_total += int(res.get("input_bytes", 0))
            worker_decode_seconds_total += float(res.get("decode_seconds_worker", 0.0))

            saved_now: List[Dict[str, Any]] = []
            while pending_n >= args.shard_size:
                shard_images, shard_labels, shard_indices = pop_pending_arrays(pending_images, pending_labels, pending_indices, args.shard_size)
                pending_n -= args.shard_size
                rec = save_crop_shard_atomic(output_dir, next_shard_index + len(new_records), shard_images, shard_labels, shard_indices, metadata, args.sha256)
                save_seconds_total += rec.save_seconds
                new_records.append(rec)
                append_jsonl(output_dir / "manifest.jsonl", asdict(rec))
                saved_now.append(asdict(rec))
                del shard_images, shard_labels, shard_indices

            if (chunk_count % args.log_every_chunks) == 0 or saved_now or cropped_total >= target_total:
                progress = write_progress(
                    "chunk",
                    {
                        "last_chunk_samples": n,
                        "last_chunk_source_index_first": int(res["source_index_first"]),
                        "last_chunk_source_index_last": int(res["source_index_last"]),
                        "last_chunk_worker_decode_sec": float(res["decode_seconds_worker"]),
                        "last_chunk_input_bytes": int(res["input_bytes"]),
                        "saved_shards_this_chunk": saved_now,
                    },
                )
                print(
                    "CROP_PROGRESS",
                    json.dumps(
                        {k: progress[k] for k in ["cropped_total", "target_total", "new_cropped", "avg_new_samples_per_sec", "interval_samples_per_sec", "eta_hms", "pending_shard_samples", "num_new_shards"]},
                        sort_keys=True,
                    ),
                    flush=True,
                )

        if pending_n > 0:
            shard_images, shard_labels, shard_indices = pop_pending_arrays(pending_images, pending_labels, pending_indices, pending_n)
            rec = save_crop_shard_atomic(output_dir, next_shard_index + len(new_records), shard_images, shard_labels, shard_indices, metadata, args.sha256)
            save_seconds_total += rec.save_seconds
            new_records.append(rec)
            append_jsonl(output_dir / "manifest.jsonl", asdict(rec))
            pending_n = 0
            del shard_images, shard_labels, shard_indices

        all_records = scan_existing_crop_shards(output_dir, strict_contiguous=True)
        total_final = sum(r.num_samples for r in all_records)
        summary = {
            "ok": total_final == target_total,
            "status": "complete" if total_final == target_total else "partial",
            "output_dir": str(output_dir),
            "target_total": target_total,
            "parquet_total_rows": total_rows,
            "cropped_total": total_final,
            "new_cropped": new_cropped,
            "num_existing_shards_at_start": len(existing),
            "num_new_shards": len(new_records),
            "num_shards_total": len(all_records),
            "elapsed_sec": time.time() - start_t,
            "avg_new_samples_per_sec": new_cropped / (time.time() - start_t) if (time.time() - start_t) > 0 else None,
            "input_bytes_total": input_bytes_total,
            "worker_decode_seconds_total": worker_decode_seconds_total,
            "save_seconds_total": save_seconds_total,
            "finished_utc": utc_now(),
            "shards_tail": [asdict(r) for r in all_records[-5:]],
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
    p.add_argument("--data-files-glob", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument("--chunk-size", type=int, default=64)
    p.add_argument("--read-batch-rows", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=32)
    p.add_argument("--prefetch-chunks", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sha256", action="store_true")
    p.add_argument("--log-every-chunks", type=int, default=8)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    summary = build_cache(args)
    if not summary.get("ok", False):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
