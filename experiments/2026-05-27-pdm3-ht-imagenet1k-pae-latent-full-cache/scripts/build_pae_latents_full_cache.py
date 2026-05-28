#!/usr/bin/env python3
"""
Resumable production ImageNet-1k -> PAE_DINOv2L_d32 latent cache builder.

Output schema is compatible with external/PAE/pae_with_generator/dataset/img_latent_dataset.py:
  - latents:      [N, 32, 16, 16]
  - latents_flip: [N, 32, 16, 16]
  - labels:       [N]

Compared with the earlier smoke builder, this script is designed for full-cache work:
  - resume by scanning completed shards;
  - atomic temp save -> verify -> rename;
  - per-shard manifest JSONL and per-batch progress JSON/JSONL;
  - single ADM center crop per image; horizontal flip is generated from the batch tensor;
  - optional ThreadPoolExecutor preprocessing/prefetch;
  - optional os._exit hard finish for HF streaming finalizer hangs.
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import glob
import hashlib
import json
import os
import re
import sys
import time
import traceback
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "external" / "PAE" / "pae_with_generator" / "tokenizer" / "pae.py").exists():
            return p
    return Path("/workspace/PDM")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
PAE_ROOT = REPO_ROOT / "external" / "PAE" / "pae_with_generator"
if str(PAE_ROOT) not in sys.path:
    sys.path.insert(0, str(PAE_ROOT))


def resolve_path_maybe_relative(path_value: str, base: Path) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    candidate = base / p
    return str(candidate) if candidate.exists() else path_value


def load_pae_params(pae_config_path: Path) -> Dict[str, Any]:
    with pae_config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = dict(cfg["stage_1"].get("params", {}))
    if "decoder_config_path" in params:
        params["decoder_config_path"] = resolve_path_maybe_relative(params["decoder_config_path"], PAE_ROOT)
    return params


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model", "ema", "state_dict", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        if all(isinstance(k, str) for k in ckpt.keys()) and any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise TypeError(f"Could not find a model state_dict in checkpoint object of type {type(ckpt)}")


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned: Dict[str, torch.Tensor] = {}
    prefixes = ("module.", "_orig_mod.", "model.")
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = name.lower()
    if key not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[key]




def cuda_set_current(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)


def cuda_reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        cuda_set_current(device)
        torch.cuda.reset_peak_memory_stats()


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        cuda_set_current(device)
        torch.cuda.synchronize()

def load_pae_model(
    pae_config_path: Path,
    pae_ckpt_path: Path,
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.nn.Module:
    from tokenizer.pae import PAE

    params = load_pae_params(pae_config_path)
    print("PAE params:", json.dumps(params, indent=2, sort_keys=True), flush=True)
    print(f"Instantiating PAE from config: {pae_config_path}", flush=True)
    model = PAE(**params)
    print(f"Loading PAE checkpoint: {pae_ckpt_path}", flush=True)
    ckpt = torch.load(str(pae_ckpt_path), map_location="cpu", weights_only=False)
    state_dict = clean_state_dict_keys(extract_state_dict(ckpt))
    msg = model.load_state_dict(state_dict, strict=False)
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    print(f"load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("missing_first20=", missing[:20], flush=True)
    if unexpected:
        print("unexpected_first20=", unexpected[:20], flush=True)
    del ckpt, state_dict
    model = model.to(device=device, dtype=model_dtype).eval()
    return model


def resolve_hf_token(token_env: str) -> Tuple[Optional[str], str]:
    """Resolve HF token without printing it.

    Priority:
      1. explicit environment variable, e.g. HF_TOKEN
      2. cached `huggingface-cli login` token via huggingface_hub.get_token()
      3. no token

    This fixes the misleading old `token_env_set=False` log after CLI login.
    """
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


def resolve_data_files_glob(data_files_glob: Optional[str]) -> List[str]:
    if not data_files_glob:
        return []
    files = sorted(glob.glob(data_files_glob))
    if not files:
        raise FileNotFoundError(f"--data-files-glob matched no files: {data_files_glob}")
    return files


def load_hf_dataset(
    dataset: str,
    split: str,
    streaming: bool,
    token_env: str,
    data_files_glob: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    from datasets import load_dataset

    local_files = resolve_data_files_glob(data_files_glob)
    if local_files:
        total_bytes = sum(Path(p).stat().st_size for p in local_files)
        kwargs: Dict[str, Any] = {
            "data_files": {split: local_files},
            "split": split,
            "streaming": streaming,
        }
        info = {
            "source_loader": "local_parquet",
            "dataset": "parquet",
            "requested_dataset": dataset,
            "split": split,
            "streaming": streaming,
            "data_files_glob": data_files_glob,
            "data_files_count": len(local_files),
            "data_files_total_bytes": total_bytes,
            "hf_token_source": "not_used_local",
            "hf_token_set": False,
        }
        print(
            "Loading local parquet dataset "
            f"split={split!r} streaming={streaming} files={len(local_files)} "
            f"total_gb={total_bytes / 1e9:.3f} glob={data_files_glob!r}",
            flush=True,
        )
        return load_dataset("parquet", **kwargs), info

    token, token_source = resolve_hf_token(token_env)
    kwargs = {"split": split, "streaming": streaming}
    if token:
        kwargs["token"] = token
    info = {
        "source_loader": "hf_dataset",
        "dataset": dataset,
        "requested_dataset": dataset,
        "split": split,
        "streaming": streaming,
        "data_files_glob": None,
        "data_files_count": 0,
        "data_files_total_bytes": 0,
        "hf_token_source": token_source,
        "hf_token_set": bool(token),
    }
    print(
        f"Loading HF dataset={dataset!r} split={split!r} streaming={streaming} "
        f"token_source={token_source} token_set={bool(token)}",
        flush=True,
    )
    return load_dataset(dataset, **kwargs), info


def adm_center_crop_uint8_chw(pil_image: Image.Image, image_size: int) -> torch.Tensor:
    """ADM center crop -> CHW uint8 tensor. One crop per source image."""
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.Resampling.BICUBIC)

    arr = np.array(pil_image, dtype=np.uint8, copy=True)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    arr = np.ascontiguousarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size, :])
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def preprocess_sample(sample: Dict[str, Any], image_key: str, label_key: str, image_size: int) -> Tuple[torch.Tensor, int]:
    image = sample[image_key]
    label = int(sample[label_key])
    return adm_center_crop_uint8_chw(image, image_size), label


def maybe_skip_dataset(ds: Any, skip_samples: int):
    if skip_samples <= 0:
        return ds
    print(f"Skipping source samples for resume: {skip_samples}", flush=True)
    if hasattr(ds, "skip"):
        return ds.skip(skip_samples)
    # Map-style fallback.
    return ds.select(range(skip_samples, len(ds)))


def preprocessed_batches(
    ds_iter: Iterator[Dict[str, Any]],
    *,
    remaining: Optional[int],
    batch_size: int,
    preprocess_workers: int,
    prefetch_batches: int,
    image_key: str,
    label_key: str,
    image_size: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    """Yield ordered batches of uint8 CHW tensors and int64 labels."""
    submitted = 0
    yielded = 0
    batch_x: List[torch.Tensor] = []
    batch_y: List[int] = []
    t_pre_start = time.time()

    def make_batch() -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        nonlocal batch_x, batch_y
        bx = torch.stack(batch_x, dim=0).contiguous()
        by = torch.tensor(batch_y, dtype=torch.long)
        meta = {
            "batch_samples": int(by.shape[0]),
            "preprocess_elapsed_sec": time.time() - t_pre_start,
        }
        batch_x, batch_y = [], []
        return bx, by, meta

    if preprocess_workers <= 1:
        for sample in ds_iter:
            if remaining is not None and submitted >= remaining:
                break
            x, y = preprocess_sample(sample, image_key, label_key, image_size)
            submitted += 1
            yielded += 1
            batch_x.append(x)
            batch_y.append(y)
            if len(batch_x) >= batch_size:
                yield make_batch()
        if batch_x:
            yield make_batch()
        return

    max_prefetch = max(batch_size, int(batch_size) * max(1, int(prefetch_batches)))
    executor = futures.ThreadPoolExecutor(max_workers=preprocess_workers, thread_name_prefix="pae_pre")
    pending: Deque[futures.Future[Tuple[torch.Tensor, int]]] = deque()
    exhausted = False

    def submit_one() -> bool:
        nonlocal submitted, exhausted
        if exhausted:
            return False
        if remaining is not None and submitted >= remaining:
            exhausted = True
            return False
        try:
            sample = next(ds_iter)
        except StopIteration:
            exhausted = True
            return False
        fut = executor.submit(preprocess_sample, sample, image_key, label_key, image_size)
        pending.append(fut)
        submitted += 1
        return True

    try:
        while len(pending) < max_prefetch and submit_one():
            pass
        while pending:
            while len(pending) < max_prefetch and submit_one():
                pass
            fut = pending.popleft()
            x, y = fut.result()
            yielded += 1
            batch_x.append(x)
            batch_y.append(y)
            if len(batch_x) >= batch_size:
                yield make_batch()
        if batch_x:
            yield make_batch()
    finally:
        for fut in pending:
            fut.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def encode_pair_from_u8_batch(
    model: torch.nn.Module,
    x_u8: torch.Tensor,
    device: torch.device,
    model_dtype: torch.dtype,
    save_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Move uint8 batch once, convert to [0,1], encode original and horizontal flip."""
    if device.type == "cuda":
        cuda_sync(device)
    t0 = time.time()
    x = x_u8.to(device=device, non_blocking=True).to(dtype=torch.float32).div_(255.0)
    if device.type == "cuda":
        cuda_sync(device)
    t_h2d = time.time() - t0

    use_autocast = device.type == "cuda" and model_dtype in (torch.bfloat16, torch.float16)
    autocast_dtype = model_dtype if use_autocast else torch.float32

    if device.type == "cuda":
        cuda_sync(device)
    t1 = time.time()
    with torch.inference_mode():
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
            z = model.encode(x)
            xf = torch.flip(x, dims=[-1])
            zf = model.encode(xf)
    if device.type == "cuda":
        cuda_sync(device)
    t_encode = time.time() - t1

    t2 = time.time()
    z_cpu = z.detach().to(dtype=save_dtype, device="cpu").contiguous()
    zf_cpu = zf.detach().to(dtype=save_dtype, device="cpu").contiguous()
    if device.type == "cuda":
        cuda_sync(device)
    t_d2h = time.time() - t2
    del x, z, zf
    return z_cpu, zf_cpu, {"h2d_sec": t_h2d, "encode_sec": t_encode, "d2h_sec": t_d2h}


def verify_safetensors_file(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "ok": False, "keys": {}, "metadata": {}}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        try:
            info["metadata"] = dict(f.metadata() or {})
        except Exception:
            info["metadata"] = {}
        for key in ("latents", "latents_flip", "labels"):
            tensor_slice = f.get_slice(key)
            info["keys"][key] = {
                "shape": list(tensor_slice.get_shape()),
                "dtype": str(tensor_slice.get_dtype()),
            }
    n = int(info["keys"]["labels"]["shape"][0])
    ok = n > 0
    ok = ok and info["keys"]["latents"]["shape"][0] == n
    ok = ok and info["keys"]["latents_flip"]["shape"][0] == n
    ok = ok and info["keys"]["latents"]["shape"] == info["keys"]["latents_flip"]["shape"]
    info["ok"] = bool(ok)
    info["num_samples"] = n
    return info


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fadvise_dontneed(path: Path) -> None:
    """Best-effort page-cache eviction for large clean safetensors files."""
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
class ShardRecord:
    shard_index: int
    path: str
    num_samples: int
    latents_shape: List[int]
    latents_dtype: str
    labels_min: int
    labels_max: int
    bytes: int
    sha256: Optional[str]
    save_seconds: float
    verify_ok: bool
    created_utc: str


def shard_path_for(output_dir: Path, shard_index: int) -> Path:
    return output_dir / f"latents_rank00_shard{shard_index:06d}.safetensors"


def save_shard_atomic(
    output_dir: Path,
    shard_index: int,
    z: torch.Tensor,
    zf: torch.Tensor,
    y: torch.Tensor,
    metadata: Dict[str, str],
    compute_sha256: bool,
) -> ShardRecord:
    output_dir.mkdir(parents=True, exist_ok=True)
    assert z.shape == zf.shape, (z.shape, zf.shape)
    assert z.shape[0] == y.shape[0], (z.shape, y.shape)
    y = y.to(dtype=torch.long).contiguous()
    final_path = shard_path_for(output_dir, shard_index)
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing shard: {final_path}")
    tmp_path = output_dir / f".{final_path.name}.tmp.{os.getpid()}"
    if tmp_path.exists():
        tmp_path.unlink()

    meta = dict(metadata)
    meta.update(
        {
            "shard_index": str(shard_index),
            "num_samples": str(int(y.shape[0])),
            "latent_dtype": str(z.dtype),
            "latent_shape": str(tuple(z.shape)),
            "labels_min": str(int(y.min().item())),
            "labels_max": str(int(y.max().item())),
            "created_utc": utc_now(),
        }
    )
    save_dict = {"latents": z.contiguous(), "latents_flip": zf.contiguous(), "labels": y}
    t0 = time.time()
    save_file(save_dict, str(tmp_path), metadata=meta)
    verify_tmp = verify_safetensors_file(tmp_path)
    if not verify_tmp["ok"]:
        raise RuntimeError(f"Temporary safetensors verification failed: {json.dumps(verify_tmp, indent=2)}")
    os.replace(tmp_path, final_path)
    verify = verify_safetensors_file(final_path)
    if not verify["ok"]:
        raise RuntimeError(f"Published safetensors verification failed: {json.dumps(verify, indent=2)}")
    digest = sha256_file(final_path) if compute_sha256 else None
    fadvise_dontneed(final_path)
    save_seconds = time.time() - t0
    rec = ShardRecord(
        shard_index=shard_index,
        path=str(final_path),
        num_samples=int(y.shape[0]),
        latents_shape=list(z.shape),
        latents_dtype=str(z.dtype),
        labels_min=int(y.min().item()),
        labels_max=int(y.max().item()),
        bytes=int(final_path.stat().st_size),
        sha256=digest,
        save_seconds=save_seconds,
        verify_ok=bool(verify["ok"]),
        created_utc=utc_now(),
    )
    print("SAVED_SHARD", json.dumps(asdict(rec), sort_keys=True), flush=True)
    return rec


_SHARD_RE = re.compile(r"shard(\d+)\.safetensors$")


def parse_shard_index(path: Path) -> int:
    m = _SHARD_RE.search(path.name)
    if not m:
        raise ValueError(f"Could not parse shard index from {path.name}")
    return int(m.group(1))


def scan_existing_shards(output_dir: Path, strict_contiguous: bool = True) -> List[ShardRecord]:
    files = sorted(p for p in output_dir.glob("*.safetensors") if ".tmp" not in p.name)
    records: List[ShardRecord] = []
    for path in files:
        idx = parse_shard_index(path)
        info = verify_safetensors_file(path)
        if not info["ok"]:
            raise RuntimeError(f"Existing shard failed verification: {path}")
        with safe_open(str(path), framework="pt", device="cpu") as f:
            labels = f.get_tensor("labels")
            lat_shape = list(f.get_slice("latents").get_shape())
            lat_dtype = str(f.get_slice("latents").get_dtype())
        records.append(
            ShardRecord(
                shard_index=idx,
                path=str(path),
                num_samples=int(labels.shape[0]),
                latents_shape=lat_shape,
                latents_dtype=lat_dtype,
                labels_min=int(labels.min().item()),
                labels_max=int(labels.max().item()),
                bytes=int(path.stat().st_size),
                sha256=None,
                save_seconds=0.0,
                verify_ok=True,
                created_utc=str(info.get("metadata", {}).get("created_utc", "")),
            )
        )
    records.sort(key=lambda r: r.shard_index)
    if strict_contiguous:
        for expected, rec in enumerate(records):
            if rec.shard_index != expected:
                raise RuntimeError(
                    f"Non-contiguous shard indices in {output_dir}: expected {expected}, got {rec.shard_index} at {rec.path}"
                )
    return records


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


class PendingShardBuffer:
    def __init__(self) -> None:
        self.zs: List[torch.Tensor] = []
        self.zfs: List[torch.Tensor] = []
        self.ys: List[torch.Tensor] = []
        self.n = 0

    def add(self, z: torch.Tensor, zf: torch.Tensor, y: torch.Tensor) -> None:
        assert z.shape[0] == zf.shape[0] == y.shape[0]
        self.zs.append(z)
        self.zfs.append(zf)
        self.ys.append(y.to(dtype=torch.long).contiguous())
        self.n += int(y.shape[0])

    def pop(self, n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if n <= 0 or n > self.n:
            raise ValueError(f"Cannot pop {n} from pending buffer with n={self.n}")
        out_z: List[torch.Tensor] = []
        out_zf: List[torch.Tensor] = []
        out_y: List[torch.Tensor] = []
        new_z: List[torch.Tensor] = []
        new_zf: List[torch.Tensor] = []
        new_y: List[torch.Tensor] = []
        remain = n
        consumed_done = False
        for z, zf, y in zip(self.zs, self.zfs, self.ys):
            cur = int(y.shape[0])
            if consumed_done:
                new_z.append(z)
                new_zf.append(zf)
                new_y.append(y)
            elif cur <= remain:
                out_z.append(z)
                out_zf.append(zf)
                out_y.append(y)
                remain -= cur
                if remain == 0:
                    consumed_done = True
            else:
                out_z.append(z[:remain].contiguous())
                out_zf.append(zf[:remain].contiguous())
                out_y.append(y[:remain].contiguous())
                new_z.append(z[remain:].contiguous())
                new_zf.append(zf[remain:].contiguous())
                new_y.append(y[remain:].contiguous())
                remain = 0
                consumed_done = True
        if remain != 0:
            raise RuntimeError("Internal PendingShardBuffer accounting error")
        self.zs, self.zfs, self.ys = new_z, new_zf, new_y
        self.n -= n
        return torch.cat(out_z, dim=0).contiguous(), torch.cat(out_zf, dim=0).contiguous(), torch.cat(out_y, dim=0).contiguous()


def cuda_mem(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda":
        return {"cuda_allocated_mb": 0.0, "cuda_reserved_mb": 0.0, "cuda_peak_allocated_mb": 0.0}
    cuda_set_current(device)
    return {
        "cuda_allocated_mb": float(torch.cuda.memory_allocated() / 1024**2),
        "cuda_reserved_mb": float(torch.cuda.memory_reserved() / 1024**2),
        "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
    }


def seconds_to_hms(sec: Optional[float]) -> Optional[str]:
    if sec is None or sec < 0 or not np.isfinite(sec):
        return None
    sec_i = int(sec)
    h, rem = divmod(sec_i, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def clear_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = ["*.safetensors", ".*.safetensors.tmp.*", "manifest.jsonl", "progress.jsonl", "progress.json", "build_summary.json", "run_config.json"]
    for pat in patterns:
        for p in output_dir.glob(pat):
            if p.is_file():
                p.unlink()


def build_latents(args: argparse.Namespace) -> Dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        print(f"OVERWRITE enabled: clearing cache artifacts in {output_dir}", flush=True)
        clear_output_dir(output_dir)

    existing = scan_existing_shards(output_dir, strict_contiguous=True)
    completed_at_start = sum(r.num_samples for r in existing)
    next_shard_index = (existing[-1].shard_index + 1) if existing else 0
    target_total = args.max_samples

    if existing and not args.resume:
        raise RuntimeError(f"Found {len(existing)} existing shards in {output_dir}; pass --resume or --overwrite.")
    if target_total is not None and completed_at_start > target_total:
        raise RuntimeError(f"Existing cache has {completed_at_start} samples > target max_samples={target_total}")

    run_config = {
        "created_utc": utc_now(),
        "argv": sys.argv,
        "args": vars(args),
        "repo_root": str(REPO_ROOT),
        "pae_root": str(PAE_ROOT),
        "existing_shards": len(existing),
        "completed_at_start": completed_at_start,
        "next_shard_index": next_shard_index,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    write_json_atomic(output_dir / "run_config.json", run_config)

    if target_total is not None and completed_at_start == target_total:
        summary = {
            "ok": True,
            "status": "already_complete",
            "output_dir": str(output_dir),
            "target_total": target_total,
            "encoded_total": completed_at_start,
            "new_encoded": 0,
            "num_shards": len(existing),
            "shards": [asdict(r) for r in existing],
            "finished_utc": utc_now(),
        }
        write_json_atomic(output_dir / "build_summary.json", summary)
        write_json_atomic(output_dir / "progress.json", summary)
        print("BUILD_SUMMARY", json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available; pass --allow-cpu only for debugging.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = dtype_from_name(args.model_dtype)
    save_dtype = dtype_from_name(args.save_dtype)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        cuda_reset_peak(device)

    model = load_pae_model(Path(args.pae_config), Path(args.pae_ckpt), device, model_dtype)

    if args.self_test_only:
        x_u8 = torch.randint(0, 256, (min(args.batch_size, 16), 3, args.image_size, args.image_size), dtype=torch.uint8)
        z, zf, timing = encode_pair_from_u8_batch(model, x_u8, device, model_dtype, save_dtype)
        summary = {
            "ok": True,
            "status": "self_test_only",
            "z_shape": list(z.shape),
            "zf_shape": list(zf.shape),
            "z_dtype": str(z.dtype),
            "timing": timing,
            **cuda_mem(device),
        }
        print("SELF_TEST_SUMMARY", json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return summary

    ds, source_info = load_hf_dataset(
        args.dataset,
        args.split,
        args.streaming,
        args.hf_token_env,
        getattr(args, "data_files_glob", None),
    )
    run_config["source_info"] = source_info
    write_json_atomic(output_dir / "run_config.json", run_config)
    source_skip = int(args.skip_samples) + int(completed_at_start)
    ds = maybe_skip_dataset(ds, source_skip)
    ds_iter = iter(ds)
    remaining = None if target_total is None else int(target_total - completed_at_start)

    metadata = {
        "dataset": args.dataset,
        "split": args.split,
        "image_size": str(args.image_size),
        "pae_config": str(args.pae_config),
        "pae_ckpt": str(args.pae_ckpt),
        "model_dtype": args.model_dtype,
        "save_dtype": args.save_dtype,
        "created_by": "build_pae_latents_full_cache.py",
        "source_skip_at_run_start": str(source_skip),
        "completed_at_run_start": str(completed_at_start),
        "batch_size": str(args.batch_size),
        "shard_size": str(args.shard_size),
        "preprocess_workers": str(args.preprocess_workers),
        "prefetch_batches": str(args.prefetch_batches),
        "source_loader": str(source_info.get("source_loader")),
        "data_files_glob": str(source_info.get("data_files_glob")),
        "data_files_count": str(source_info.get("data_files_count")),
        "hf_token_source": str(source_info.get("hf_token_source")),
    }

    pending = PendingShardBuffer()
    new_shards: List[ShardRecord] = []
    encoded_total = completed_at_start
    new_encoded = 0
    batch_index = 0
    started = time.time()
    last_log_t = started
    last_log_encoded = 0
    encode_time_total = 0.0
    h2d_time_total = 0.0
    d2h_time_total = 0.0
    preprocess_wall_meta_total = 0.0
    status = "running"

    def write_progress(event: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        nonlocal last_log_t, last_log_encoded
        now = time.time()
        elapsed = now - started
        avg_sps = (new_encoded / elapsed) if elapsed > 0 else 0.0
        interval_sec = now - last_log_t
        interval_samples = new_encoded - last_log_encoded
        interval_sps = interval_samples / interval_sec if interval_sec > 0 else 0.0
        eta_sec = None
        if target_total is not None and avg_sps > 0:
            eta_sec = max(0.0, (target_total - encoded_total) / avg_sps)
        payload: Dict[str, Any] = {
            "event": event,
            "status": status,
            "time_utc": utc_now(),
            "pid": os.getpid(),
            "output_dir": str(output_dir),
            "completed_at_start": completed_at_start,
            "source_skip_at_start": source_skip,
            "encoded_total": encoded_total,
            "new_encoded": new_encoded,
            "target_total": target_total,
            "remaining_to_target": None if target_total is None else max(0, target_total - encoded_total),
            "next_shard_index": next_shard_index + len(new_shards),
            "num_existing_shards": len(existing),
            "num_new_shards": len(new_shards),
            "pending_shard_samples": pending.n,
            "batch_index": batch_index,
            "batch_size": args.batch_size,
            "shard_size": args.shard_size,
            "preprocess_workers": args.preprocess_workers,
            "prefetch_batches": args.prefetch_batches,
            "source_loader": source_info.get("source_loader"),
            "data_files_glob": source_info.get("data_files_glob"),
            "data_files_count": source_info.get("data_files_count"),
            "hf_token_source": source_info.get("hf_token_source"),
            "elapsed_sec": elapsed,
            "avg_new_samples_per_sec": avg_sps,
            "interval_samples_per_sec": interval_sps,
            "eta_sec": eta_sec,
            "eta_hms": seconds_to_hms(eta_sec),
            "encode_time_total_sec": encode_time_total,
            "h2d_time_total_sec": h2d_time_total,
            "d2h_time_total_sec": d2h_time_total,
            "preprocess_wall_meta_total_sec": preprocess_wall_meta_total,
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
        for x_u8, y, batch_meta in preprocessed_batches(
            ds_iter,
            remaining=remaining,
            batch_size=args.batch_size,
            preprocess_workers=args.preprocess_workers,
            prefetch_batches=args.prefetch_batches,
            image_key=args.image_key,
            label_key=args.label_key,
            image_size=args.image_size,
        ):
            if target_total is not None and encoded_total >= target_total:
                break
            if target_total is not None and encoded_total + int(y.shape[0]) > target_total:
                keep = target_total - encoded_total
                x_u8 = x_u8[:keep].contiguous()
                y = y[:keep].contiguous()
            batch_index += 1
            t_batch = time.time()
            z, zf, timing = encode_pair_from_u8_batch(model, x_u8, device, model_dtype, save_dtype)
            batch_elapsed = time.time() - t_batch
            h2d_time_total += timing["h2d_sec"]
            encode_time_total += timing["encode_sec"]
            d2h_time_total += timing["d2h_sec"]
            preprocess_wall_meta_total = float(batch_meta.get("preprocess_elapsed_sec", preprocess_wall_meta_total))

            pending.add(z, zf, y)
            n_batch = int(y.shape[0])
            encoded_total += n_batch
            new_encoded += n_batch

            saved_now: List[Dict[str, Any]] = []
            while pending.n >= args.shard_size:
                shard_z, shard_zf, shard_y = pending.pop(args.shard_size)
                shard_idx = next_shard_index + len(new_shards)
                rec = save_shard_atomic(output_dir, shard_idx, shard_z, shard_zf, shard_y, metadata, args.sha256)
                new_shards.append(rec)
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
                    "saved_shards_this_batch": saved_now,
                },
            )
            print("PROGRESS", json.dumps({k: progress[k] for k in ["encoded_total", "target_total", "new_encoded", "avg_new_samples_per_sec", "interval_samples_per_sec", "eta_hms", "pending_shard_samples", "num_new_shards", "cuda_peak_allocated_mb"]}, sort_keys=True), flush=True)

            if target_total is not None and encoded_total >= target_total:
                break

        if pending.n > 0:
            shard_z, shard_zf, shard_y = pending.pop(pending.n)
            shard_idx = next_shard_index + len(new_shards)
            rec = save_shard_atomic(output_dir, shard_idx, shard_z, shard_zf, shard_y, metadata, args.sha256)
            new_shards.append(rec)
            append_jsonl(output_dir / "manifest.jsonl", asdict(rec))
            del shard_z, shard_zf, shard_y

        status = "finished"
        elapsed = time.time() - started
        all_records = scan_existing_shards(output_dir, strict_contiguous=True)
        encoded_scanned = sum(r.num_samples for r in all_records)
        ok = encoded_scanned == encoded_total and encoded_total > 0 and all(r.verify_ok for r in all_records)
        if target_total is not None:
            ok = ok and (encoded_total == target_total)
        if args.expected_total is not None and target_total is None:
            ok = ok and (encoded_total == args.expected_total)

        summary = {
            "ok": bool(ok),
            "status": status,
            "dataset": args.dataset,
            "split": args.split,
            "output_dir": str(output_dir),
            "target_total": target_total,
            "expected_total": args.expected_total,
            "completed_at_start": completed_at_start,
            "source_skip_at_start": source_skip,
            "new_encoded": new_encoded,
            "encoded_total": encoded_total,
            "encoded_scanned": encoded_scanned,
            "num_existing_shards_at_start": len(existing),
            "num_new_shards": len(new_shards),
            "num_shards_total": len(all_records),
            "new_shards": [asdict(r) for r in new_shards],
            "all_shards_brief": [
                {"shard_index": r.shard_index, "path": r.path, "num_samples": r.num_samples, "bytes": r.bytes}
                for r in all_records
            ],
            "elapsed_seconds": elapsed,
            "new_samples_per_sec": new_encoded / elapsed if elapsed > 0 else 0.0,
            "encode_time_total_sec": encode_time_total,
            "h2d_time_total_sec": h2d_time_total,
            "d2h_time_total_sec": d2h_time_total,
            "preprocess_wall_meta_total_sec": preprocess_wall_meta_total,
            "total_bytes": sum(r.bytes for r in all_records),
            **cuda_mem(device),
            "finished_utc": utc_now(),
        }
        write_json_atomic(output_dir / "build_summary.json", summary)
        write_progress("finished", {"summary_path": str(output_dir / "build_summary.json"), "ok": bool(ok)})
        print("BUILD_SUMMARY", json.dumps(summary, indent=2, sort_keys=True), flush=True)
        if not ok:
            raise RuntimeError("Build completed but summary ok=False")
        return summary
    except BaseException as exc:
        status = "failed"
        fail_payload = {
            "ok": False,
            "status": status,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "encoded_total": encoded_total,
            "new_encoded": new_encoded,
            "output_dir": str(output_dir),
            "time_utc": utc_now(),
            **cuda_mem(device),
        }
        write_json_atomic(output_dir / "progress.json", fail_payload)
        append_jsonl(output_dir / "progress.jsonl", {"event": "failed", **fail_payload})
        print("BUILD_FAILED", json.dumps(fail_payload, indent=2, sort_keys=True), flush=True)
        raise


def verify_access(args: argparse.Namespace) -> Dict[str, Any]:
    ds, source_info = load_hf_dataset(
        args.dataset,
        args.split,
        args.streaming,
        args.hf_token_env,
        getattr(args, "data_files_glob", None),
    )
    sample = next(iter(ds))
    image = sample.get(args.image_key)
    label = int(sample.get(args.label_key))
    out = {
        "ok": True,
        "sample_keys": list(sample.keys()),
        "image_type": type(image).__name__,
        "image_size": list(getattr(image, "size", [])),
        "image_mode": getattr(image, "mode", None),
        "label": label,
        "source_info": source_info,
    }
    print("ACCESS_CHECK", json.dumps(out, indent=2, sort_keys=True), flush=True)
    return out


def hard_finish(exit_code: int = 0) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ILSVRC/imagenet-1k")
    p.add_argument("--split", default="train")
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hf-token-env", default="HF_TOKEN")
    p.add_argument(
        "--data-files-glob",
        default=None,
        help=(
            "Optional local parquet glob. If set, load_dataset('parquet', data_files=glob) "
            "is used instead of streaming from the HF repo. This is the raw-first/offline path."
        ),
    )
    p.add_argument("--pae-config", default=str(PAE_ROOT / "configs" / "pae" / "PAE_DINOv2L_d32.yaml"))
    p.add_argument("--pae-ckpt", default="/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument("--max-samples", type=int, default=None, help="Target total samples in output, including resumed samples.")
    p.add_argument("--expected-total", type=int, default=None, help="Expected dataset size for reporting/verification.")
    p.add_argument("--skip-samples", type=int, default=0, help="Additional source samples to skip before resume skip.")
    p.add_argument("--image-key", default="image")
    p.add_argument("--label-key", default="label")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    p.add_argument("--save-dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    p.add_argument("--preprocess-workers", type=int, default=12)
    p.add_argument("--prefetch-batches", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--sha256", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--access-check-only", action="store_true")
    p.add_argument("--self-test-only", action="store_true")
    p.add_argument("--hard-exit", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("ARGS", json.dumps(vars(args), indent=2, sort_keys=True), flush=True)
    try:
        if args.access_check_only:
            result = verify_access(args)
        else:
            result = build_latents(args)
        exit_code = 0 if result.get("ok", False) else 1
    except Exception:
        traceback.print_exc()
        exit_code = 1
    if args.hard_exit:
        hard_finish(exit_code)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
