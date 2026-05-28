#!/usr/bin/env python3
"""Small Inception/FID/MMD helpers for PDM-3-HT image-space eval.

This module is intentionally experiment-local.  It reuses LightningDiT's
FID-compatible InceptionV3 implementation/weights, but avoids a hard SciPy
dependency by computing the Frechet covariance term with torch.linalg.

Input conventions:

* reference cache: safetensors shards with key ``images`` in uint8 CHW
  [N, 3, 256, 256], labels/source_indices optional;
* generated samples: PNG/JPEG directory, values interpreted as RGB [0, 255];
* Inception input: float32 RGB [0, 1], resized/normalized inside the
  LightningDiT/pytorch-fid Inception wrapper.

The resulting ``mu``/``sigma`` are compatible in spirit with pytorch-fid style
statistics generated from the same Inception weights and the same real-image
crop policy.  They are not uploaded by default.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TF
from PIL import Image
from safetensors import safe_open


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".pgm", ".png", ".ppm", ".tif", ".tiff", ".webp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def repo_root_from_script() -> Path:
    # scripts/ -> experiment/ -> experiments/ -> repo root
    return Path(__file__).resolve().parents[3]


def load_lightningdit_fid_module() -> Any:
    """Load external/LightningDiT/tools/calculate_fid.py without SciPy.

    The external module imports ``from scipy import linalg`` at import time.
    We only need its Inception classes and FID weights URL, not SciPy's sqrtm.
    If SciPy is absent, install a tiny placeholder module before import.
    """
    try:
        import scipy  # type: ignore  # noqa: F401
    except Exception:
        scipy_mod = types.ModuleType("scipy")
        linalg_mod = types.ModuleType("scipy.linalg")
        scipy_mod.linalg = linalg_mod  # type: ignore[attr-defined]
        sys.modules.setdefault("scipy", scipy_mod)
        sys.modules.setdefault("scipy.linalg", linalg_mod)

    path = repo_root_from_script() / "external" / "LightningDiT" / "tools" / "calculate_fid.py"
    if not path.exists():
        raise FileNotFoundError(f"LightningDiT FID script not found: {path}")
    spec = importlib.util.spec_from_file_location("pdm_ldit_calculate_fid", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import LightningDiT FID module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pdm_ldit_calculate_fid"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_inception(*, dims: int, device: torch.device) -> torch.nn.Module:
    mod = load_lightningdit_fid_module()
    block_idx = mod.InceptionV3.BLOCK_INDEX_BY_DIM[int(dims)]
    model = mod.InceptionV3([block_idx]).to(device).eval()
    model.requires_grad_(False)
    return model


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).lower() == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device_arg)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device_arg}")
    return dev


def list_image_files(images_dir: Path, *, recursive: bool = False) -> List[Path]:
    if not images_dir.exists():
        raise FileNotFoundError(images_dir)
    def keep(p: Path) -> bool:
        if not (p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
            return False
        # Decoder sidecar contact sheets are useful for humans but must not be
        # counted as generated samples for FID/MMD.
        if p.stem in {"preview_grid", "contact_sheet"} or p.name.startswith("preview_"):
            return False
        return True

    if recursive:
        files = [p for p in images_dir.rglob("*") if keep(p)]
    else:
        files = [p for p in images_dir.iterdir() if keep(p)]
    return sorted(files)


class ImageFileDataset(torch.utils.data.Dataset):
    def __init__(self, files: Sequence[Path]) -> None:
        self.files = list(files)
        self.to_tensor = TF.ToTensor()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.files[idx]).convert("RGB")
        return self.to_tensor(img)


def inception_forward(model: torch.nn.Module, x01: torch.Tensor) -> torch.Tensor:
    """Return [B, D] Inception features from RGB float tensor in [0, 1]."""
    with torch.inference_mode():
        pred = model(x01)[0]
        if pred.ndim == 4 and (pred.shape[2] != 1 or pred.shape[3] != 1):
            pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        if pred.ndim == 4:
            pred = pred.squeeze(3).squeeze(2)
        if pred.ndim != 2:
            raise RuntimeError(f"Expected 2D Inception features, got shape={tuple(pred.shape)}")
    return pred.float()


def features_from_image_dir(
    images_dir: Path,
    *,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_images: Optional[int] = None,
    recursive: bool = False,
    log_every_batches: int = 0,
    log_prefix: str = "INCEPTION_FEATURES",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    files = list_image_files(images_dir, recursive=recursive)
    if max_images is not None:
        files = files[: int(max_images)]
    if not files:
        raise RuntimeError(f"No image files found in {images_dir}")
    ds = ImageFileDataset(files)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )
    feats: List[np.ndarray] = []
    total = len(files)
    for batch_idx, batch in enumerate(dl):
        if int(log_every_batches) > 0 and (batch_idx % int(log_every_batches) == 0):
            done = min(batch_idx * int(batch_size), total)
            print(f"{log_prefix} dir={images_dir} batch={batch_idx} images={done}/{total}", flush=True)
        batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
        feats.append(inception_forward(model, batch).cpu().numpy().astype(np.float32, copy=False))
    arr = np.concatenate(feats, axis=0)
    meta = {
        "images_dir": str(images_dir),
        "num_images": int(arr.shape[0]),
        "first_image": str(files[0]),
        "last_image": str(files[-1]),
    }
    return arr, meta


def safetensor_image_shards(cache_dir: Path, file_glob: str) -> List[Path]:
    files = sorted(cache_dir.glob(file_glob))
    if not files:
        raise FileNotFoundError(f"No shards matching {file_glob!r} under {cache_dir}")
    return files


def shard_num_images(path: Path, image_key: str = "images") -> int:
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if image_key not in f.keys():
            raise KeyError(f"{path} missing key={image_key!r}; keys={list(f.keys())}")
        shape = f.get_slice(image_key).get_shape()
    if len(shape) != 4:
        raise RuntimeError(f"Expected {image_key} shape [N,3,H,W], got {shape} in {path}")
    return int(shape[0])


def iter_safetensor_image_batches(
    shards: Sequence[Path],
    *,
    image_key: str,
    batch_size: int,
    max_samples: Optional[int] = None,
) -> Iterator[Tuple[int, torch.Tensor, Path]]:
    """Yield (global_start_index, uint8_chw_batch_cpu, shard_path)."""
    seen = 0
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            z = f.get_slice(image_key)
            shape = z.get_shape()
            n = int(shape[0])
            if max_samples is not None:
                remaining = int(max_samples) - seen
                if remaining <= 0:
                    break
                n = min(n, remaining)
            for s in range(0, n, int(batch_size)):
                e = min(n, s + int(batch_size))
                batch = z[s:e].contiguous()
                yield seen + s, batch, shard
            seen += n
            if max_samples is not None and seen >= int(max_samples):
                break


@dataclass
class RunningMoments:
    dims: int
    count: int = 0

    def __post_init__(self) -> None:
        self.sum = np.zeros((self.dims,), dtype=np.float64)
        self.sum_outer = np.zeros((self.dims, self.dims), dtype=np.float64)

    def update(self, feats: np.ndarray) -> None:
        if feats.ndim != 2 or feats.shape[1] != self.dims:
            raise RuntimeError(f"Bad features shape {feats.shape}; expected [N,{self.dims}]")
        x = feats.astype(np.float64, copy=False)
        self.count += int(x.shape[0])
        self.sum += x.sum(axis=0)
        self.sum_outer += x.T @ x

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.count < 2:
            raise RuntimeError(f"Need at least two samples for covariance, got {self.count}")
        mu = self.sum / float(self.count)
        cov = (self.sum_outer - float(self.count) * np.outer(mu, mu)) / float(self.count - 1)
        cov = (cov + cov.T) * 0.5
        return mu.astype(np.float64), cov.astype(np.float64)


def covariance_from_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if features.ndim != 2 or features.shape[0] < 2:
        raise RuntimeError(f"Need feature matrix [N,D] with N>=2, got shape={features.shape}")
    x = features.astype(np.float64, copy=False)
    mu = x.mean(axis=0)
    xc = x - mu
    sigma = (xc.T @ xc) / float(x.shape[0] - 1)
    sigma = (sigma + sigma.T) * 0.5
    return mu, sigma


def save_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
    tmp.replace(path)


def load_reference_stats(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as f:
        mu = f["mu"].astype(np.float64)
        sigma = f["sigma"].astype(np.float64)
        meta: Dict[str, Any] = {}
        for key in f.files:
            if key in {"mu", "sigma"}:
                continue
            arr = f[key]
            if arr.shape == ():
                meta[key] = arr.item()
            elif arr.size <= 16:
                meta[key] = arr.tolist()
            else:
                meta[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    return mu, sigma, meta


def trace_sqrt_product_lowrank(
    *,
    features1: np.ndarray,
    mu1: np.ndarray,
    sigma2: np.ndarray,
    device: torch.device,
) -> float:
    """Trace sqrt(Sigma_1 Sigma_2) with Sigma_1 from low-rank sample features.

    If Sigma_1 = X^T X / (n-1), the non-zero eigenvalues of Sigma_1 Sigma_2
    equal those of X Sigma_2 X^T / (n-1). This is exact and much faster for
    small generated eval batches.
    """
    n = int(features1.shape[0])
    if n < 2:
        raise RuntimeError("Low-rank Frechet path needs n>=2")
    x = torch.as_tensor(features1.astype(np.float64, copy=False) - mu1[None, :], dtype=torch.float64, device=device)
    s2 = torch.as_tensor(sigma2, dtype=torch.float64, device=device)
    gram = (x @ s2 @ x.T) / float(n - 1)
    gram = (gram + gram.T) * 0.5
    vals = torch.linalg.eigvalsh(gram)
    return float(torch.sqrt(torch.clamp(vals, min=0)).sum().detach().cpu().item())


def trace_sqrt_product_full(sigma1: np.ndarray, sigma2: np.ndarray, *, device: torch.device) -> float:
    a = torch.as_tensor(sigma1, dtype=torch.float64, device=device)
    b = torch.as_tensor(sigma2, dtype=torch.float64, device=device)
    a = (a + a.T) * 0.5
    b = (b + b.T) * 0.5
    evals, evecs = torch.linalg.eigh(a)
    evals = torch.clamp(evals, min=0)
    sqrt_a = (evecs * torch.sqrt(evals).unsqueeze(0)) @ evecs.T
    middle = sqrt_a @ b @ sqrt_a
    middle = (middle + middle.T) * 0.5
    vals = torch.linalg.eigvalsh(middle)
    return float(torch.sqrt(torch.clamp(vals, min=0)).sum().detach().cpu().item())


def frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    *,
    features1: Optional[np.ndarray] = None,
    sqrt_device: torch.device | str = "cpu",
) -> float:
    mu1 = np.asarray(mu1, dtype=np.float64)
    mu2 = np.asarray(mu2, dtype=np.float64)
    sigma1 = np.asarray(sigma1, dtype=np.float64)
    sigma2 = np.asarray(sigma2, dtype=np.float64)
    if mu1.shape != mu2.shape:
        raise RuntimeError(f"Mean shape mismatch: {mu1.shape} vs {mu2.shape}")
    if sigma1.shape != sigma2.shape:
        raise RuntimeError(f"Cov shape mismatch: {sigma1.shape} vs {sigma2.shape}")
    dev = resolve_device(str(sqrt_device)) if not isinstance(sqrt_device, torch.device) else sqrt_device
    diff = mu1 - mu2
    if features1 is not None and features1.shape[0] < features1.shape[1]:
        tr_covmean = trace_sqrt_product_lowrank(features1=features1, mu1=mu1, sigma2=sigma2, device=dev)
    else:
        tr_covmean = trace_sqrt_product_full(sigma1, sigma2, device=dev)
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * tr_covmean)
    # Negative values can occur only from numerical error.
    return max(fid, 0.0)


def _poly3_sum(x: torch.Tensor, y: torch.Tensor, *, dims: int) -> torch.Tensor:
    return ((x @ y.T) / float(dims) + 1.0).pow(3).sum()


def polynomial_mmd2_unbiased(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    device: torch.device,
    chunk_size: int = 2048,
) -> float:
    """Unbiased MMD^2 with KID's degree-3 polynomial kernel."""
    if x_np.shape[0] < 2 or y_np.shape[0] < 2:
        raise RuntimeError(f"Need both sample counts >=2 for MMD, got {x_np.shape[0]} and {y_np.shape[0]}")
    dims = int(x_np.shape[1])
    x = torch.as_tensor(x_np.astype(np.float32, copy=False), device=device)
    y = torch.as_tensor(y_np.astype(np.float32, copy=False), device=device)
    n = int(x.shape[0])
    m = int(y.shape[0])

    def within_sum(a: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float64, device=device)
        for s in range(0, int(a.shape[0]), int(chunk_size)):
            e = min(int(a.shape[0]), s + int(chunk_size))
            k = ((a[s:e] @ a.T) / float(dims) + 1.0).pow(3).double()
            # Remove diagonal entries for rows s:e.
            diag_cols = torch.arange(s, e, device=device)
            row_ids = torch.arange(0, e - s, device=device)
            k[row_ids, diag_cols] = 0.0
            total += k.sum()
        return total

    def cross_sum(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        total = torch.zeros((), dtype=torch.float64, device=device)
        for s in range(0, int(a.shape[0]), int(chunk_size)):
            e = min(int(a.shape[0]), s + int(chunk_size))
            total += _poly3_sum(a[s:e], b, dims=dims).double()
        return total

    k_xx = within_sum(x) / float(n * (n - 1))
    k_yy = within_sum(y) / float(m * (m - 1))
    k_xy = cross_sum(x, y) / float(n * m)
    return float((k_xx + k_yy - 2.0 * k_xy).detach().cpu().item())


def load_feature_bank(path: Path, *, key: str = "features", max_ref: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as f:
        feats = f[key].astype(np.float32)
        meta: Dict[str, Any] = {}
        for k in f.files:
            if k == key:
                continue
            arr = f[k]
            if arr.shape == ():
                meta[k] = arr.item()
            elif arr.size <= 32:
                meta[k] = arr.tolist()
            else:
                meta[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
    if max_ref is not None and feats.shape[0] > int(max_ref):
        feats = feats[: int(max_ref)]
        meta["truncated_to_max_ref"] = int(max_ref)
    return feats, meta


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
