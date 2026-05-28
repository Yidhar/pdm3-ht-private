#!/usr/bin/env python3
"""Decode B3/MeanFlow PAE latent samples and write compact image-space smoke metrics.

This is intentionally a *small* step-10k wiring/eval helper, not a replacement for
full 50k Inception FID. It verifies that the EMA latent sampler, PAE decoder, image
save path, and basic distribution-comparison metrics are connected before a long run.

Outputs under eval_dir by default:
  images/generated/*.png
  images/real_ref/*.png
  images/generated_grid.png
  images/real_ref_grid.png
  decoded_samples.npz
  compact_metrics.json
  compact_metrics.md

Metrics named ``compact_*`` use deterministic random-projection pixel features and
are not official ADM/clean-fid numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw
from safetensors import safe_open
from safetensors.torch import load_file
from scipy import linalg


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
        params["decoder_config_path"] = resolve_path_maybe_relative(str(params["decoder_config_path"]), PAE_ROOT)
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
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[key]


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_pae_model(pae_config_path: Path, pae_ckpt_path: Path, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
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
    model = model.to(device=device, dtype=dtype).eval()
    return model


def safe_load_tensor(path: Path, key: str) -> torch.Tensor:
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        return sf.get_tensor(key)


def load_generated_latents(path: Path, key: str = "samples") -> Tuple[torch.Tensor, Optional[torch.Tensor], List[str]]:
    tensors = load_file(str(path), device="cpu")
    if key not in tensors:
        raise KeyError(f"{path} has keys {sorted(tensors.keys())}, expected key={key!r}")
    labels = tensors.get("labels")
    return tensors[key].float().contiguous(), (labels.long().contiguous() if labels is not None else None), sorted(tensors.keys())


def list_shards(data_dir: Path, glob_pat: str = "*.safetensors") -> List[Path]:
    files = sorted([p for p in data_dir.glob(glob_pat) if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No shard files matched {data_dir / glob_pat}")
    return files


def read_shard_len(path: Path) -> int:
    with safe_open(str(path), framework="pt", device="cpu") as sf:
        return int(sf.get_tensor("labels").shape[0])


def load_real_latent_reference(
    data_dir: Path,
    n: int,
    seed: int,
    use_flip: bool = False,
    shard_glob: str = "*.safetensors",
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Deterministically sample n real PAE latents without materializing the full cache."""
    files = list_shards(data_dir, shard_glob)
    # Use a stable per-file length scan; 313 shards is cheap and avoids assumptions for the short last shard.
    lengths = [read_shard_len(p) for p in files]
    total = int(sum(lengths))
    if n > total:
        raise ValueError(f"Requested {n} real refs but cache only has {total}")
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, size=n, replace=False))
    latents: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    refs: List[Dict[str, Any]] = []
    cursor = 0
    pos = 0
    key = "latents_flip" if use_flip else "latents"
    for shard_idx, (path, length) in enumerate(zip(files, lengths)):
        next_cursor = cursor + length
        take: List[int] = []
        while pos < len(indices) and cursor <= int(indices[pos]) < next_cursor:
            take.append(int(indices[pos]) - cursor)
            pos += 1
        if take:
            idx = torch.tensor(take, dtype=torch.long)
            with safe_open(str(path), framework="pt", device="cpu") as sf:
                z = sf.get_tensor(key).index_select(0, idx).float().contiguous()
                y = sf.get_tensor("labels").index_select(0, idx).long().contiguous()
            latents.append(z)
            labels.append(y)
            for local_i, global_i in zip(take, indices[pos - len(take) : pos]):
                refs.append({"global_index": int(global_i), "shard": path.name, "shard_index": shard_idx, "local_index": int(local_i), "key": key})
        cursor = next_cursor
        if pos >= len(indices):
            break
    if pos != len(indices):
        raise RuntimeError(f"Only collected {pos}/{len(indices)} real indices")
    return torch.cat(latents, dim=0), torch.cat(labels, dim=0), refs


@torch.no_grad()
def decode_latents(
    model: torch.nn.Module,
    latents: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
) -> torch.Tensor:
    outs: List[torch.Tensor] = []
    total = int(latents.shape[0])
    for start in range(0, total, batch_size):
        z = latents[start : start + batch_size].to(device=device, dtype=dtype, non_blocking=False)
        if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
            with torch.autocast(device_type="cuda", dtype=dtype):
                x = model.decode(z)
        else:
            x = model.decode(z)
        x = torch.clamp(x.float(), 0.0, 1.0).detach().cpu()
        outs.append(x)
        del z, x
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return torch.cat(outs, dim=0)


def tensor_to_uint8_images(x: torch.Tensor) -> np.ndarray:
    x = torch.clamp(x, 0.0, 1.0)
    arr = (x.mul(255.0).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().numpy())
    return arr


def save_images(arr: np.ndarray, labels: Optional[torch.Tensor], out_dir: Path, prefix: str) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    lab = labels.cpu().numpy().tolist() if labels is not None else [None] * int(arr.shape[0])
    for i, img in enumerate(arr):
        suffix = f"_label{int(lab[i]):03d}" if lab[i] is not None else ""
        path = out_dir / f"{prefix}_{i:06d}{suffix}.png"
        Image.fromarray(img).save(path)
        paths.append(str(path))
    return paths


def make_grid(arr: np.ndarray, labels: Optional[torch.Tensor], path: Path, cols: int = 8, annotate: bool = True) -> None:
    n, h, w, c = arr.shape
    rows = int(math.ceil(n / cols))
    label_h = 14 if annotate else 0
    grid = Image.new("RGB", (cols * w, rows * (h + label_h)), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid)
    labs = labels.cpu().numpy().tolist() if labels is not None else [None] * n
    for i, img in enumerate(arr):
        r, col = divmod(i, cols)
        x0 = col * w
        y0 = r * (h + label_h)
        grid.paste(Image.fromarray(img), (x0, y0))
        if annotate:
            txt = f"{i:02d}" + (f" y={int(labs[i])}" if labs[i] is not None else "")
            draw.text((x0 + 2, y0 + h), txt, fill=(0, 0, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def image_summary(arr: np.ndarray) -> Dict[str, Any]:
    x = arr.astype(np.float32) / 255.0
    flat = x.reshape(-1, 3)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "finite": bool(np.isfinite(x).all()),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "channel_mean_rgb": [float(v) for v in flat.mean(axis=0)],
        "channel_std_rgb": [float(v) for v in flat.std(axis=0, ddof=0)],
    }


def latent_summary(z: torch.Tensor) -> Dict[str, Any]:
    zf = z.float()
    return {
        "shape": list(z.shape),
        "dtype": str(z.dtype),
        "finite": bool(torch.isfinite(zf).all().item()),
        "mean": float(zf.mean().item()),
        "std": float(zf.std(unbiased=False).item()),
        "min": float(zf.min().item()),
        "max": float(zf.max().item()),
    }


def random_projection_features(arr: np.ndarray, feature_dim: int, image_size: int, seed: int) -> np.ndarray:
    # arr: NHWC uint8 [0,255]. Downsample on CPU using torch interpolate, then deterministic RP.
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    if int(x.shape[-1]) != image_size or int(x.shape[-2]) != image_size:
        x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    flat = x.flatten(1).numpy().astype(np.float64)
    rng = np.random.default_rng(seed)
    proj = rng.normal(0.0, 1.0 / math.sqrt(flat.shape[1]), size=(flat.shape[1], feature_dim)).astype(np.float64)
    feats = flat @ proj
    return feats.astype(np.float64)


def covariance(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= 1:
        return np.eye(x.shape[1], dtype=np.float64) * 1e-6
    return np.cov(x, rowvar=False).astype(np.float64)


def frechet_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
    mu_x, mu_y = x.mean(axis=0), y.mean(axis=0)
    sx, sy = covariance(x), covariance(y)
    sx = sx + np.eye(sx.shape[0], dtype=np.float64) * eps
    sy = sy + np.eye(sy.shape[0], dtype=np.float64) * eps
    covmean, _ = linalg.sqrtm(sx @ sy, disp=False)
    if not np.isfinite(covmean).all():
        covmean = linalg.sqrtm((sx + np.eye(sx.shape[0]) * eps * 10) @ (sy + np.eye(sy.shape[0]) * eps * 10))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu_x - mu_y
    val = diff.dot(diff) + np.trace(sx + sy - 2.0 * covmean)
    return float(np.real(val))


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x2 = np.sum(x * x, axis=1, keepdims=True)
    y2 = np.sum(y * y, axis=1, keepdims=True).T
    return np.maximum(x2 + y2 - 2.0 * (x @ y.T), 0.0)


def mmd_rbf_unbiased(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    z = np.concatenate([x, y], axis=0)
    d = squared_distances(z, z)
    tri = d[np.triu_indices_from(d, k=1)]
    bandwidth2 = float(np.median(tri[tri > 0])) if np.any(tri > 0) else 1.0
    if not np.isfinite(bandwidth2) or bandwidth2 <= 0:
        bandwidth2 = 1.0
    kxx = np.exp(-squared_distances(x, x) / (2.0 * bandwidth2))
    kyy = np.exp(-squared_distances(y, y) / (2.0 * bandwidth2))
    kxy = np.exp(-squared_distances(x, y) / (2.0 * bandwidth2))
    n, m = x.shape[0], y.shape[0]
    xx = (kxx.sum() - np.trace(kxx)) / max(n * (n - 1), 1)
    yy = (kyy.sum() - np.trace(kyy)) / max(m * (m - 1), 1)
    xy = kxy.mean()
    return float(xx + yy - 2.0 * xy), bandwidth2


def kid_poly3_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    d = float(x.shape[1])
    kxx = (x @ x.T / d + 1.0) ** 3
    kyy = (y @ y.T / d + 1.0) ** 3
    kxy = (x @ y.T / d + 1.0) ** 3
    n, m = x.shape[0], y.shape[0]
    xx = (kxx.sum() - np.trace(kxx)) / max(n * (n - 1), 1)
    yy = (kyy.sum() - np.trace(kyy)) / max(m * (m - 1), 1)
    xy = kxy.mean()
    return float(xx + yy - 2.0 * xy)


def compact_metrics(gen_arr: np.ndarray, real_arr: np.ndarray, feature_dim: int, feature_image_size: int, seed: int) -> Dict[str, Any]:
    gen_f = random_projection_features(gen_arr, feature_dim=feature_dim, image_size=feature_image_size, seed=seed)
    real_f = random_projection_features(real_arr, feature_dim=feature_dim, image_size=feature_image_size, seed=seed)
    fid = frechet_distance(real_f, gen_f)
    mmd, bw2 = mmd_rbf_unbiased(real_f, gen_f)
    kid = kid_poly3_unbiased(real_f, gen_f)
    return {
        "feature_method": "uint8_rgb_downsample_random_projection",
        "feature_image_size": int(feature_image_size),
        "feature_dim": int(feature_dim),
        "feature_seed": int(seed),
        "compact_fid_rp": fid,
        "compact_mmd_rbf": mmd,
        "compact_mmd_rbf_bandwidth2": bw2,
        "compact_kid_poly3": kid,
        "official_inception_fid": None,
        "official_metric_note": "compact_* are deterministic random-projection image-space smoke metrics, not official ADM/clean-fid Inception metrics.",
    }


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def write_markdown(path: Path, rec: Dict[str, Any]) -> None:
    m = rec["metrics"]
    lines = [
        "# Step compact decode/eval\n\n",
        f"- created_at_utc: `{rec['created_at_utc']}`\n",
        f"- sample_latents: `{rec['sample_latents_path']}`\n",
        f"- output_dir: `{rec['output_dir']}`\n",
        f"- device: `{rec['device']}`; dtype: `{rec['dtype']}`; decode_batch_size: `{rec['decode_batch_size']}`\n",
        f"- generated_images: `{rec['generated_image_dir']}`\n",
        f"- real_ref_images: `{rec['real_image_dir']}`\n",
        f"- generated_grid: `{rec['generated_grid_path']}`\n",
        f"- real_ref_grid: `{rec['real_grid_path']}`\n",
        "\n## Compact metrics\n\n",
        "| metric | value |\n|---|---:|\n",
        f"| compact_fid_rp{m['feature_dim']} | `{m['compact_fid_rp']}` |\n",
        f"| compact_mmd_rbf_rp{m['feature_dim']} | `{m['compact_mmd_rbf']}` |\n",
        f"| compact_kid_poly3_rp{m['feature_dim']} | `{m['compact_kid_poly3']}` |\n",
        f"| mmd_bandwidth2 | `{m['compact_mmd_rbf_bandwidth2']}` |\n",
        "\n> Note: compact metrics are random-projection image-space smoke metrics, not official Inception FID/KID. They are for step-10k pipeline wiring.\n\n",
        "## Image stats\n\n",
        "| split | mean | std | min | max | finite |\n|---|---:|---:|---:|---:|---|\n",
        f"| generated | `{rec['generated_image_summary']['mean']}` | `{rec['generated_image_summary']['std']}` | `{rec['generated_image_summary']['min']}` | `{rec['generated_image_summary']['max']}` | `{rec['generated_image_summary']['finite']}` |\n",
        f"| real_ref | `{rec['real_image_summary']['mean']}` | `{rec['real_image_summary']['std']}` | `{rec['real_image_summary']['min']}` | `{rec['real_image_summary']['max']}` | `{rec['real_image_summary']['finite']}` |\n",
        "\n## Latent stats\n\n",
        "| split | shape | mean | std | min | max | finite |\n|---|---|---:|---:|---:|---:|---|\n",
        f"| generated | `{rec['generated_latent_summary']['shape']}` | `{rec['generated_latent_summary']['mean']}` | `{rec['generated_latent_summary']['std']}` | `{rec['generated_latent_summary']['min']}` | `{rec['generated_latent_summary']['max']}` | `{rec['generated_latent_summary']['finite']}` |\n",
        f"| real_ref | `{rec['real_latent_summary']['shape']}` | `{rec['real_latent_summary']['mean']}` | `{rec['real_latent_summary']['std']}` | `{rec['real_latent_summary']['min']}` | `{rec['real_latent_summary']['max']}` | `{rec['real_latent_summary']['finite']}` |\n",
    ]
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-latents", type=Path, required=True, help="sample_latents.safetensors from trainer eval")
    p.add_argument("--output-dir", type=Path, default=None, help="defaults to sample_latents parent")
    p.add_argument("--pae-config", type=Path, default=REPO_ROOT / "external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml")
    p.add_argument("--pae-ckpt", type=Path, default=REPO_ROOT / "data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt")
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_full")
    p.add_argument("--num-real", type=int, default=64)
    p.add_argument("--real-seed", type=int, default=20260528)
    p.add_argument("--real-use-flip", action="store_true")
    p.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, or auto; default cpu to avoid disturbing active H100 training")
    p.add_argument("--dtype", default="fp32", help="fp32/bf16/fp16; CPU should use fp32")
    p.add_argument("--decode-batch-size", type=int, default=2)
    p.add_argument("--feature-dim", type=int, default=128)
    p.add_argument("--feature-image-size", type=int, default=32)
    p.add_argument("--feature-seed", type=int, default=20260528)
    p.add_argument("--max-samples", type=int, default=0, help="optional cap for generated samples")
    args = p.parse_args()

    started = time.perf_counter()
    out_dir = args.output_dir or args.sample_latents.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print(f"CPU decode requested with {dtype}; using fp32 for CPU numerical/kernel compatibility", flush=True)
        dtype = torch.float32
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True

    gen_z, gen_labels, gen_keys = load_generated_latents(args.sample_latents)
    if args.max_samples and args.max_samples > 0:
        gen_z = gen_z[: args.max_samples]
        if gen_labels is not None:
            gen_labels = gen_labels[: args.max_samples]
    n = int(gen_z.shape[0])
    n_real = min(int(args.num_real), n)
    if n_real <= 0:
        raise ValueError("num-real must be positive")
    real_z, real_labels, real_refs = load_real_latent_reference(args.data_dir, n_real, args.real_seed, use_flip=args.real_use_flip)
    if n_real != n:
        gen_z_metric = gen_z[:n_real]
        gen_labels_metric = gen_labels[:n_real] if gen_labels is not None else None
    else:
        gen_z_metric = gen_z
        gen_labels_metric = gen_labels

    model = load_pae_model(args.pae_config, args.pae_ckpt, device, dtype)
    gen_x = decode_latents(model, gen_z, device, dtype, args.decode_batch_size)
    # Metric reference only needs n_real images. Keep generated image saving for all generated samples.
    real_x = decode_latents(model, real_z, device, dtype, args.decode_batch_size)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    gen_arr = tensor_to_uint8_images(gen_x)
    real_arr = tensor_to_uint8_images(real_x)

    image_root = out_dir / "images"
    gen_dir = image_root / "generated"
    real_dir = image_root / "real_ref"
    gen_paths = save_images(gen_arr, gen_labels, gen_dir, "gen")
    real_paths = save_images(real_arr, real_labels, real_dir, "real")
    gen_grid = image_root / "generated_grid.png"
    real_grid = image_root / "real_ref_grid.png"
    make_grid(gen_arr, gen_labels, gen_grid, cols=8, annotate=True)
    make_grid(real_arr, real_labels, real_grid, cols=8, annotate=True)

    npz_path = out_dir / "decoded_samples.npz"
    np.savez_compressed(
        npz_path,
        generated=gen_arr,
        real_ref=real_arr,
        generated_labels=(gen_labels.cpu().numpy() if gen_labels is not None else np.array([], dtype=np.int64)),
        real_labels=real_labels.cpu().numpy(),
    )

    metrics = compact_metrics(gen_arr[:n_real], real_arr, args.feature_dim, args.feature_image_size, args.feature_seed)
    rec: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "status": "ok",
        "repo_root": str(REPO_ROOT),
        "pae_root": str(PAE_ROOT),
        "sample_latents_path": str(args.sample_latents),
        "sample_latents_keys": gen_keys,
        "output_dir": str(out_dir),
        "generated_image_dir": str(gen_dir),
        "real_image_dir": str(real_dir),
        "generated_grid_path": str(gen_grid),
        "real_grid_path": str(real_grid),
        "decoded_npz_path": str(npz_path),
        "pae_config": str(args.pae_config),
        "pae_ckpt": str(args.pae_ckpt),
        "data_dir": str(args.data_dir),
        "device": str(device),
        "dtype": str(dtype),
        "decode_batch_size": int(args.decode_batch_size),
        "num_generated": int(gen_arr.shape[0]),
        "num_real_ref": int(real_arr.shape[0]),
        "real_seed": int(args.real_seed),
        "real_use_flip": bool(args.real_use_flip),
        "real_refs_first16": real_refs[:16],
        "generated_image_paths_first16": gen_paths[:16],
        "real_image_paths_first16": real_paths[:16],
        "generated_latent_summary": latent_summary(gen_z),
        "real_latent_summary": latent_summary(real_z),
        "generated_image_summary": image_summary(gen_arr),
        "real_image_summary": image_summary(real_arr),
        "metrics": metrics,
        "elapsed_sec": float(time.perf_counter() - started),
    }
    save_json(out_dir / "compact_metrics.json", rec)
    write_markdown(out_dir / "compact_metrics.md", rec)
    print("COMPACT_EVAL", json.dumps(rec, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
