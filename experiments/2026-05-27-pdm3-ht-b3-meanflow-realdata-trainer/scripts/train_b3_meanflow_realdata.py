#!/usr/bin/env python3
"""Phase-0/B3 MeanFlow real-data trainer over PAE latent safetensors.

This script is intentionally experiment-local.  It reuses the already validated
B3 MeanFlow LightningDiT model and transport implementation from
``external/PAE/pae_with_generator`` while adding the production-loop engineering
needed for real ImageNet-1k latent training:

* scalable shard-indexed DataLoader over safetensors latent cache;
* class-conditional label injection;
* bf16 backbone + fp32 live-parameter JVP target;
* periodic checkpoint save/resume;
* periodic eval hook: latent sampling + optional external FID command;
* metrics JSONL + summary JSON/Markdown.
"""

from __future__ import annotations

import argparse
import bisect
import copy
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("DISABLE_XFORMERS", "1")

import numpy as np
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader, Dataset

PAE_GEN_ROOT = Path(os.environ.get("PAE_GEN_ROOT", "/workspace/PDM/external/PAE/pae_with_generator")).resolve()
if str(PAE_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(PAE_GEN_ROOT))

# Reuse the validated B3 model/transport/trainer utility layer.
from train_meanflow_dit import (  # noqa: E402
    COMPILE_UNWRAPPED,
    build_model,
    configure_sdpa,
    cuda_sync,
    git_commit,
    grad_all_finite,
    load_config,
    make_model_kwargs,
    now_utc,
    peak_mb,
    requires_grad,
    reset_peak,
    resolve_device,
    separate_weight_decay,
    set_seed,
    setup_logger,
    update_ema,
)
from transport.meanflow_transport import MIXED_BF16_BACKBONE_FP32_JVP, create_meanflow_transport  # noqa: E402


@dataclass
class ShardInfo:
    path: str
    num_samples: int
    start: int
    end: int
    latent_shape: List[int]
    has_latents_flip: bool


class ShardIndexedLatentDataset(Dataset):
    """Index PAE latent safetensor shards without materializing a 1.28M-entry map.

    Expected shard keys:
        latents       [N, C, H, W]
        latents_flip  [N, C, H, W]  (optional but expected for current cache)
        labels        [N]

    The shard list is snapshotted at construction time.  This is deliberate: the
    full-cache builder may still be appending new shards, while a training run
    should see a stable epoch length and a reproducible file list.
    """

    def __init__(
        self,
        data_dir: str | Path,
        *,
        file_glob: str = "*.safetensors",
        flip_prob: float = 0.5,
        latent_norm: bool = False,
        latent_multiplier: float = 1.0,
        latent_stats_path: Optional[str | Path] = None,
        max_shards: Optional[int] = None,
        max_samples: Optional[int] = None,
        skip_unreadable: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.file_glob = file_glob
        self.flip_prob = float(flip_prob)
        self.latent_norm = bool(latent_norm)
        self.latent_multiplier = float(latent_multiplier)
        self.max_samples = int(max_samples) if max_samples is not None else None
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive when provided")
        self.skip_unreadable = bool(skip_unreadable)
        self.skipped_files: List[Dict[str, str]] = []
        self.shards: List[ShardInfo] = []
        self.ends: List[int] = []
        self.latent_shape: Optional[List[int]] = None

        if not (0.0 <= self.flip_prob <= 1.0):
            raise ValueError("flip_prob must be in [0, 1]")
        if not self.data_dir.exists():
            raise FileNotFoundError(f"latent data_dir does not exist: {self.data_dir}")

        files = sorted(self.data_dir.glob(self.file_glob))
        if max_shards is not None:
            files = files[: int(max_shards)]
        if not files:
            raise FileNotFoundError(f"no files matching {self.file_glob!r} in {self.data_dir}")

        total = 0
        source_total_seen = 0
        for path in files:
            if self.max_samples is not None and total >= self.max_samples:
                break
            try:
                with safe_open(str(path), framework="pt", device="cpu") as f:
                    keys = set(f.keys())
                    missing = {"latents", "labels"} - keys
                    if missing:
                        raise KeyError(f"missing keys {sorted(missing)}")
                    labels_shape = list(f.get_slice("labels").get_shape())
                    latents_shape = list(f.get_slice("latents").get_shape())
                    if len(labels_shape) != 1:
                        raise ValueError(f"labels must be [N], got {labels_shape}")
                    if len(latents_shape) != 4:
                        raise ValueError(f"latents must be [N,C,H,W], got {latents_shape}")
                    if labels_shape[0] != latents_shape[0]:
                        raise ValueError(f"labels/latents N mismatch: {labels_shape[0]} vs {latents_shape[0]}")
                    if "latents_flip" in keys:
                        flip_shape = list(f.get_slice("latents_flip").get_shape())
                        if flip_shape != latents_shape:
                            raise ValueError(f"latents_flip shape mismatch: {flip_shape} vs {latents_shape}")
                    shard_latent_shape = latents_shape[1:]
                    if self.latent_shape is None:
                        self.latent_shape = shard_latent_shape
                    elif self.latent_shape != shard_latent_shape:
                        raise ValueError(f"latent shape changed: {self.latent_shape} vs {shard_latent_shape} in {path}")
                    source_n = int(labels_shape[0])
                    source_total_seen += source_n
                    if self.max_samples is not None:
                        remaining = int(self.max_samples) - total
                        if remaining <= 0:
                            break
                        n = min(source_n, remaining)
                    else:
                        n = source_n
                    if n <= 0:
                        continue
                    info = ShardInfo(
                        path=str(path),
                        num_samples=n,
                        start=total,
                        end=total + n,
                        latent_shape=shard_latent_shape,
                        has_latents_flip="latents_flip" in keys,
                    )
                    self.shards.append(info)
                    total += n
                    self.ends.append(total)
            except Exception as exc:  # noqa: BLE001
                if not self.skip_unreadable:
                    raise
                self.skipped_files.append({"path": str(path), "error_type": type(exc).__name__, "error": str(exc)})
        if not self.shards:
            raise RuntimeError(f"all candidate shards were skipped in {self.data_dir}; skipped={self.skipped_files[:5]}")
        self.total = total
        self.source_total_seen = int(source_total_seen)

        self._latent_mean: Optional[torch.Tensor] = None
        self._latent_std: Optional[torch.Tensor] = None
        if self.latent_norm:
            stats_path = Path(latent_stats_path).resolve() if latent_stats_path else self.data_dir / "latents_stats.pt"
            stats = self._load_or_compute_stats(stats_path)
            self._latent_mean = stats["mean"].float()
            self._latent_std = stats["std"].float().clamp_min(1e-6)

    def _load_or_compute_stats(self, stats_path: Path, max_samples: int = 10000) -> Dict[str, torch.Tensor]:
        if stats_path.exists():
            return torch.load(stats_path, map_location="cpu", weights_only=False)
        rng = random.Random(20260527)
        sample_n = min(max_samples, self.total)
        indices = rng.sample(range(self.total), sample_n)
        sums: Optional[torch.Tensor] = None
        sq_sums: Optional[torch.Tensor] = None
        count = 0
        for idx in indices:
            x, _ = self._get_item_no_norm(idx, tensor_key="latents")
            xf = x.float()
            dims = [1, 2]
            cur_sum = xf.sum(dim=dims, keepdim=True)
            cur_sq = (xf * xf).sum(dim=dims, keepdim=True)
            if sums is None:
                sums = cur_sum
                sq_sums = cur_sq
            else:
                sums += cur_sum
                sq_sums += cur_sq
            count += int(xf.shape[1] * xf.shape[2])
        assert sums is not None and sq_sums is not None
        mean = (sums / count).view(1, -1, 1, 1)
        var = (sq_sums / count).view(1, -1, 1, 1) - mean * mean
        std = var.clamp_min(1e-12).sqrt()
        stats = {"mean": mean, "std": std, "num_samples": sample_n, "created_at_utc": now_utc()}
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, stats_path)
        return stats

    def __len__(self) -> int:
        return int(self.total)

    def _locate(self, idx: int) -> Tuple[ShardInfo, int]:
        if idx < 0:
            idx += self.total
        if idx < 0 or idx >= self.total:
            raise IndexError(idx)
        shard_i = bisect.bisect_right(self.ends, idx)
        shard = self.shards[shard_i]
        return shard, idx - shard.start

    def _get_item_no_norm(self, idx: int, tensor_key: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        shard, local_idx = self._locate(idx)
        key = tensor_key
        if key is None:
            use_flip = shard.has_latents_flip and random.random() < self.flip_prob
            key = "latents_flip" if use_flip else "latents"
        with safe_open(shard.path, framework="pt", device="cpu") as f:
            feature = f.get_slice(key)[local_idx : local_idx + 1].squeeze(0)
            label = f.get_slice("labels")[local_idx : local_idx + 1].squeeze(0)
        return feature, label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feature, label = self._get_item_no_norm(idx)
        if self.latent_norm:
            assert self._latent_mean is not None and self._latent_std is not None
            feature = (feature.float().unsqueeze(0) - self._latent_mean) / self._latent_std
            feature = feature.squeeze(0)
        feature = feature * self.latent_multiplier
        return feature, label.long()

    def label_preview(self, max_labels: int = 2048) -> Dict[str, Any]:
        labels: List[torch.Tensor] = []
        remaining = min(int(max_labels), self.total)
        for shard in self.shards:
            if remaining <= 0:
                break
            take = min(remaining, shard.num_samples)
            with safe_open(shard.path, framework="pt", device="cpu") as f:
                labels.append(f.get_slice("labels")[:take].long().cpu())
            remaining -= take
        if not labels:
            return {}
        y = torch.cat(labels, dim=0)
        uniq = torch.unique(y)
        return {
            "num_previewed": int(y.numel()),
            "min": int(y.min().item()),
            "max": int(y.max().item()),
            "unique_count": int(uniq.numel()),
            "first_32": [int(v) for v in y[:32].tolist()],
        }

    def summary(self, *, include_shards: int = 8) -> Dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "file_glob": self.file_glob,
            "num_shards": len(self.shards),
            "total_samples": int(self.total),
            "max_samples": self.max_samples,
            "source_total_seen_before_truncation": int(self.source_total_seen),
            "latent_shape": self.latent_shape,
            "flip_prob": self.flip_prob,
            "latent_norm": self.latent_norm,
            "latent_multiplier": self.latent_multiplier,
            "skip_unreadable": self.skip_unreadable,
            "skipped_files_count": len(self.skipped_files),
            "skipped_files_first": self.skipped_files[:8],
            "first_shards": [asdict(s) for s in self.shards[:include_shards]],
            "last_shards": [asdict(s) for s in self.shards[-min(include_shards, len(self.shards)) :]],
            "snapshotted_at_utc": now_utc(),
        }

    def write_snapshot(self, path: str | Path) -> None:
        payload = self.summary(include_shards=len(self.shards))
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def cycle_loader(loader: DataLoader) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def dump_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    return state


def _coerce_rng_byte_tensor(x: Any, *, cpu: bool) -> torch.Tensor:
    """Return a uint8 RNG state tensor on the requested device.

    Checkpoints are loaded with ``map_location=device`` in the trainer, so the
    saved CPU RNG ByteTensor can be remapped to CUDA on resume.
    ``torch.set_rng_state`` requires a CPU ByteTensor; normalize here so
    checkpoint resume is robust across map_location choices and PyTorch 2.9.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.uint8)
    if x.dtype != torch.uint8:
        x = x.to(dtype=torch.uint8)
    return x.cpu() if cpu else x


def set_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "numpy_random" in state:
        np.random.set_state(state["numpy_random"])
    if "torch_cpu" in state:
        torch.set_rng_state(_coerce_rng_byte_tensor(state["torch_cpu"], cpu=True))
    if torch.cuda.is_available() and "torch_cuda_all" in state:
        try:
            torch.cuda.set_rng_state_all([_coerce_rng_byte_tensor(s, cpu=False) for s in state["torch_cuda_all"]])
        except Exception:
            pass


def atomic_symlink_latest(latest_path: Path, target_path: Path) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = latest_path.with_name(latest_path.name + ".tmp")
    try:
        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        os.symlink(target_path.name, tmp)
        os.replace(tmp, latest_path)
    except OSError:
        # Filesystems without symlink support: fall back to copying.
        shutil.copy2(target_path, latest_path)


def checkpoint_step_from_name(path: Path) -> int:
    m = re.search(r"step_(\d+)\.pt$", path.name)
    return int(m.group(1)) if m else -1


def prune_checkpoints(ckpt_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=checkpoint_step_from_name)
    for old in ckpts[:-keep_last]:
        try:
            old.unlink()
        except FileNotFoundError:
            pass


def save_checkpoint(
    ckpt_dir: Path,
    *,
    step: int,
    model: nn.Module,
    ema: Optional[nn.Module],
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    dataset_summary: Dict[str, Any],
    keep_last: int,
    logger: Any,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:08d}.pt"
    tmp = ckpt_dir / f".step_{step:08d}.pt.tmp"
    state = {
        "created_at_utc": now_utc(),
        "step": int(step),
        "model": model.state_dict(),
        "ema": ema.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scaler": None,  # bf16 path uses no GradScaler.
        "config": cfg,
        "dataset_summary": dataset_summary,
        "rng_state": get_rng_state(),
        "torch_version": torch.__version__,
    }
    torch.save(state, tmp)
    os.replace(tmp, path)
    atomic_symlink_latest(ckpt_dir / "latest.pt", path)
    prune_checkpoints(ckpt_dir, keep_last)
    logger.info("checkpoint saved step=%d path=%s latest=%s", step, path, ckpt_dir / "latest.pt")
    return path


def resolve_resume_path(output_dir: Path, resume_from: Optional[str]) -> Optional[Path]:
    if not resume_from:
        return None
    if str(resume_from).lower() == "auto":
        p = output_dir / "checkpoints" / "latest.pt"
        return p if p.exists() or p.is_symlink() else None
    return Path(resume_from).resolve()


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema: Optional[nn.Module],
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    restore_rng: bool,
    logger: Any,
) -> int:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if restore_rng:
        set_rng_state(ckpt.get("rng_state", {}))
    step = int(ckpt.get("step", 0))
    logger.info("resumed checkpoint path=%s step=%d restore_rng=%s", path, step, restore_rng)
    return step


def lr_schedule_type(opt_cfg: Dict[str, Any]) -> str:
    """Return the configured LR schedule type while supporting old flat configs."""
    sched_cfg = opt_cfg.get("scheduler", {})
    if isinstance(sched_cfg, dict):
        return str(sched_cfg.get("type", sched_cfg.get("name", opt_cfg.get("lr_schedule", "constant")))).lower()
    if sched_cfg:
        return str(sched_cfg).lower()
    return str(opt_cfg.get("lr_schedule", "constant")).lower()


def lr_for_step(opt_cfg: Dict[str, Any], *, step: int, max_steps: int) -> float:
    """Step-indexed LR scheduler.

    Backward compatible default is constant ``optimizer.lr``.  Cosine configs can
    be specified either as the older flat fragment:

        optimizer:
          lr: 0.000075
          min_lr: 0.0000075
          scheduler: cosine
          warmup_steps: 13333
          decay_end_step: 1070000

    or as a nested schedule:

        optimizer:
          lr: 0.000075
          scheduler:
            type: cosine
            min_lr: 0.0000075
            warmup_steps: 13333
            end_step: 1070000
    """
    base_lr = float(opt_cfg.get("lr", opt_cfg.get("base_lr", 1e-4)))
    sched_type = lr_schedule_type(opt_cfg)
    if sched_type in {"", "none", "constant", "null"}:
        return base_lr
    if sched_type not in {"cosine", "warmup_cosine", "cosine_warmup"}:
        raise ValueError(f"unsupported optimizer scheduler={sched_type!r}")

    sched_cfg = opt_cfg.get("scheduler", {})
    if not isinstance(sched_cfg, dict):
        sched_cfg = {}
    min_lr = float(sched_cfg.get("min_lr", opt_cfg.get("min_lr", 0.0)))
    warmup_steps = int(sched_cfg.get("warmup_steps", opt_cfg.get("warmup_steps", 0)))
    end_step = int(
        sched_cfg.get(
            "end_step",
            sched_cfg.get("decay_end_step", opt_cfg.get("decay_end_step", max_steps)),
        )
    )
    step_i = max(0, int(step))
    if warmup_steps > 0 and step_i < warmup_steps:
        return base_lr * float(step_i) / float(max(1, warmup_steps))
    decay_start = max(0, warmup_steps)
    decay_end = max(decay_start + 1, end_step)
    progress = min(1.0, max(0.0, float(step_i - decay_start) / float(decay_end - decay_start)))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


@torch.no_grad()
def sample_meanflow_latents(
    *,
    model: nn.Module,
    transport: Any,
    labels: torch.Tensor,
    latent_shape: Sequence[int],
    num_steps: int,
    precision_mode: str,
    device: torch.device,
) -> torch.Tensor:
    """Simple deterministic Euler sampler in latent space.

    Orientation follows ``meanflow_transport.py``: data at t=0, noise at t=1,
    and one step from t to r uses ``z_r = z_t - (t-r) * u_theta(z_t,r,t,y)``.
    """
    was_training = model.training
    model.eval()
    try:
        b = int(labels.shape[0])
        c, h, w = [int(v) for v in latent_shape]
        z = torch.randn(b, c, h, w, device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)
        force_drop_ids = torch.zeros_like(labels, dtype=torch.long, device=device)
        times = torch.linspace(1.0, 0.0, int(num_steps) + 1, device=device, dtype=torch.float32)
        for i in range(int(num_steps)):
            t_val = times[i]
            r_val = times[i + 1]
            t = torch.full((b,), float(t_val.item()), device=device, dtype=torch.float32)
            r = torch.full((b,), float(r_val.item()), device=device, dtype=torch.float32)
            model_kwargs = {"y": labels, "force_drop_ids": force_drop_ids}
            u = transport.call_model(model, z, r, t, model_kwargs, mode=precision_mode)
            z = z - (t - r).view(b, 1, 1, 1) * u.float()
        return z.detach().cpu()
    finally:
        model.train(was_training)


def make_eval_labels(eval_cfg: Dict[str, Any], *, num_samples: int, num_classes: int, device: torch.device) -> torch.Tensor:
    fixed = eval_cfg.get("fixed_labels")
    if fixed:
        vals = [int(v) for v in fixed]
        reps = (num_samples + len(vals) - 1) // len(vals)
        labels = (vals * reps)[:num_samples]
        return torch.tensor(labels, device=device, dtype=torch.long)
    return torch.randint(0, int(num_classes), (int(num_samples),), device=device, dtype=torch.long)


def parse_fid_from_text(text: str) -> Optional[float]:
    # JSON-ish first.
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            for key in ("fid", "FID", "fid50k", "FID50K"):
                if key in obj:
                    return float(obj[key])
        except Exception:
            pass
    m = re.search(r"(?i)\bfid(?:50k)?\b[^0-9+\-.eE]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
    if m:
        return float(m.group(1))
    return None


def run_fid_command(
    *,
    command_template: str,
    sample_dir: Path,
    output_dir: Path,
    step: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    if not command_template:
        return {"fid_status": "skipped_no_command", "fid": None}
    command = command_template.format(sample_dir=str(sample_dir), output_dir=str(output_dir), step=step)
    started = time.perf_counter()
    proc = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=timeout_sec, cwd=str(PAE_GEN_ROOT))
    elapsed = time.perf_counter() - started
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    (sample_dir / "fid_command.txt").write_text(command + "\n", encoding="utf-8")
    (sample_dir / "fid_stdout_stderr.txt").write_text(text, encoding="utf-8")
    fid = parse_fid_from_text(text)
    return {
        "fid_status": "ok" if proc.returncode == 0 else "command_failed",
        "fid": fid,
        "fid_command": command,
        "fid_returncode": proc.returncode,
        "fid_elapsed_sec": elapsed,
    }


def run_eval(
    *,
    step: int,
    eval_cfg: Dict[str, Any],
    output_dir: Path,
    model: nn.Module,
    ema: Optional[nn.Module],
    transport: Any,
    latent_shape: Sequence[int],
    num_classes: int,
    device: torch.device,
    logger: Any,
) -> Dict[str, Any]:
    eval_dir = output_dir / "eval" / f"step_{step:08d}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    num_samples = int(eval_cfg.get("num_samples", 16))
    sample_steps = int(eval_cfg.get("sample_steps", 8))
    precision_mode = eval_cfg.get("precision_mode", "bf16_autocast")
    use_ema = bool(eval_cfg.get("use_ema", True)) and ema is not None
    eval_model = ema if use_ema else model
    labels = make_eval_labels(eval_cfg, num_samples=num_samples, num_classes=num_classes, device=device)

    cuda_sync(device)
    reset_peak(device)
    started = time.perf_counter()
    record: Dict[str, Any] = {
        "type": "eval",
        "step": int(step),
        "created_at_utc": now_utc(),
        "eval_dir": str(eval_dir),
        "num_samples": num_samples,
        "sample_steps": sample_steps,
        "precision_mode": precision_mode,
        "use_ema": use_ema,
    }
    try:
        samples = sample_meanflow_latents(
            model=eval_model,
            transport=transport,
            labels=labels,
            latent_shape=latent_shape,
            num_steps=sample_steps,
            precision_mode=precision_mode,
            device=device,
        )
        cuda_sync(device)
        sample_path = eval_dir / "sample_latents.safetensors"
        save_file({"samples": samples.contiguous().float(), "labels": labels.detach().cpu().long().contiguous()}, str(sample_path))
        record.update(
            {
                "sample_status": "ok",
                "sample_path": str(sample_path),
                "sample_shape": list(samples.shape),
                "sample_finite": bool(torch.isfinite(samples).all().item()),
                "sample_mean": float(samples.float().mean().item()),
                "sample_std": float(samples.float().std().item()),
                "sample_elapsed_sec": time.perf_counter() - started,
                "sample_peak_memory_mb": peak_mb(device),
                "label_min": int(labels.min().item()),
                "label_max": int(labels.max().item()),
                "label_unique_count": int(torch.unique(labels).numel()),
            }
        )
        fid_rec = run_fid_command(
            command_template=str(eval_cfg.get("fid_command_template", "") or ""),
            sample_dir=eval_dir,
            output_dir=output_dir,
            step=step,
            timeout_sec=int(eval_cfg.get("fid_timeout_sec", 3600)),
        )
        record.update(fid_rec)
    except subprocess.TimeoutExpired as exc:
        record.update({"sample_status": record.get("sample_status", "unknown"), "fid_status": "timeout", "fid_error": str(exc), "fid": None})
    except Exception as exc:  # noqa: BLE001
        record.update({"sample_status": "error", "error_type": type(exc).__name__, "error": str(exc), "fid_status": "not_run", "fid": None})
    finally:
        record["elapsed_sec"] = time.perf_counter() - started
    save_json(eval_dir / "eval_record.json", record)
    logger.info(
        "eval step=%d sample_status=%s fid_status=%s fid=%s sample_path=%s",
        step,
        record.get("sample_status"),
        record.get("fid_status"),
        record.get("fid"),
        record.get("sample_path"),
    )
    return record


def create_summary_markdown(output_dir: Path, payload: Dict[str, Any]) -> None:
    result = payload.get("result", {})
    dataset = payload.get("dataset", {})
    gate = payload.get("gate", {})
    lines: List[str] = []
    lines.append("# B3 MeanFlow Real-Data Trainer Summary\n\n")
    lines.append(f"- created_at_utc: `{payload.get('created_at_utc')}`\n")
    lines.append(f"- output_dir: `{output_dir}`\n")
    lines.append(f"- torch: `{payload.get('torch_version')}`\n")
    lines.append(f"- device: `{payload.get('device')}`; gpu: `{payload.get('cuda_device_name')}`\n")
    lines.append(f"- PAE generator root: `{payload.get('pae_gen_root')}`\n")
    lines.append(f"- compile_unwrapped: `{payload.get('compile_unwrapped')}`\n")
    lines.append("\n## Dataset snapshot\n\n")
    lines.append(f"- data_dir: `{dataset.get('data_dir')}`\n")
    lines.append(f"- total_samples: `{dataset.get('total_samples')}`\n")
    lines.append(f"- num_shards: `{dataset.get('num_shards')}`\n")
    lines.append(f"- latent_shape: `{dataset.get('latent_shape')}`\n")
    lines.append(f"- skipped_files_count: `{dataset.get('skipped_files_count')}`\n")
    lines.append(f"- label_preview: `{payload.get('label_preview')}`\n")
    lines.append("\n## Gate\n\n")
    for k, v in gate.items():
        lines.append(f"- `{k}`: `{v}`\n")
    lines.append("\n## Training\n\n")
    rows = {
        "start_step": result.get("start_step"),
        "final_step": result.get("final_step"),
        "requested_max_steps": result.get("requested_max_steps"),
        "optimizer_steps_this_run": result.get("optimizer_steps_this_run"),
        "final_loss": result.get("final_loss"),
        "loss_all_finite": result.get("loss_all_finite"),
        "grad_all_finite": result.get("grad_all_finite"),
        "class_cond_all_steps": result.get("class_cond_all_steps"),
        "mean_realized_r_eq_t_fraction": result.get("mean_realized_r_eq_t_fraction"),
        "loop_peak_memory_mb": result.get("loop_peak_memory_mb"),
        "total_elapsed_sec": result.get("total_elapsed_sec"),
    }
    lines.append("| item | value |\n|---|---:|\n")
    for k, v in rows.items():
        lines.append(f"| {k} | `{v}` |\n")
    lines.append("\n## Checkpoints\n\n")
    for p in result.get("checkpoints", []):
        lines.append(f"- `{p}`\n")
    lines.append("\n## FD audits\n\n")
    lines.append("| step | fd_rel_err | all-r=t target-v max | jvp MB | fd MB | r=t frac |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|\n")
    for a in result.get("fd_audits", []):
        lines.append(
            f"| {a.get('step')} | `{a.get('fd_rel_err_full_jvp')}` | `{a.get('all_r_eq_t_degenerate_target_minus_v_max_abs')}` | "
            f"`{a.get('jvp_peak_memory_mb')}` | `{a.get('fd_peak_memory_mb')}` | `{a.get('realized_r_eq_t_fraction')}` |\n"
        )
    lines.append("\n## Eval records\n\n")
    lines.append("| step | sample_status | sample_path | fid_status | fid |\n")
    lines.append("|---:|---|---|---|---:|\n")
    for e in result.get("eval_records", []):
        lines.append(f"| {e.get('step')} | `{e.get('sample_status')}` | `{e.get('sample_path')}` | `{e.get('fid_status')}` | `{e.get('fid')}` |\n")
    lines.append("\n## Interpretation\n\n")
    lines.append("- `class_cond_all_steps=True` means ImageNet labels were read from safetensors and passed into `model_kwargs['y']` each step.\n")
    lines.append("- `live_param_jvp_target=True` and `ema_used_for_target_jvp=False` are inherited from the transport diagnostics.\n")
    lines.append("- Smoke FID is intentionally skipped unless `eval.fid_command_template` is configured; the loop still writes sample latent tensors and an eval record.\n")
    (output_dir / "train_summary.md").write_text("".join(lines), encoding="utf-8")


def build_gate(result: Dict[str, Any], eval_cfg: Dict[str, Any]) -> Dict[str, Any]:
    fd_audits = result.get("fd_audits", [])
    last_fd = fd_audits[-1] if fd_audits else {}
    fd_rel = last_fd.get("fd_rel_err_full_jvp")
    degen = last_fd.get("all_r_eq_t_degenerate_target_minus_v_max_abs")
    eval_required = bool(eval_cfg.get("enabled", False))
    eval_ok = (not eval_required) or any(e.get("sample_status") == "ok" for e in result.get("eval_records", []))
    ckpt_ok = bool(result.get("checkpoints")) and bool(result.get("latest_checkpoint_exists"))
    fd_ok = fd_rel is not None and float(fd_rel) < 1e-2
    degen_ok = degen is not None and float(degen) < 1e-7
    train_ok = (
        result.get("final_step") == result.get("requested_max_steps")
        and result.get("loss_all_finite") is True
        and result.get("grad_all_finite") is True
        and result.get("u_du_target_all_finite") is True
    )
    practical = bool(
        train_ok
        and result.get("class_cond_all_steps") is True
        and result.get("target_detached_all_steps") is True
        and result.get("jvp_param_source_all_live") is True
        and result.get("mixed_modes_ok") is True
        and fd_ok
        and degen_ok
        and ckpt_ok
        and eval_ok
        and result.get("loop_peak_memory_mb") is not None
    )
    return {
        "train_ok": train_ok,
        "class_cond_ok": result.get("class_cond_all_steps") is True,
        "live_param_jvp_target": result.get("jvp_param_source_all_live") is True,
        "ema_not_used_for_jvp_target": result.get("ema_used_for_target_any_step") is False,
        "mixed_precision_recipe_ok": result.get("mixed_modes_ok") is True,
        "fd_ok_lt_1e_2": fd_ok,
        "fd_rel_err_full_jvp": fd_rel,
        "degenerate_ok_lt_1e_7": degen_ok,
        "all_r_eq_t_degenerate_target_minus_v_max_abs": degen,
        "checkpoint_ok": ckpt_ok,
        "eval_sample_ok": eval_ok,
        "memory_logged": result.get("loop_peak_memory_mb") is not None,
        "practical_gate_pass": practical,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    opt_cfg = cfg.get("optimizer", {})
    meanflow_cfg = cfg.get("meanflow", {})
    eval_cfg = cfg.get("eval", {})

    seed = int(train_cfg.get("global_seed", train_cfg.get("seed", 20260527)))
    set_seed(seed)
    device = resolve_device(train_cfg.get("device", "auto"))
    allow_tf32 = bool(train_cfg.get("allow_tf32", False))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    except Exception:
        pass
    sdpa_info = configure_sdpa(train_cfg.get("sdpa_kernel", "math"), device)

    output_dir = Path(train_cfg.get("output_dir", "output_meanflow")) / train_cfg.get("exp_name", "b3_meanflow_realdata")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    metrics_jsonl = output_dir / "metrics.jsonl"
    resume_path = resolve_resume_path(output_dir, train_cfg.get("resume_from"))
    if metrics_jsonl.exists() and resume_path is None:
        metrics_jsonl.unlink()

    logger.info("B3 MeanFlow real-data trainer starting")
    logger.info("config=%s", args.config)
    logger.info("output_dir=%s", output_dir)
    logger.info(
        "device=%s torch=%s cuda=%s gpu=%s",
        device,
        torch.__version__,
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    )
    logger.info("sdpa=%s allow_tf32=%s compile_unwrapped=%s", sdpa_info, allow_tf32, COMPILE_UNWRAPPED)
    dump_yaml(output_dir / "config_resolved.yaml", cfg)

    dataset = ShardIndexedLatentDataset(
        data_dir=data_cfg["data_path"],
        file_glob=data_cfg.get("file_glob", "*.safetensors"),
        flip_prob=float(data_cfg.get("flip_prob", 0.5)),
        latent_norm=bool(data_cfg.get("latent_norm", False)),
        latent_multiplier=float(data_cfg.get("latent_multiplier", 1.0)),
        latent_stats_path=data_cfg.get("latent_stats_path"),
        max_shards=data_cfg.get("max_shards"),
        max_samples=data_cfg.get("max_samples"),
        skip_unreadable=bool(data_cfg.get("skip_unreadable", True)),
    )
    dataset_summary = dataset.summary(include_shards=8)
    dataset.write_snapshot(output_dir / "dataset_snapshot.json")
    label_preview = dataset.label_preview(int(data_cfg.get("label_preview_count", 2048)))
    expected_total = data_cfg.get("expected_total")
    if expected_total is not None and int(expected_total) != len(dataset):
        logger.warning("dataset length %d != expected_total %s", len(dataset), expected_total)
    logger.info(
        "dataset=%s shards=%d length=%d latent_shape=%s label_preview=%s skipped=%d",
        data_cfg["data_path"],
        len(dataset.shards),
        len(dataset),
        dataset.latent_shape,
        label_preview,
        len(dataset.skipped_files),
    )

    batch_size = int(train_cfg.get("global_batch_size", train_cfg.get("batch_size", 4)))
    num_workers = int(data_cfg.get("num_workers", 0))
    loader_kwargs: Dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": bool(data_cfg.get("shuffle", True)),
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": bool(data_cfg.get("drop_last", True)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(data_cfg.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(data_cfg.get("prefetch_factor", 2))
    loader = DataLoader(dataset, **loader_kwargs)
    data_iter = cycle_loader(loader)

    downsample_ratio = int(cfg.get("vae", {}).get("downsample_ratio", 16))
    image_size = int(data_cfg.get("image_size", 256))
    latent_size = image_size // downsample_ratio
    if dataset.latent_shape is not None and int(dataset.latent_shape[-1]) != latent_size:
        logger.warning("config latent_size=%d but dataset latent_shape=%s", latent_size, dataset.latent_shape)

    model = build_model(cfg, latent_size).to(device=device, dtype=torch.float32)
    model.train()
    ema: Optional[nn.Module] = None
    ema_decay = float(train_cfg.get("ema_decay", 0.9999))
    if bool(train_cfg.get("use_ema", True)):
        ema = copy.deepcopy(model).to(device=device, dtype=torch.float32)
        requires_grad(ema, False)
        ema.eval()
        update_ema(ema, model, decay=0.0)
    model_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model=%s params=%.4fM trainable=%.4fM ema=%s", model_cfg.get("model_type"), model_params / 1e6, trainable_params / 1e6, ema is not None)

    transport = create_meanflow_transport(
        precision_recipe=meanflow_cfg.get("precision_recipe", MIXED_BF16_BACKBONE_FP32_JVP),
        equal_prob=float(meanflow_cfg.get("equal_prob", 0.75)),
        time_sampler=meanflow_cfg.get("time_sampler", "ltg"),
        t_low=float(meanflow_cfg.get("t_low", 0.05)),
        t_high=float(meanflow_cfg.get("t_high", 0.95)),
        lognorm_mu=float(meanflow_cfg.get("lognorm_mu", -0.4)),
        lognorm_sigma=float(meanflow_cfg.get("lognorm_sigma", 1.0)),
        fd_eps=float(meanflow_cfg.get("fd_eps", 1e-2)),
        device_type=device.type,
    )
    logger.info(
        "meanflow precision=%s backbone=%s jvp=%s equal_prob=%s sampler=%s",
        transport.precision_recipe,
        transport.backbone_forward_mode,
        transport.jvp_target_mode,
        transport.equal_prob,
        transport.time_sampler,
    )

    opt = torch.optim.AdamW(
        separate_weight_decay(model, float(opt_cfg.get("weight_decay", 0.0))),
        lr=float(opt_cfg.get("lr", 1e-4)),
        betas=(float(opt_cfg.get("beta1", 0.9)), float(opt_cfg.get("beta2", 0.95))),
    )

    start_step = 0
    if resume_path is not None:
        start_step = load_checkpoint(
            resume_path,
            model=model,
            ema=ema,
            optimizer=opt,
            device=device,
            restore_rng=bool(train_cfg.get("restore_rng", True)),
            logger=logger,
        )

    max_steps = int(train_cfg.get("max_steps", 8))
    log_every = int(train_cfg.get("log_every", 1))
    fd_audit_every = int(meanflow_cfg.get("fd_audit_every", max_steps))
    checkpoint_every = int(train_cfg.get("checkpoint_every", train_cfg.get("ckpt_every", 0)))
    keep_last = int(train_cfg.get("keep_last_checkpoints", 3))
    eval_every = int(eval_cfg.get("every_steps", 0)) if bool(eval_cfg.get("enabled", False)) else 0
    grad_clip = float(opt_cfg.get("max_grad_norm", train_cfg.get("grad_clip", 1.0)))
    ckpt_dir = output_dir / "checkpoints"
    sched_type = lr_schedule_type(opt_cfg)
    logger.info(
        "optimizer lr=%s scheduler=%s min_lr=%s warmup_steps=%s decay_end_step=%s",
        opt_cfg.get("lr", 1e-4),
        sched_type,
        opt_cfg.get("min_lr", opt_cfg.get("scheduler", {}).get("min_lr") if isinstance(opt_cfg.get("scheduler", {}), dict) else None),
        opt_cfg.get("warmup_steps", opt_cfg.get("scheduler", {}).get("warmup_steps") if isinstance(opt_cfg.get("scheduler", {}), dict) else None),
        opt_cfg.get("decay_end_step", opt_cfg.get("scheduler", {}).get("end_step") if isinstance(opt_cfg.get("scheduler", {}), dict) else None),
    )

    steps: List[Dict[str, Any]] = []
    fd_audits: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []
    checkpoints: List[str] = []
    loop_peak = 0.0
    optimizer_steps_this_run = 0
    start_all = time.perf_counter()

    if start_step >= max_steps:
        logger.warning("start_step=%d >= max_steps=%d; no training iterations will run", start_step, max_steps)

    for step in range(start_step + 1, max_steps + 1):
        current_lr = lr_for_step(opt_cfg, step=step, max_steps=max_steps)
        set_optimizer_lr(opt, current_lr)
        x, y = next(data_iter)
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, non_blocking=True).long()
        model_kwargs = make_model_kwargs(y, float(model_cfg.get("class_dropout_prob", 0.0)), int(data_cfg.get("num_classes", 1000)))
        opt.zero_grad(set_to_none=True)
        rec: Dict[str, Any] = {
            "type": "train_step",
            "step": int(step),
            "created_at_utc": now_utc(),
            "batch_size": int(x.shape[0]),
            "optimizer_lr": float(current_lr),
            "optimizer_lr_schedule": sched_type,
            "label_min": int(y.min().item()),
            "label_max": int(y.max().item()),
            "label_unique_count": int(torch.unique(y).numel()),
            "class_cond_injected": "y" in model_kwargs and tuple(model_kwargs["y"].shape) == tuple(y.shape),
            "force_drop_count": int(model_kwargs["force_drop_ids"].sum().item()) if "force_drop_ids" in model_kwargs else None,
            "force_drop_total": int(model_kwargs["force_drop_ids"].numel()) if "force_drop_ids" in model_kwargs else None,
        }
        step_start = time.perf_counter()
        try:
            terms = transport.training_losses(model, x, model_kwargs, return_diagnostics=True, log_memory=True)
            diag = terms.get("diagnostics", {})
            loss = terms["loss"].mean()
            rec.update(diag)
            rec["loss"] = float(loss.detach().float().cpu())

            reset_peak(device)
            cuda_sync(device)
            backward_start = time.perf_counter()
            loss.backward()
            cuda_sync(device)
            rec["backward_sec"] = time.perf_counter() - backward_start
            rec["backward_peak_memory_mb"] = peak_mb(device)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_norm_f = float(grad_norm.detach().float().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
            rec["grad_norm_before_clip"] = grad_norm_f
            rec["grad_clip_threshold"] = grad_clip
            rec["grad_is_finite"] = math.isfinite(grad_norm_f) and grad_all_finite(model)
            rec["loss_finite"] = bool(torch.isfinite(loss.detach()).cpu()) and bool(diag.get("loss_finite", False))
            if rec["grad_is_finite"] and rec["loss_finite"]:
                opt.step()
                if ema is not None:
                    update_ema(ema, model, ema_decay)
                optimizer_steps_this_run += 1
                rec["optimizer_step_applied"] = True
            else:
                rec["optimizer_step_applied"] = False
                rec["stopped_reason"] = "nonfinite_loss_or_grad"
        except Exception as exc:  # noqa: BLE001
            rec.update({"status": "error", "error_type": type(exc).__name__, "error_message": str(exc), "loss_finite": False, "grad_is_finite": False, "optimizer_step_applied": False})
        finally:
            cuda_sync(device)
            rec["elapsed_sec"] = time.perf_counter() - step_start
            candidates = [rec.get("target_jvp_peak_memory_mb"), rec.get("backbone_forward_peak_memory_mb"), rec.get("backward_peak_memory_mb"), peak_mb(device)]
            rec["peak_memory_mb"] = max(float(v) for v in candidates if v is not None) if device.type == "cuda" else None
            if rec["peak_memory_mb"] is not None:
                loop_peak = max(loop_peak, float(rec["peak_memory_mb"]))
            steps.append(rec)
            append_jsonl(metrics_jsonl, rec)

        if step % log_every == 0:
            logger.info(
                "step=%d loss=%.6g lr=%.6g labels=[%s,%s] uniq=%s rt=%.3f targetMB=%s fwdMB=%s bwdMB=%s grad=%.6g class_cond=%s",
                step,
                rec.get("loss", float("nan")),
                rec.get("optimizer_lr", float("nan")),
                rec.get("label_min"),
                rec.get("label_max"),
                rec.get("label_unique_count"),
                rec.get("realized_r_eq_t_fraction", float("nan")),
                rec.get("target_jvp_peak_memory_mb"),
                rec.get("backbone_forward_peak_memory_mb"),
                rec.get("backward_peak_memory_mb"),
                rec.get("grad_norm_before_clip", float("nan")),
                rec.get("class_cond_injected"),
            )

        if fd_audit_every > 0 and (step == 1 or step % fd_audit_every == 0 or step == max_steps):
            audit_kwargs = make_model_kwargs(y, float(model_cfg.get("class_dropout_prob", 0.0)), int(data_cfg.get("num_classes", 1000)))
            audit = transport.fd_audit(model, x, audit_kwargs, log_memory=True)
            audit["type"] = "fd_audit"
            audit["step"] = int(step)
            audit["created_at_utc"] = now_utc()
            fd_audits.append(audit)
            append_jsonl(metrics_jsonl, audit)
            logger.info(
                "FD audit step=%d fd_rel=%.6g degen=%.6g forced_non_degen=%s jvpMB=%s fdMB=%s",
                step,
                audit.get("fd_rel_err_full_jvp", float("nan")),
                audit.get("all_r_eq_t_degenerate_target_minus_v_max_abs", float("nan")),
                audit.get("audit_forced_non_degenerate"),
                audit.get("jvp_peak_memory_mb"),
                audit.get("fd_peak_memory_mb"),
            )

        should_ckpt = checkpoint_every > 0 and (step % checkpoint_every == 0 or step == max_steps)
        if should_ckpt:
            ckpt_path = save_checkpoint(
                ckpt_dir,
                step=step,
                model=model,
                ema=ema,
                optimizer=opt,
                cfg=cfg,
                dataset_summary=dataset_summary,
                keep_last=keep_last,
                logger=logger,
            )
            checkpoints.append(str(ckpt_path))
            append_jsonl(metrics_jsonl, {"type": "checkpoint", "step": step, "path": str(ckpt_path), "created_at_utc": now_utc()})

        should_eval = eval_every > 0 and (step % eval_every == 0 or (step == 1 and bool(eval_cfg.get("run_at_step_1", False))) or step == max_steps)
        if should_eval:
            eval_rec = run_eval(
                step=step,
                eval_cfg=eval_cfg,
                output_dir=output_dir,
                model=model,
                ema=ema,
                transport=transport,
                latent_shape=dataset.latent_shape or [int(cfg.get("vae", {}).get("latent_dim", 32)), latent_size, latent_size],
                num_classes=int(data_cfg.get("num_classes", 1000)),
                device=device,
                logger=logger,
            )
            eval_records.append(eval_rec)
            append_jsonl(metrics_jsonl, eval_rec)

        if rec.get("status") == "error" or not rec.get("loss_finite", False) or not rec.get("grad_is_finite", False):
            logger.error("stopping at step=%d due to nonfinite/error rec=%s", step, rec)
            break

    losses = [s.get("loss") for s in steps if isinstance(s.get("loss"), (float, int))]
    realized = [s.get("realized_r_eq_t_fraction") for s in steps if isinstance(s.get("realized_r_eq_t_fraction"), (float, int))]
    actual_degen = [s.get("actual_rt_samples_target_minus_v_max_abs") for s in steps if isinstance(s.get("actual_rt_samples_target_minus_v_max_abs"), (float, int))]
    final_step = steps[-1]["step"] if steps else start_step
    latest_path = ckpt_dir / "latest.pt"
    result: Dict[str, Any] = {
        "start_step": start_step,
        "requested_max_steps": max_steps,
        "final_step": final_step,
        "optimizer_steps_this_run": optimizer_steps_this_run,
        "recorded_steps_this_run": len(steps),
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "loss_all_finite": all(bool(s.get("loss_finite", False)) for s in steps) if steps else False,
        "grad_all_finite": all(bool(s.get("grad_is_finite", False)) for s in steps) if steps else False,
        "u_du_target_all_finite": all(bool(s.get("u_finite", False) and s.get("du_finite", False) and s.get("target_finite", False)) for s in steps) if steps else False,
        "target_detached_all_steps": all(s.get("target_requires_grad") is False for s in steps) if steps else False,
        "jvp_param_source_all_live": all(s.get("jvp_param_source") == "live" for s in steps) if steps else False,
        "ema_used_for_target_any_step": any(s.get("ema_used_for_target_jvp") is True for s in steps),
        "sample_level_sampler": all(s.get("r_t_sampling_granularity") == "sample_level" for s in steps) if steps else False,
        "mixed_modes_ok": all(
            s.get("backbone_forward_mode") == "bf16_autocast" and s.get("jvp_target_mode") == "fp32" and s.get("fd_audit_mode") == "fp32"
            for s in steps
        ) if steps else False,
        "class_cond_all_steps": all(s.get("class_cond_injected") is True for s in steps) if steps else False,
        "mean_realized_r_eq_t_fraction": sum(float(x) for x in realized) / len(realized) if realized else None,
        "max_actual_rt_samples_target_minus_v_abs": max(actual_degen) if actual_degen else None,
        "loop_peak_memory_mb": loop_peak if device.type == "cuda" else None,
        "steps": steps,
        "fd_audits": fd_audits,
        "eval_records": eval_records,
        "checkpoints": checkpoints,
        "latest_checkpoint": str(latest_path),
        "latest_checkpoint_exists": bool(latest_path.exists() or latest_path.is_symlink()),
        "total_elapsed_sec": time.perf_counter() - start_all,
    }
    payload: Dict[str, Any] = {
        "created_at_utc": now_utc(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "pae_gen_root": str(PAE_GEN_ROOT),
        "pae_git_commit": git_commit(PAE_GEN_ROOT.parents[0]),
        "script": str(Path(__file__).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "config": cfg,
        "sdpa_info": sdpa_info,
        "allow_tf32": allow_tf32,
        "compile_unwrapped": COMPILE_UNWRAPPED,
        "model_params": model_params,
        "trainable_params": trainable_params,
        "ema_present": ema is not None,
        "ema_decay": ema_decay if ema is not None else None,
        "dataset": dataset_summary,
        "label_preview": label_preview,
        "precision_recipe": {
            "mode": transport.precision_recipe,
            "backbone_forward_mode": transport.backbone_forward_mode,
            "jvp_target_mode": transport.jvp_target_mode,
            "fd_audit_mode": transport.fd_audit_mode,
            "jvp_param_source": "live",
            "ema_used_for_target_jvp": False,
            "target_detached": True,
        },
        "result": result,
    }
    payload["gate"] = build_gate(result, eval_cfg)
    save_json(output_dir / "train_summary.json", payload)
    create_summary_markdown(output_dir, payload)
    logger.info(
        "DONE practical_gate=%s final_step=%s final_loss=%s loop_peak=%sMB output=%s",
        payload["gate"].get("practical_gate_pass"),
        result.get("final_step"),
        result.get("final_loss"),
        result.get("loop_peak_memory_mb"),
        output_dir,
    )
    logger.info("wrote %s", output_dir / "train_summary.json")
    logger.info("wrote %s", output_dir / "train_summary.md")


if __name__ == "__main__":
    main()
