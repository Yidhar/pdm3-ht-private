#!/usr/bin/env python3
"""
PDM-3-HT Experiment 6: Phase 0 / B3 MeanFlow mixed-precision integration smoke.

This script is the first B3 (global/sample-level MeanFlow) engineering smoke
built from the Experiment-5 precision recipe:

    bf16/autocast trainable backbone forward/backward
    + fp32 no-autocast JVP/FD target path
    + target = (v - du).detach()
    + live parameters for target JVP (EMA is not used for target construction)
    + sample-level scalar (r, t) sampler with default 75% r=t
    + math SDPA and TF32 disabled for FD/JVP audits
    + no activation checkpoint around the JVP path

Scope:
    The upstream PAE training transport currently implements standard velocity
    / Flow-Matching loss, not MeanFlow.  This smoke therefore provides a
    Phase-0/B3 integration scaffold: official PAE LightningDiT submodules are
    reused, but the forward signature is extended to u_theta(z_t, r, t, y) by
    conditioning on both t and (t-r).  The input is synthetic PAE-stat latent
    data, not a real ImageNet latent shard.  The output is a correctness and
    telemetry smoke result, not an FID/FDr claim.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The official PAE files decorate several methods with @torch.compile.  For a
# forward-mode AD/JVP correctness smoke, Dynamo adds latency and may clash with
# transforms, so disable it unless explicitly requested.
DYNAMO_REQUESTED = "--enable-dynamo" in sys.argv
if not DYNAMO_REQUESTED:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# xFormers is not needed for this smoke and is not guaranteed to be installed.
os.environ.setdefault("XFORMERS_DISABLED", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp
from torch.utils.checkpoint import checkpoint

PAE_GEN_ROOT = Path(os.environ.get("PAE_GEN_ROOT", "/workspace/PDM/external/PAE/pae_with_generator"))
if str(PAE_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(PAE_GEN_ROOT))

try:
    import models.lightningdit as lightningdit_module
    from models.lightningdit import LabelEmbedder, LightningDiT, LightningDiTBlock, TimestepEmbedder
    from models.swiglu_ffn import SwiGLUFFN
except Exception as exc:  # noqa: BLE001
    raise RuntimeError(
        f"Failed to import official PAE LightningDiT modules from {PAE_GEN_ROOT}. "
        "Check PYTHONPATH and dependencies."
    ) from exc


def _plain_modulate(x: torch.Tensor, shift: Optional[torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
    """Plain Python replacement for official @torch.compile modulate."""
    if shift is None:
        return x * (1 + scale.unsqueeze(1))
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _unwrap_torch_compile_for_forward_ad() -> List[str]:
    """Use original Python forward bodies when official methods are compile-wrapped."""
    unwrapped: List[str] = []
    if DYNAMO_REQUESTED:
        return unwrapped

    # Replace the module-global modulate used by unwrapped LightningDiTBlock.
    lightningdit_module.modulate = _plain_modulate
    unwrapped.append("models.lightningdit.modulate->plain_python")

    for cls in (SwiGLUFFN, TimestepEmbedder, LabelEmbedder, LightningDiTBlock):
        fn = getattr(cls, "forward", None)
        wrapped = getattr(fn, "__wrapped__", None)
        if wrapped is not None:
            setattr(cls, "forward", wrapped)
            unwrapped.append(f"{cls.__name__}.forward")
    return unwrapped


COMPILE_UNWRAPPED = _unwrap_torch_compile_for_forward_ad()


@dataclass
class Config:
    seed: int = 20260527
    batch_size: int = 4
    channels: int = 32
    grid: int = 16
    patch_size: int = 1
    width: int = 128
    heads: int = 4
    depth: int = 2
    mlp_ratio: float = 4.0
    num_classes: int = 1000
    class_dropout_prob: float = 0.0
    device: str = "auto"
    mode: str = "bf16_backbone_fp32_jvp"
    train_steps: int = 8
    fd_eps: float = 1e-2
    run_fd: bool = True
    t_low: float = 0.05
    t_high: float = 0.95
    equal_prob: float = 0.75
    time_sampler: str = "ltg"
    ltg_mu: float = -0.4
    ltg_sigma: float = 1.0
    lr: float = 1e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    use_qknorm: bool = False
    use_swiglu: bool = True
    use_rmsnorm: bool = True
    use_rope: bool = True
    use_abs_pos: bool = True
    fused_attn: bool = True
    sdpa_kernel: str = "math"
    use_checkpoint: bool = False
    checkpoint_reentrant: bool = False
    init_scheme: str = "stress_nonzero"
    adaln_init_std: float = 0.02
    final_linear_init: str = "xavier"
    latent_mode: str = "pae_stats_synthetic"
    latent_smooth_passes: int = 2
    latent_multiplier: float = 1.0
    pae_stats_path: str = "/workspace/PDM/external/PAE_hf/Latent-stats/PAE_DINOv2L.pt"
    create_ema: bool = True
    ema_decay: float = 0.9999
    output_dir: Optional[str] = None
    enable_dynamo: bool = False
    allow_tf32: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--grid", type=int, default=16, help="latent H=W grid; PAE f16 ImageNet256 is typically 16")
    p.add_argument("--patch-size", type=int, default=1)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--class-dropout-prob", type=float, default=0.0)
    p.add_argument(
        "--mode",
        type=str,
        default="bf16_backbone_fp32_jvp",
        choices=["fp32", "bf16_autocast", "bf16_backbone_fp32_jvp"],
    )
    p.add_argument("--train-steps", type=int, default=8)
    p.add_argument("--fd-eps", type=float, default=1e-2)
    p.add_argument("--no-fd", action="store_true")
    p.add_argument("--t-low", type=float, default=0.05)
    p.add_argument("--t-high", type=float, default=0.95)
    p.add_argument("--equal-prob", type=float, default=0.75)
    p.add_argument("--time-sampler", type=str, default="ltg", choices=["uniform", "ltg"])
    p.add_argument("--ltg-mu", type=float, default=-0.4)
    p.add_argument("--ltg-sigma", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--use-qknorm", action="store_true")
    p.add_argument("--no-swiglu", dest="use_swiglu", action="store_false")
    p.add_argument("--no-rmsnorm", dest="use_rmsnorm", action="store_false")
    p.add_argument("--no-rope", dest="use_rope", action="store_false")
    p.add_argument("--no-abs-pos", dest="use_abs_pos", action="store_false")
    p.add_argument("--no-fused-attn", dest="fused_attn", action="store_false")
    p.add_argument("--sdpa-kernel", type=str, default="math", choices=["math", "default"])
    p.add_argument("--use-checkpoint", action="store_true", help="diagnostic only; rejected for target JVP path")
    p.add_argument("--checkpoint-reentrant", action="store_true")
    p.add_argument("--init-scheme", type=str, default="stress_nonzero", choices=["stress_nonzero", "official_zero"])
    p.add_argument("--adaln-init-std", type=float, default=0.02)
    p.add_argument("--final-linear-init", type=str, default="xavier", choices=["xavier", "official_zero"])
    p.add_argument(
        "--latent-mode",
        type=str,
        default="pae_stats_synthetic",
        choices=["gaussian", "correlated", "pae_stats_synthetic", "pae_stats_raw_synthetic"],
    )
    p.add_argument("--latent-smooth-passes", type=int, default=2)
    p.add_argument("--latent-multiplier", type=float, default=1.0)
    p.add_argument("--pae-stats-path", type=str, default="/workspace/PDM/external/PAE_hf/Latent-stats/PAE_DINOv2L.pt")
    p.add_argument("--no-ema", dest="create_ema", action="store_false")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--enable-dynamo", action="store_true")
    p.add_argument("--allow-tf32", action="store_true")
    p.set_defaults(use_swiglu=True, use_rmsnorm=True, use_rope=True, use_abs_pos=True, fused_attn=True, create_ema=True)
    return p.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    if args.width % args.heads != 0:
        raise ValueError("width must be divisible by heads")
    if args.use_rope and ((args.width // args.heads) % 4 != 0):
        raise ValueError("for RoPE, width/heads should be divisible by 4")
    if args.grid % args.patch_size != 0:
        raise ValueError("grid must be divisible by patch_size")
    if not (0.0 <= args.equal_prob <= 1.0):
        raise ValueError("equal_prob must be in [0, 1]")
    if not (0.0 <= args.t_low < args.t_high <= 1.0):
        raise ValueError("need 0 <= t_low < t_high <= 1")
    if args.use_checkpoint and args.mode == MIXED_BF16_BACKBONE_FP32_JVP_MODE:
        # This script supports checkpointing only as a diagnostic; the official
        # recipe disallows checkpoint around JVP.  We fail fast to avoid a false
        # PASS in the exact task requested by the user.
        raise ValueError("mixed B3 recipe requires no checkpoint around the JVP path; do not pass --use-checkpoint")
    return Config(
        seed=args.seed,
        batch_size=args.batch_size,
        channels=args.channels,
        grid=args.grid,
        patch_size=args.patch_size,
        width=args.width,
        heads=args.heads,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        num_classes=args.num_classes,
        class_dropout_prob=args.class_dropout_prob,
        device=args.device,
        mode=args.mode,
        train_steps=args.train_steps,
        fd_eps=args.fd_eps,
        run_fd=not args.no_fd,
        t_low=args.t_low,
        t_high=args.t_high,
        equal_prob=args.equal_prob,
        time_sampler=args.time_sampler,
        ltg_mu=args.ltg_mu,
        ltg_sigma=args.ltg_sigma,
        lr=args.lr,
        grad_clip=args.grad_clip,
        weight_decay=args.weight_decay,
        use_qknorm=args.use_qknorm,
        use_swiglu=args.use_swiglu,
        use_rmsnorm=args.use_rmsnorm,
        use_rope=args.use_rope,
        use_abs_pos=args.use_abs_pos,
        fused_attn=args.fused_attn,
        sdpa_kernel=args.sdpa_kernel,
        use_checkpoint=args.use_checkpoint,
        checkpoint_reentrant=args.checkpoint_reentrant,
        init_scheme=args.init_scheme,
        adaln_init_std=args.adaln_init_std,
        final_linear_init=args.final_linear_init,
        latent_mode=args.latent_mode,
        latent_smooth_passes=args.latent_smooth_passes,
        latent_multiplier=args.latent_multiplier,
        pae_stats_path=args.pae_stats_path,
        create_ema=args.create_ema,
        ema_decay=args.ema_decay,
        output_dir=args.output_dir,
        enable_dynamo=args.enable_dynamo,
        allow_tf32=args.allow_tf32,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(s: str) -> torch.device:
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if s == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device(s)


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def allocated_memory_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.memory_allocated(device) / (1024**2))


def norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().float()).cpu())


def rel(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> float:
    return norm(reference - candidate) / (norm(reference) + eps)


def finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().cpu())


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    y = x.detach().float().flatten()
    if y.numel() == 0:
        return {"mean": float("nan"), "std": float("nan")}
    qs = torch.quantile(y.abs(), torch.tensor([0.5, 0.9, 0.95, 0.99], device=y.device))
    return {
        "mean": float(y.mean().cpu()),
        "std": float(y.std(unbiased=False).cpu()),
        "abs_mean": float(y.abs().mean().cpu()),
        "abs_median": float(qs[0].cpu()),
        "abs_p90": float(qs[1].cpu()),
        "abs_p95": float(qs[2].cpu()),
        "abs_p99": float(qs[3].cpu()),
        "abs_max": float(y.abs().max().cpu()),
    }


def fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, str)):
        return str(x)
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def git_commit(path: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL)
        return out.strip()
    except Exception:
        return None


def configure_sdpa(kernel: str, device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {"requested": kernel, "device": str(device)}
    if device.type != "cuda" or not hasattr(torch.backends, "cuda"):
        return info
    cuda_backend = torch.backends.cuda
    if kernel == "math":
        if hasattr(cuda_backend, "enable_flash_sdp"):
            cuda_backend.enable_flash_sdp(False)
        if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
            cuda_backend.enable_mem_efficient_sdp(False)
        if hasattr(cuda_backend, "enable_math_sdp"):
            cuda_backend.enable_math_sdp(True)
        if hasattr(cuda_backend, "enable_cudnn_sdp"):
            cuda_backend.enable_cudnn_sdp(False)
    else:
        if hasattr(cuda_backend, "enable_flash_sdp"):
            cuda_backend.enable_flash_sdp(True)
        if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
            cuda_backend.enable_mem_efficient_sdp(True)
        if hasattr(cuda_backend, "enable_math_sdp"):
            cuda_backend.enable_math_sdp(True)
        if hasattr(cuda_backend, "enable_cudnn_sdp"):
            cuda_backend.enable_cudnn_sdp(True)
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
        fn = getattr(cuda_backend, name, None)
        if callable(fn):
            try:
                info[name] = bool(fn())
            except Exception:
                pass
    return info


class PaeStats:
    def __init__(self, path: Path):
        self.path = path
        self.available = path.exists()
        self.mean_cpu: Optional[torch.Tensor] = None
        self.std_cpu: Optional[torch.Tensor] = None
        self.meta: Dict[str, Any] = {"path": str(path), "available": self.available}
        if self.available:
            obj = torch.load(path, map_location="cpu")
            if not isinstance(obj, dict) or "mean" not in obj or "std" not in obj:
                raise ValueError(f"PAE stats file has unexpected format: {path}")
            self.mean_cpu = obj["mean"].float().contiguous()
            self.std_cpu = obj["std"].float().contiguous()
            self.meta.update(
                {
                    "mean_shape": list(self.mean_cpu.shape),
                    "std_shape": list(self.std_cpu.shape),
                    "mean_dtype_original": str(obj["mean"].dtype) if torch.is_tensor(obj["mean"]) else None,
                    "std_dtype_original": str(obj["std"].dtype) if torch.is_tensor(obj["std"]) else None,
                    "mean_stats": tensor_stats(self.mean_cpu),
                    "std_stats": tensor_stats(self.std_cpu),
                }
            )

    def to_device(self, device: torch.device, channels: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.available or self.mean_cpu is None or self.std_cpu is None:
            raise FileNotFoundError(f"PAE stats not found: {self.path}")
        if self.mean_cpu.shape[1] != channels or self.std_cpu.shape[1] != channels:
            raise ValueError(f"PAE stats channels {self.mean_cpu.shape[1]} != requested channels {channels}")
        return self.mean_cpu.to(device), self.std_cpu.to(device)


def smooth_chw_noise(batch: int, channels: int, grid: int, device: torch.device, passes: int) -> torch.Tensor:
    x = torch.randn(batch, channels, grid, grid, device=device, dtype=torch.float32)
    for _ in range(max(0, passes)):
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    x = x - x.mean(dim=(2, 3), keepdim=True)
    x = x / (x.std(dim=(2, 3), keepdim=True, unbiased=False) + 1e-6)
    return x


def sample_t_scalar(cfg: Config, batch: int, device: torch.device) -> torch.Tensor:
    if cfg.time_sampler == "uniform":
        q = torch.rand(batch, device=device, dtype=torch.float32)
    elif cfg.time_sampler == "ltg":
        logits = cfg.ltg_mu + cfg.ltg_sigma * torch.randn(batch, device=device, dtype=torch.float32)
        q = torch.sigmoid(logits)
    else:
        raise ValueError(cfg.time_sampler)
    return cfg.t_low + (cfg.t_high - cfg.t_low) * q


def sample_problem(cfg: Config, device: torch.device, pae_stats: PaeStats) -> Dict[str, Any]:
    b, c, g = cfg.batch_size, cfg.channels, cfg.grid
    latent_meta: Dict[str, Any] = {
        "latent_mode": cfg.latent_mode,
        "r_t_sampling_granularity": "sample_level",
        "configured_equal_prob": cfg.equal_prob,
    }
    if cfg.latent_mode == "gaussian":
        z0 = torch.randn(b, c, g, g, device=device, dtype=torch.float32)
        latent_meta["source"] = "standard_gaussian_chw"
    elif cfg.latent_mode == "correlated":
        z0 = smooth_chw_noise(b, c, g, device, cfg.latent_smooth_passes)
        latent_meta["source"] = "spatially_smoothed_unit_variance_synthetic"
    elif cfg.latent_mode in {"pae_stats_synthetic", "pae_stats_raw_synthetic"}:
        mean, std = pae_stats.to_device(device, c)
        noise = smooth_chw_noise(b, c, g, device, cfg.latent_smooth_passes)
        raw = mean + std * noise
        if cfg.latent_mode == "pae_stats_synthetic":
            z0 = (raw - mean) / (std + 1e-12) * cfg.latent_multiplier
            latent_meta["source"] = "official_pae_stats_synthetic_normalized_not_real_imagenet_latent"
        else:
            z0 = raw
            latent_meta["source"] = "official_pae_stats_synthetic_raw_not_real_imagenet_latent"
        latent_meta["pae_stats_path"] = str(pae_stats.path)
        latent_meta["latent_multiplier"] = cfg.latent_multiplier
        latent_meta["raw_stats"] = tensor_stats(raw)
        latent_meta["noise_stats"] = tensor_stats(noise)
    else:
        raise ValueError(cfg.latent_mode)

    eps = torch.randn(b, c, g, g, device=device, dtype=torch.float32)
    v = eps - z0
    t = sample_t_scalar(cfg, b, device)
    r_candidate = torch.rand(b, device=device, dtype=torch.float32) * t
    eq_mask = torch.rand(b, device=device) < cfg.equal_prob
    r = torch.where(eq_mask, t, r_candidate)
    zt = (1.0 - t[:, None, None, None]) * z0 + t[:, None, None, None] * eps
    y = torch.randint(0, cfg.num_classes, (b,), device=device, dtype=torch.long)
    latent_meta.update(
        {
            "r_eq_t_count": int(eq_mask.detach().sum().cpu()),
            "r_eq_t_total": int(eq_mask.numel()),
            "realized_r_eq_t_fraction": float(eq_mask.float().mean().cpu()),
            "y_min": int(y.min().detach().cpu()),
            "y_max": int(y.max().detach().cpu()),
        }
    )
    return {"z0": z0, "eps": eps, "v": v, "r": r, "t": t, "zt": zt, "y": y, "eq_mask": eq_mask, "latent_meta": latent_meta}


def init_linear_xavier(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def init_linear_normal(linear: nn.Linear, std: float) -> None:
    nn.init.normal_(linear.weight, std=std)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class B3MeanFlowLightningDiTSmokeModel(nn.Module):
    """Official PAE LightningDiT submodules with sample-level MeanFlow conditioning.

    Forward signature:
        u_theta(z_t, r, t, y)

    Conditioning follows the MeanFlow B3 baseline convention of exposing both
    endpoint time and interval length: c = emb(t) + emb(t-r) + emb(y).
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.core = LightningDiT(
            input_size=cfg.grid,
            patch_size=cfg.patch_size,
            in_channels=cfg.channels,
            hidden_size=cfg.width,
            depth=cfg.depth,
            num_heads=cfg.heads,
            mlp_ratio=cfg.mlp_ratio,
            class_dropout_prob=cfg.class_dropout_prob,
            num_classes=cfg.num_classes,
            learn_sigma=False,
            use_qknorm=cfg.use_qknorm,
            use_swiglu=cfg.use_swiglu,
            use_rope=cfg.use_rope,
            use_rmsnorm=cfg.use_rmsnorm,
            use_checkpoint=False,  # controlled manually below; JVP recipe rejects checkpoint.
            use_abs_pos=cfg.use_abs_pos,
        )
        self.delta_embedder = TimestepEmbedder(cfg.width)
        self.use_checkpoint = cfg.use_checkpoint
        self.checkpoint_reentrant = cfg.checkpoint_reentrant
        self._apply_smoke_initialization()

    @staticmethod
    def modulate_sample(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def _apply_smoke_initialization(self) -> None:
        # The official DiT init zeros block AdaLN and final output for stable long
        # training.  That makes a tiny randomly initialized smoke nearly trivial
        # for JVP (du≈0).  The default smoke init makes time dependence observable
        # while keeping the architecture/submodules official.  Passing
        # --init-scheme official_zero reproduces the zero-init behavior.
        if self.cfg.init_scheme == "official_zero":
            return
        if self.cfg.init_scheme != "stress_nonzero":
            raise ValueError(self.cfg.init_scheme)
        for block in self.core.blocks:
            init_linear_normal(block.adaLN_modulation[-1], self.cfg.adaln_init_std)
        init_linear_normal(self.core.final_layer.adaLN_modulation[-1], self.cfg.adaln_init_std)
        if self.cfg.final_linear_init == "xavier":
            init_linear_xavier(self.core.final_layer.linear)
        elif self.cfg.final_linear_init == "official_zero":
            nn.init.zeros_(self.core.final_layer.linear.weight)
            if self.core.final_layer.linear.bias is not None:
                nn.init.zeros_(self.core.final_layer.linear.bias)
        else:
            raise ValueError(self.cfg.final_linear_init)
        init_linear_normal(self.delta_embedder.mlp[0], 0.02)
        init_linear_normal(self.delta_embedder.mlp[2], 0.02)

    def final_layer_once(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Avoid calling official FinalLayer.forward directly because this checked
        # out file contains a duplicate self.linear call.  We still use the
        # official submodules: norm_final, adaLN_modulation and linear.
        shift, scale = self.core.final_layer.adaLN_modulation(c).chunk(2, dim=1)
        x = self.modulate_sample(self.core.final_layer.norm_final(x), shift, scale)
        return self.core.final_layer.linear(x)

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"expected z [B,C,H,W], got {tuple(z.shape)}")
        if r.shape != t.shape or r.ndim != 1:
            raise ValueError(f"expected sample-level r,t [B], got r={tuple(r.shape)} t={tuple(t.shape)}")
        x = self.core.x_embedder(z)
        if self.core.use_abs_pos:
            x = x + self.core.pos_embed.to(dtype=x.dtype, device=x.device)
        delta = t - r
        c = self.core.t_embedder(t) + self.delta_embedder(delta) + self.core.y_embedder(y, self.training)
        for block in self.core.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x, c, self.core.feat_rope, use_reentrant=self.checkpoint_reentrant)
            else:
                x = block(x, c, self.core.feat_rope)
        x = self.final_layer_once(x, c)
        x = self.core.unpatchify(x)
        return x


def call_model(model: nn.Module, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor, y: torch.Tensor, mode: str, device: torch.device) -> torch.Tensor:
    if mode == "fp32":
        return model(z, r, t, y)
    if mode == "bf16_autocast":
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return model(z, r, t, y).float()
        return model(z, r, t, y)
    raise ValueError(mode)


MIXED_BF16_BACKBONE_FP32_JVP_MODE = "bf16_backbone_fp32_jvp"


def is_mixed_bf16_backbone_fp32_jvp(mode: str) -> bool:
    return mode == MIXED_BF16_BACKBONE_FP32_JVP_MODE


def backbone_forward_mode(mode: str) -> str:
    if is_mixed_bf16_backbone_fp32_jvp(mode):
        return "bf16_autocast"
    return mode


def jvp_target_mode(mode: str) -> str:
    if is_mixed_bf16_backbone_fp32_jvp(mode):
        return "fp32"
    return mode


def full_bundle_jvp(
    model: nn.Module,
    zt: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    mode: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    delta = t - r
    z_dot = delta[:, None, None, None] * v
    r_dot = torch.zeros_like(r)
    t_dot = delta

    def fn(z: torch.Tensor, rr: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        return call_model(model, z, rr, tt, y, mode, device)

    u, du = jvp(fn, (zt, r, t), (z_dot, r_dot, t_dot))
    return u, du, (z_dot, r_dot, t_dot)


def central_fd(
    model: nn.Module,
    primals: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    tangents: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    y: torch.Tensor,
    eps: float,
    mode: str,
    device: torch.device,
) -> torch.Tensor:
    z, r, t = primals
    z_dot, r_dot, t_dot = tangents
    with torch.no_grad():
        u_plus = call_model(model, z + eps * z_dot, r + eps * r_dot, t + eps * t_dot, y, mode, device)
        u_minus = call_model(model, z - eps * z_dot, r - eps * r_dot, t - eps * t_dot, y, mode, device)
    return (u_plus - u_minus) / (2.0 * eps)


def params_with_grad_all_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad.detach()).all().cpu()):
            return False
    return True


def requires_grad(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(flag)


@torch.no_grad()
def update_ema(ema: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = dict(ema.named_parameters())
    model_params = dict(model.named_parameters())
    for name, p_ema in ema_params.items():
        p = model_params[name]
        p_ema.mul_(decay).add_(p.detach(), alpha=1.0 - decay)
    ema_buffers = dict(ema.named_buffers())
    model_buffers = dict(model.named_buffers())
    for name, b_ema in ema_buffers.items():
        b = model_buffers[name]
        if torch.is_floating_point(b_ema):
            b_ema.copy_(b.detach())
        else:
            b_ema.copy_(b)


@torch.no_grad()
def live_ema_l2(model: nn.Module, ema: Optional[nn.Module]) -> Optional[float]:
    if ema is None:
        return None
    acc = torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)
    for p, q in zip(model.parameters(), ema.parameters()):
        acc = acc + torch.sum((p.detach().float() - q.detach().float()) ** 2)
    return float(torch.sqrt(acc).cpu())


def short_train_loop(
    cfg: Config,
    model: nn.Module,
    ema: Optional[nn.Module],
    device: torch.device,
    pae_stats: PaeStats,
) -> Dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps: List[Dict[str, Any]] = []
    loop_peak = 0.0
    completed = 0
    model.train()
    if ema is not None:
        ema.eval()
    mixed_recipe = is_mixed_bf16_backbone_fp32_jvp(cfg.mode)
    u_mode = backbone_forward_mode(cfg.mode)
    du_mode = jvp_target_mode(cfg.mode)

    for step in range(1, cfg.train_steps + 1):
        opt.zero_grad(set_to_none=True)
        problem = sample_problem(cfg, device, pae_stats)
        zt: torch.Tensor = problem["zt"]
        r: torch.Tensor = problem["r"]
        t: torch.Tensor = problem["t"]
        y: torch.Tensor = problem["y"]
        v: torch.Tensor = problem["v"]
        eq_mask: torch.Tensor = problem["eq_mask"]
        latent_meta: Dict[str, Any] = problem["latent_meta"]
        reset_peak_memory(device)
        cuda_sync(device)
        start = time.perf_counter()
        rec: Dict[str, Any] = {
            "step": step,
            "precision_recipe": cfg.mode,
            "backbone_forward_mode": u_mode,
            "jvp_target_mode": du_mode,
            "jvp_param_source": "live",
            "ema_present": ema is not None,
            "ema_decay": cfg.ema_decay if ema is not None else None,
            "ema_live_l2_before_step": live_ema_l2(model, ema),
            "target_detached": True,
            "r_t_sampling_granularity": "sample_level",
            "configured_equal_prob": cfg.equal_prob,
            "r_eq_t_count": latent_meta["r_eq_t_count"],
            "r_eq_t_total": latent_meta["r_eq_t_total"],
            "realized_r_eq_t_fraction": latent_meta["realized_r_eq_t_fraction"],
            "latent_meta": latent_meta,
        }
        try:
            if mixed_recipe:
                reset_peak_memory(device)
                cuda_sync(device)
                target_start = time.perf_counter()
                with torch.no_grad():
                    _u_jvp, du, _ = full_bundle_jvp(model, zt, r, t, y, v, du_mode, device)
                    target = (v - du).detach()
                cuda_sync(device)
                rec["target_jvp_sec"] = time.perf_counter() - target_start
                rec["target_jvp_peak_memory_mb"] = peak_memory_mb(device)
                rec["target_jvp_u_finite"] = finite_tensor(_u_jvp)
                rec["target_requires_grad"] = bool(target.requires_grad)
                del _u_jvp

                reset_peak_memory(device)
                cuda_sync(device)
                backbone_start = time.perf_counter()
                u = call_model(model, zt, r, t, y, u_mode, device)
                cuda_sync(device)
                rec["backbone_forward_sec"] = time.perf_counter() - backbone_start
                rec["backbone_forward_peak_memory_mb"] = peak_memory_mb(device)
            else:
                u, du, _ = full_bundle_jvp(model, zt, r, t, y, v, cfg.mode, device)
                target = (v - du).detach()
                rec["target_requires_grad"] = bool(target.requires_grad)

            if bool(eq_mask.any().detach().cpu()):
                rec["actual_rt_samples_target_minus_v_max_abs"] = float((target[eq_mask] - v[eq_mask]).detach().float().abs().max().cpu())
            else:
                rec["actual_rt_samples_target_minus_v_max_abs"] = None

            loss = F.mse_loss(u.float(), target.float())
            rec.update(
                {
                    "loss": float(loss.detach().float().cpu()),
                    "u_finite": finite_tensor(u),
                    "du_finite": finite_tensor(du),
                    "target_finite": finite_tensor(target),
                    "loss_finite": bool(torch.isfinite(loss.detach()).cpu()),
                    "du_over_v_norm_ratio": norm(du) / (norm(v) + 1e-12),
                }
            )
            if rec["loss_finite"]:
                reset_peak_memory(device)
                cuda_sync(device)
                backward_start = time.perf_counter()
                loss.backward()
                cuda_sync(device)
                rec["backward_sec"] = time.perf_counter() - backward_start
                rec["backward_peak_memory_mb"] = peak_memory_mb(device)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                grad_norm_value = float(grad_norm.detach().float().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
                grad_finite = math.isfinite(grad_norm_value) and params_with_grad_all_finite(model)
                rec.update(
                    {
                        "grad_norm_before_clip": grad_norm_value,
                        "grad_clip_threshold": cfg.grad_clip,
                        "grad_is_finite": grad_finite,
                    }
                )
                if grad_finite:
                    opt.step()
                    if ema is not None:
                        update_ema(ema, model, cfg.ema_decay)
                    rec["ema_live_l2_after_step"] = live_ema_l2(model, ema)
                    completed += 1
                else:
                    rec["stopped_reason"] = "nonfinite_gradient"
            else:
                rec["grad_is_finite"] = False
                rec["stopped_reason"] = "nonfinite_loss"
        except Exception as exc:  # noqa: BLE001
            rec.update({"status": "error", "error_type": type(exc).__name__, "error_message": str(exc), "grad_is_finite": False})
        finally:
            cuda_sync(device)
            rec["elapsed_sec"] = time.perf_counter() - start
            memory_candidates = [
                rec.get("target_jvp_peak_memory_mb"),
                rec.get("backbone_forward_peak_memory_mb"),
                rec.get("backward_peak_memory_mb"),
                peak_memory_mb(device),
            ]
            rec["peak_memory_mb"] = max(float(x) for x in memory_candidates if x is not None) if device.type == "cuda" else None
            if rec["peak_memory_mb"] is not None:
                loop_peak = max(loop_peak, float(rec["peak_memory_mb"]))
            steps.append(rec)

        if rec.get("status") == "error" or not rec.get("loss_finite", False) or not rec.get("grad_is_finite", False):
            break

    losses = [s.get("loss") for s in steps if isinstance(s.get("loss"), (int, float))]
    realized = [s.get("realized_r_eq_t_fraction") for s in steps if isinstance(s.get("realized_r_eq_t_fraction"), (int, float))]
    actual_degens = [s.get("actual_rt_samples_target_minus_v_max_abs") for s in steps if isinstance(s.get("actual_rt_samples_target_minus_v_max_abs"), (int, float))]
    return {
        "requested_steps": cfg.train_steps,
        "n_completed_optimizer_steps": completed,
        "n_recorded_steps": len(steps),
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "loss_all_finite": all(bool(s.get("loss_finite", False)) for s in steps) if steps else False,
        "grad_all_finite": all(bool(s.get("grad_is_finite", False)) for s in steps) if steps else False,
        "u_du_target_all_finite": all(bool(s.get("u_finite", False) and s.get("du_finite", False) and s.get("target_finite", False)) for s in steps) if steps else False,
        "target_detached_all_steps": all(s.get("target_requires_grad") is False for s in steps) if steps else False,
        "jvp_param_source_all_live": all(s.get("jvp_param_source") == "live" for s in steps) if steps else False,
        "mean_realized_r_eq_t_fraction": sum(float(x) for x in realized) / len(realized) if realized else None,
        "max_actual_rt_samples_target_minus_v_abs": max(actual_degens) if actual_degens else None,
        "loop_peak_memory_mb": loop_peak if device.type == "cuda" else None,
        "precision_recipe": cfg.mode,
        "backbone_forward_mode": u_mode,
        "jvp_target_mode": du_mode,
        "steps": steps,
    }


def run_smoke(cfg: Config, device: torch.device, pae_stats: PaeStats) -> Dict[str, Any]:
    set_seed(cfg.seed)
    model = B3MeanFlowLightningDiTSmokeModel(cfg).to(device=device, dtype=torch.float32)
    ema: Optional[nn.Module] = None
    if cfg.create_ema:
        ema = copy.deepcopy(model).to(device=device, dtype=torch.float32)
        requires_grad(ema, False)
        ema.eval()

    model_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    u_mode = backbone_forward_mode(cfg.mode)
    du_mode = jvp_target_mode(cfg.mode)
    record: Dict[str, Any] = {
        "status": "ok",
        "phase0_baseline": "B3_MeanFlow_global_sample_level",
        "implementation_scope": "integration_smoke_scaffold_not_full_imagenet_training",
        "official_component_claim": "official_PAE_LightningDiT_submodules_with_sample_level_MeanFlow_conditioning_wrapper",
        "model_params": model_params,
        "trainable_params": trainable_params,
        "ema_present": ema is not None,
        "ema_decay": cfg.ema_decay if ema is not None else None,
        "jvp_param_source": "live",
        "target_detached_for_training": True,
        "r_t_sampling_granularity": "sample_level",
        "configured_equal_prob": cfg.equal_prob,
        "precision_recipe": {
            "mode": cfg.mode,
            "backbone_forward_mode": u_mode,
            "jvp_target_mode": du_mode,
            "fd_audit_mode": du_mode,
            "target_detached_for_training": True,
            "jvp_param_source": "live",
            "ema_used_for_target_jvp": False,
            "mixed_recipe_claim": (
                "bf16/autocast backbone forward/backward plus fp32 no-autocast JVP/FD target path"
                if is_mixed_bf16_backbone_fp32_jvp(cfg.mode)
                else "single-path precision"
            ),
        },
    }

    try:
        problem = sample_problem(cfg, device, pae_stats)
        zt: torch.Tensor = problem["zt"]
        r: torch.Tensor = problem["r"]
        t: torch.Tensor = problem["t"]
        y: torch.Tensor = problem["y"]
        v: torch.Tensor = problem["v"]
        eq_mask: torch.Tensor = problem["eq_mask"]
        latent_meta: Dict[str, Any] = problem["latent_meta"]
        record["latent_meta"] = latent_meta
        record["realized_r_eq_t_fraction"] = latent_meta["realized_r_eq_t_fraction"]
        record["input_stats"] = {
            "zt": tensor_stats(zt),
            "v": tensor_stats(v),
            "r": tensor_stats(r),
            "t": tensor_stats(t),
            "delta": tensor_stats(t - r),
        }

        # Ordinary forward audit in the same mode that will carry gradients.
        model.eval()
        reset_peak_memory(device)
        cuda_sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            u_forward = call_model(model, zt, r, t, y, u_mode, device)
        cuda_sync(device)
        record["forward_only_sec"] = time.perf_counter() - start
        record["forward_only_peak_memory_mb"] = peak_memory_mb(device)
        record["forward_only_mode"] = u_mode
        record["forward_output_finite"] = finite_tensor(u_forward)
        record["forward_output_stats"] = tensor_stats(u_forward)

        if is_mixed_bf16_backbone_fp32_jvp(cfg.mode):
            reset_peak_memory(device)
            cuda_sync(device)
            ref_start = time.perf_counter()
            with torch.no_grad():
                u_forward_fp32_ref = call_model(model, zt, r, t, y, "fp32", device)
            cuda_sync(device)
            record["fp32_reference_forward_sec"] = time.perf_counter() - ref_start
            record["fp32_reference_forward_peak_memory_mb"] = peak_memory_mb(device)
            record["bf16_forward_vs_fp32_rel_err"] = rel(u_forward_fp32_ref, u_forward)
            record["bf16_forward_minus_fp32_abs_norm"] = norm(u_forward - u_forward_fp32_ref)
            record["fp32_reference_forward_norm"] = norm(u_forward_fp32_ref)
            del u_forward_fp32_ref

        # Full-bundle scalar/sample-level JVP audit path.
        reset_peak_memory(device)
        cuda_sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            u, du, tangents = full_bundle_jvp(model, zt, r, t, y, v, du_mode, device)
        cuda_sync(device)
        record["jvp_sec"] = time.perf_counter() - start
        record["jvp_peak_memory_mb"] = peak_memory_mb(device)
        record["jvp_mode"] = du_mode
        record["allocated_after_jvp_mb"] = allocated_memory_mb(device)
        target = (v - du).detach()
        record.update(
            {
                "u_norm": norm(u),
                "v_norm": norm(v),
                "du_norm": norm(du),
                "target_norm": norm(target),
                "du_over_v_norm_ratio": norm(du) / (norm(v) + 1e-12),
                "u_finite": finite_tensor(u),
                "du_finite": finite_tensor(du),
                "target_finite": finite_tensor(target),
                "target_requires_grad": bool(target.requires_grad),
                "u_stats": tensor_stats(u),
                "du_stats": tensor_stats(du),
                "target_stats": tensor_stats(target),
            }
        )
        if bool(eq_mask.any().detach().cpu()):
            record["actual_rt_samples_target_minus_v_max_abs"] = float((target[eq_mask] - v[eq_mask]).detach().float().abs().max().cpu())
        else:
            record["actual_rt_samples_target_minus_v_max_abs"] = None

        if cfg.run_fd:
            reset_peak_memory(device)
            cuda_sync(device)
            start = time.perf_counter()
            du_fd = central_fd(model, (zt, r, t), tangents, y, cfg.fd_eps, du_mode, device)
            cuda_sync(device)
            record["fd_sec"] = time.perf_counter() - start
            record["fd_peak_memory_mb"] = peak_memory_mb(device)
            record["fd_mode"] = du_mode
            record["fd_rel_err_full_jvp"] = rel(du_fd, du)
            record["fd_abs_err_norm"] = norm(du_fd - du)
            record["fd_norm"] = norm(du_fd)
            record["fd_finite"] = finite_tensor(du_fd)
        else:
            record["fd_skipped"] = True

        # All-sample r=t degeneracy audit.  Tangent must be exactly zero, so the
        # MeanFlow target must reduce to FM velocity v.
        with torch.no_grad():
            t_eq = t.clone()
            r_eq = t_eq.clone()
            _u_eq, du_eq, _ = full_bundle_jvp(model, zt, r_eq, t_eq, y, v, du_mode, device)
            target_eq = (v - du_eq).detach()
        record["all_r_eq_t_degenerate_du_norm"] = norm(du_eq)
        record["all_r_eq_t_degenerate_target_minus_v_max_abs"] = float((target_eq - v).detach().float().abs().max().cpu())
        record["all_r_eq_t_degenerate_finite"] = finite_tensor(du_eq) and finite_tensor(target_eq)

        if cfg.train_steps > 0:
            record["short_train"] = short_train_loop(cfg, model, ema, device, pae_stats)
        else:
            record["short_train_skipped"] = True

        bad = not (
            record.get("forward_output_finite", False)
            and record.get("u_finite", False)
            and record.get("du_finite", False)
            and record.get("target_finite", False)
        )
        if record.get("fd_finite") is False:
            bad = True
        if record.get("short_train", {}).get("grad_all_finite") is False:
            bad = True
        if record.get("short_train", {}).get("loss_all_finite") is False:
            bad = True
        record["has_nan_or_inf"] = bool(bad)
    except Exception as exc:  # noqa: BLE001
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)
        record["has_nan_or_inf"] = True
    finally:
        del model
        if ema is not None:
            del ema
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return record


def gate_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    train = record.get("short_train", {}) if isinstance(record.get("short_train"), dict) else {}
    memory_required = [
        record.get("jvp_peak_memory_mb"),
        record.get("fd_peak_memory_mb"),
        record.get("forward_only_peak_memory_mb"),
        train.get("loop_peak_memory_mb"),
    ]
    memory_logged = all(x is not None for x in memory_required) if record.get("device", "cuda") != "cpu" else True
    fd_ok = (record.get("fd_rel_err_full_jvp") is not None) and float(record.get("fd_rel_err_full_jvp")) < 1e-2
    degen_ok = (record.get("all_r_eq_t_degenerate_target_minus_v_max_abs") is not None) and float(record.get("all_r_eq_t_degenerate_target_minus_v_max_abs")) < 1e-7
    target_detached = record.get("target_requires_grad") is False and train.get("target_detached_all_steps") is True
    live_param_ok = record.get("jvp_param_source") == "live" and train.get("jvp_param_source_all_live") is True
    sampler_ok = record.get("r_t_sampling_granularity") == "sample_level" and train.get("mean_realized_r_eq_t_fraction") is not None
    mixed_modes_ok = (
        record.get("precision_recipe", {}).get("backbone_forward_mode") == "bf16_autocast"
        and record.get("precision_recipe", {}).get("jvp_target_mode") == "fp32"
        and record.get("precision_recipe", {}).get("fd_audit_mode") == "fp32"
    )
    train_ok = (
        train.get("n_completed_optimizer_steps") == train.get("requested_steps")
        and train.get("loss_all_finite") is True
        and train.get("grad_all_finite") is True
        and train.get("u_du_target_all_finite") is True
    )
    practical = bool(
        record.get("status") == "ok"
        and not record.get("has_nan_or_inf", True)
        and fd_ok
        and degen_ok
        and target_detached
        and live_param_ok
        and sampler_ok
        and mixed_modes_ok
        and train_ok
        and memory_logged
    )
    return {
        "status_ok": record.get("status") == "ok",
        "fd_ok_lt_1e_2": fd_ok,
        "fd_rel_err_full_jvp": record.get("fd_rel_err_full_jvp"),
        "degenerate_ok_lt_1e_7": degen_ok,
        "all_r_eq_t_degenerate_target_minus_v_max_abs": record.get("all_r_eq_t_degenerate_target_minus_v_max_abs"),
        "target_detached": target_detached,
        "live_param_jvp_target": live_param_ok,
        "sample_level_sampler": sampler_ok,
        "mixed_modes_ok": mixed_modes_ok,
        "train_ok": train_ok,
        "memory_logged": memory_logged,
        "no_nan_or_inf": not record.get("has_nan_or_inf", True),
        "practical_gate_pass": practical,
        "strict_gate_pass": practical,
    }


def write_summary(out_dir: Path, payload: Dict[str, Any]) -> None:
    cfg = payload["config"]
    record = payload["result"]
    gate = payload["gate"]
    train = record.get("short_train", {}) if isinstance(record.get("short_train"), dict) else {}
    lines: List[str] = []
    lines.append("# Experiment 6 — Phase 0 / B3 MeanFlow Mixed-Precision Integration Smoke 结果\n\n")
    lines.append(f"- 生成时间：`{payload['created_at_utc']}`\n")
    lines.append(f"- torch：`{payload['torch_version']}`；cuda_available：`{payload['cuda_available']}`；device：`{payload['device']}`；GPU：`{payload.get('cuda_device_name')}`\n")
    lines.append(f"- PAE repo：`{payload['pae_gen_root']}`；commit：`{payload.get('pae_git_commit')}`\n")
    lines.append("- scope：Phase 0 / B3 MeanFlow integration smoke scaffold；不声明完整 ImageNet/FID/FDr。\n")
    lines.append("- official modules：`LightningDiT` / `PatchEmbed` / `LightningDiTBlock` / `Attention` / `RMSNorm` / `SwiGLU` / `FinalLayer` / `VisionRotaryEmbeddingFast`。\n")
    lines.append("- MeanFlow conditioning：sample-level `u_theta(z_t, r, t, y)`，conditioning 使用 `emb(t) + emb(t-r) + emb(y)`。\n")
    lines.append("- precision recipe：`bf16_backbone_fp32_jvp` = bf16/autocast backbone forward/backward + fp32 no-autocast JVP/FD target path；target detach；JVP 使用 live params。\n")
    lines.append(f"- model：C/grid/patch/width/heads/depth = `{cfg['channels']}/{cfg['grid']}/{cfg['patch_size']}/{cfg['width']}/{cfg['heads']}/{cfg['depth']}`；params：`{record.get('model_params')}`\n")
    lines.append(f"- sampler：`{record.get('r_t_sampling_granularity')}`；configured equal_prob=`{cfg['equal_prob']}`；audit realized=`{fmt(record.get('realized_r_eq_t_fraction'))}`；train mean realized=`{fmt(train.get('mean_realized_r_eq_t_fraction'))}`\n")
    lines.append(f"- SDPA/TF32/checkpoint：sdpa=`{cfg['sdpa_kernel']}`；allow_tf32=`{cfg['allow_tf32']}`；use_checkpoint=`{cfg['use_checkpoint']}`\n")
    lines.append(f"- EMA：present=`{record.get('ema_present')}`；decay=`{record.get('ema_decay')}`；target JVP uses EMA=`False`；param_source=`{record.get('jvp_param_source')}`\n")
    lines.append(f"- PAE latent stats：`{cfg['pae_stats_path']}`；available=`{payload.get('pae_stats', {}).get('available')}`；latent_mode=`{cfg['latent_mode']}`\n")
    lines.append("\n## Gate 摘要\n\n")
    for k, v in gate.items():
        lines.append(f"- `{k}`: `{v}`\n")
    lines.append("\n## 核心指标\n\n")
    lines.append("| item | value |\n|---|---:|\n")
    rows = {
        "FD rel err full JVP": record.get("fd_rel_err_full_jvp"),
        "bf16-vs-fp32 fwd rel": record.get("bf16_forward_vs_fp32_rel_err"),
        "du/v norm ratio": record.get("du_over_v_norm_ratio"),
        "all r=t target-v max abs": record.get("all_r_eq_t_degenerate_target_minus_v_max_abs"),
        "actual r=t samples target-v max abs": record.get("actual_rt_samples_target_minus_v_max_abs"),
        "short train completed steps": train.get("n_completed_optimizer_steps"),
        "short train final loss": train.get("final_loss"),
        "short train grad all finite": train.get("grad_all_finite"),
        "target detached all steps": train.get("target_detached_all_steps"),
        "JVP peak MB": record.get("jvp_peak_memory_mb"),
        "FD peak MB": record.get("fd_peak_memory_mb"),
        "bf16 backbone forward peak MB": record.get("forward_only_peak_memory_mb"),
        "short train loop peak MB": train.get("loop_peak_memory_mb"),
    }
    for k, v in rows.items():
        lines.append(f"| {k} | `{fmt(v)}` |\n")
    lines.append("\n## 短训 step telemetry\n\n")
    lines.append("| step | loss | r=t frac | target JVP MB | backbone MB | backward MB | grad norm | actual r=t target-v max |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in train.get("steps", []):
        lines.append(
            f"| {s.get('step')} | {fmt(s.get('loss'))} | {fmt(s.get('realized_r_eq_t_fraction'))} | "
            f"{fmt(s.get('target_jvp_peak_memory_mb'))} | {fmt(s.get('backbone_forward_peak_memory_mb'))} | "
            f"{fmt(s.get('backward_peak_memory_mb'))} | {fmt(s.get('grad_norm_before_clip'))} | "
            f"{fmt(s.get('actual_rt_samples_target_minus_v_max_abs'))} |\n"
        )
    lines.append("\n## 解释与限制\n\n")
    lines.append("- `FD rel err` 是 fp32 JVP target path 与 fp32 central finite difference 的一致性指标；它才是 JVP correctness 的关键指标。\n")
    lines.append("- `bf16-vs-fp32 fwd rel` 是 trainable bf16/autocast backbone forward 与 fp32 forward 的普通前向差异；target 用 fp32 计算，训练用 bf16 forward 拟合该 detached target，这是 mixed-precision 训练的预期状态。\n")
    lines.append("- `r=t` 退化项中 `delta=t-r=0`，因此 JVP tangent 为零，target 应严格退化为 `v`；本 smoke 同时记录全 batch 强制 r=t 与实际 sampler 中 r=t 样本。\n")
    lines.append("- 上游 PAE `transport.training_losses` 尚未被改成生产级 MeanFlow loss；本实验产物是 B3 baseline 的可执行 smoke/scaffold，可作为后续正式训练循环补丁的依据。\n")
    lines.append("- 默认 `stress_nonzero` 初始化用于让 tiny smoke 中 JVP/FD 非平凡；若切换 `official_zero`，初始 JVP 可能接近零，适合测试官方初始化但不适合作为数值压测。\n")
    (out_dir / "b3_meanflow_mixed_precision_smoke_summary.md").write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    try:
        torch.set_float32_matmul_precision("high" if cfg.allow_tf32 else "highest")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.allow_tf32)
    device = resolve_device(cfg.device)
    sdpa_info = configure_sdpa(cfg.sdpa_kernel, device)
    out_dir = Path(cfg.output_dir) if cfg.output_dir else Path(__file__).resolve().parents[1] / "results" / "smoke_b3_mixed"
    out_dir.mkdir(parents=True, exist_ok=True)
    pae_stats = PaeStats(Path(cfg.pae_stats_path))

    print(f"[env] device={device} torch={torch.__version__} cuda={torch.cuda.is_available()} sdpa={sdpa_info}", flush=True)
    print(f"[env] PAE_GEN_ROOT={PAE_GEN_ROOT} compile_unwrapped={COMPILE_UNWRAPPED} dynamo_requested={DYNAMO_REQUESTED}", flush=True)
    print(f"[env] pae_stats={pae_stats.meta}", flush=True)
    print(
        f"[run] B3 MeanFlow smoke mode={cfg.mode} grid={cfg.grid} batch={cfg.batch_size} width={cfg.width} "
        f"depth={cfg.depth} train_steps={cfg.train_steps} equal_prob={cfg.equal_prob}",
        flush=True,
    )

    total_start = time.perf_counter()
    result = run_smoke(cfg, device, pae_stats)
    result["device"] = str(device)
    payload: Dict[str, Any] = {
        "created_at_utc": now_utc(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "pae_gen_root": str(PAE_GEN_ROOT),
        "pae_git_commit": git_commit(PAE_GEN_ROOT.parents[0]) if PAE_GEN_ROOT.exists() else None,
        "config": asdict(cfg),
        "sdpa_info": sdpa_info,
        "allow_tf32": cfg.allow_tf32,
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32) if torch.cuda.is_available() else None,
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32) if torch.cuda.is_available() else None,
        "compile_unwrapped": COMPILE_UNWRAPPED,
        "torchdynamo_disable_env": os.environ.get("TORCHDYNAMO_DISABLE"),
        "pae_stats": pae_stats.meta,
        "total_elapsed_sec": time.perf_counter() - total_start,
        "result": result,
    }
    payload["gate"] = gate_summary(result)
    metrics_path = out_dir / "b3_meanflow_mixed_precision_smoke_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(out_dir, payload)
    train = result.get("short_train", {}) if isinstance(result.get("short_train"), dict) else {}
    print(
        f"[done] status={result.get('status')} fd={fmt(result.get('fd_rel_err_full_jvp'))} "
        f"bf16_vs_fp32={fmt(result.get('bf16_forward_vs_fp32_rel_err'))} "
        f"degen={fmt(result.get('all_r_eq_t_degenerate_target_minus_v_max_abs'))} "
        f"train_steps={train.get('n_completed_optimizer_steps')}/{train.get('requested_steps')} "
        f"final_loss={fmt(train.get('final_loss'))} loop_peak={fmt(train.get('loop_peak_memory_mb'))}MB",
        flush=True,
    )
    print(f"[write] {metrics_path}", flush=True)
    print(f"[write] {out_dir / 'b3_meanflow_mixed_precision_smoke_summary.md'}", flush=True)
    print(f"[gate] practical={payload['gate']['practical_gate_pass']} strict={payload['gate']['strict_gate_pass']}", flush=True)


if __name__ == "__main__":
    main()
