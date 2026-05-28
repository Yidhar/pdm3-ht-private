#!/usr/bin/env python3
"""
PDM-3-HT Experiment 4: Real LightningDiT Block / PAE Latent / Short-Train Stress.

This experiment moves beyond the Experiment-3 proxy and uses official PAE repository
LightningDiT components:
  - LightningDiTBlock submodules: Attention, RMSNorm/LayerNorm, SwiGLU/Mlp, AdaLN.
  - FinalLayer submodules for final norm/projection.
  - VisionRotaryEmbeddingFast and official 2-D sin-cos position embedding.

The official LightningDiT block expects image-level conditioning c: [B, D]. For
per-patch MeanFlow we need token-level (r_i, t_i) conditioning. Therefore this
script wraps the official block submodules with a minimal token-wise AdaLN path:
  adaLN_modulation(c_tokens): [B, N, D] -> [B, N, 6D], chunked along dim=-1.
The attention / MLP / norm / RoPE implementation itself remains official.

Latents:
  - gaussian: z0 ~ N(0,1)
  - correlated: spatially smoothed unit-variance latent noise
  - pae_stats_synthetic: raw = official_mean + official_std * smooth_noise,
    then normalized as the PAE generator training path does: (raw-mean)/std.
    This is NOT a real ImageNet PAE latent shard; it is PAE-stat-conditioned
    synthetic stress input and is recorded as such.

JVP formula (full bundle over z_t, r, t):
    delta = t - r
    z_dot = delta[..., None] * v
    r_dot = 0
    t_dot = delta
    u, du = jvp(model, (z_t, r, t), (z_dot, r_dot, t_dot))
    target = (v - du).detach()
"""

from __future__ import annotations

import argparse
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

# The official PAE files decorate several methods with @torch.compile.  In this
# audit the target is forward-mode AD/JVP correctness, not Dynamo compilation.
# Disable by default to avoid first-call compile latency and transform clashes;
# pass --enable-dynamo to leave it enabled in a separate diagnostic process.
DYNAMO_REQUESTED = "--enable-dynamo" in sys.argv
if not DYNAMO_REQUESTED:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

# xFormers is not required for the official fallback SwiGLU FFN and is not
# installed in this environment; make the dependency choice explicit.
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
    from models.lightningdit import FinalLayer, LightningDiTBlock, get_2d_sincos_pos_embed
    from models.pos_embed import VisionRotaryEmbeddingFast
    from models.swiglu_ffn import SwiGLUFFN
except Exception as exc:  # noqa: BLE001 - fail with actionable message.
    raise RuntimeError(
        f"Failed to import official PAE LightningDiT modules from {PAE_GEN_ROOT}. "
        "Check PYTHONPATH and dependencies (timm, einops, fairscale)."
    ) from exc


def _unwrap_torch_compile_for_forward_ad() -> List[str]:
    """Use original official Python forward bodies when @torch.compile wrapped them."""
    unwrapped: List[str] = []
    if DYNAMO_REQUESTED:
        return unwrapped
    for cls in (SwiGLUFFN, LightningDiTBlock, FinalLayer):
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
    batch_size: int = 1
    channels: int = 32
    width: int = 128
    heads: int = 4
    depth: int = 2
    mlp_ratio: float = 4.0
    grids: Tuple[int, ...] = (32,)
    modes: Tuple[str, ...] = ("fp32",)
    latent_modes: Tuple[str, ...] = ("pae_stats_synthetic",)
    device: str = "auto"
    train_steps: int = 16
    fd_eps: float = 1e-2
    fd_max_tokens: int = 1024
    run_fd: bool = True
    t_low: float = 0.05
    t_high: float = 0.95
    equal_prob: float = 0.75
    time_sampler: str = "ltg"
    ltg_mu: float = 0.0
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
    adaln_init: str = "small"
    final_adaln_init: str = "small"
    latent_smooth_passes: int = 2
    latent_multiplier: float = 1.0
    pae_stats_path: str = "/workspace/PDM/external/PAE_hf/Latent-stats/PAE_DINOv2L.pt"
    output_dir: Optional[str] = None
    enable_dynamo: bool = False
    allow_tf32: bool = False


def parse_csv_int(s: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in s.split(",") if x.strip())


def parse_csv_str(s: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in s.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--grids", type=str, default="32", help="comma-separated square grid sides")
    p.add_argument("--modes", type=str, default="fp32", help="comma-separated: fp32,bf16_autocast")
    p.add_argument(
        "--latent-modes",
        type=str,
        default="pae_stats_synthetic",
        help="comma-separated: gaussian,correlated,pae_stats_synthetic,pae_stats_raw_synthetic",
    )
    p.add_argument("--train-steps", type=int, default=16)
    p.add_argument("--fd-eps", type=float, default=1e-2)
    p.add_argument("--fd-max-tokens", type=int, default=1024)
    p.add_argument("--no-fd", action="store_true")
    p.add_argument("--t-low", type=float, default=0.05)
    p.add_argument("--t-high", type=float, default=0.95)
    p.add_argument("--equal-prob", type=float, default=0.75)
    p.add_argument("--time-sampler", type=str, default="ltg", choices=["uniform", "ltg"])
    p.add_argument("--ltg-mu", type=float, default=0.0)
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
    p.add_argument("--use-checkpoint", action="store_true")
    p.add_argument("--checkpoint-reentrant", action="store_true")
    p.add_argument("--adaln-init", type=str, default="small", choices=["small", "zero", "xavier"])
    p.add_argument("--final-adaln-init", type=str, default="small", choices=["small", "zero", "xavier"])
    p.add_argument("--latent-smooth-passes", type=int, default=2)
    p.add_argument("--latent-multiplier", type=float, default=1.0)
    p.add_argument("--pae-stats-path", type=str, default="/workspace/PDM/external/PAE_hf/Latent-stats/PAE_DINOv2L.pt")
    p.add_argument("--enable-dynamo", action="store_true", help="do not set TORCHDYNAMO_DISABLE=1; diagnostic only")
    p.add_argument("--allow-tf32", action="store_true", help="allow TF32 matmul/cudnn; default disables it for FD/JVP audits")
    p.set_defaults(use_swiglu=True, use_rmsnorm=True, use_rope=True, use_abs_pos=True, fused_attn=True)
    return p.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    modes = parse_csv_str(args.modes)
    allowed_modes = {"fp32", "bf16_autocast"}
    bad_modes = set(modes) - allowed_modes
    if bad_modes:
        raise ValueError(f"unsupported modes: {sorted(bad_modes)}")

    latent_modes = parse_csv_str(args.latent_modes)
    allowed_latents = {"gaussian", "correlated", "pae_stats_synthetic", "pae_stats_raw_synthetic"}
    bad_latents = set(latent_modes) - allowed_latents
    if bad_latents:
        raise ValueError(f"unsupported latent modes: {sorted(bad_latents)}")

    if args.width % args.heads != 0:
        raise ValueError("width must be divisible by heads")
    if args.use_rope and ((args.width // args.heads) % 4 != 0):
        # VisionRotaryEmbeddingFast creates 2-D rotations whose dim is hidden/head/2.
        # It works best when head_dim/2 is even.
        raise ValueError("for RoPE, width/heads should be divisible by 4")
    if not (0.0 <= args.equal_prob <= 1.0):
        raise ValueError("equal_prob must be in [0, 1]")
    if not (0.0 <= args.t_low < args.t_high <= 1.0):
        raise ValueError("need 0 <= t_low < t_high <= 1")

    return Config(
        seed=args.seed,
        batch_size=args.batch_size,
        channels=args.channels,
        width=args.width,
        heads=args.heads,
        depth=args.depth,
        mlp_ratio=args.mlp_ratio,
        grids=parse_csv_int(args.grids),
        modes=modes,
        latent_modes=latent_modes,
        device=args.device,
        train_steps=args.train_steps,
        fd_eps=args.fd_eps,
        fd_max_tokens=args.fd_max_tokens,
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
        adaln_init=args.adaln_init,
        final_adaln_init=args.final_adaln_init,
        latent_smooth_passes=args.latent_smooth_passes,
        latent_multiplier=args.latent_multiplier,
        pae_stats_path=args.pae_stats_path,
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


def rel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return norm(a - b) / (norm(a) + eps)


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
        # Keep the official F.scaled_dot_product_attention call, but force the
        # math backend, which is the most transparent path for forward AD audits.
        if hasattr(cuda_backend, "enable_flash_sdp"):
            cuda_backend.enable_flash_sdp(False)
        if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
            cuda_backend.enable_mem_efficient_sdp(False)
        if hasattr(cuda_backend, "enable_math_sdp"):
            cuda_backend.enable_math_sdp(True)
        if hasattr(cuda_backend, "enable_cudnn_sdp"):
            cuda_backend.enable_cudnn_sdp(False)
    elif kernel == "default":
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


def chw_to_tokens(x: torch.Tensor) -> torch.Tensor:
    return x.flatten(2).transpose(1, 2).contiguous()


def sample_t(cfg: Config, batch: int, n_tokens: int, device: torch.device) -> torch.Tensor:
    if cfg.time_sampler == "uniform":
        q = torch.rand(batch, n_tokens, device=device, dtype=torch.float32)
    elif cfg.time_sampler == "ltg":
        logits = cfg.ltg_mu + cfg.ltg_sigma * torch.randn(batch, n_tokens, device=device, dtype=torch.float32)
        q = torch.sigmoid(logits)
    else:
        raise ValueError(cfg.time_sampler)
    return cfg.t_low + (cfg.t_high - cfg.t_low) * q


def sample_problem(cfg: Config, grid: int, latent_mode: str, device: torch.device, pae_stats: PaeStats) -> Dict[str, torch.Tensor | Dict[str, Any]]:
    b, n, c = cfg.batch_size, grid * grid, cfg.channels
    latent_meta: Dict[str, Any] = {"latent_mode": latent_mode}

    if latent_mode == "gaussian":
        z0 = torch.randn(b, n, c, device=device, dtype=torch.float32)
        latent_meta["source"] = "standard_gaussian_tokens"
    elif latent_mode == "correlated":
        z0 = chw_to_tokens(smooth_chw_noise(b, c, grid, device, cfg.latent_smooth_passes))
        latent_meta["source"] = "spatially_smoothed_unit_variance_synthetic"
    elif latent_mode in {"pae_stats_synthetic", "pae_stats_raw_synthetic"}:
        mean, std = pae_stats.to_device(device, c)
        noise = smooth_chw_noise(b, c, grid, device, cfg.latent_smooth_passes)
        raw = mean + std * noise
        if latent_mode == "pae_stats_synthetic":
            # Matches the PAE generator training config latent_norm: true, latent_multiplier: 1.0.
            z0_chw = (raw - mean) / (std + 1e-12) * cfg.latent_multiplier
            latent_meta["source"] = "official_pae_stats_synthetic_normalized_not_real_imagenet_latent"
        else:
            z0_chw = raw
            latent_meta["source"] = "official_pae_stats_synthetic_raw_not_real_imagenet_latent"
        latent_meta["pae_stats_path"] = str(pae_stats.path)
        latent_meta["latent_multiplier"] = cfg.latent_multiplier
        latent_meta["raw_stats"] = tensor_stats(raw)
        latent_meta["noise_stats"] = tensor_stats(noise)
        z0 = chw_to_tokens(z0_chw)
    else:
        raise ValueError(latent_mode)

    eps = torch.randn(b, n, c, device=device, dtype=torch.float32)
    v = eps - z0
    t = sample_t(cfg, b, n, device)
    r = torch.rand(b, n, device=device, dtype=torch.float32) * t
    if cfg.equal_prob > 0:
        mask = torch.rand(b, n, device=device) < cfg.equal_prob
        r = torch.where(mask, t, r)
        latent_meta["r_eq_t_fraction_realized"] = float(mask.float().mean().cpu())
    else:
        latent_meta["r_eq_t_fraction_realized"] = 0.0
    zt = (1.0 - t[..., None]) * z0 + t[..., None] * eps
    return {"z0": z0, "eps": eps, "v": v, "r": r, "t": t, "zt": zt, "latent_meta": latent_meta}


class PerTokenMeanFlowEmbedder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    @staticmethod
    def features(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        delta = t - r
        return torch.stack(
            [
                r,
                t,
                delta,
                r * t,
                delta.square(),
                torch.sin(math.pi * r),
                torch.cos(math.pi * r),
                torch.sin(math.pi * t),
                torch.cos(math.pi * t),
                torch.sin(math.pi * delta),
            ],
            dim=-1,
        )

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(self.features(r, t))


class OfficialBlockPerTokenWrapper(nn.Module):
    """Token-wise AdaLN wrapper around an official PAE LightningDiTBlock."""

    def __init__(self, block: LightningDiTBlock):
        super().__init__()
        self.block = block

    @staticmethod
    def modulate_token(x: torch.Tensor, shift: Optional[torch.Tensor], scale: torch.Tensor) -> torch.Tensor:
        if shift is None:
            return x * (1.0 + scale)
        return x * (1.0 + scale) + shift

    def forward(self, x: torch.Tensor, c_tokens: torch.Tensor, feat_rope: Optional[nn.Module] = None) -> torch.Tensor:
        if self.block.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.block.adaLN_modulation(c_tokens).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.block.adaLN_modulation(c_tokens).chunk(6, dim=-1)

        a = self.block.attn(self.modulate_token(self.block.norm1(x), shift_msa, scale_msa), rope=feat_rope)
        x = x + gate_msa * a
        m = self.block.mlp(self.modulate_token(self.block.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp * m
        return x


class OfficialFinalLayerPerTokenWrapper(nn.Module):
    """Token-wise wrapper around official FinalLayer submodules."""

    def __init__(self, final_layer: FinalLayer):
        super().__init__()
        self.final_layer = final_layer

    @staticmethod
    def modulate_token(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale) + shift

    def forward(self, x: torch.Tensor, c_tokens: torch.Tensor) -> torch.Tensor:
        shift, scale = self.final_layer.adaLN_modulation(c_tokens).chunk(2, dim=-1)
        x = self.modulate_token(self.final_layer.norm_final(x), shift, scale)
        return self.final_layer.linear(x)


def init_linear_basic(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_adaln_linear(linear: nn.Linear, mode: str) -> None:
    if mode == "zero":
        nn.init.zeros_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
    elif mode == "small":
        nn.init.normal_(linear.weight, std=0.02)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
    elif mode == "xavier":
        nn.init.xavier_uniform_(linear.weight)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
    else:
        raise ValueError(mode)


class PerPatchLightningDiTStressModel(nn.Module):
    """Official-block PAE-shape stress model with per-token MeanFlow conditioning."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.channels = cfg.channels
        self.width = cfg.width
        self.heads = cfg.heads
        self.use_abs_pos = cfg.use_abs_pos
        self.use_rope = cfg.use_rope
        self.use_checkpoint = cfg.use_checkpoint
        self.checkpoint_reentrant = cfg.checkpoint_reentrant
        self.in_proj = nn.Linear(cfg.channels, cfg.width)
        self.time_embed = PerTokenMeanFlowEmbedder(cfg.width)
        self.blocks = nn.ModuleList(
            [
                OfficialBlockPerTokenWrapper(
                    LightningDiTBlock(
                        cfg.width,
                        cfg.heads,
                        mlp_ratio=cfg.mlp_ratio,
                        use_qknorm=cfg.use_qknorm,
                        use_swiglu=cfg.use_swiglu,
                        use_rmsnorm=cfg.use_rmsnorm,
                        fused_attn=cfg.fused_attn,
                    )
                )
                for _ in range(cfg.depth)
            ]
        )
        self.final = OfficialFinalLayerPerTokenWrapper(FinalLayer(cfg.width, patch_size=1, out_channels=cfg.channels, use_rmsnorm=cfg.use_rmsnorm))
        self.ropes = nn.ModuleDict()
        if cfg.use_rope:
            half_head_dim = cfg.width // cfg.heads // 2
            for g in cfg.grids:
                self.ropes[str(g)] = VisionRotaryEmbeddingFast(dim=half_head_dim, pt_seq_len=g)
        self._pos_cache_cpu: Dict[int, torch.Tensor] = {}

        # Use official-like Xavier initialization for Linear modules, then set
        # AdaLN to small non-zero by default. Full official LightningDiT zeros
        # AdaLN/output for stable training; here small non-zero modulation is
        # intentional so the JVP stress sees time/patch conditioning signal.
        self.apply(init_linear_basic)
        for w in self.blocks:
            init_adaln_linear(w.block.adaLN_modulation[-1], cfg.adaln_init)
        init_adaln_linear(self.final.final_layer.adaLN_modulation[-1], cfg.final_adaln_init)

    def abs_pos(self, grid: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if grid not in self._pos_cache_cpu:
            pos = get_2d_sincos_pos_embed(self.width, grid)
            self._pos_cache_cpu[grid] = torch.from_numpy(pos).float().unsqueeze(0).contiguous()
        return self._pos_cache_cpu[grid].to(device=device, dtype=dtype)

    def rope_for_grid(self, grid: int) -> Optional[nn.Module]:
        if not self.use_rope:
            return None
        key = str(grid)
        if key not in self.ropes:
            raise KeyError(f"No RoPE module for grid={grid}; configured grids={list(self.ropes.keys())}")
        return self.ropes[key]

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, n, c = z.shape
        if c != self.channels:
            raise ValueError(f"channels {c} != configured {self.channels}")
        grid = math.isqrt(n)
        if grid * grid != n:
            raise ValueError(f"n_tokens must be square, got {n}")
        h = self.in_proj(z)
        if self.use_abs_pos:
            h = h + self.abs_pos(grid, h.device, h.dtype)
        c_tokens = self.time_embed(r, t)
        feat_rope = self.rope_for_grid(grid)
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                # Diagnostic path. Forward AD + checkpoint may be unsupported; failures are recorded.
                h = checkpoint(
                    lambda h_, c_: block(h_, c_, feat_rope),
                    h,
                    c_tokens,
                    use_reentrant=self.checkpoint_reentrant,
                )
            else:
                h = block(h, c_tokens, feat_rope)
        return self.final(h, c_tokens)


def call_model(model: nn.Module, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor, mode: str, device: torch.device) -> torch.Tensor:
    if mode == "fp32":
        return model(z, r, t)
    if mode == "bf16_autocast":
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return model(z, r, t).float()
        # CPU autocast support varies; keep fp32 on CPU but preserve label.
        return model(z, r, t)
    raise ValueError(mode)


def full_bundle_jvp(model: nn.Module, zt: torch.Tensor, r: torch.Tensor, t: torch.Tensor, v: torch.Tensor, mode: str, device: torch.device):
    delta = t - r
    z_dot = delta[..., None] * v
    r_dot = torch.zeros_like(r)
    t_dot = delta

    def fn(z: torch.Tensor, rr: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        return call_model(model, z, rr, tt, mode, device)

    u, du = jvp(fn, (zt, r, t), (z_dot, r_dot, t_dot))
    return u, du, (z_dot, r_dot, t_dot)


def central_fd(model: nn.Module, primals, tangents, eps: float, mode: str, device: torch.device) -> torch.Tensor:
    z, r, t = primals
    z_dot, r_dot, t_dot = tangents
    with torch.no_grad():
        u_plus = call_model(model, z + eps * z_dot, r + eps * r_dot, t + eps * t_dot, mode, device)
        u_minus = call_model(model, z - eps * z_dot, r - eps * r_dot, t - eps * t_dot, mode, device)
    return (u_plus - u_minus) / (2.0 * eps)


def params_with_grad_all_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad.detach()).all().cpu()):
            return False
    return True


def short_train_loop(
    cfg: Config,
    model: nn.Module,
    grid: int,
    latent_mode: str,
    mode: str,
    device: torch.device,
    pae_stats: PaeStats,
) -> Dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps: List[Dict[str, Any]] = []
    loop_peak = 0.0
    completed = 0
    model.train()
    for step in range(1, cfg.train_steps + 1):
        opt.zero_grad(set_to_none=True)
        problem = sample_problem(cfg, grid, latent_mode, device, pae_stats)
        zt = problem["zt"]  # type: ignore[index]
        r = problem["r"]  # type: ignore[index]
        t = problem["t"]  # type: ignore[index]
        v = problem["v"]  # type: ignore[index]
        assert isinstance(zt, torch.Tensor) and isinstance(r, torch.Tensor) and isinstance(t, torch.Tensor) and isinstance(v, torch.Tensor)

        reset_peak_memory(device)
        cuda_sync(device)
        start = time.perf_counter()
        rec: Dict[str, Any] = {"step": step}
        try:
            u, du, _ = full_bundle_jvp(model, zt, r, t, v, mode, device)
            target = (v - du).detach()
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
                loss.backward()
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
                    completed += 1
                else:
                    rec["stopped_reason"] = "nonfinite_gradient"
            else:
                rec["grad_is_finite"] = False
                rec["stopped_reason"] = "nonfinite_loss"
        except Exception as exc:  # noqa: BLE001 - record stress failure as data.
            rec.update({"status": "error", "error_type": type(exc).__name__, "error_message": str(exc), "grad_is_finite": False})
        finally:
            cuda_sync(device)
            rec["elapsed_sec"] = time.perf_counter() - start
            rec["peak_memory_mb"] = peak_memory_mb(device)
            if rec["peak_memory_mb"] is not None:
                loop_peak = max(loop_peak, float(rec["peak_memory_mb"]))
            steps.append(rec)

        if rec.get("status") == "error" or not rec.get("loss_finite", False) or not rec.get("grad_is_finite", False):
            break

    losses = [s.get("loss") for s in steps if isinstance(s.get("loss"), (int, float))]
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
        "loop_peak_memory_mb": loop_peak if device.type == "cuda" else None,
        "steps": steps,
    }


def run_one(cfg: Config, grid: int, latent_mode: str, mode: str, device: torch.device, pae_stats: PaeStats) -> Dict[str, Any]:
    n = grid * grid
    seed = cfg.seed + 1009 * grid + 9176 * cfg.depth + 13 * cfg.width
    seed += {"gaussian": 0, "correlated": 31, "pae_stats_synthetic": 71, "pae_stats_raw_synthetic": 73}[latent_mode]
    seed += {"fp32": 0, "bf16_autocast": 101}[mode]
    if cfg.use_checkpoint:
        seed += 503
    set_seed(seed)

    model = PerPatchLightningDiTStressModel(cfg).to(device=device, dtype=torch.float32)
    model_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    record: Dict[str, Any] = {
        "grid": grid,
        "n_tokens": n,
        "latent_mode": latent_mode,
        "mode": mode,
        "seed": seed,
        "model_params": model_params,
        "trainable_params": trainable_params,
        "status": "ok",
        "device": str(device),
        "official_component_claim": "official_PAE_LightningDiTBlock_submodules_with_per_token_AdaLN_wrapper",
    }

    try:
        problem = sample_problem(cfg, grid, latent_mode, device, pae_stats)
        zt = problem["zt"]  # type: ignore[index]
        r = problem["r"]  # type: ignore[index]
        t = problem["t"]  # type: ignore[index]
        v = problem["v"]  # type: ignore[index]
        latent_meta = problem["latent_meta"]  # type: ignore[index]
        assert isinstance(zt, torch.Tensor) and isinstance(r, torch.Tensor) and isinstance(t, torch.Tensor) and isinstance(v, torch.Tensor)
        record["latent_meta"] = latent_meta
        record["input_stats"] = {
            "zt": tensor_stats(zt),
            "v": tensor_stats(v),
            "r": tensor_stats(r),
            "t": tensor_stats(t),
            "delta": tensor_stats(t - r),
        }

        # Forward-only baseline.
        model.eval()
        reset_peak_memory(device)
        cuda_sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            u_forward = call_model(model, zt, r, t, mode, device)
        cuda_sync(device)
        record["forward_only_sec"] = time.perf_counter() - start
        record["forward_only_peak_memory_mb"] = peak_memory_mb(device)
        record["forward_output_finite"] = finite_tensor(u_forward)
        record["forward_output_stats"] = tensor_stats(u_forward)

        # Full-bundle JVP audit path.
        reset_peak_memory(device)
        cuda_sync(device)
        start = time.perf_counter()
        with torch.no_grad():
            u, du, tangents = full_bundle_jvp(model, zt, r, t, v, mode, device)
        cuda_sync(device)
        record["jvp_sec"] = time.perf_counter() - start
        record["jvp_peak_memory_mb"] = peak_memory_mb(device)
        record["allocated_after_jvp_mb"] = allocated_memory_mb(device)
        target = v - du
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
                "u_stats": tensor_stats(u),
                "du_stats": tensor_stats(du),
                "target_stats": tensor_stats(target),
            }
        )

        if cfg.run_fd and n <= cfg.fd_max_tokens:
            reset_peak_memory(device)
            cuda_sync(device)
            start = time.perf_counter()
            du_fd = central_fd(model, (zt, r, t), tangents, cfg.fd_eps, mode, device)
            cuda_sync(device)
            record["fd_sec"] = time.perf_counter() - start
            record["fd_peak_memory_mb"] = peak_memory_mb(device)
            record["fd_rel_err_full_jvp"] = rel(du_fd, du)
            record["fd_abs_err_norm"] = norm(du_fd - du)
            record["fd_norm"] = norm(du_fd)
            record["fd_finite"] = finite_tensor(du_fd)
        else:
            record["fd_skipped"] = True

        # r=t degeneracy sanity check: tangent vector should be exactly zero.
        with torch.no_grad():
            t_eq = t.clone()
            r_eq = t_eq.clone()
            _u_eq, du_eq, _ = full_bundle_jvp(model, zt, r_eq, t_eq, v, mode, device)
            target_eq = v - du_eq
        record["degenerate_du_norm"] = norm(du_eq)
        record["degenerate_target_minus_v_max_abs"] = float((target_eq - v).detach().float().abs().max().cpu())
        record["degenerate_finite"] = finite_tensor(du_eq) and finite_tensor(target_eq)

        # Short training stress.
        if cfg.train_steps > 0:
            train_stats = short_train_loop(cfg, model, grid, latent_mode, mode, device, pae_stats)
            record["short_train"] = train_stats
        else:
            record["short_train_skipped"] = True

        bad = not (record.get("forward_output_finite", False) and record.get("u_finite", False) and record.get("du_finite", False) and record.get("target_finite", False))
        if record.get("fd_finite") is False:
            bad = True
        if record.get("short_train", {}).get("grad_all_finite") is False:
            bad = True
        if record.get("short_train", {}).get("loss_all_finite") is False:
            bad = True
        record["has_nan_or_inf"] = bool(bad)

    except Exception as exc:  # noqa: BLE001 - experiment should record failures as data.
        record["status"] = "error"
        record["error_type"] = type(exc).__name__
        record["error_message"] = str(exc)
        record["has_nan_or_inf"] = True
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return record


def gate_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    errors = [r for r in results if r.get("status") != "ok"]
    fp32 = [r for r in ok if r.get("mode") == "fp32"]
    fp32_1024 = [r for r in fp32 if r.get("n_tokens") == 1024]
    fd_vals = [r["fd_rel_err_full_jvp"] for r in fp32_1024 if "fd_rel_err_full_jvp" in r]
    max_fd = max(fd_vals) if fd_vals else None
    max_degen = max((r.get("degenerate_target_minus_v_max_abs", float("inf")) for r in fp32_1024), default=None)
    train_all_finite = all(
        r.get("short_train", {}).get("grad_all_finite", False)
        and r.get("short_train", {}).get("loss_all_finite", False)
        and r.get("short_train", {}).get("u_du_target_all_finite", False)
        for r in fp32_1024
    ) if fp32_1024 else False
    jvp_all_finite = all(r.get("du_finite", False) and r.get("target_finite", False) and r.get("u_finite", False) for r in fp32_1024) if fp32_1024 else False
    no_nan = all(not r.get("has_nan_or_inf", True) for r in fp32_1024) if fp32_1024 else False
    strict = bool(
        fp32_1024
        and jvp_all_finite
        and train_all_finite
        and no_nan
        and max_fd is not None
        and max_fd < 1e-2
        and max_degen is not None
        and max_degen < 1e-7
    )
    practical = bool(
        fp32_1024
        and jvp_all_finite
        and train_all_finite
        and no_nan
        and max_fd is not None
        and max_fd < 5e-2
        and max_degen is not None
        and max_degen < 1e-7
    )
    return {
        "n_results": len(results),
        "n_ok": len(ok),
        "n_error": len(errors),
        "fp32_result_count": len(fp32),
        "fp32_1024_result_count": len(fp32_1024),
        "max_fp32_1024_fd_rel_err": max_fd,
        "max_fp32_1024_degenerate_target_minus_v_max_abs": max_degen,
        "fp32_1024_train_all_finite": train_all_finite,
        "fp32_1024_full_jvp_all_finite": jvp_all_finite,
        "fp32_1024_no_nan_or_inf": no_nan,
        "strict_gate_pass": strict,
        "practical_gate_pass": practical,
        "error_modes": [
            {
                "grid": r.get("grid"),
                "latent_mode": r.get("latent_mode"),
                "mode": r.get("mode"),
                "error_type": r.get("error_type"),
                "error_message": r.get("error_message"),
            }
            for r in errors
        ],
    }


def write_summary(out_dir: Path, payload: Dict[str, Any]) -> None:
    cfg = payload["config"]
    gate = payload["gate"]
    results = payload["results"]
    lines: List[str] = []
    lines.append("# Experiment 4 — Real LightningDiT Block / PAE Latent / Short-Train Stress 结果\n\n")
    lines.append(f"- 生成时间：`{payload['created_at_utc']}`\n")
    lines.append(f"- torch：`{payload['torch_version']}`；cuda_available：`{payload['cuda_available']}`；device：`{payload['device']}`\n")
    lines.append(f"- PAE repo：`{payload['pae_gen_root']}`；commit：`{payload.get('pae_git_commit')}`\n")
    lines.append(f"- official modules：`LightningDiTBlock` / `Attention` / `RMSNorm` / `SwiGLU` / `FinalLayer` / `VisionRotaryEmbeddingFast`\n")
    lines.append(f"- per-token wrapper：官方 block 子模块保留，AdaLN conditioning 从 `[B,D]` 改为 `[B,N,D]` 并沿 `dim=-1` chunk。\n")
    lines.append(f"- C/width/heads/depth：`{cfg['channels']}/{cfg['width']}/{cfg['heads']}/{cfg['depth']}`；mlp_ratio：`{cfg['mlp_ratio']}`\n")
    lines.append(f"- grids：`{cfg['grids']}`；modes：`{cfg['modes']}`；latent_modes：`{cfg['latent_modes']}`\n")
    lines.append(f"- train_steps：`{cfg['train_steps']}`；time_sampler：`{cfg['time_sampler']}`；equal_prob(r=t)：`{cfg['equal_prob']}`\n")
    lines.append(f"- RoPE/abs_pos/RMSNorm/SwiGLU/SDPA：`{cfg['use_rope']}/{cfg['use_abs_pos']}/{cfg['use_rmsnorm']}/{cfg['use_swiglu']}/{cfg['fused_attn']}`；sdpa_kernel：`{cfg['sdpa_kernel']}`；allow_tf32：`{cfg['allow_tf32']}`\n")
    lines.append(f"- torch.compile：enable_dynamo=`{cfg['enable_dynamo']}`；unwrapped=`{payload.get('compile_unwrapped')}`\n")
    lines.append(f"- PAE latent stats：`{cfg['pae_stats_path']}`；available：`{payload.get('pae_stats', {}).get('available')}`\n")
    lines.append("\n## Gate 摘要\n\n")
    lines.append(f"- fp32 N=1024 max FD relative error：`{gate['max_fp32_1024_fd_rel_err']}`\n")
    lines.append(f"- fp32 N=1024 max r=t target-v max abs：`{gate['max_fp32_1024_degenerate_target_minus_v_max_abs']}`\n")
    lines.append(f"- fp32 N=1024 full JVP finite：`{gate['fp32_1024_full_jvp_all_finite']}`\n")
    lines.append(f"- fp32 N=1024 short train finite：`{gate['fp32_1024_train_all_finite']}`\n")
    lines.append(f"- practical gate pass：`{gate['practical_gate_pass']}`\n")
    lines.append(f"- strict gate pass：`{gate['strict_gate_pass']}`\n")
    if gate["error_modes"]:
        lines.append("\n### 记录到的错误模式\n\n")
        for e in gate["error_modes"]:
            lines.append(f"- grid={e['grid']} latent={e['latent_mode']} mode={e['mode']}：{e['error_type']} — `{e['error_message']}`\n")
    lines.append("\n## 核心指标表\n\n")
    lines.append("| grid | N | latent | mode | status | params | FD rel err | du/v | JVP sec | FD sec | FWD sec | train steps | final loss | train peak MB | JVP peak MB | degen max | NaN/Inf |\n")
    lines.append("|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for r in results:
        train = r.get("short_train", {})
        lines.append(
            f"| {r.get('grid')} | {r.get('n_tokens')} | {r.get('latent_mode')} | {r.get('mode')} | {r.get('status')} | "
            f"{r.get('model_params')} | {fmt(r.get('fd_rel_err_full_jvp'))} | {fmt(r.get('du_over_v_norm_ratio'))} | "
            f"{fmt(r.get('jvp_sec'))} | {fmt(r.get('fd_sec'))} | {fmt(r.get('forward_only_sec'))} | "
            f"{train.get('n_completed_optimizer_steps')} | {fmt(train.get('final_loss'))} | {fmt(train.get('loop_peak_memory_mb'))} | "
            f"{fmt(r.get('jvp_peak_memory_mb'))} | {fmt(r.get('degenerate_target_minus_v_max_abs'))} | {r.get('has_nan_or_inf')} |\n"
        )
    lines.append("\n## 解释与限制\n\n")
    lines.append("- 本实验是 **真实官方 LightningDiT block 子模块集成**，不是完整官方 ImageNet 训练栈；核心差异是 per-token MeanFlow conditioning wrapper。\n")
    lines.append("- `pae_stats_synthetic` 使用官方 PAE DINOv2L latent mean/std 构造并按 `latent_norm: true` 归一化；它不是 ImageNet PAE latent shard。\n")
    lines.append("- `sdpa_kernel=math` 时仍调用官方 `F.scaled_dot_product_attention` 分支，但强制数学后端以便 forward AD/JVP 可审计；flash/mem-efficient 后端应另做兼容性探针。\n")
    lines.append("- bf16/autocast 仅为稳定性诊断；JVP correctness gate 以 fp32 + central finite difference 为准。\n")
    lines.append("- 该 stress 不给出 FID/FDr 或生成质量结论，也不证明 XL/1 全规模长训稳定性。\n")
    (out_dir / "lightningdit_block_pae_shorttrain_summary.md").write_text("".join(lines), encoding="utf-8")


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
    out_dir = Path(cfg.output_dir) if cfg.output_dir else Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    pae_stats = PaeStats(Path(cfg.pae_stats_path))

    print(f"[env] device={device} torch={torch.__version__} cuda={torch.cuda.is_available()} sdpa={sdpa_info}", flush=True)
    print(f"[env] PAE_GEN_ROOT={PAE_GEN_ROOT} compile_unwrapped={COMPILE_UNWRAPPED} dynamo_requested={DYNAMO_REQUESTED}", flush=True)
    print(f"[env] pae_stats={pae_stats.meta}", flush=True)

    set_seed(cfg.seed)
    results: List[Dict[str, Any]] = []
    total_start = time.perf_counter()
    for latent_mode in cfg.latent_modes:
        for grid in cfg.grids:
            for mode in cfg.modes:
                print(
                    f"[run] grid={grid} N={grid*grid} latent={latent_mode} mode={mode} "
                    f"width={cfg.width} heads={cfg.heads} depth={cfg.depth} train_steps={cfg.train_steps}",
                    flush=True,
                )
                rec = run_one(cfg, grid, latent_mode, mode, device, pae_stats)
                results.append(rec)
                if rec.get("status") == "ok":
                    train = rec.get("short_train", {})
                    print(
                        f"[done] grid={grid} mode={mode} fd={fmt(rec.get('fd_rel_err_full_jvp'))} "
                        f"du/v={fmt(rec.get('du_over_v_norm_ratio'))} degen={fmt(rec.get('degenerate_target_minus_v_max_abs'))} "
                        f"jvp_sec={fmt(rec.get('jvp_sec'))} train_steps={train.get('n_completed_optimizer_steps')} "
                        f"final_loss={fmt(train.get('final_loss'))} peak={fmt(train.get('loop_peak_memory_mb'))}MB",
                        flush=True,
                    )
                else:
                    print(f"[error] grid={grid} mode={mode} {rec.get('error_type')}: {rec.get('error_message')}", flush=True)

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
        "results": results,
    }
    payload["gate"] = gate_summary(results)
    metrics_path = out_dir / "lightningdit_block_pae_shorttrain_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(out_dir, payload)
    print(f"[write] {metrics_path}", flush=True)
    print(f"[write] {out_dir / 'lightningdit_block_pae_shorttrain_summary.md'}", flush=True)
    print(f"[gate] practical={payload['gate']['practical_gate_pass']} strict={payload['gate']['strict_gate_pass']}", flush=True)


if __name__ == "__main__":
    main()
