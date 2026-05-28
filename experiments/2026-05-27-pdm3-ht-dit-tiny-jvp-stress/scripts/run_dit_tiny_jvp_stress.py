#!/usr/bin/env python3
"""
PDM-3-HT Experiment 3: LightningDiT-Tiny / PAE-shape Full-Bundle JVP Stress Test.

This is a DiT-like proxy rather than the official LightningDiT implementation.
It keeps the ingredients needed for the JVP risk audit:
  - PAE-like tokens: B x (H*W) x C, default C=32, H=W=16/32.
  - global self-attention coupling.
  - per-token time conditioning via AdaLN-like modulation.
  - full-bundle torch.func.jvp over (z_t, r_vec, t_vec).
  - finite-difference audit and MeanFlow-style stop-gradient training step.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp


@dataclass
class Config:
    seed: int = 20260527
    batch_size: int = 1
    channels: int = 32
    width: int = 128
    heads: int = 4
    depth: int = 2
    grids: Tuple[int, ...] = (16, 32)
    modes: Tuple[str, ...] = ("fp32",)
    latent_modes: Tuple[str, ...] = ("gaussian",)
    device: str = "auto"
    fd_eps: float = 1e-2
    fd_max_tokens: int = 1024
    run_fd: bool = True
    run_backward: bool = True
    t_low: float = 0.05
    t_high: float = 0.95
    equal_prob: float = 0.0
    lr: float = 1e-4
    grad_clip: float = 1.0
    output_dir: Optional[str] = None


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
    p.add_argument("--grids", type=str, default="16,32", help="comma-separated square grid sides")
    p.add_argument("--modes", type=str, default="fp32", choices=None,
                   help="comma-separated: fp32,bf16_autocast")
    p.add_argument("--latent-modes", type=str, default="gaussian", help="comma-separated: gaussian,correlated")
    p.add_argument("--fd-eps", type=float, default=1e-2)
    p.add_argument("--fd-max-tokens", type=int, default=1024)
    p.add_argument("--no-fd", action="store_true")
    p.add_argument("--no-backward", action="store_true")
    p.add_argument("--t-low", type=float, default=0.05)
    p.add_argument("--t-high", type=float, default=0.95)
    p.add_argument("--equal-prob", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    return p.parse_args()


def make_config(args: argparse.Namespace) -> Config:
    modes = parse_csv_str(args.modes)
    allowed_modes = {"fp32", "bf16_autocast"}
    bad_modes = set(modes) - allowed_modes
    if bad_modes:
        raise ValueError(f"unsupported modes: {sorted(bad_modes)}")
    latent_modes = parse_csv_str(args.latent_modes)
    allowed_latents = {"gaussian", "correlated"}
    bad_latents = set(latent_modes) - allowed_latents
    if bad_latents:
        raise ValueError(f"unsupported latent modes: {sorted(bad_latents)}")
    return Config(
        seed=args.seed,
        batch_size=args.batch_size,
        channels=args.channels,
        width=args.width,
        heads=args.heads,
        depth=args.depth,
        grids=parse_csv_int(args.grids),
        modes=modes,
        latent_modes=latent_modes,
        device=args.device,
        fd_eps=args.fd_eps,
        fd_max_tokens=args.fd_max_tokens,
        run_fd=not args.no_fd,
        run_backward=not args.no_backward,
        t_low=args.t_low,
        t_high=args.t_high,
        equal_prob=args.equal_prob,
        lr=args.lr,
        grad_clip=args.grad_clip,
        output_dir=args.output_dir,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
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
    return float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))


def allocated_memory_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.memory_allocated(device) / (1024 ** 2))


def norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().float()).cpu())


def rel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return norm(a - b) / (norm(a) + eps)


def finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().cpu())


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    y = x.detach().float().flatten()
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


class PerTokenConditioner(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, width),
            nn.SiLU(),
            nn.Linear(width, 6 * width),
        )
        # AdaLN-zero style: start with small modulation so the proxy is stable but non-trivial.
        nn.init.normal_(self.net[-1].weight, std=0.02)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def features(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        delta = t - r
        return torch.stack([
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
        ], dim=-1)

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        return self.net(self.features(r, t)).chunk(6, dim=-1)


class DiTAttention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        assert width % heads == 0
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(logits, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d)
        return self.proj(y)


class DiTTinyBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.attn = DiTAttention(width, heads)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, mlp_ratio * width),
            nn.GELU(),
            nn.Linear(mlp_ratio * width, width),
        )
        self.cond = PerTokenConditioner(width)

    @staticmethod
    def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale) + shift

    def forward(self, x: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.cond(r, t)
        a = self.attn(self.modulate(self.norm1(x), shift_a, scale_a))
        x = x + torch.tanh(gate_a) * a
        m = self.mlp(self.modulate(self.norm2(x), shift_m, scale_m))
        x = x + torch.tanh(gate_m) * m
        return x


class DiTTinyPAEProxy(nn.Module):
    """LightningDiT-like tiny proxy with per-token AdaLN time conditioning."""

    def __init__(self, channels: int, width: int, heads: int, depth: int, max_grid: int = 64):
        super().__init__()
        self.channels = channels
        self.width = width
        self.in_proj = nn.Linear(channels, width)
        self.blocks = nn.ModuleList([DiTTinyBlock(width, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, channels)
        # Learned spatial embedding table to mimic DiT positional information while keeping variable grid support.
        self.pos_table = nn.Parameter(torch.zeros(1, max_grid * max_grid, width))
        nn.init.normal_(self.pos_table, std=0.02)

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, n, c = z.shape
        h = self.in_proj(z)
        h = h + self.pos_table[:, :n, :].to(dtype=h.dtype, device=h.device)
        for block in self.blocks:
            h = block(h, r, t)
        return self.out(self.out_norm(h))


def smooth_latent(x: torch.Tensor, grid: int) -> torch.Tensor:
    """Make a simple spatially correlated latent proxy using avg-pool smoothing."""
    b, n, c = x.shape
    y = x.transpose(1, 2).reshape(b, c, grid, grid)
    y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
    y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
    # Re-normalize roughly to unit variance to keep comparison fair.
    y = y / (y.flatten(1).std(dim=1).view(b, 1, 1, 1) + 1e-6)
    return y.flatten(2).transpose(1, 2).contiguous()


def sample_problem(cfg: Config, grid: int, latent_mode: str, device: torch.device) -> Dict[str, torch.Tensor]:
    b, n, c = cfg.batch_size, grid * grid, cfg.channels
    z0 = torch.randn(b, n, c, device=device, dtype=torch.float32)
    eps = torch.randn(b, n, c, device=device, dtype=torch.float32)
    if latent_mode == "correlated":
        z0 = smooth_latent(z0, grid)
        # Keep eps white noise, matching diffusion endpoint.
    elif latent_mode != "gaussian":
        raise ValueError(latent_mode)
    v = eps - z0
    t = cfg.t_low + (cfg.t_high - cfg.t_low) * torch.rand(b, n, device=device, dtype=torch.float32)
    r = torch.rand(b, n, device=device, dtype=torch.float32) * t
    if cfg.equal_prob > 0:
        mask = torch.rand(b, n, device=device) < cfg.equal_prob
        r = torch.where(mask, t, r)
    zt = (1.0 - t[..., None]) * z0 + t[..., None] * eps
    return {"z0": z0, "eps": eps, "v": v, "r": r, "t": t, "zt": zt}


def call_model(model: nn.Module, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor, mode: str, device: torch.device) -> torch.Tensor:
    if mode == "fp32":
        return model(z, r, t)
    if mode == "bf16_autocast":
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                return model(z, r, t).float()
        # CPU autocast support varies; keep fp32 on CPU but label records device.
        return model(z, r, t)
    raise ValueError(mode)


def full_bundle_jvp(model: nn.Module, zt: torch.Tensor, r: torch.Tensor, t: torch.Tensor, v: torch.Tensor, mode: str, device: torch.device):
    delta = t - r
    z_dot = delta[..., None] * v
    r_dot = torch.zeros_like(r)
    t_dot = delta

    def fn(z, rr, tt):
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


def optimizer_step_stress(
    cfg: Config,
    model: nn.Module,
    problem: Dict[str, torch.Tensor],
    mode: str,
    device: torch.device,
) -> Dict[str, float | bool]:
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    opt.zero_grad(set_to_none=True)
    zt, r, t, v = problem["zt"], problem["r"], problem["t"], problem["v"]
    reset_peak_memory(device)
    cuda_sync(device)
    start = time.perf_counter()
    u, du, _ = full_bundle_jvp(model, zt, r, t, v, mode, device)
    target = (v - du).detach()
    loss = F.mse_loss(u.float(), target.float())
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()
    cuda_sync(device)
    elapsed = time.perf_counter() - start
    grad_norm_value = float(grad_norm.detach().float().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm)
    return {
        "loss": float(loss.detach().float().cpu()),
        "grad_norm_before_clip": grad_norm_value,
        "grad_clip_threshold": cfg.grad_clip,
        "grad_is_finite": math.isfinite(grad_norm_value),
        "elapsed_sec": elapsed,
        "peak_memory_mb": peak_memory_mb(device),
    }


def run_one(cfg: Config, grid: int, latent_mode: str, mode: str, device: torch.device) -> Dict:
    n = grid * grid
    seed = cfg.seed + n + (17 if latent_mode == "correlated" else 0) + (101 if mode == "bf16_autocast" else 0)
    set_seed(seed)
    model = DiTTinyPAEProxy(cfg.channels, cfg.width, cfg.heads, cfg.depth, max_grid=max(64, max(cfg.grids))).to(device=device, dtype=torch.float32)
    model.train()
    params = sum(p.numel() for p in model.parameters())
    problem = sample_problem(cfg, grid, latent_mode, device)
    zt, r, t, v = problem["zt"], problem["r"], problem["t"], problem["v"]

    record: Dict = {
        "grid": grid,
        "n_tokens": n,
        "latent_mode": latent_mode,
        "mode": mode,
        "seed": seed,
        "model_params": params,
        "status": "ok",
        "device": str(device),
    }

    try:
        # Forward-only baseline timing.
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
        record.update({
            "u_norm": norm(u),
            "v_norm": norm(v),
            "du_norm": norm(du),
            "target_norm": norm(target),
            "du_over_v_norm_ratio": norm(du) / (norm(v) + 1e-12),
            "u_finite": finite_tensor(u),
            "du_finite": finite_tensor(du),
            "target_finite": finite_tensor(target),
            "du_stats": tensor_stats(du),
            "target_stats": tensor_stats(target),
        })

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

        # r=t degeneracy sanity check.
        with torch.no_grad():
            t_eq = t.clone()
            r_eq = t_eq.clone()
            u_eq, du_eq, _ = full_bundle_jvp(model, zt, r_eq, t_eq, v, mode, device)
            target_eq = v - du_eq
        record["degenerate_du_norm"] = norm(du_eq)
        record["degenerate_target_minus_v_max_abs"] = float((target_eq - v).detach().float().abs().max().cpu())
        record["degenerate_finite"] = finite_tensor(du_eq) and finite_tensor(target_eq)

        if cfg.run_backward:
            model.train()
            # Re-sample problem for training-like step to avoid sharing no_grad tensors is not necessary,
            # but keeps the train-step independent from the audit path.
            train_problem = sample_problem(cfg, grid, latent_mode, device)
            train_stats = optimizer_step_stress(cfg, model, train_problem, mode, device)
            record["train_step"] = train_stats
        else:
            record["train_step_skipped"] = True

        bad = not (record.get("u_finite", False) and record.get("du_finite", False) and record.get("target_finite", False))
        if record.get("fd_finite") is False:
            bad = True
        if record.get("train_step", {}).get("grad_is_finite") is False:
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


def gate_summary(results: List[Dict]) -> Dict:
    fp32 = [r for r in results if r.get("status") == "ok" and r.get("mode") == "fp32"]
    ok_all = [r for r in results if r.get("status") == "ok"]
    errors = [r for r in results if r.get("status") != "ok"]
    fd_vals = [r["fd_rel_err_full_jvp"] for r in fp32 if "fd_rel_err_full_jvp" in r]
    max_fd = max(fd_vals) if fd_vals else None
    max_degen = max((r.get("degenerate_target_minus_v_max_abs", float("inf")) for r in fp32), default=None)
    train_finite = all(r.get("train_step", {}).get("grad_is_finite", False) for r in fp32 if "train_step" in r)
    full_jvp_all_shapes = all(r.get("status") == "ok" and r.get("du_finite", False) for r in fp32)
    has_256 = any(r.get("n_tokens") == 256 for r in fp32)
    has_1024 = any(r.get("n_tokens") == 1024 for r in fp32)
    strict_pass = bool(
        fp32
        and not errors
        and full_jvp_all_shapes
        and has_256
        and has_1024
        and (max_fd is not None and max_fd < 1e-2)
        and (max_degen is not None and max_degen == 0.0)
        and train_finite
    )
    practical_pass = bool(
        fp32
        and full_jvp_all_shapes
        and has_256
        and has_1024
        and (max_fd is not None and max_fd < 5e-2)
        and (max_degen is not None and max_degen < 1e-7)
        and train_finite
    )
    return {
        "n_results": len(results),
        "n_ok": len(ok_all),
        "n_error": len(errors),
        "fp32_result_count": len(fp32),
        "max_fp32_fd_rel_err": max_fd,
        "max_fp32_degenerate_target_minus_v_max_abs": max_degen,
        "fp32_train_grad_all_finite": train_finite,
        "fp32_full_jvp_all_shapes_finite": full_jvp_all_shapes,
        "has_fp32_256": has_256,
        "has_fp32_1024": has_1024,
        "strict_gate_pass": strict_pass,
        "practical_gate_pass": practical_pass,
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


def write_summary(out_dir: Path, payload: Dict) -> None:
    results = payload["results"]
    gate = payload["gate"]
    lines: List[str] = []
    lines.append("# DiT-Tiny / PAE-shape Full-Bundle JVP Stress Test 结果\n\n")
    lines.append(f"- 生成时间：{payload['created_at_utc']}\n")
    cfg = payload["config"]
    lines.append(f"- seed：`{cfg['seed']}`\n")
    lines.append(f"- grids：`{cfg['grids']}`\n")
    lines.append(f"- modes：`{cfg['modes']}`\n")
    lines.append(f"- latent_modes：`{cfg['latent_modes']}`\n")
    lines.append(f"- proxy：C={cfg['channels']}, width={cfg['width']}, heads={cfg['heads']}, depth={cfg['depth']}\n")
    lines.append(f"- torch：`{payload['torch_version']}`；cuda_available：`{payload['cuda_available']}`\n")
    lines.append("\n## Gate 摘要\n\n")
    lines.append(f"- fp32 max FD relative error：`{gate['max_fp32_fd_rel_err']}`\n")
    lines.append(f"- fp32 max r=t target-v max abs：`{gate['max_fp32_degenerate_target_minus_v_max_abs']}`\n")
    lines.append(f"- fp32 full JVP all tested shapes finite：`{gate['fp32_full_jvp_all_shapes_finite']}`\n")
    lines.append(f"- fp32 train-step gradients finite：`{gate['fp32_train_grad_all_finite']}`\n")
    lines.append(f"- practical gate pass：`{gate['practical_gate_pass']}`\n")
    lines.append(f"- strict gate pass：`{gate['strict_gate_pass']}`\n")
    if gate["error_modes"]:
        lines.append("\n### 记录到的错误模式\n\n")
        for e in gate["error_modes"]:
            lines.append(f"- grid={e['grid']} latent={e['latent_mode']} mode={e['mode']}：{e['error_type']} — `{e['error_message']}`\n")
    lines.append("\n## 核心指标表\n\n")
    lines.append("| grid | N | latent | mode | status | params | fd rel err | du/v | jvp sec | fwd sec | train sec | jvp peak MB | train peak MB | grad finite | grad norm | NaN/Inf |\n")
    lines.append("|---:|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|\n")
    for r in results:
        train = r.get("train_step", {})
        lines.append(
            f"| {r.get('grid')} | {r.get('n_tokens')} | {r.get('latent_mode')} | {r.get('mode')} | {r.get('status')} | "
            f"{r.get('model_params')} | {fmt(r.get('fd_rel_err_full_jvp'))} | {fmt(r.get('du_over_v_norm_ratio'))} | "
            f"{fmt(r.get('jvp_sec'))} | {fmt(r.get('forward_only_sec'))} | {fmt(train.get('elapsed_sec'))} | "
            f"{fmt(r.get('jvp_peak_memory_mb'))} | {fmt(train.get('peak_memory_mb'))} | {train.get('grad_is_finite')} | "
            f"{fmt(train.get('grad_norm_before_clip'))} | {r.get('has_nan_or_inf')} |\n"
        )
    lines.append("\n## 解释要点\n\n")
    lines.append("- 本实验使用 LightningDiT-like tiny proxy，不是官方 LightningDiT；它用于隔离验证 PAE-shape token、全局 attention coupling、per-token AdaLN 与 full-bundle JVP 的工程风险。\n")
    lines.append("- 若 fp32 的 FD error < 1e-2 且 N=1024 train-step finite，则说明 full-bundle JVP 在该 proxy 上具备继续放大 depth/width 的资格。\n")
    lines.append("- 若 bf16_autocast 与 fp32 差距显著或报错，应视为 mixed-precision 风险；真实训练应优先保持 JVP path fp32，并单独做 autocast 兼容性工程。\n")
    (out_dir / "dit_tiny_jvp_stress_summary.md").write_text("".join(lines), encoding="utf-8")


def fmt(x) -> str:
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


def main() -> None:
    args = parse_args()
    cfg = make_config(args)
    device = resolve_device(cfg.device)
    out_dir = Path(cfg.output_dir) if cfg.output_dir else Path(__file__).resolve().parents[1] / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    results: List[Dict] = []
    total_start = time.perf_counter()
    for latent_mode in cfg.latent_modes:
        for grid in cfg.grids:
            for mode in cfg.modes:
                print(f"[run] grid={grid} N={grid*grid} latent={latent_mode} mode={mode} device={device} width={cfg.width} depth={cfg.depth}", flush=True)
                rec = run_one(cfg, grid, latent_mode, mode, device)
                results.append(rec)
                if rec.get("status") == "ok":
                    train = rec.get("train_step", {})
                    print(
                        f"[done] grid={grid} mode={mode} fd={fmt(rec.get('fd_rel_err_full_jvp'))} "
                        f"du/v={fmt(rec.get('du_over_v_norm_ratio'))} jvp_sec={fmt(rec.get('jvp_sec'))} "
                        f"peak={fmt(rec.get('jvp_peak_memory_mb'))}MB grad_finite={train.get('grad_is_finite')}",
                        flush=True,
                    )
                else:
                    print(f"[error] grid={grid} mode={mode} {rec.get('error_type')}: {rec.get('error_message')}", flush=True)

    payload = {
        "created_at_utc": now_utc(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "config": asdict(cfg),
        "total_elapsed_sec": time.perf_counter() - total_start,
        "results": results,
    }
    payload["gate"] = gate_summary(results)
    metrics_path = out_dir / "dit_tiny_jvp_stress_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(out_dir, payload)
    print(f"[write] {metrics_path}", flush=True)
    print(f"[write] {out_dir / 'dit_tiny_jvp_stress_summary.md'}", flush=True)
    print(f"[gate] practical={payload['gate']['practical_gate_pass']} strict={payload['gate']['strict_gate_pass']}", flush=True)


if __name__ == "__main__":
    main()
