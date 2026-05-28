#!/usr/bin/env python3
"""
PDM-3-HT Full-Bundle JVP Toy Harness

Purpose
-------
Verify the first-step Per-Patch MeanFlow implementation before any large-model
training:

    delta = t_vec - r_vec
    z_dot = delta[..., None] * v
    r_dot = 0
    t_dot = delta
    u_tgt = v - du_dc

where du_dc is computed by one full-bundle torch.func.jvp over the coupled model
inputs (z_t, r_vec, t_vec).

This script intentionally uses a tiny self-attention model so that u_i depends
on other patches/tokens. It audits:
  - full-bundle JVP vs central finite difference;
  - r=t degeneration to Flow Matching;
  - full-bundle JVP vs diagonal-only per-patch approximation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
from torch.func import jvp


@dataclass
class Config:
    seed: int = 20260527
    batch_size: int = 2
    channels: int = 8
    width: int = 32
    heads: int = 4
    depth: int = 2
    n_tokens_list: Tuple[int, ...] = (4, 16, 64)
    dtype: str = "float32"
    device: str = "auto"
    fd_eps_list: Tuple[float, ...] = (1e-2, 1e-3, 1e-4)
    t_low: float = 0.05
    t_high: float = 0.95
    equal_prob: float = 0.0


class TinyCoupledModel(nn.Module):
    """Small transformer-like model with explicit token coupling.

    Inputs:
      z: [B, N, C]
      r: [B, N]
      t: [B, N]
    Output:
      u: [B, N, C]
    """

    def __init__(self, channels: int, width: int, heads: int, depth: int):
        super().__init__()
        assert width % heads == 0, "width must be divisible by heads"
        self.channels = channels
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.depth = depth

        self.in_proj = nn.Linear(channels + 6, width)
        self.blocks = nn.ModuleList([TinyAttentionBlock(width, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, channels)

    def time_features(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Smooth deterministic features; no random embeddings needed for toy audit.
        dt = t - r
        return torch.stack(
            [
                r,
                t,
                dt,
                torch.sin(math.pi * t),
                torch.cos(math.pi * t),
                torch.sin(math.pi * dt),
            ],
            dim=-1,
        )

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, self.time_features(r, t)], dim=-1)
        h = self.in_proj(h)
        for block in self.blocks:
            h = block(h)
        return self.out(self.out_norm(h))


class TinyAttentionBlock(nn.Module):
    """Custom attention block using basic ops to keep forward-mode AD transparent."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        b, n, d = h.shape
        x = self.norm1(h)
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # [B, N, H, Dh]
        q = q.transpose(1, 2)        # [B, H, N, Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn_logits, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d)
        h = h + self.proj(y)
        h = h + self.mlp(self.norm2(h))
        return h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    parser.add_argument("--seed", type=int, default=20260527)
    parser.add_argument("--n-tokens", type=str, default="4,16,64")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--depth", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return torch.device(device_arg)


def resolve_dtype(dtype_arg: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[dtype_arg]


def sample_problem(
    b: int,
    n: int,
    c: int,
    device: torch.device,
    dtype: torch.dtype,
    t_low: float,
    t_high: float,
    equal_prob: float = 0.0,
) -> Dict[str, torch.Tensor]:
    z0 = torch.randn(b, n, c, device=device, dtype=dtype)
    eps = torch.randn(b, n, c, device=device, dtype=dtype)
    v = eps - z0
    t = t_low + (t_high - t_low) * torch.rand(b, n, device=device, dtype=dtype)
    alpha = torch.rand(b, n, device=device, dtype=dtype)
    r = alpha * t
    if equal_prob > 0:
        mask = torch.rand(b, n, device=device) < equal_prob
        r = torch.where(mask, t, r)
    zt = (1.0 - t[..., None]) * z0 + t[..., None] * eps
    return {"z0": z0, "eps": eps, "v": v, "r": r, "t": t, "zt": zt}


def full_bundle_jvp(model: nn.Module, zt: torch.Tensor, r: torch.Tensor, t: torch.Tensor, v: torch.Tensor):
    delta = t - r
    z_dot = delta[..., None] * v
    r_dot = torch.zeros_like(r)
    t_dot = delta

    def fn(z, rr, tt):
        return model(z, rr, tt)

    u, du_dc = jvp(fn, (zt, r, t), (z_dot, r_dot, t_dot))
    return u, du_dc, (z_dot, r_dot, t_dot)


def central_finite_difference(model: nn.Module, primals, tangents, eps: float) -> torch.Tensor:
    z, r, t = primals
    z_dot, r_dot, t_dot = tangents

    def fn(zz, rr, tt):
        return model(zz, rr, tt)

    u_plus = fn(z + eps * z_dot, r + eps * r_dot, t + eps * t_dot)
    u_minus = fn(z - eps * z_dot, r - eps * r_dot, t - eps * t_dot)
    return (u_plus - u_minus) / (2.0 * eps)


def diagonal_patch_jvp(model: nn.Module, zt: torch.Tensor, r: torch.Tensor, t: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute diagonal-only per-patch JVP by N tiny JVPs.

    For output token i, keep tangent only on input token i and collect derivative
    of output token i. This deliberately ignores cross-patch terms from j != i.
    """
    b, n, c = zt.shape
    delta = t - r
    z_dot_full = delta[..., None] * v
    t_dot_full = delta
    r_dot_zero = torch.zeros_like(r)
    diag = torch.zeros_like(zt)

    def fn(z, rr, tt):
        return model(z, rr, tt)

    for i in range(n):
        z_dot = torch.zeros_like(zt)
        t_dot = torch.zeros_like(t)
        z_dot[:, i, :] = z_dot_full[:, i, :]
        t_dot[:, i] = t_dot_full[:, i]
        _, du_i_all = jvp(fn, (zt, r, t), (z_dot, r_dot_zero, t_dot))
        diag[:, i, :] = du_i_all[:, i, :]
    return diag


def norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach()).cpu())


def rel_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return norm(a - b) / (norm(a) + eps)


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    xd = x.detach().flatten()
    return {
        "mean": float(xd.mean().cpu()),
        "std": float(xd.std(unbiased=False).cpu()),
        "min": float(xd.min().cpu()),
        "max": float(xd.max().cpu()),
        "norm": norm(x),
    }


def run_one_config(cfg: Config, n_tokens: int, device: torch.device, dtype: torch.dtype) -> Dict:
    # Re-seed per N for reproducibility while making configs distinct.
    set_seed(cfg.seed + n_tokens)
    model = TinyCoupledModel(cfg.channels, cfg.width, cfg.heads, cfg.depth).to(device=device, dtype=dtype)
    model.eval()

    problem = sample_problem(
        cfg.batch_size,
        n_tokens,
        cfg.channels,
        device,
        dtype,
        cfg.t_low,
        cfg.t_high,
        cfg.equal_prob,
    )
    zt, r, t, v = problem["zt"], problem["r"], problem["t"], problem["v"]

    with torch.no_grad():
        # no_grad does not disable forward-mode AD in torch.func.jvp; it only
        # avoids building reverse-mode graphs for this audit.
        u, du_full, tangents = full_bundle_jvp(model, zt, r, t, v)
        du_diag = diagonal_patch_jvp(model, zt, r, t, v)

        fd_errors = {}
        for eps in cfg.fd_eps_list:
            du_fd = central_finite_difference(model, (zt, r, t), tangents, eps)
            fd_errors[str(eps)] = {
                "rel_err_fd_vs_jvp": rel_err(du_fd, du_full),
                "du_fd_norm": norm(du_fd),
            }

        delta = t - r
        u_tgt = v - du_full
        u_tgt_diag = v - du_diag
        cross_ratio = norm(du_full - du_diag) / (norm(du_full) + 1e-12)
        target_cross_ratio = norm(u_tgt - u_tgt_diag) / (norm(u_tgt) + 1e-12)

        # r=t degeneracy: tangent should be zero, so du_dc=0 and target=v.
        t_equal = cfg.t_low + (cfg.t_high - cfg.t_low) * torch.rand_like(t)
        r_equal = t_equal.clone()
        zt_equal = (1.0 - t_equal[..., None]) * problem["z0"] + t_equal[..., None] * problem["eps"]
        u_eq, du_eq, _ = full_bundle_jvp(model, zt_equal, r_equal, t_equal, v)
        u_tgt_eq = v - du_eq

        result = {
            "n_tokens": n_tokens,
            "batch_size": cfg.batch_size,
            "channels": cfg.channels,
            "width": cfg.width,
            "heads": cfg.heads,
            "depth": cfg.depth,
            "device": str(device),
            "dtype": str(dtype),
            "fd_errors": fd_errors,
            "full_jvp_stats": tensor_stats(du_full),
            "diag_jvp_stats": tensor_stats(du_diag),
            "v_stats": tensor_stats(v),
            "u_pred_stats": tensor_stats(u),
            "u_target_stats": tensor_stats(u_tgt),
            "delta_stats": tensor_stats(delta),
            "cross_ratio_full_vs_diag_jvp": cross_ratio,
            "target_cross_ratio_full_vs_diag": target_cross_ratio,
            "degeneracy_r_eq_t": {
                "du_eq_norm": norm(du_eq),
                "du_eq_max_abs": float(du_eq.detach().abs().max().cpu()),
                "target_minus_v_norm": norm(u_tgt_eq - v),
                "target_minus_v_max_abs": float((u_tgt_eq - v).detach().abs().max().cpu()),
                "u_eq_norm": norm(u_eq),
            },
            "has_nan_or_inf": bool(
                torch.isnan(du_full).any()
                or torch.isinf(du_full).any()
                or torch.isnan(du_diag).any()
                or torch.isinf(du_diag).any()
                or torch.isnan(u_tgt).any()
                or torch.isinf(u_tgt).any()
            ),
        }
    return result


def write_summary(out_dir: Path, cfg: Config, results: List[Dict]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: List[str] = []
    lines.append("# JVP Toy Harness 实验结果\n")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- device：{results[0]['device'] if results else 'NA'}")
    lines.append(f"- dtype：{results[0]['dtype'] if results else 'NA'}")
    lines.append(f"- seed：{cfg.seed}")
    lines.append("")
    lines.append("## 结论摘要\n")

    max_best_fd_err = 0.0
    max_degen_abs = 0.0
    has_nan = False
    for r in results:
        best_fd = min(v["rel_err_fd_vs_jvp"] for v in r["fd_errors"].values())
        max_best_fd_err = max(max_best_fd_err, best_fd)
        max_degen_abs = max(max_degen_abs, r["degeneracy_r_eq_t"]["target_minus_v_max_abs"])
        has_nan = has_nan or r["has_nan_or_inf"]

    pass_fd = max_best_fd_err < 0.05
    pass_degen = max_degen_abs < 1e-6 if "float32" in str(results[0]["dtype"]) else max_degen_abs < 1e-10
    pass_nan = not has_nan

    lines.append(f"- finite-difference vs full JVP 最大 best relative error：`{max_best_fd_err:.6g}` → {'PASS' if pass_fd else 'CHECK'}")
    lines.append(f"- r=t degeneracy 最大 target-v 误差：`{max_degen_abs:.6g}` → {'PASS' if pass_degen else 'CHECK'}")
    lines.append(f"- NaN/Inf：`{has_nan}` → {'PASS' if pass_nan else 'FAIL'}")
    lines.append("")
    lines.append("## 汇总表\n")
    lines.append("| N tokens | best FD rel err | FD err by eps | cross_ratio full-vs-diag JVP | target cross_ratio | r=t max abs | NaN/Inf |")
    lines.append("|---:|---:|---|---:|---:|---:|---|")
    for r in results:
        fd_items = [(eps, v["rel_err_fd_vs_jvp"]) for eps, v in r["fd_errors"].items()]
        best_fd = min(x[1] for x in fd_items)
        fd_str = ", ".join([f"eps={eps}: {val:.3g}" for eps, val in fd_items])
        lines.append(
            f"| {r['n_tokens']} | {best_fd:.6g} | {fd_str} | "
            f"{r['cross_ratio_full_vs_diag_jvp']:.6g} | "
            f"{r['target_cross_ratio_full_vs_diag']:.6g} | "
            f"{r['degeneracy_r_eq_t']['target_minus_v_max_abs']:.6g} | "
            f"{r['has_nan_or_inf']} |"
        )
    lines.append("")
    lines.append("## 解释\n")
    lines.append("- `best FD rel err` 用 central finite difference 审计 full-bundle `torch.func.jvp`。小于 5% 说明 tangent 与公式实现基本正确。")
    lines.append("- `cross_ratio full-vs-diag JVP = ||J_full - J_diag|| / ||J_full||`。该值越大，说明 diagonal per-patch approximation 漏掉的 cross-patch coupling 越多。")
    lines.append("- `r=t max abs` 检查 MeanFlow target 是否退化为 Flow Matching target `v`。")
    lines.append("")
    lines.append("## 下一步建议\n")
    if pass_fd and pass_degen and pass_nan:
        lines.append("1. 保留当前 toy harness 作为回归测试。")
        lines.append("2. 增加 local/stop-context JVP mode，与 full/diagonal 做三方对比。")
        lines.append("3. 将同一 JVP audit 接入 DiT-Tiny/PAE latent proxy。")
    else:
        lines.append("1. 暂停进入 DiT-Tiny/PAE 集成。")
        lines.append("2. 优先检查 tangent 是否重复乘 delta、finite difference eps 是否合适、dtype 是否导致误差。")
    lines.append("")

    (out_dir / "jvp_toy_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    out_dir = Path(args.output_dir) if args.output_dir else root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        seed=args.seed,
        batch_size=args.batch_size,
        channels=args.channels,
        width=args.width,
        heads=args.heads,
        depth=args.depth,
        n_tokens_list=tuple(int(x.strip()) for x in args.n_tokens.split(",") if x.strip()),
        dtype=args.dtype,
        device=args.device,
    )

    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)

    results: List[Dict] = []
    for n in cfg.n_tokens_list:
        print(f"[run] N={n} device={device} dtype={dtype}", flush=True)
        result = run_one_config(cfg, n, device, dtype)
        results.append(result)
        best_fd = min(v["rel_err_fd_vs_jvp"] for v in result["fd_errors"].values())
        print(
            f"[done] N={n} best_fd_rel_err={best_fd:.6g} "
            f"cross_ratio={result['cross_ratio_full_vs_diag_jvp']:.6g} "
            f"r_eq_t_max_abs={result['degeneracy_r_eq_t']['target_minus_v_max_abs']:.6g} "
            f"nan={result['has_nan_or_inf']}",
            flush=True,
        )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "config": asdict(cfg),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "results": results,
    }
    json_path = out_dir / "jvp_toy_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(out_dir, cfg, results)
    print(f"[write] {json_path}")
    print(f"[write] {out_dir / 'jvp_toy_summary.md'}")


if __name__ == "__main__":
    main()
