#!/usr/bin/env python3
"""
PDM-3-HT Local / Stop-Context JVP Toy Experiment

This script extends the first full-bundle JVP toy harness by comparing:
  - full-bundle JVP: all token tangents are active;
  - diagonal JVP: only the same token tangent is active for each output token;
  - local-window JVP: only spatial neighbors within Chebyshev radius R are active
    for each output token.

The local-window mode approximates a stop-context / local-JVP fallback: forward
attention remains globally coupled, but the JVP tangent outside the local window
is stopped when collecting each output token's derivative.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

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
    radii: Tuple[int, ...] = (0, 1, 2, 3)
    dtype: str = "float32"
    device: str = "auto"
    fd_eps: float = 1e-2
    t_low: float = 0.05
    t_high: float = 0.95


class TinyCoupledModel(nn.Module):
    def __init__(self, channels: int, width: int, heads: int, depth: int):
        super().__init__()
        assert width % heads == 0
        self.in_proj = nn.Linear(channels + 6, width)
        self.blocks = nn.ModuleList([TinyAttentionBlock(width, heads) for _ in range(depth)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, channels)

    @staticmethod
    def time_features(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        dt = t - r
        return torch.stack([
            r,
            t,
            dt,
            torch.sin(math.pi * t),
            torch.cos(math.pi * t),
            torch.sin(math.pi * dt),
        ], dim=-1)

    def forward(self, z: torch.Tensor, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = torch.cat([z, self.time_features(r, t)], dim=-1)
        h = self.in_proj(h)
        for block in self.blocks:
            h = block(h)
        return self.out(self.out_norm(h))


class TinyAttentionBlock(nn.Module):
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
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(logits, dim=-1)
        y = torch.matmul(attn, v).transpose(1, 2).reshape(b, n, d)
        h = h + self.proj(y)
        h = h + self.mlp(self.norm2(h))
        return h


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--seed", type=int, default=20260527)
    p.add_argument("--n-tokens", type=str, default="4,16,64")
    p.add_argument("--radii", type=str, default="0,1,2,3")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--channels", type=int, default=8)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--fd-eps", type=float, default=1e-2)
    return p.parse_args()


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


def resolve_dtype(s: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[s]


def sample_problem(cfg: Config, n: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    b, c = cfg.batch_size, cfg.channels
    z0 = torch.randn(b, n, c, device=device, dtype=dtype)
    eps = torch.randn(b, n, c, device=device, dtype=dtype)
    v = eps - z0
    t = cfg.t_low + (cfg.t_high - cfg.t_low) * torch.rand(b, n, device=device, dtype=dtype)
    r = torch.rand(b, n, device=device, dtype=dtype) * t
    zt = (1.0 - t[..., None]) * z0 + t[..., None] * eps
    return {"z0": z0, "eps": eps, "v": v, "r": r, "t": t, "zt": zt}


def full_bundle_jvp(model: nn.Module, zt: torch.Tensor, r: torch.Tensor, t: torch.Tensor, v: torch.Tensor):
    delta = t - r
    z_dot = delta[..., None] * v
    r_dot = torch.zeros_like(r)
    t_dot = delta

    def fn(z, rr, tt):
        return model(z, rr, tt)

    u, du = jvp(fn, (zt, r, t), (z_dot, r_dot, t_dot))
    return u, du, (z_dot, r_dot, t_dot)


def central_fd(model: nn.Module, primals, tangents, eps: float) -> torch.Tensor:
    z, r, t = primals
    z_dot, r_dot, t_dot = tangents
    u_plus = model(z + eps * z_dot, r + eps * r_dot, t + eps * t_dot)
    u_minus = model(z - eps * z_dot, r - eps * r_dot, t - eps * t_dot)
    return (u_plus - u_minus) / (2.0 * eps)


def grid_side(n: int) -> int:
    s = int(round(math.sqrt(n)))
    if s * s != n:
        raise ValueError(f"n_tokens must be square for local grid, got {n}")
    return s


def local_mask_indices(n: int, idx: int, radius: int) -> List[int]:
    s = grid_side(n)
    y, x = divmod(idx, s)
    ids = []
    for yy in range(max(0, y - radius), min(s, y + radius + 1)):
        for xx in range(max(0, x - radius), min(s, x + radius + 1)):
            ids.append(yy * s + xx)
    return ids


def average_coverage(n: int, radius: int) -> float:
    return sum(len(local_mask_indices(n, i, radius)) for i in range(n)) / float(n * n)


def local_window_jvp(model: nn.Module, zt: torch.Tensor, r: torch.Tensor, t: torch.Tensor, v: torch.Tensor, radius: int):
    """Collect per-output derivatives with tangent restricted to local window."""
    b, n, c = zt.shape
    delta = t - r
    z_dot_full = delta[..., None] * v
    t_dot_full = delta
    r_dot_zero = torch.zeros_like(r)
    out = torch.zeros_like(zt)

    def fn(z, rr, tt):
        return model(z, rr, tt)

    for i in range(n):
        ids = local_mask_indices(n, i, radius)
        z_dot = torch.zeros_like(zt)
        t_dot = torch.zeros_like(t)
        z_dot[:, ids, :] = z_dot_full[:, ids, :]
        t_dot[:, ids] = t_dot_full[:, ids]
        _, du_all = jvp(fn, (zt, r, t), (z_dot, r_dot_zero, t_dot))
        out[:, i, :] = du_all[:, i, :]
    return out


def norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach()).cpu())


def rel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    return norm(a - b) / (norm(a) + eps)


def run_one(cfg: Config, n: int, device: torch.device, dtype: torch.dtype) -> Dict:
    set_seed(cfg.seed + n)
    model = TinyCoupledModel(cfg.channels, cfg.width, cfg.heads, cfg.depth).to(device=device, dtype=dtype)
    model.eval()
    problem = sample_problem(cfg, n, device, dtype)
    zt, r, t, v = problem["zt"], problem["r"], problem["t"], problem["v"]

    with torch.no_grad():
        u, du_full, tangents = full_bundle_jvp(model, zt, r, t, v)
        du_fd = central_fd(model, (zt, r, t), tangents, cfg.fd_eps)
        fd_rel_err = rel(du_fd, du_full)

        radius_results = []
        diag_du = None
        for radius in cfg.radii:
            # Skip radii larger than grid side; still valid but redundant if radius >= side-1.
            du_local = local_window_jvp(model, zt, r, t, v, radius)
            if radius == 0:
                diag_du = du_local
            err_vs_full = rel(du_full, du_local)
            target_full = v - du_full
            target_local = v - du_local
            target_err_vs_full = rel(target_full, target_local)
            radius_results.append({
                "radius": radius,
                "coverage_fraction_avg": average_coverage(n, radius),
                "err_vs_full_jvp": err_vs_full,
                "target_err_vs_full": target_err_vs_full,
                "du_local_norm": norm(du_local),
            })

        if diag_du is None:
            diag_du = local_window_jvp(model, zt, r, t, v, 0)
        diag_err = rel(du_full, diag_du)
        for rr in radius_results:
            rr["improvement_vs_diag_error_ratio"] = rr["err_vs_full_jvp"] / (diag_err + 1e-12)
            rr["error_reduction_vs_diag"] = 1.0 - rr["improvement_vs_diag_error_ratio"]

        # degeneracy check still included.
        t_equal = cfg.t_low + (cfg.t_high - cfg.t_low) * torch.rand_like(t)
        r_equal = t_equal.clone()
        zt_equal = (1.0 - t_equal[..., None]) * problem["z0"] + t_equal[..., None] * problem["eps"]
        _, du_eq, _ = full_bundle_jvp(model, zt_equal, r_equal, t_equal, v)
        target_eq = v - du_eq

        return {
            "n_tokens": n,
            "grid_side": grid_side(n),
            "device": str(device),
            "dtype": str(dtype),
            "fd_eps": cfg.fd_eps,
            "fd_rel_err_full_jvp": fd_rel_err,
            "full_jvp_norm": norm(du_full),
            "diag_err_vs_full_jvp": diag_err,
            "radius_results": radius_results,
            "degeneracy_r_eq_t": {
                "du_eq_norm": norm(du_eq),
                "target_minus_v_max_abs": float((target_eq - v).detach().abs().max().cpu()),
            },
            "has_nan_or_inf": bool(
                torch.isnan(du_full).any()
                or torch.isinf(du_full).any()
                or any(torch.isnan(torch.tensor(rr["err_vs_full_jvp"])) for rr in radius_results)
            ),
        }


def write_summary(out_dir: Path, cfg: Config, results: List[Dict]) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = []
    lines.append("# Local / Stop-Context JVP Toy 实验结果\n")
    lines.append(f"- 生成时间：{now}")
    lines.append(f"- seed：{cfg.seed}")
    lines.append(f"- radii：{list(cfg.radii)}")
    if results:
        lines.append(f"- device：{results[0]['device']}")
        lines.append(f"- dtype：{results[0]['dtype']}")
    lines.append("")

    max_fd = max(r["fd_rel_err_full_jvp"] for r in results)
    max_degen = max(r["degeneracy_r_eq_t"]["target_minus_v_max_abs"] for r in results)
    has_nan = any(r["has_nan_or_inf"] for r in results)
    lines.append("## Gate 结论\n")
    lines.append(f"- full-bundle JVP vs finite difference 最大相对误差：`{max_fd:.6g}` → {'PASS' if max_fd < 0.05 else 'CHECK'}")
    lines.append(f"- r=t degeneracy 最大 target-v 误差：`{max_degen:.6g}` → {'PASS' if max_degen < 1e-6 else 'CHECK'}")
    lines.append(f"- NaN/Inf：`{has_nan}` → {'PASS' if not has_nan else 'FAIL'}")
    lines.append("")

    lines.append("## Radius 对比表\n")
    lines.append("| N | grid | radius | avg coverage | err vs full JVP | target err vs full | error ratio vs diag | error reduction vs diag |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        for rr in r["radius_results"]:
            lines.append(
                f"| {r['n_tokens']} | {r['grid_side']}x{r['grid_side']} | {rr['radius']} | "
                f"{rr['coverage_fraction_avg']:.6g} | "
                f"{rr['err_vs_full_jvp']:.6g} | "
                f"{rr['target_err_vs_full']:.6g} | "
                f"{rr['improvement_vs_diag_error_ratio']:.6g} | "
                f"{rr['error_reduction_vs_diag']:.6g} |"
            )
    lines.append("")

    lines.append("## 关键观察\n")
    lines.append("- radius=0 是 diagonal / self-token-only JVP；其 error ratio vs diag 应为 1。")
    lines.append("- radius 越大，local-window tangent 覆盖越多 token，理论上应更接近 full-bundle JVP。")
    lines.append("- 如果 radius=1 就能显著降低 error，说明 stop-context/local JVP 有潜力作为 full JVP 不稳定时的 fallback。")
    lines.append("- 如果 radius=1/2 仍接近 diagonal，则说明 cross-patch coupling 主要来自长程 attention，local fallback 可能不足。")
    lines.append("")

    # Compact recommendation based on N=64, if present.
    n64 = next((x for x in results if x["n_tokens"] == 64), None)
    lines.append("## 初步建议\n")
    if n64:
        diag = next(x for x in n64["radius_results"] if x["radius"] == 0)
        r1 = next((x for x in n64["radius_results"] if x["radius"] == 1), None)
        r2 = next((x for x in n64["radius_results"] if x["radius"] == 2), None)
        if r1 and r1["error_reduction_vs_diag"] > 0.25:
            lines.append("- 在 N=64 toy setting 中，radius=1 已明显优于 diagonal；建议下一步在 DiT-Tiny proxy 中加入 local-JVP ablation。")
        elif r2 and r2["error_reduction_vs_diag"] > 0.25:
            lines.append("- radius=1 收益有限，但 radius=2 明显优于 diagonal；真实模型中可测试 5x5 local tangent window。")
        else:
            lines.append("- local window 对 full JVP 的近似收益有限；若真实 DiT 也如此，fallback 应优先考虑 JVP-free consistency / SplitMeanFlow，而不是 local JVP。")
    else:
        lines.append("- 未包含 N=64，建议补跑更接近 2D grid 的 token scale。")
    lines.append("")

    (out_dir / "jvp_local_summary.md").write_text("\n".join(lines), encoding="utf-8")


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
        radii=tuple(int(x.strip()) for x in args.radii.split(",") if x.strip()),
        dtype=args.dtype,
        device=args.device,
        fd_eps=args.fd_eps,
    )
    device = resolve_device(cfg.device)
    dtype = resolve_dtype(cfg.dtype)
    results = []
    for n in cfg.n_tokens_list:
        print(f"[run] N={n} radii={cfg.radii} device={device} dtype={dtype}", flush=True)
        res = run_one(cfg, n, device, dtype)
        results.append(res)
        r_summ = ", ".join([
            f"R{rr['radius']}:err={rr['err_vs_full_jvp']:.4g},red={rr['error_reduction_vs_diag']:.3g}"
            for rr in res["radius_results"]
        ])
        print(
            f"[done] N={n} fd_rel={res['fd_rel_err_full_jvp']:.4g} "
            f"degen={res['degeneracy_r_eq_t']['target_minus_v_max_abs']:.3g} {r_summ}",
            flush=True,
        )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "config": asdict(cfg),
        "results": results,
    }
    json_path = out_dir / "jvp_local_metrics.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(out_dir, cfg, results)
    print(f"[write] {json_path}", flush=True)
    print(f"[write] {out_dir / 'jvp_local_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
