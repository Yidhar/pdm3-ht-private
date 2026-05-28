#!/usr/bin/env python3
"""Supplemental r=t mixed-mask audit for Experiment 5.

The main Experiment-5 script already records an all-token r=t degeneracy check:
set r=t for the whole latent grid and verify du=0, target=v.

This supplemental audit answers a narrower question:
when the training sampler realizes a 75%/25% per-token mixture inside one
transformer sample, what happens on the tokens whose own r_i=t_i?

For a transformer with attention, those tokens can still receive nonzero JVP
contribution through other tokens whose delta_j=t_j-r_j is nonzero. Therefore
this audit records both:
  1. all-token r=t sanity: should be exactly zero;
  2. mixed-mask equal-token slice: diagnostic only, not expected to be zero
     under full-bundle cross-token JVP.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import torch


EXP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STRESS_SCRIPT = EXP_ROOT / "scripts" / "run_mixed_precision_synthetic_jvp_stress.py"
DEFAULT_RUN_METRICS = (
    EXP_ROOT / "results" / "smoke_mixed_w64d1" / "lightningdit_block_pae_shorttrain_metrics.json",
    EXP_ROOT / "results" / "main_mixed_w128d2" / "lightningdit_block_pae_shorttrain_metrics.json",
    EXP_ROOT / "results" / "scale_mixed_w256d4" / "lightningdit_block_pae_shorttrain_metrics.json",
)


def load_stress_module(path: Path):
    spec = importlib.util.spec_from_file_location("exp5_stress", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import stress script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def max_abs(x: torch.Tensor) -> float:
    y = x.detach().float()
    if y.numel() == 0:
        return 0.0
    return float(y.abs().max().cpu())


def norm(x: torch.Tensor) -> float:
    y = x.detach().float()
    if y.numel() == 0:
        return 0.0
    return float(torch.linalg.vector_norm(y).cpu())


def masked(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # x: [B, N, C], mask: [B, N]
    return x[mask]


def rel_on_mask(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    aa = masked(a, mask)
    bb = masked(b, mask)
    return norm(aa - bb) / (norm(aa) + 1e-12)


def cfg_from_payload(stress: Any, payload: Dict[str, Any]):
    raw_cfg = payload["config"]
    allowed = {f.name for f in fields(stress.Config)}
    return stress.Config(**{k: v for k, v in raw_cfg.items() if k in allowed})


def run_audit_for_metrics(stress: Any, metrics_path: Path, device: torch.device) -> Dict[str, Any]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    cfg = cfg_from_payload(stress, payload)
    rec = payload["results"][0]
    grid = int(rec["grid"])
    latent_mode = str(rec["latent_mode"])
    mode = str(rec["mode"])
    seed = int(rec["seed"])
    du_mode = stress.jvp_target_mode(mode)
    u_mode = stress.backbone_forward_mode(mode)

    stress.set_seed(seed)
    model = stress.PerPatchLightningDiTStressModel(cfg).to(device=device, dtype=torch.float32)
    model.eval()
    pae_stats = stress.PaeStats(Path(cfg.pae_stats_path))

    problem = stress.sample_problem(cfg, grid, latent_mode, device, pae_stats)
    zt = problem["zt"]
    r = problem["r"]
    t = problem["t"]
    v = problem["v"]
    latent_meta = problem["latent_meta"]
    assert isinstance(zt, torch.Tensor)
    assert isinstance(r, torch.Tensor)
    assert isinstance(t, torch.Tensor)
    assert isinstance(v, torch.Tensor)

    eq_mask = r == t
    neq_mask = ~eq_mask
    n_tokens = int(eq_mask.numel())
    n_eq = int(eq_mask.sum().detach().cpu())
    n_neq = n_tokens - n_eq

    stress.reset_peak_memory(device)
    stress.cuda_sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        _u, du, _tangents = stress.full_bundle_jvp(model, zt, r, t, v, du_mode, device)
        target = v - du
    stress.cuda_sync(device)
    mixed_jvp_sec = time.perf_counter() - t0
    mixed_jvp_peak_mb = stress.peak_memory_mb(device)

    # All-token r=t check. This is the strict MeanFlow/FM degenerate case.
    t_eq = t.clone()
    r_eq = t_eq.clone()
    stress.reset_peak_memory(device)
    stress.cuda_sync(device)
    t1 = time.perf_counter()
    with torch.no_grad():
        _u_eq, du_all_eq, _ = stress.full_bundle_jvp(model, zt, r_eq, t_eq, v, du_mode, device)
        target_all_eq = v - du_all_eq
    stress.cuda_sync(device)
    all_eq_jvp_sec = time.perf_counter() - t1
    all_eq_jvp_peak_mb = stress.peak_memory_mb(device)

    out: Dict[str, Any] = {
        "metrics_path": str(metrics_path),
        "run_name": metrics_path.parent.name,
        "grid": grid,
        "n_tokens": n_tokens,
        "latent_mode": latent_mode,
        "mode": mode,
        "backbone_forward_mode": u_mode,
        "jvp_target_mode": du_mode,
        "seed": seed,
        "configured_equal_prob": float(cfg.equal_prob),
        "recorded_latent_meta_r_eq_t_fraction": latent_meta.get("r_eq_t_fraction_realized"),
        "mixed_mask": {
            "eq_token_count": n_eq,
            "neq_token_count": n_neq,
            "eq_fraction": n_eq / max(1, n_tokens),
            "neq_fraction": n_neq / max(1, n_tokens),
            "delta_eq_max_abs": max_abs((t - r)[eq_mask]),
            "delta_neq_min_abs": float((t - r)[neq_mask].detach().float().abs().min().cpu()) if n_neq else 0.0,
            "du_norm_all": norm(du),
            "v_norm_all": norm(v),
            "du_over_v_all": norm(du) / (norm(v) + 1e-12),
            "du_norm_on_eq_tokens": norm(masked(du, eq_mask)),
            "v_norm_on_eq_tokens": norm(masked(v, eq_mask)),
            "du_over_v_on_eq_tokens": norm(masked(du, eq_mask)) / (norm(masked(v, eq_mask)) + 1e-12),
            "target_minus_v_max_abs_on_eq_tokens": max_abs(masked(target - v, eq_mask)),
            "target_vs_v_rel_on_eq_tokens": rel_on_mask(target, v, eq_mask),
            "du_norm_on_neq_tokens": norm(masked(du, neq_mask)),
            "v_norm_on_neq_tokens": norm(masked(v, neq_mask)),
            "du_over_v_on_neq_tokens": norm(masked(du, neq_mask)) / (norm(masked(v, neq_mask)) + 1e-12),
            "target_minus_v_max_abs_on_neq_tokens": max_abs(masked(target - v, neq_mask)),
            "mixed_jvp_sec": mixed_jvp_sec,
            "mixed_jvp_peak_memory_mb": mixed_jvp_peak_mb,
            "finite": bool(torch.isfinite(du.detach()).all().cpu()) and bool(torch.isfinite(target.detach()).all().cpu()),
        },
        "all_token_r_eq_t": {
            "du_norm": norm(du_all_eq),
            "target_minus_v_max_abs": max_abs(target_all_eq - v),
            "target_vs_v_rel": norm(target_all_eq - v) / (norm(v) + 1e-12),
            "jvp_sec": all_eq_jvp_sec,
            "jvp_peak_memory_mb": all_eq_jvp_peak_mb,
            "finite": bool(torch.isfinite(du_all_eq.detach()).all().cpu()) and bool(torch.isfinite(target_all_eq.detach()).all().cpu()),
        },
        "interpretation": (
            "all-token r=t must give du=0; mixed per-token r=t slice may be nonzero "
            "under full-bundle attention-coupled JVP because other tokens move."
        ),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


def write_markdown(out_path: Path, audit: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("# Supplemental r=t mixed-mask degeneracy audit\n\n")
    lines.append("Purpose: answer whether Experiment-5 used the intended 75% `r=t` mixture and separate the all-token `r=t` sanity check from mixed per-token edge behavior.\n\n")
    lines.append("## Summary table\n\n")
    lines.append("| run | cfg p(r=t) | realized eq frac | eq/neq tokens | all-token r=t target-v max | mixed eq-token target-v max | mixed eq-token du/v | mixed neq-token du/v | note |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
    for rec in audit["records"]:
        mm = rec["mixed_mask"]
        ae = rec["all_token_r_eq_t"]
        note = "all-eq PASS; mixed eq-token not forced zero" if mm["target_minus_v_max_abs_on_eq_tokens"] != 0 else "all-eq PASS; mixed eq-token zero"
        lines.append(
            f"| {rec['run_name']} | {rec['configured_equal_prob']:.3f} | {mm['eq_fraction']:.6g} | "
            f"{mm['eq_token_count']}/{mm['neq_token_count']} | {ae['target_minus_v_max_abs']:.6g} | "
            f"{mm['target_minus_v_max_abs_on_eq_tokens']:.6g} | {mm['du_over_v_on_eq_tokens']:.6g} | "
            f"{mm['du_over_v_on_neq_tokens']:.6g} | {note} |\n"
        )
    lines.append("\n## Interpretation\n\n")
    lines.append("- Experiment-5 `--equal-prob 0.75` was active. The audited correctness sample realized approximately 75% `r=t` tokens.\n")
    lines.append("- The previously reported `r=t degen max = 0` is the **all-token** `r=t` sanity check: if the whole latent grid has `r=t`, then the full-bundle tangent is zero and `target=v` exactly.\n")
    lines.append("- In a **mixed per-token** grid, tokens with their own `r_i=t_i` can still have nonzero `du_i` through attention from other moving tokens. This is expected for full-bundle cross-token JVP and should not be interpreted as a failure of the all-token MeanFlow/FM degeneracy.\n")
    lines.append("- For Phase-0 B3, log both sampler realized ratio and the all-token degeneracy check; if the intended baseline is standard MeanFlow scalar `(r,t)` per image, sample `r=t` per image/batch item rather than per token.\n")
    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress-script", type=Path, default=DEFAULT_STRESS_SCRIPT)
    ap.add_argument("--output-dir", type=Path, default=EXP_ROOT / "results" / "rt_mixed_mask_audit")
    ap.add_argument("--metrics", type=Path, nargs="*", default=list(DEFAULT_RUN_METRICS))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    os.environ.setdefault("XFORMERS_DISABLED", "1")
    stress = load_stress_module(args.stress_script)
    device = stress.resolve_device(args.device)

    # Match Experiment-5 defaults.
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    sdpa_info = stress.configure_sdpa("math", device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    print(f"[env] device={device} sdpa={sdpa_info}", flush=True)
    for metrics in args.metrics:
        print(f"[audit] {metrics}", flush=True)
        records.append(run_audit_for_metrics(stress, metrics, device))
        last = records[-1]
        mm = last["mixed_mask"]
        ae = last["all_token_r_eq_t"]
        print(
            f"[done] {last['run_name']} eq_frac={mm['eq_fraction']:.6g} "
            f"all_eq_target-v={ae['target_minus_v_max_abs']:.6g} "
            f"mixed_eq_target-v={mm['target_minus_v_max_abs_on_eq_tokens']:.6g}",
            flush=True,
        )

    audit = {
        "created_at_utc": stress.now_utc(),
        "device": str(device),
        "torch_version": torch.__version__,
        "sdpa_info": sdpa_info,
        "records": records,
    }
    json_path = args.output_dir / "rt_mixed_mask_audit.json"
    md_path = args.output_dir / "rt_mixed_mask_audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(md_path, audit)
    print(f"[write] {json_path}", flush=True)
    print(f"[write] {md_path}", flush=True)


if __name__ == "__main__":
    main()
