#!/usr/bin/env python3
"""Dedicated MeanFlow JVP-vs-finite-difference audit for B3/PAE DiT checkpoints.

This script is intentionally read-only w.r.t. training state.  It loads a saved
checkpoint, constructs one small deterministic non-degenerate batch, and compares
JVP against central finite differences across several numerics modes and eps
values.  It is meant for post-checkpoint audits when the main trainer is paused
or enough GPU memory is available.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("DISABLE_XFORMERS", "1")

import numpy as np
import torch
import yaml

PAE_GEN_ROOT = Path(os.environ.get("PAE_GEN_ROOT", "/workspace/PDM/external/PAE/pae_with_generator")).resolve()
if str(PAE_GEN_ROOT) not in sys.path:
    sys.path.insert(0, str(PAE_GEN_ROOT))

from train_meanflow_dit import (  # noqa: E402
    build_model,
    configure_sdpa,
    cuda_sync,
    load_config,
    make_model_kwargs,
    now_utc,
    peak_mb,
    reset_peak,
    resolve_device,
    set_seed,
)
from transport.meanflow_transport import MIXED_BF16_BACKBONE_FP32_JVP, create_meanflow_transport  # noqa: E402


def import_train_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("b3_realdata_trainer", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import trainer module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def tensor_norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.detach().float()).cpu())


def rel_err(reference: torch.Tensor, candidate: torch.Tensor, eps: float = 1e-12) -> float:
    return tensor_norm(reference - candidate) / (tensor_norm(reference) + eps)


def finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x.detach()).all().cpu())


def set_numerics(*, sdpa_kernel: str, allow_tf32: bool, device: torch.device) -> Dict[str, Any]:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    except Exception:
        pass
    info = configure_sdpa(sdpa_kernel, device)
    info["allow_tf32"] = bool(allow_tf32)
    try:
        info["float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:
        pass
    return info


def mode_specs(names: Sequence[str]) -> List[Dict[str, Any]]:
    all_modes = {
        "A": {
            "mode": "A",
            "description": "full/global math SDPA + TF32 off; JVP/FD math SDPA fp32",
            "sdpa_kernel": "math",
            "allow_tf32": False,
        },
        "B": {
            "mode": "B",
            "description": "default/global train SDPA + TF32 off; JVP/FD forced math SDPA fp32",
            "sdpa_kernel": "default",
            "allow_tf32": False,
        },
        "C": {
            "mode": "C",
            "description": "current fast numerics: default/global train SDPA + TF32 on; JVP/FD forced math SDPA fp32 with TF32 allowed",
            "sdpa_kernel": "default",
            "allow_tf32": True,
        },
    }
    out = []
    for name in names:
        key = name.strip().upper()
        if key not in all_modes:
            raise ValueError(f"unknown mode {name!r}; choose from A/B/C")
        out.append(all_modes[key])
    return out


def choose_indices(total: int, batch_size: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    # Spread over the full cache rather than taking the first N contiguous rows.
    # Replacement is disabled for normal ImageNet sizes.
    idx = rng.choice(total, size=int(batch_size), replace=False)
    return [int(i) for i in idx.tolist()]


def load_fixed_batch(dataset: Any, indices: Sequence[int], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    # Use canonical non-flipped latents if the experiment dataset exposes the helper.
    for idx in indices:
        if hasattr(dataset, "_get_item_no_norm"):
            x, y = dataset._get_item_no_norm(int(idx), tensor_key="latents")  # noqa: SLF001 - experiment audit helper
            if getattr(dataset, "latent_norm", False):
                x = (x.float().unsqueeze(0) - dataset._latent_mean) / dataset._latent_std  # noqa: SLF001
                x = x.squeeze(0)
            x = x * float(getattr(dataset, "latent_multiplier", 1.0))
        else:
            x, y = dataset[int(idx)]
        xs.append(x.float())
        ys.append(y.long().reshape(()))
    xb = torch.stack(xs, dim=0).to(device=device, dtype=torch.float32)
    yb = torch.stack(ys, dim=0).to(device=device, dtype=torch.long)
    return xb, yb


def sample_non_degenerate_path(transport: Any, x_data: torch.Tensor, *, seed: int) -> Dict[str, torch.Tensor]:
    # Make noise/r/t deterministic and independent of which audit mode runs first.
    torch.manual_seed(int(seed))
    if x_data.device.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
    old_equal_prob = float(transport.equal_prob)
    try:
        transport.equal_prob = 0.0
        path = transport.sample_path(x_data)
    finally:
        transport.equal_prob = old_equal_prob
    # Guard against floating-point coincidence and make non-degeneracy explicit.
    r = path["r"].clone()
    t = path["t"].clone()
    eq = torch.isclose(r, t)
    if bool(eq.any().detach().cpu()):
        r[eq] = 0.5 * t[eq]
    path["r"] = r
    path["t"] = t
    path["eq_mask"] = torch.zeros_like(path["eq_mask"], dtype=torch.bool)
    return path


def run_one_mode(
    *,
    spec: Dict[str, Any],
    model: torch.nn.Module,
    transport: Any,
    path: Dict[str, torch.Tensor],
    model_kwargs: Dict[str, Any],
    eps_values: Sequence[float],
    device: torch.device,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sdpa_info = set_numerics(sdpa_kernel=spec["sdpa_kernel"], allow_tf32=bool(spec["allow_tf32"]), device=device)
    model.eval()
    cuda_sync(device)
    z_t, r, t, v = path["z_t"], path["r"], path["t"], path["v"]
    batch = int(z_t.shape[0])

    # JVP once per mode; FD is swept over eps against the same tangent.
    reset_peak(device)
    cuda_sync(device)
    jvp_start = time.perf_counter()
    with torch.no_grad():
        u, du, tangents = transport.full_jvp(model, z_t, r, t, v, model_kwargs, transport.jvp_target_mode)
    cuda_sync(device)
    jvp_sec = time.perf_counter() - jvp_start
    jvp_peak = peak_mb(device)
    du_norm = tensor_norm(du)
    u_norm = tensor_norm(u)
    u_finite = finite(u)
    du_finite = finite(du)

    # Optional train-pred diagnostic: this is the piece affected by global SDPA and
    # bf16/TF32 fast numerics, unlike JVP/FD which intentionally force math SDPA.
    reset_peak(device)
    cuda_sync(device)
    backbone_start = time.perf_counter()
    with torch.no_grad():
        u_backbone = transport.call_model(model, z_t, r, t, model_kwargs, transport.backbone_forward_mode)
    cuda_sync(device)
    backbone_sec = time.perf_counter() - backbone_start
    backbone_peak = peak_mb(device)
    backbone_rel = rel_err(u, u_backbone)

    # Degenerate invariant check on exactly the same z/v/t and labels.
    with torch.no_grad():
        r_eq = t.clone()
        _u_eq, du_eq, _ = transport.full_jvp(model, z_t, r_eq, t, v, model_kwargs, transport.jvp_target_mode)
        target_eq = (v - du_eq).detach()
    degen_du_norm = tensor_norm(du_eq)
    degen_target_minus_v = float((target_eq - v).detach().float().abs().max().cpu())
    degen_finite = finite(du_eq) and finite(target_eq)

    for eps in eps_values:
        reset_peak(device)
        cuda_sync(device)
        fd_start = time.perf_counter()
        du_fd = transport.central_fd(model, (z_t, r, t), tangents, model_kwargs, float(eps), transport.fd_audit_mode)
        cuda_sync(device)
        fd_sec = time.perf_counter() - fd_start
        fd_peak = peak_mb(device)
        fd_norm = tensor_norm(du_fd)
        abs_err = tensor_norm(du_fd - du)
        fd_rel = rel_err(du_fd, du)
        row: Dict[str, Any] = {
            "type": "dedicated_fd_jvp_audit",
            "created_at_utc": now_utc(),
            "mode": spec["mode"],
            "mode_description": spec["description"],
            "global_sdpa_kernel": spec["sdpa_kernel"],
            "allow_tf32": bool(spec["allow_tf32"]),
            "sdpa_info": sdpa_info,
            "precision_recipe": transport.precision_recipe,
            "backbone_forward_mode": transport.backbone_forward_mode,
            "jvp_target_mode": transport.jvp_target_mode,
            "fd_audit_mode": transport.fd_audit_mode,
            "fd_eps": float(eps),
            "batch_size": batch,
            "force_non_degenerate": True,
            "r_eq_t_count": 0,
            "r_eq_t_total": batch,
            "realized_r_eq_t_fraction": 0.0,
            "target_jvp_effective_batch": batch,
            "target_jvp_skipped_equal_batch": 0,
            "actual_rt_samples_target_minus_v_max_abs": None,
            "u_finite": u_finite,
            "du_finite": du_finite,
            "fd_finite": finite(du_fd),
            "target_detached": True,
            "jvp_sec": jvp_sec,
            "jvp_peak_memory_mb": jvp_peak,
            "fd_sec": fd_sec,
            "fd_peak_memory_mb": fd_peak,
            "u_norm": u_norm,
            "du_norm": du_norm,
            "fd_norm": fd_norm,
            "fd_abs_err_norm": abs_err,
            "fd_rel_err_full_jvp": fd_rel,
            "backbone_forward_sec": backbone_sec,
            "backbone_forward_peak_memory_mb": backbone_peak,
            "backbone_forward_vs_fp32_jvp_u_rel_err": backbone_rel,
            "backbone_u_finite": finite(u_backbone),
            "all_r_eq_t_degenerate_du_norm": degen_du_norm,
            "all_r_eq_t_degenerate_target_minus_v_max_abs": degen_target_minus_v,
            "all_r_eq_t_degenerate_finite": degen_finite,
        }
        row["no_nan_or_inf"] = all(bool(row.get(k, False)) for k in ("u_finite", "du_finite", "fd_finite", "all_r_eq_t_degenerate_finite", "backbone_u_finite"))
        rows.append(row)
        del du_fd
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows


def write_outputs(out_dir: Path, payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "fd_jvp_modes_audit.json"
    jsonl_path = out_dir / "fd_jvp_modes_audit.jsonl"
    md_path = out_dir / "fd_jvp_modes_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in payload["rows"]:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    lines = []
    lines.append("# Dedicated FD/JVP Audit\n\n")
    meta = payload["meta"]
    lines.append(f"- created_at_utc: `{meta['created_at_utc']}`\n")
    lines.append(f"- checkpoint: `{meta['checkpoint']}`\n")
    lines.append(f"- checkpoint_step: `{meta.get('checkpoint_step')}`\n")
    lines.append(f"- batch_size: `{meta['batch_size']}`\n")
    lines.append(f"- indices: `{meta['indices']}`\n")
    lines.append(f"- eps_values: `{meta['eps_values']}`\n")
    lines.append("\n| mode | TF32 | global SDPA | eps | fd_rel | abs_err | du_norm | fd_norm | no_nan | backbone_rel | degen_target-v |\n")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|\n")
    for r in payload["rows"]:
        lines.append(
            f"| {r['mode']} | {r['allow_tf32']} | {r['global_sdpa_kernel']} | {r['fd_eps']:.0e} | "
            f"{r['fd_rel_err_full_jvp']:.6g} | {r['fd_abs_err_norm']:.6g} | {r['du_norm']:.6g} | {r['fd_norm']:.6g} | "
            f"{r['no_nan_or_inf']} | {r['backbone_forward_vs_fp32_jvp_u_rel_err']:.6g} | "
            f"{r['all_r_eq_t_degenerate_target_minus_v_max_abs']:.6g} |\n"
        )
    md_path.write_text("".join(lines), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "jsonl": str(jsonl_path), "md": str(md_path)}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trainer-script", default="/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/scripts/train_b3_meanflow_realdata.py")
    parser.add_argument("--output-dir", default="/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fd_jvp_dedicated_audit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026052801)
    parser.add_argument("--path-seed", type=int, default=2026052802)
    parser.add_argument("--eps", type=float, nargs="+", default=[1e-2, 3e-3, 1e-3])
    parser.add_argument("--modes", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-ema", action="store_true", help="Audit EMA weights instead of live model weights.")
    parser.add_argument("--indices", type=int, nargs="*", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_mod = import_train_module(Path(args.trainer_script))
    set_seed(int(args.seed))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed) % (2**32 - 1))
    device = resolve_device(args.device)

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    meanflow_cfg = cfg.get("meanflow", {})
    train_cfg = cfg.get("train", {})

    # Load a deterministic fixed batch from canonical (non-flipped) latents.
    dataset = train_mod.ShardIndexedLatentDataset(
        data_dir=data_cfg["data_path"],
        file_glob=data_cfg.get("file_glob", "*.safetensors"),
        flip_prob=0.0,
        latent_norm=bool(data_cfg.get("latent_norm", False)),
        latent_multiplier=float(data_cfg.get("latent_multiplier", 1.0)),
        latent_stats_path=data_cfg.get("latent_stats_path"),
        max_shards=data_cfg.get("max_shards"),
        skip_unreadable=bool(data_cfg.get("skip_unreadable", True)),
    )
    indices = list(args.indices) if args.indices else choose_indices(len(dataset), int(args.batch_size), int(args.seed))
    if len(indices) != int(args.batch_size):
        raise ValueError("number of --indices must equal --batch-size")
    x, y = load_fixed_batch(dataset, indices, device)

    downsample_ratio = int(cfg.get("vae", {}).get("downsample_ratio", 16))
    image_size = int(data_cfg.get("image_size", 256))
    latent_size = image_size // downsample_ratio

    # Build/load on CPU first to avoid GPU duplication of the checkpoint payload.
    model = build_model(cfg, latent_size)
    ckpt_path = Path(args.checkpoint).resolve()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    key = "ema" if args.use_ema else "model"
    if args.use_ema and ckpt.get("ema") is None:
        raise RuntimeError("--use-ema requested but checkpoint has no ema state")
    model.load_state_dict(ckpt[key])
    checkpoint_step = int(ckpt.get("step", -1))
    del ckpt
    gc.collect()
    model.to(device=device, dtype=torch.float32)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    transport = create_meanflow_transport(
        precision_recipe=meanflow_cfg.get("precision_recipe", MIXED_BF16_BACKBONE_FP32_JVP),
        equal_prob=0.0,
        time_sampler=meanflow_cfg.get("time_sampler", "ltg"),
        t_low=float(meanflow_cfg.get("t_low", 0.05)),
        t_high=float(meanflow_cfg.get("t_high", 0.95)),
        lognorm_mu=float(meanflow_cfg.get("lognorm_mu", -0.4)),
        lognorm_sigma=float(meanflow_cfg.get("lognorm_sigma", 1.0)),
        fd_eps=float(meanflow_cfg.get("fd_eps", 1e-2)),
        device_type=device.type,
    )
    model_kwargs = make_model_kwargs(y, 0.0, int(data_cfg.get("num_classes", 1000)))
    # Explicitly pin dropout mask to all-zero; this is already true with prob=0.
    if "force_drop_ids" in model_kwargs:
        model_kwargs["force_drop_ids"].zero_()

    path = sample_non_degenerate_path(transport, x, seed=int(args.path_seed))
    modes = mode_specs(args.modes)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    for spec in modes:
        rows.extend(run_one_mode(spec=spec, model=model, transport=transport, path=path, model_kwargs=model_kwargs, eps_values=args.eps, device=device))
    cuda_sync(device)
    elapsed = time.perf_counter() - started

    payload: Dict[str, Any] = {
        "meta": {
            "type": "dedicated_fd_jvp_audit_summary",
            "created_at_utc": now_utc(),
            "config": str(Path(args.config).resolve()),
            "checkpoint": str(ckpt_path),
            "checkpoint_step": checkpoint_step,
            "use_ema": bool(args.use_ema),
            "device": str(device),
            "torch_version": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "batch_size": int(x.shape[0]),
            "indices": indices,
            "labels": [int(v) for v in y.detach().cpu().tolist()],
            "latent_shape": list(x.shape[1:]),
            "seed": int(args.seed),
            "path_seed": int(args.path_seed),
            "eps_values": [float(v) for v in args.eps],
            "modes": [m["mode"] for m in modes],
            "elapsed_sec": elapsed,
            "model_type": model_cfg.get("model_type"),
            "train_allow_tf32_config": bool(train_cfg.get("allow_tf32", False)),
            "train_sdpa_kernel_config": train_cfg.get("sdpa_kernel"),
        },
        "rows": rows,
    }
    write_outputs(Path(args.output_dir), payload)


if __name__ == "__main__":
    main()
