#!/usr/bin/env python3
"""Single-process Phase-0/B3 MeanFlow trainer for PAE latent DiT.

This trainer is deliberately separate from upstream ``train_dit.py`` so the
standard PAE Flow-Matching baseline remains untouched.  It is suitable for
Phase-0 engineering smoke and single-GPU short runs; distributed long-training
can be added after this path is validated.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("DISABLE_XFORMERS", "1")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from dataset.img_latent_dataset import ImgLatentDataset
from models.meanflow_lightningdit import B3MeanFlowLightningDiT, B3MeanFlowLightningDiT_models, COMPILE_UNWRAPPED
from transport.meanflow_transport import MIXED_BF16_BACKBONE_FP32_JVP, create_meanflow_transport


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    elif kernel == "default":
        if hasattr(cuda_backend, "enable_flash_sdp"):
            cuda_backend.enable_flash_sdp(True)
        if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
            cuda_backend.enable_mem_efficient_sdp(True)
        if hasattr(cuda_backend, "enable_math_sdp"):
            cuda_backend.enable_math_sdp(True)
        if hasattr(cuda_backend, "enable_cudnn_sdp"):
            cuda_backend.enable_cudnn_sdp(True)
    else:
        raise ValueError("sdpa kernel must be math or default")
    for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
        fn = getattr(cuda_backend, name, None)
        if callable(fn):
            try:
                info[name] = bool(fn())
            except Exception:
                pass
    return info


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_mb(device: torch.device) -> Optional[float]:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / (1024**2))


def git_commit(path: Path) -> Optional[str]:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_meanflow_dit")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(out_dir / "log.txt")
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def separate_weight_decay(module: nn.Module, default_decay: float) -> List[Dict[str, Any]]:
    decay: List[nn.Parameter] = []
    no_decay: List[nn.Parameter] = []
    for name, param in module.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or ".norm" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": default_decay}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        if not param.requires_grad:
            continue
        ema_params[name].mul_(decay).add_(param.detach(), alpha=1.0 - decay)
    ema_buffers = OrderedDict(ema_model.named_buffers())
    model_buffers = OrderedDict(model.named_buffers())
    for name, buf in model_buffers.items():
        if name in ema_buffers:
            ema_buffers[name].copy_(buf.detach())


def requires_grad(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(flag)


def cycle_loader(loader: DataLoader) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def build_model(cfg: Dict[str, Any], latent_size: int) -> nn.Module:
    model_cfg = cfg["model"]
    vae_cfg = cfg.get("vae", {})
    data_cfg = cfg.get("data", {})
    model_type = model_cfg.get("model_type", "B3MeanFlowLightningDiT-XL/1")
    common = dict(
        input_size=latent_size,
        in_channels=model_cfg.get("in_chans", vae_cfg.get("latent_dim", 32)),
        num_classes=data_cfg.get("num_classes", 1000),
        class_dropout_prob=model_cfg.get("class_dropout_prob", 0.0),
        learn_sigma=model_cfg.get("learn_sigma", False),
        use_qknorm=model_cfg.get("use_qknorm", False),
        use_swiglu=model_cfg.get("use_swiglu", True),
        use_rope=model_cfg.get("use_rope", True),
        use_rmsnorm=model_cfg.get("use_rmsnorm", True),
        wo_shift=model_cfg.get("wo_shift", False),
        use_checkpoint=model_cfg.get("use_checkpoint", False),
        use_abs_pos=model_cfg.get("use_abs_pos", True),
        checkpoint_reentrant=model_cfg.get("checkpoint_reentrant", False),
        meanflow_init_scheme=model_cfg.get("meanflow_init_scheme", "official_zero"),
        meanflow_adaln_init_std=model_cfg.get("meanflow_adaln_init_std", 0.02),
        meanflow_final_linear_init=model_cfg.get("meanflow_final_linear_init", "official_zero"),
    )
    if common["use_checkpoint"] and cfg.get("meanflow", {}).get("precision_recipe", MIXED_BF16_BACKBONE_FP32_JVP) == MIXED_BF16_BACKBONE_FP32_JVP:
        raise ValueError("mixed MeanFlow recipe forbids activation checkpoint around JVP path")
    if model_type == "B3MeanFlowLightningDiT-Custom":
        return B3MeanFlowLightningDiT(
            patch_size=model_cfg.get("patch_size", 1),
            hidden_size=model_cfg.get("hidden_size", 128),
            depth=model_cfg.get("depth", 2),
            num_heads=model_cfg.get("num_heads", 4),
            mlp_ratio=model_cfg.get("mlp_ratio", 4.0),
            **common,
        )
    if model_type not in B3MeanFlowLightningDiT_models:
        raise KeyError(f"unknown model_type={model_type!r}; choices={sorted(B3MeanFlowLightningDiT_models)}")
    # Factory model types encode depth/width/heads/patch_size in the name.
    return B3MeanFlowLightningDiT_models[model_type](mlp_ratio=model_cfg.get("mlp_ratio", 4.0), **common)


def make_model_kwargs(y: torch.Tensor, class_dropout_prob: float, num_classes: int) -> Dict[str, torch.Tensor]:
    if class_dropout_prob > 0:
        force_drop_ids = (torch.rand(y.shape, device=y.device) < class_dropout_prob).long()
    else:
        force_drop_ids = torch.zeros(y.shape, device=y.device, dtype=torch.long)
    # If dropout is disabled, force_drop_ids must stay all-zero because the label
    # embedding table has no CFG extra row.
    if class_dropout_prob <= 0:
        force_drop_ids.zero_()
    return {"y": y.long(), "force_drop_ids": force_drop_ids}


def grad_all_finite(model: nn.Module) -> bool:
    for p in model.parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad.detach()).all().cpu()):
            return False
    return True


def create_summary(out_dir: Path, payload: Dict[str, Any]) -> None:
    result = payload["result"]
    gate = payload["gate"]
    steps = result.get("steps", [])
    fd_audits = result.get("fd_audits", [])
    lines: List[str] = []
    lines.append("# Phase 0 / B3 MeanFlow Production Trainer Smoke Summary\n\n")
    lines.append(f"- created_at_utc: `{payload['created_at_utc']}`\n")
    lines.append(f"- torch: `{payload['torch_version']}`; device: `{payload['device']}`; GPU: `{payload.get('cuda_device_name')}`\n")
    lines.append(f"- PAE root: `{payload['pae_gen_root']}`; commit: `{payload.get('pae_git_commit')}`\n")
    lines.append(f"- compile_unwrapped: `{payload.get('compile_unwrapped')}`\n")
    lines.append("- scope: non-invasive trainer integration smoke; not ImageNet FID/FDr.\n")
    lines.append("- model path: `models/meanflow_lightningdit.py`; loss path: `transport/meanflow_transport.py`; trainer: `train_meanflow_dit.py`.\n")
    lines.append("- recipe: bf16/autocast backbone train forward/backward + fp32/no-autocast live-param JVP target; target detach; EMA not used for target.\n")
    lines.append("\n## Gate\n\n")
    for k, v in gate.items():
        lines.append(f"- `{k}`: `{v}`\n")
    lines.append("\n## Core metrics\n\n")
    rows = {
        "completed optimizer steps": result.get("completed_steps"),
        "final loss": result.get("final_loss"),
        "loss all finite": result.get("loss_all_finite"),
        "grad all finite": result.get("grad_all_finite"),
        "target detached all steps": result.get("target_detached_all_steps"),
        "jvp param source all live": result.get("jvp_param_source_all_live"),
        "mean realized r=t fraction": result.get("mean_realized_r_eq_t_fraction"),
        "max actual r=t target-v abs": result.get("max_actual_rt_samples_target_minus_v_abs"),
        "best/last FD rel err": gate.get("fd_rel_err_full_jvp"),
        "all-r=t target-v max abs": gate.get("all_r_eq_t_degenerate_target_minus_v_max_abs"),
        "loop peak memory MB": result.get("loop_peak_memory_mb"),
    }
    lines.append("| item | value |\n|---|---:|\n")
    for k, v in rows.items():
        lines.append(f"| {k} | `{v}` |\n")
    lines.append("\n## FD audits\n\n")
    lines.append("| step | fd_rel_err | degen target-v max | forced non-degen | JVP MB | FD MB | r=t frac |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|\n")
    for a in fd_audits:
        lines.append(
            f"| {a.get('step')} | `{a.get('fd_rel_err_full_jvp')}` | `{a.get('all_r_eq_t_degenerate_target_minus_v_max_abs')}` | "
            f"`{a.get('audit_forced_non_degenerate')}` | `{a.get('jvp_peak_memory_mb')}` | "
            f"`{a.get('fd_peak_memory_mb')}` | `{a.get('realized_r_eq_t_fraction')}` |\n"
        )
    lines.append("\n## Train telemetry\n\n")
    lines.append("| step | loss | r=t frac | bf16-vs-fp32 fwd rel | target JVP MB | backbone MB | backward MB | grad norm | actual r=t target-v max |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for s in steps:
        lines.append(
            f"| {s.get('step')} | `{s.get('loss')}` | `{s.get('realized_r_eq_t_fraction')}` | `{s.get('bf16_forward_vs_fp32_rel_err')}` | "
            f"`{s.get('target_jvp_peak_memory_mb')}` | `{s.get('backbone_forward_peak_memory_mb')}` | `{s.get('backward_peak_memory_mb')}` | "
            f"`{s.get('grad_norm_before_clip')}` | `{s.get('actual_rt_samples_target_minus_v_max_abs')}` |\n"
        )
    lines.append("\n## Interpretation\n\n")
    lines.append("- FD rel err validates the fp32 JVP target path against central finite difference.\n")
    lines.append("- bf16-vs-fp32 forward rel is ordinary mixed-precision forward drift; the detached target is fp32 while trainable backbone is bf16/autocast.\n")
    lines.append("- `r=t` samples have zero tangent (`t-r=0`), so target must reduce exactly to `v`; this is audited on actual sampler samples and forced all-r=t batches.\n")
    lines.append("- `force_drop_ids` is sampled once per batch and reused by target JVP and train forward, avoiding class-dropout condition mismatch.\n")
    (out_dir / "meanflow_train_summary.md").write_text("".join(lines), encoding="utf-8")


def gate_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    fd_audits = result.get("fd_audits", [])
    last_fd = fd_audits[-1] if fd_audits else {}
    fd_rel = last_fd.get("fd_rel_err_full_jvp")
    degen = last_fd.get("all_r_eq_t_degenerate_target_minus_v_max_abs")
    fd_ok = fd_rel is not None and float(fd_rel) < 1e-2
    degen_ok = degen is not None and float(degen) < 1e-7
    memory_logged = result.get("loop_peak_memory_mb") is not None and all(
        s.get("target_jvp_peak_memory_mb") is not None and s.get("backbone_forward_peak_memory_mb") is not None and s.get("backward_peak_memory_mb") is not None
        for s in result.get("steps", [])
    )
    train_ok = (
        result.get("completed_steps") == result.get("requested_steps")
        and result.get("loss_all_finite") is True
        and result.get("grad_all_finite") is True
        and result.get("u_du_target_all_finite") is True
    )
    practical = bool(
        train_ok
        and fd_ok
        and degen_ok
        and result.get("target_detached_all_steps") is True
        and result.get("jvp_param_source_all_live") is True
        and result.get("sample_level_sampler") is True
        and result.get("mixed_modes_ok") is True
        and memory_logged
        and result.get("no_nan_or_inf") is True
    )
    return {
        "train_ok": train_ok,
        "fd_ok_lt_1e_2": fd_ok,
        "fd_rel_err_full_jvp": fd_rel,
        "degenerate_ok_lt_1e_7": degen_ok,
        "all_r_eq_t_degenerate_target_minus_v_max_abs": degen,
        "target_detached": result.get("target_detached_all_steps") is True,
        "live_param_jvp_target": result.get("jvp_param_source_all_live") is True,
        "sample_level_sampler": result.get("sample_level_sampler") is True,
        "mixed_modes_ok": result.get("mixed_modes_ok") is True,
        "memory_logged": memory_logged,
        "no_nan_or_inf": result.get("no_nan_or_inf") is True,
        "practical_gate_pass": practical,
        "strict_gate_pass": practical,
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

    output_dir = Path(train_cfg.get("output_dir", "output_meanflow")) / train_cfg.get("exp_name", "b3_meanflow")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir)
    logger.info("Phase0/B3 MeanFlow trainer starting")
    logger.info("config=%s", args.config)
    logger.info("device=%s torch=%s cuda=%s gpu=%s", device, torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    logger.info("sdpa=%s allow_tf32=%s compile_unwrapped=%s", sdpa_info, allow_tf32, COMPILE_UNWRAPPED)

    downsample_ratio = int(cfg.get("vae", {}).get("downsample_ratio", 16))
    image_size = int(data_cfg.get("image_size", 256))
    latent_size = image_size // downsample_ratio
    dataset = ImgLatentDataset(
        data_dir=data_cfg["data_path"],
        latent_norm=bool(data_cfg.get("latent_norm", False)),
        latent_multiplier=float(data_cfg.get("latent_multiplier", 1.0)),
    )
    batch_size = int(train_cfg.get("global_batch_size", train_cfg.get("batch_size", 4)))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    logger.info("dataset=%s length=%s batch_size=%s latent_size=%s", data_cfg["data_path"], len(dataset), batch_size, latent_size)

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
    max_steps = int(train_cfg.get("max_steps", 8))
    log_every = int(train_cfg.get("log_every", 1))
    fd_audit_every = int(meanflow_cfg.get("fd_audit_every", max_steps))
    grad_clip = float(opt_cfg.get("max_grad_norm", train_cfg.get("grad_clip", 1.0)))
    metrics_jsonl = output_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        metrics_jsonl.unlink()

    steps: List[Dict[str, Any]] = []
    fd_audits: List[Dict[str, Any]] = []
    completed = 0
    loop_peak = 0.0
    data_iter = cycle_loader(loader)
    start_all = time.perf_counter()

    for step in range(1, max_steps + 1):
        x, y = next(data_iter)
        x = x.to(device=device, dtype=torch.float32, non_blocking=True)
        y = y.to(device=device, non_blocking=True).long()
        model_kwargs = make_model_kwargs(y, float(model_cfg.get("class_dropout_prob", 0.0)), int(data_cfg.get("num_classes", 1000)))
        opt.zero_grad(set_to_none=True)
        rec: Dict[str, Any] = {"step": step, "created_at_utc": now_utc()}
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
                completed += 1
            else:
                rec["stopped_reason"] = "nonfinite_loss_or_grad"
        except Exception as exc:  # noqa: BLE001
            rec.update({"status": "error", "error_type": type(exc).__name__, "error_message": str(exc), "loss_finite": False, "grad_is_finite": False})
        finally:
            cuda_sync(device)
            rec["elapsed_sec"] = time.perf_counter() - step_start
            candidates = [rec.get("target_jvp_peak_memory_mb"), rec.get("backbone_forward_peak_memory_mb"), rec.get("backward_peak_memory_mb"), peak_mb(device)]
            rec["peak_memory_mb"] = max(float(v) for v in candidates if v is not None) if device.type == "cuda" else None
            if rec["peak_memory_mb"] is not None:
                loop_peak = max(loop_peak, float(rec["peak_memory_mb"]))
            steps.append(rec)
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"type": "train_step", **rec}, ensure_ascii=False) + "\n")

        if step % log_every == 0:
            logger.info(
                "step=%d loss=%.6g rt=%.3f fd? targetMB=%s fwdMB=%s bwdMB=%s grad=%.6g",
                step,
                rec.get("loss", float("nan")),
                rec.get("realized_r_eq_t_fraction", float("nan")),
                rec.get("target_jvp_peak_memory_mb"),
                rec.get("backbone_forward_peak_memory_mb"),
                rec.get("backward_peak_memory_mb"),
                rec.get("grad_norm_before_clip", float("nan")),
            )

        if fd_audit_every > 0 and (step == 1 or step % fd_audit_every == 0 or step == max_steps):
            audit_kwargs = make_model_kwargs(y, float(model_cfg.get("class_dropout_prob", 0.0)), int(data_cfg.get("num_classes", 1000)))
            audit = transport.fd_audit(model, x, audit_kwargs, log_memory=True)
            audit["step"] = step
            fd_audits.append(audit)
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"type": "fd_audit", **audit}, ensure_ascii=False) + "\n")
            logger.info(
                "FD audit step=%d fd_rel=%.6g degen=%.6g forced_non_degen=%s jvpMB=%s fdMB=%s",
                step,
                audit.get("fd_rel_err_full_jvp", float("nan")),
                audit.get("all_r_eq_t_degenerate_target_minus_v_max_abs", float("nan")),
                audit.get("audit_forced_non_degenerate"),
                audit.get("jvp_peak_memory_mb"),
                audit.get("fd_peak_memory_mb"),
            )

        if rec.get("status") == "error" or not rec.get("loss_finite", False) or not rec.get("grad_is_finite", False):
            break

    losses = [s.get("loss") for s in steps if isinstance(s.get("loss"), (float, int))]
    realized = [s.get("realized_r_eq_t_fraction") for s in steps if isinstance(s.get("realized_r_eq_t_fraction"), (float, int))]
    actual_degen = [s.get("actual_rt_samples_target_minus_v_max_abs") for s in steps if isinstance(s.get("actual_rt_samples_target_minus_v_max_abs"), (float, int))]
    result: Dict[str, Any] = {
        "requested_steps": max_steps,
        "completed_steps": completed,
        "recorded_steps": len(steps),
        "final_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "max_loss": max(losses) if losses else None,
        "loss_all_finite": all(bool(s.get("loss_finite", False)) for s in steps) if steps else False,
        "grad_all_finite": all(bool(s.get("grad_is_finite", False)) for s in steps) if steps else False,
        "u_du_target_all_finite": all(bool(s.get("u_finite", False) and s.get("du_finite", False) and s.get("target_finite", False)) for s in steps) if steps else False,
        "target_detached_all_steps": all(s.get("target_requires_grad") is False for s in steps) if steps else False,
        "jvp_param_source_all_live": all(s.get("jvp_param_source") == "live" for s in steps) if steps else False,
        "sample_level_sampler": all(s.get("r_t_sampling_granularity") == "sample_level" for s in steps) if steps else False,
        "mixed_modes_ok": all(
            s.get("backbone_forward_mode") == "bf16_autocast" and s.get("jvp_target_mode") == "fp32" and s.get("fd_audit_mode") == "fp32"
            for s in steps
        ) if steps else False,
        "mean_realized_r_eq_t_fraction": sum(float(x) for x in realized) / len(realized) if realized else None,
        "max_actual_rt_samples_target_minus_v_abs": max(actual_degen) if actual_degen else None,
        "loop_peak_memory_mb": loop_peak if device.type == "cuda" else None,
        "no_nan_or_inf": all(
            bool(s.get("loss_finite", False) and s.get("grad_is_finite", False) and s.get("u_finite", False) and s.get("du_finite", False) and s.get("target_finite", False))
            for s in steps
        ) if steps else False,
        "steps": steps,
        "fd_audits": fd_audits,
        "total_elapsed_sec": time.perf_counter() - start_all,
    }
    payload: Dict[str, Any] = {
        "created_at_utc": now_utc(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device": str(device),
        "pae_gen_root": str(Path(__file__).resolve().parent),
        "pae_git_commit": git_commit(Path(__file__).resolve().parents[1]),
        "config_path": str(Path(args.config).resolve()),
        "config": cfg,
        "sdpa_info": sdpa_info,
        "allow_tf32": allow_tf32,
        "compile_unwrapped": COMPILE_UNWRAPPED,
        "model_params": model_params,
        "trainable_params": trainable_params,
        "ema_present": ema is not None,
        "ema_decay": ema_decay if ema is not None else None,
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
    payload["gate"] = gate_summary(result)
    (output_dir / "meanflow_train_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    create_summary(output_dir, payload)
    logger.info("DONE gate practical=%s strict=%s final_loss=%s loop_peak=%sMB", payload["gate"]["practical_gate_pass"], payload["gate"]["strict_gate_pass"], result.get("final_loss"), result.get("loop_peak_memory_mb"))
    logger.info("wrote %s", output_dir / "meanflow_train_metrics.json")
    logger.info("wrote %s", output_dir / "meanflow_train_summary.md")


if __name__ == "__main__":
    main()
