#!/usr/bin/env python3
"""Generate a large deterministic EMA latent sample set from a B3 MeanFlow checkpoint.

This is the missing first half of a paper-comparable image-space FID pass: the
trainer's normal eval emits only 64 latents for smoke/convergence diagnostics.
For a 5k/50k FID anchor, run this helper from a saved checkpoint to produce a
larger ``sample_latents_*.safetensors`` file, then feed that file to
``decode_and_inception_eval_step.py``.

Operational note for the single-H100 workstation: the active training process
uses most H100 memory, so CUDA sampling should be done during a short controlled
pause at a checkpoint boundary.  The resulting latent file is small enough that
CPU decode/Inception can then run in the background after training resumes.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("XFORMERS_DISABLED", "1")
os.environ.setdefault("DISABLE_XFORMERS", "1")

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Reuse the exact experiment-local model/transport helpers used by training.
from train_b3_meanflow_realdata import (  # noqa: E402
    build_model,
    configure_sdpa,
    create_meanflow_transport,
    load_config,
    now_utc,
    requires_grad,
    resolve_device,
    sample_meanflow_latents,
    set_seed,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def atomic_save_latents(path: Path, samples: torch.Tensor, labels: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    save_file({"samples": samples.contiguous().float(), "labels": labels.contiguous().long()}, str(tmp))
    os.replace(tmp, path)


def checkpoint_step(path: Path) -> Optional[int]:
    name = path.name
    if name == "latest.pt":
        try:
            return checkpoint_step(path.resolve())
        except Exception:
            return None
    if name.startswith("step_") and name.endswith(".pt"):
        try:
            return int(name[len("step_") : -len(".pt")])
        except ValueError:
            return None
    return None


def output_root_from_config(cfg: Dict[str, Any]) -> Path:
    train_cfg = cfg.get("train", {})
    return Path(train_cfg.get("output_dir", "output_meanflow")) / train_cfg.get("exp_name", "b3_meanflow_realdata")


def latent_shape_from_config(cfg: Dict[str, Any]) -> List[int]:
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    vae_cfg = cfg.get("vae", {})
    image_size = int(data_cfg.get("image_size", 256))
    downsample_ratio = int(vae_cfg.get("downsample_ratio", 16))
    latent_dim = int(vae_cfg.get("latent_dim", model_cfg.get("in_chans", 32)))
    latent_size = image_size // downsample_ratio
    return [latent_dim, latent_size, latent_size]


def make_labels(
    *,
    num_samples: int,
    num_classes: int,
    fixed_labels: Optional[Sequence[int]],
    generator: torch.Generator,
) -> torch.Tensor:
    if fixed_labels:
        vals = [int(v) for v in fixed_labels]
        if not vals:
            raise ValueError("fixed_labels was provided but empty")
        reps = (int(num_samples) + len(vals) - 1) // len(vals)
        return torch.tensor((vals * reps)[: int(num_samples)], dtype=torch.long)
    return torch.randint(0, int(num_classes), (int(num_samples),), generator=generator, dtype=torch.long)


def load_label_pool_from_latent_subset(
    *,
    latent_cache: Path,
    shard_glob: str,
    max_samples: int,
) -> torch.Tensor:
    """Load labels from the first-N latent-cache samples.

    This is used by small-subset fit probes so generated-label marginal matches
    the training subset instead of assuming a perfectly uniform ImageNet prior.
    The pool is tiny for current probes (e.g. 64k int64 labels), so materializing
    it is simpler and more robust than building a global-index map.
    """
    files = sorted([p for p in latent_cache.glob(shard_glob) if p.is_file()])
    if not files:
        raise FileNotFoundError(f"No latent shards matched {latent_cache / shard_glob}")
    remaining = int(max_samples)
    if remaining <= 0:
        raise ValueError("label subset max_samples must be positive")
    chunks: List[torch.Tensor] = []
    for path in files:
        if remaining <= 0:
            break
        with safe_open(str(path), framework="pt", device="cpu") as sf:
            if "labels" not in sf.keys():
                raise KeyError(f"{path} missing key 'labels'; keys={list(sf.keys())}")
            n = int(sf.get_slice("labels").get_shape()[0])
            take = min(n, remaining)
            chunks.append(sf.get_slice("labels")[:take].long().cpu().contiguous())
        remaining -= take
    if not chunks:
        raise RuntimeError(f"No labels loaded from {latent_cache}")
    pool = torch.cat(chunks, dim=0)
    if pool.numel() < int(max_samples):
        raise ValueError(f"Requested label pool of {max_samples} but only found {pool.numel()} samples")
    return pool


def make_labels_from_latent_subset(
    *,
    num_samples: int,
    latent_cache: Path,
    shard_glob: str,
    max_samples: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    pool = load_label_pool_from_latent_subset(latent_cache=latent_cache, shard_glob=shard_glob, max_samples=max_samples)
    choice = torch.randint(0, int(pool.numel()), (int(num_samples),), generator=generator, dtype=torch.long)
    labels = pool.index_select(0, choice).long().contiguous()
    hist = torch.bincount(pool.long(), minlength=1000).float()
    nonzero = int((hist > 0).sum().item())
    meta = {
        "label_source": "latent_subset",
        "label_latent_cache": str(latent_cache),
        "label_latent_shard_glob": shard_glob,
        "label_subset_max_samples": int(max_samples),
        "label_pool_size": int(pool.numel()),
        "label_pool_unique_count": nonzero,
        "label_pool_min_count_nonzero": int(hist[hist > 0].min().item()) if nonzero else None,
        "label_pool_max_count": int(hist.max().item()) if hist.numel() else None,
        "sampled_label_unique_count": int(torch.unique(labels).numel()),
    }
    return labels, meta


def load_model_and_optional_ema(
    *,
    cfg: Dict[str, Any],
    checkpoint: Path,
    device: torch.device,
    use_ema: bool,
) -> tuple[torch.nn.Module, Optional[torch.nn.Module], int, Dict[str, Any]]:
    data_cfg = cfg.get("data", {})
    vae_cfg = cfg.get("vae", {})
    image_size = int(data_cfg.get("image_size", 256))
    downsample_ratio = int(vae_cfg.get("downsample_ratio", 16))
    latent_size = image_size // downsample_ratio

    # Load checkpoint tensors on CPU first.  Full training checkpoints include a
    # large optimizer state; mapping directly to CUDA would waste H100 memory.
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    step = int(ckpt.get("step", checkpoint_step(checkpoint) or 0))

    model = build_model(cfg, latent_size).to(dtype=torch.float32)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ema = None
    if use_ema:
        if ckpt.get("ema") is None:
            raise ValueError(f"Checkpoint {checkpoint} does not contain EMA weights but --use-ema was requested")
        ema = copy.deepcopy(model).to(dtype=torch.float32)
        ema.load_state_dict(ckpt["ema"])
        requires_grad(ema, False)
        ema.eval()

    # Keep a small, JSON-safe checkpoint metadata subset for provenance, then
    # release the full optimizer state before moving models to CUDA.
    ckpt_meta = {
        "checkpoint_created_at_utc": ckpt.get("created_at_utc"),
        "checkpoint_step": step,
        "checkpoint_torch_version": ckpt.get("torch_version"),
        "checkpoint_has_ema": ckpt.get("ema") is not None,
        "checkpoint_has_optimizer": ckpt.get("optimizer") is not None,
    }
    del ckpt
    gc.collect()

    model = model.to(device=device, dtype=torch.float32)
    if ema is not None:
        ema = ema.to(device=device, dtype=torch.float32)
    return model, ema, step, ckpt_meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None, help="default: <output_dir>/checkpoints/latest.pt")
    parser.add_argument("--output-latents", type=Path, default=None, help="default: <output_dir>/eval/step_XXXXXXXX/sample_latents_5k.safetensors")
    parser.add_argument("--metadata-json", type=Path, default=None, help="default: output-latents sibling .json")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--sample-batch-size", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=0, help="0: use config eval.sample_steps")
    parser.add_argument("--precision-mode", default="", help="empty: use config eval.precision_mode")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026052850)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--label-source",
        choices=["auto", "uniform", "latent-subset"],
        default="auto",
        help="auto preserves old behavior: config fixed_labels if present, otherwise uniform over classes.",
    )
    parser.add_argument("--label-latent-cache", type=Path, default=None, help="default: config data.data_path")
    parser.add_argument("--label-latent-shard-glob", default="", help="default: config data.file_glob")
    parser.add_argument("--label-subset-max-samples", type=int, default=0, help="default: config data.max_samples")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    cfg = load_config(str(args.config))
    train_cfg = cfg.get("train", {})
    eval_cfg = cfg.get("eval", {})
    data_cfg = cfg.get("data", {})
    meanflow_cfg = cfg.get("meanflow", {})

    output_root = output_root_from_config(cfg)
    checkpoint = args.checkpoint or (output_root / "checkpoints" / "latest.pt")
    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    inferred_step = checkpoint_step(checkpoint)
    # If latest.pt points to a real step file, prefer the resolved name.
    try:
        resolved_step = checkpoint_step(checkpoint.resolve())
        if resolved_step is not None:
            inferred_step = resolved_step
    except Exception:
        pass
    step_for_path = int(inferred_step or 0)

    output_latents = args.output_latents
    if output_latents is None:
        output_latents = output_root / "eval" / f"step_{step_for_path:08d}" / f"sample_latents_{int(args.num_samples)}.safetensors"
    output_latents = output_latents.resolve()
    metadata_json = args.metadata_json or output_latents.with_suffix(".json")
    metadata_json = metadata_json.resolve()

    if output_latents.exists() and not args.overwrite:
        print(f"Latent sample already exists and --overwrite not set: {output_latents}", flush=True)
        return 0

    seed = int(args.seed)
    set_seed(seed)
    label_gen = torch.Generator(device="cpu")
    label_gen.manual_seed(seed + 17)

    device = resolve_device(args.device if args.device else train_cfg.get("device", "auto"))
    allow_tf32 = bool(train_cfg.get("allow_tf32", False))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    try:
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    except Exception:
        pass
    sdpa_info = configure_sdpa(train_cfg.get("sdpa_kernel", "default"), device)

    sample_steps = int(args.sample_steps or eval_cfg.get("sample_steps", 32))
    precision_mode = str(args.precision_mode or eval_cfg.get("precision_mode", "bf16_autocast"))
    latent_shape = latent_shape_from_config(cfg)
    num_samples = int(args.num_samples)
    batch_size = int(args.sample_batch_size)
    num_classes = int(data_cfg.get("num_classes", 1000))
    label_meta: Dict[str, Any] = {"label_source": args.label_source}
    if args.label_source == "latent-subset":
        label_latent_cache = (args.label_latent_cache or Path(data_cfg["data_path"])).resolve()
        label_shard_glob = str(args.label_latent_shard_glob or data_cfg.get("file_glob", "*.safetensors"))
        label_subset_max = int(args.label_subset_max_samples or data_cfg.get("max_samples") or data_cfg.get("expected_total") or 0)
        labels_all, label_meta = make_labels_from_latent_subset(
            num_samples=num_samples,
            latent_cache=label_latent_cache,
            shard_glob=label_shard_glob,
            max_samples=label_subset_max,
            generator=label_gen,
        )
    else:
        fixed = eval_cfg.get("fixed_labels") if args.label_source == "auto" else None
        labels_all = make_labels(
            num_samples=num_samples,
            num_classes=num_classes,
            fixed_labels=fixed,
            generator=label_gen,
        )
        label_meta = {
            "label_source": "config_fixed_labels" if fixed else "uniform",
            "fixed_labels_count": len(fixed) if fixed else 0,
            "sampled_label_unique_count": int(torch.unique(labels_all).numel()),
        }

    print(
        json.dumps(
            {
                "event": "large_sample_start",
                "created_at_utc": utc_now(),
                "checkpoint": str(checkpoint),
                "output_latents": str(output_latents),
                "num_samples": num_samples,
                "sample_batch_size": batch_size,
                "sample_steps": sample_steps,
                "precision_mode": precision_mode,
                "device": str(device),
                "allow_tf32": allow_tf32,
                "sdpa_info": sdpa_info,
                "use_ema": bool(args.use_ema),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    model, ema, ckpt_step, ckpt_meta = load_model_and_optional_ema(cfg=cfg, checkpoint=checkpoint, device=device, use_ema=bool(args.use_ema))
    eval_model = ema if bool(args.use_ema) and ema is not None else model
    transport = create_meanflow_transport(
        precision_recipe=meanflow_cfg.get("precision_recipe", "bf16_backbone_fp32_jvp"),
        equal_prob=float(meanflow_cfg.get("equal_prob", 0.75)),
        time_sampler=meanflow_cfg.get("time_sampler", "ltg"),
        t_low=float(meanflow_cfg.get("t_low", 0.05)),
        t_high=float(meanflow_cfg.get("t_high", 0.95)),
        lognorm_mu=float(meanflow_cfg.get("lognorm_mu", -0.4)),
        lognorm_sigma=float(meanflow_cfg.get("lognorm_sigma", 1.0)),
        fd_eps=float(meanflow_cfg.get("fd_eps", 1e-2)),
        device_type=device.type,
    )

    chunks: List[torch.Tensor] = []
    peak_mb = None
    sampled = 0
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        labels = labels_all[start:end].to(device=device, dtype=torch.long)
        z = sample_meanflow_latents(
            model=eval_model,
            transport=transport,
            labels=labels,
            latent_shape=latent_shape,
            num_steps=sample_steps,
            precision_mode=precision_mode,
            device=device,
        )
        chunks.append(z.cpu().float())
        sampled = end
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mb = max(float(torch.cuda.max_memory_allocated(device) / (1024**2)), float(peak_mb or 0.0))
        if sampled == num_samples or sampled % max(batch_size * 5, 1) == 0:
            print(
                json.dumps(
                    {
                        "event": "large_sample_progress",
                        "created_at_utc": utc_now(),
                        "sampled": sampled,
                        "num_samples": num_samples,
                        "elapsed_sec": time.perf_counter() - started,
                        "cuda_peak_memory_mb": peak_mb,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    samples = torch.cat(chunks, dim=0)
    finite = bool(torch.isfinite(samples).all().item())
    sample_mean = float(samples.float().mean().item())
    sample_std = float(samples.float().std().item())
    atomic_save_latents(output_latents, samples, labels_all)

    label_unique = int(torch.unique(labels_all).numel())
    rec: Dict[str, Any] = {
        "created_at_utc": now_utc(),
        "status": "ok",
        "config": str(args.config.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_step": int(ckpt_step),
        "output_latents": str(output_latents),
        "num_samples": num_samples,
        "sample_batch_size": batch_size,
        "sample_steps": sample_steps,
        "precision_mode": precision_mode,
        "use_ema": bool(args.use_ema),
        "seed": seed,
        "device": str(device),
        "allow_tf32": allow_tf32,
        "sdpa_info": sdpa_info,
        "transport_precision_recipe": transport.precision_recipe,
        "latent_shape_per_sample": latent_shape,
        "sample_shape": list(samples.shape),
        "sample_finite": finite,
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "label_min": int(labels_all.min().item()),
        "label_max": int(labels_all.max().item()),
        "label_unique_count": label_unique,
        **label_meta,
        "cuda_peak_memory_mb": peak_mb,
        "elapsed_sec": float(time.perf_counter() - started),
        **ckpt_meta,
    }
    save_json(metadata_json, rec)
    print("LARGE_SAMPLE_DONE", json.dumps(rec, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
