#!/usr/bin/env python3
"""B3 trainer eval hook for FLUX.2 VAE image-space decoding.

The trainer calls this through ``eval.fid_command_template`` with placeholders
``{sample_dir}``, ``{output_dir}``, and ``{step}``. This hook decodes
``sample_latents.safetensors`` to PNGs and optionally runs a user-provided FID
command over the decoded image directory.

No fake FID is reported: when no inner FID command is configured, the script
prints a JSON record with ``fid: null`` and ``image_eval_status:
\"decoded_no_fid\"``. The trainer will still capture stdout/stderr and record
that the external command returned successfully.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
_DECODE_PATH = SCRIPT_DIR / "decode_flux2vae_latents.py"
spec = importlib.util.spec_from_file_location("decode_flux2vae_latents", _DECODE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Could not load decoder from {_DECODE_PATH}")
decoder = importlib.util.module_from_spec(spec)
sys.modules["decode_flux2vae_latents"] = decoder
spec.loader.exec_module(decoder)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_fid(text: str) -> Optional[float]:
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            for key in ("fid", "FID", "fid50k", "FID50K"):
                if key in obj and obj[key] is not None:
                    return float(obj[key])
        except Exception:
            pass
    m = re.search(r"(?i)\bfid(?:50k)?\b[^0-9+\-.eE]*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)", text)
    if m:
        return float(m.group(1))
    return None


def parse_inner_metrics(text: str) -> Dict[str, Any]:
    """Extract a compact metric dictionary from an inner command's output.

    The real metrics command prints one JSON object with keys such as
    ``fid``, ``mmd2`` and ``kid``.  Keep this parser permissive so existing
    FID-only commands continue to work.
    """
    metrics: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if isinstance(obj, dict) and ("fid" in obj or "FID" in obj or "mmd2" in obj or "kid" in obj):
            metrics = obj
    out: Dict[str, Any] = {}
    key_map = {
        "fid": ("fid", "FID", "fid50k", "FID50K"),
        "mmd2": ("mmd2", "mmd2_poly3"),
        "kid": ("kid",),
        "kid_x1000": ("kid_x1000",),
        "ref_features_used": ("ref_features_used",),
        "num_images": ("num_images",),
        "metrics_elapsed_sec": ("elapsed_sec",),
    }
    for out_key, candidates in key_map.items():
        for key in candidates:
            if key in metrics and metrics[key] is not None:
                try:
                    val = metrics[key]
                    out[out_key] = float(val) if out_key not in {"ref_features_used", "num_images"} else int(val)
                except Exception:
                    out[out_key] = metrics[key]
                break
    if metrics:
        out["inner_metrics_json"] = metrics
    return out


def run_inner_fid(template: str, *, images_dir: Path, sample_dir: Path, output_dir: Path, step: int, timeout: int) -> Dict[str, Any]:
    if not template:
        return {"fid_status": "decoded_no_fid", "fid": None}
    command = template.format(images_dir=str(images_dir), sample_dir=str(sample_dir), output_dir=str(output_dir), step=step)
    started = time.perf_counter()
    proc = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=timeout)
    elapsed = time.perf_counter() - started
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    (sample_dir / "flux2vae_imagespace_fid_command.txt").write_text(command + "\n", encoding="utf-8")
    (sample_dir / "flux2vae_imagespace_fid_stdout_stderr.txt").write_text(text, encoding="utf-8")
    fid = parse_fid(text)
    metrics = parse_inner_metrics(text)
    if fid is None and metrics.get("fid") is not None:
        fid = float(metrics["fid"])
    result = {
        "fid_status": "ok" if proc.returncode == 0 else "fid_command_failed",
        "fid": fid,
        "inner_fid_command": command,
        "inner_fid_returncode": proc.returncode,
        "inner_fid_elapsed_sec": elapsed,
    }
    for key in ("mmd2", "kid", "kid_x1000", "ref_features_used", "num_images", "metrics_elapsed_sec", "inner_metrics_json"):
        if key in metrics:
            result[key] = metrics[key]
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--latents-name", default="sample_latents.safetensors")
    p.add_argument("--images-subdir", default="decoded_flux2vae_png")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--model-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--repo-id", default="diffusers/FLUX.2-dev-bnb-4bit")
    p.add_argument("--subfolder", default="vae")
    p.add_argument("--vae-class", default="AutoencoderKLFlux2")
    p.add_argument("--revision", default="")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--fid-command-template", default="", help="optional inner FID command; placeholders: {images_dir} {sample_dir} {output_dir} {step}")
    p.add_argument("--fid-timeout-sec", type=int, default=3600)
    p.add_argument("--pre-decode-scale", type=float, default=1.0, help="scalar inverse scale applied to sampled trainer latents before FLUX VAE decode")
    p.add_argument("--pre-decode-shift", type=float, default=0.0, help="scalar inverse shift applied after pre-decode-scale before FLUX VAE decode")
    p.add_argument("--pre-decode-stats-path", default="", help="optional trainer stats .pt for per-channel inverse decode")
    p.add_argument("--save-grid", action="store_true")
    p.add_argument("--decode-manifest-mode", default="first_last", choices=["full", "first_last", "none"], help="manifest policy forwarded to decode_summary; first_last avoids huge JSON for large evals")
    return p


def main() -> int:
    args = build_parser().parse_args()
    sample_dir = Path(args.sample_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    latents_path = sample_dir / args.latents_name
    images_dir = sample_dir / args.images_subdir
    started = time.perf_counter()
    record: Dict[str, Any] = {
        "created_at_utc": utc_now(),
        "image_eval_backend": "flux2vae",
        "sample_dir": str(sample_dir),
        "output_dir": str(output_dir),
        "step": int(args.step),
        "latents_path": str(latents_path),
        "images_dir": str(images_dir),
        "pre_decode_scale": float(args.pre_decode_scale),
        "pre_decode_shift": float(args.pre_decode_shift),
        "pre_decode_stats_path": str(args.pre_decode_stats_path or ""),
    }
    try:
        decode_args = argparse.Namespace(
            latents=str(latents_path),
            output_dir=str(images_dir),
            tensor_key="samples",
            max_images=args.max_images,
            start_index=0,
            batch_size=int(args.batch_size),
            device=args.device,
            model_dtype=args.model_dtype,
            repo_id=args.repo_id,
            subfolder=args.subfolder,
            revision=args.revision,
            vae_class=args.vae_class,
            local_files_only=bool(args.local_files_only),
            output_range="minus1_1",
            pre_decode_scale=float(args.pre_decode_scale),
            pre_decode_shift=float(args.pre_decode_shift),
            pre_decode_stats_path=str(args.pre_decode_stats_path or ""),
            save_grid=bool(args.save_grid),
            manifest_mode=args.decode_manifest_mode,
            preview_max_images=16,
            preview_columns=4,
        )
        summary = decoder.decode_latent_file(decode_args)
        record.update(
            {
                "image_eval_status": "decoded",
                "num_images": summary.get("num_images"),
                "decode_summary": str(images_dir / "decode_summary.json"),
                "first_image": summary.get("first_image"),
                "last_image": summary.get("last_image"),
                "decode_elapsed_sec": summary.get("decode_elapsed_sec"),
                "decode_cuda_peak_memory_mb": summary.get("cuda_peak_memory_mb"),
                "decode_manifest_mode": summary.get("images_manifest_mode"),
                "decode_manifest_count": summary.get("images_manifest_count"),
                "decode_manifest_omitted": summary.get("images_manifest_omitted"),
                "decode_pre_decode_scale": summary.get("pre_decode_scale"),
                "decode_pre_decode_shift": summary.get("pre_decode_shift"),
                "decode_pre_decode_stats_path": summary.get("pre_decode_stats_path"),
                "pre_decode_channel_stats_path": summary.get("pre_decode_channel_stats_path"),
                "pre_decode_channel_stats_num_samples": summary.get("pre_decode_channel_stats_num_samples"),
                "pre_decode_channel_stats_created_at_utc": summary.get("pre_decode_channel_stats_created_at_utc"),
                "pre_decode_channel_mean_summary": summary.get("pre_decode_channel_mean_summary"),
                "pre_decode_channel_std_summary": summary.get("pre_decode_channel_std_summary"),
                "sample_space_mean": summary.get("sample_space_mean"),
                "sample_space_std": summary.get("sample_space_std"),
                "sample_space_rms": summary.get("sample_space_rms"),
                "sample_space_absmax": summary.get("sample_space_absmax"),
                "decode_space_mean": summary.get("decode_space_mean"),
                "decode_space_std": summary.get("decode_space_std"),
                "decode_space_rms": summary.get("decode_space_rms"),
                "decode_space_absmax": summary.get("decode_space_absmax"),
                "reference_raw_flux_mean": summary.get("reference_raw_flux_mean"),
                "reference_raw_flux_std": summary.get("reference_raw_flux_std"),
                "reference_raw_flux_rms": summary.get("reference_raw_flux_rms"),
                "sample_space_std_over_reference_raw_flux_std": summary.get("sample_space_std_over_reference_raw_flux_std"),
                "decode_space_std_over_reference_raw_flux_std": summary.get("decode_space_std_over_reference_raw_flux_std"),
                "sample_space_rms_over_reference_raw_flux_rms": summary.get("sample_space_rms_over_reference_raw_flux_rms"),
                "decode_space_rms_over_reference_raw_flux_rms": summary.get("decode_space_rms_over_reference_raw_flux_rms"),
            }
        )
        record.update(
            run_inner_fid(
                args.fid_command_template,
                images_dir=images_dir,
                sample_dir=sample_dir,
                output_dir=output_dir,
                step=int(args.step),
                timeout=int(args.fid_timeout_sec),
            )
        )
        if record.get("fid") is None and record.get("fid_status") == "decoded_no_fid":
            record["image_eval_status"] = "decoded_no_fid"
        record["elapsed_sec"] = time.perf_counter() - started
        (sample_dir / "flux2vae_imagespace_eval_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True))
        return 0
    except subprocess.TimeoutExpired as exc:
        record.update({"image_eval_status": "fid_timeout", "fid_status": "timeout", "fid": None, "error": str(exc), "elapsed_sec": time.perf_counter() - started})
        (sample_dir / "flux2vae_imagespace_eval_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(record, sort_keys=True), file=sys.stderr)
        return 3
    except Exception as exc:  # noqa: BLE001
        record.update({"image_eval_status": "error", "fid_status": "not_run", "fid": None, "error_type": type(exc).__name__, "error": str(exc), "elapsed_sec": time.perf_counter() - started})
        try:
            (sample_dir / "flux2vae_imagespace_eval_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception:
            pass
        print(json.dumps(record, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
