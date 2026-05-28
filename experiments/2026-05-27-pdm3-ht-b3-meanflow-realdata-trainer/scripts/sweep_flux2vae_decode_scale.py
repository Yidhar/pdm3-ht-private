#!/usr/bin/env python3
"""Sweep FLUX.2 VAE pre-decode scales for existing B3 sample latents.

This is a diagnostic helper for the FLUX.2 route.  It reuses an already
generated ``sample_latents.safetensors`` file, creates one isolated sample
directory per scale, then calls ``eval_flux2vae_imagespace.py`` with:

    z_raw = (z_sample * scale + shift) * channel_std + channel_mean

when ``--pre-decode-stats-path`` is supplied.  Per-scale sample directories are
intentional: ``eval_flux2vae_imagespace.py`` writes fixed record filenames into
``sample_dir`` (``flux2vae_imagespace_eval_record.json`` and inner FID logs), so
running multiple scales against the same directory would overwrite records.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL_SCRIPT = SCRIPT_DIR / "eval_flux2vae_imagespace.py"
DEFAULT_METRICS_SCRIPT = SCRIPT_DIR / "eval_imagespace_fid_mmd.py"
DEFAULT_REF_STATS = Path(
    "/workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/"
    "imagenet256_train_adm_inception2048_stats.npz"
)
DEFAULT_REF_FEATURES = Path(
    "/workspace/PDM/data/reference_stats/imagenet256_adm_train_inception2048/"
    "imagenet256_train_adm_inception2048_mmd8192_features.npz"
)


def utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_scales(text: str) -> List[float]:
    vals: List[float] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("--scales produced an empty list")
    return vals


def scale_slug(scale: float) -> str:
    s = f"{float(scale):.6g}"
    return "s" + s.replace("-", "m").replace("+", "").replace(".", "p")


def ensure_sample_latents_link(source: Path, dest: Path) -> str:
    """Create a cheap local reference to source at dest.

    Hardlink is preferred because it avoids a 129MB copy while keeping the
    per-scale directory self-contained if the source path later changes.
    Symlink/copy are fallbacks for cross-device or permission edge cases.
    """
    source = source.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        try:
            if dest.resolve() == source:
                return "existing"
        except Exception:
            pass
        dest.unlink()
    try:
        os.link(source, dest)
        return "hardlink"
    except OSError:
        try:
            os.symlink(source, dest)
            return "symlink"
        except OSError:
            shutil.copy2(source, dest)
            return "copy"


def build_inner_metrics_template(
    *,
    metrics_script: Path,
    ref_stats: Path,
    ref_features: Optional[Path],
    batch_size: int,
    num_workers: int,
    device: str,
    sqrt_device: str,
    mmd_device: str,
    mmd_max_ref: int,
    allow_tf32: bool,
) -> str:
    parts = [
        sys.executable,
        str(metrics_script),
        "--images-dir",
        "{images_dir}",
        "--ref-stats",
        str(ref_stats),
        "--output-json",
        "{sample_dir}/fid_mmd_metrics_real_imagenet256_1024.json",
        "--batch-size",
        str(int(batch_size)),
        "--num-workers",
        str(int(num_workers)),
        "--device",
        str(device),
        "--sqrt-device",
        str(sqrt_device),
        "--mmd-device",
        str(mmd_device),
    ]
    if ref_features is not None and str(ref_features):
        parts.extend(["--ref-features", str(ref_features), "--mmd-max-ref", str(int(mmd_max_ref))])
    if allow_tf32:
        parts.append("--allow-tf32")
    # No shell quoting is needed here: this entire string is passed as one
    # argument to eval_flux2vae_imagespace.py, which later runs it with shell=True
    # after filling {images_dir}/{sample_dir}.
    return " ".join(parts)


def read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compact_record(scale: float, sample_dir: Path, record: Optional[Dict[str, Any]], returncode: int, elapsed_sec: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "scale": float(scale),
        "scale_slug": scale_slug(scale),
        "sample_dir": str(sample_dir),
        "returncode": int(returncode),
        "elapsed_sec": float(elapsed_sec),
        "status": "ok" if returncode == 0 and record else "failed",
    }
    if record:
        for key in [
            "image_eval_status",
            "fid_status",
            "fid",
            "mmd2",
            "kid",
            "kid_x1000",
            "num_images",
            "sample_space_mean",
            "sample_space_std",
            "sample_space_rms",
            "sample_space_absmax",
            "decode_space_mean",
            "decode_space_std",
            "decode_space_rms",
            "decode_space_absmax",
            "reference_raw_flux_std",
            "decode_space_std_over_reference_raw_flux_std",
            "decode_space_rms_over_reference_raw_flux_rms",
            "decode_elapsed_sec",
            "inner_fid_elapsed_sec",
            "metrics_elapsed_sec",
            "decode_cuda_peak_memory_mb",
            "images_dir",
            "decode_summary",
            "inner_fid_command",
        ]:
            if key in record:
                out[key] = record[key]
    return out


def markdown_summary(payload: Dict[str, Any]) -> str:
    rows = payload.get("results", [])
    lines: List[str] = []
    lines.append("# FLUX.2 VAE decode-scale sweep")
    lines.append("")
    lines.append(f"- created_at_utc: `{payload.get('created_at_utc')}`")
    lines.append(f"- source_sample_latents: `{payload.get('source_sample_latents')}`")
    lines.append(f"- output_dir: `{payload.get('output_dir')}`")
    lines.append(f"- max_images: `{payload.get('max_images')}`")
    lines.append(f"- pre_decode_stats_path: `{payload.get('pre_decode_stats_path')}`")
    lines.append("")
    lines.append("| scale | status | FID ↓ | MMD2/KID ↓ | sample std | decode std | decode/ref std | images |")
    lines.append("|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        def fmt(key: str, nd: int = 6) -> str:
            val = r.get(key)
            if val is None:
                return ""
            if isinstance(val, float):
                return f"{val:.{nd}f}"
            return str(val)

        lines.append(
            "| "
            + " | ".join(
                [
                    fmt("scale", 4),
                    str(r.get("status", "")),
                    fmt("fid", 4),
                    fmt("mmd2", 6),
                    fmt("sample_space_std", 6),
                    fmt("decode_space_std", 6),
                    fmt("decode_space_std_over_reference_raw_flux_std", 6),
                    str(r.get("num_images", "")),
                ]
            )
            + " |"
        )
    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("fid") is not None]
    if ok_rows:
        best_fid = min(ok_rows, key=lambda r: float(r["fid"]))
        best_mmd = min(ok_rows, key=lambda r: float(r.get("mmd2", float("inf"))))
        lines.append("")
        lines.append(
            f"- best_fid_scale: `{best_fid.get('scale')}` "
            f"(FID `{float(best_fid['fid']):.6f}`, decode/ref std "
            f"`{float(best_fid.get('decode_space_std_over_reference_raw_flux_std', float('nan'))):.6f}`)"
        )
        lines.append(
            f"- best_mmd_scale: `{best_mmd.get('scale')}` "
            f"(MMD2 `{float(best_mmd.get('mmd2')):.6f}`)"
        )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-sample-dir", default="", help="directory containing sample_latents.safetensors")
    p.add_argument("--source-latents", default="", help="explicit sample_latents.safetensors path; overrides --source-sample-dir")
    p.add_argument("--latents-name", default="sample_latents.safetensors")
    p.add_argument("--output-dir", required=True, help="sweep output directory; per-scale dirs are created below it")
    p.add_argument("--step", type=int, default=1000)
    p.add_argument("--scales", default="0.45,0.55,0.64,0.70,0.80,1.0")
    p.add_argument("--pre-decode-shift", type=float, default=0.0)
    p.add_argument("--pre-decode-stats-path", required=True, help="channel mean/std stats for inverse normalized FLUX decode")
    p.add_argument("--max-images", type=int, default=1024)
    p.add_argument("--decode-batch-size", type=int, default=8)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dtype", default="fp32", choices=["fp32", "bf16", "fp16"])
    p.add_argument("--repo-id", default="diffusers/FLUX.2-dev-bnb-4bit")
    p.add_argument("--subfolder", default="vae")
    p.add_argument("--vae-class", default="AutoencoderKLFlux2")
    p.add_argument("--revision", default="")
    p.add_argument("--local-files-only", action="store_true", help="pass --local-files-only to FLUX VAE loader")
    p.add_argument("--save-grid", action="store_true")
    p.add_argument("--decode-manifest-mode", default="first_last", choices=["full", "first_last", "none"])
    p.add_argument("--eval-script", default=str(DEFAULT_EVAL_SCRIPT))
    p.add_argument("--metrics-script", default=str(DEFAULT_METRICS_SCRIPT))
    p.add_argument("--ref-stats", default=str(DEFAULT_REF_STATS))
    p.add_argument("--ref-features", default=str(DEFAULT_REF_FEATURES))
    p.add_argument("--metrics-batch-size", type=int, default=64)
    p.add_argument("--metrics-num-workers", type=int, default=4)
    p.add_argument("--sqrt-device", default="cuda:0")
    p.add_argument("--mmd-device", default="cuda:0")
    p.add_argument("--mmd-max-ref", type=int, default=8192)
    p.add_argument("--allow-metrics-tf32", action="store_true")
    p.add_argument("--fid-timeout-sec", type=int, default=7200)
    p.add_argument("--stop-on-fail", action="store_true")
    p.add_argument("--overwrite-images", action="store_true", help="remove an existing per-scale decoded image dir before running")
    return p


def main() -> int:
    args = build_parser().parse_args()
    started = time.perf_counter()
    scales = parse_scales(args.scales)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_latents = Path(args.source_latents).resolve() if args.source_latents else Path(args.source_sample_dir).resolve() / args.latents_name
    if not source_latents.exists():
        raise FileNotFoundError(f"source sample latents not found: {source_latents}")

    eval_script = Path(args.eval_script).resolve()
    metrics_script = Path(args.metrics_script).resolve()
    ref_stats = Path(args.ref_stats).resolve()
    ref_features = Path(args.ref_features).resolve() if args.ref_features else None
    pre_decode_stats = Path(args.pre_decode_stats_path).resolve()
    for required in [eval_script, metrics_script, ref_stats, pre_decode_stats]:
        if not required.exists():
            raise FileNotFoundError(required)
    if ref_features is not None and not ref_features.exists():
        raise FileNotFoundError(ref_features)

    inner_template = build_inner_metrics_template(
        metrics_script=metrics_script,
        ref_stats=ref_stats,
        ref_features=ref_features,
        batch_size=int(args.metrics_batch_size),
        num_workers=int(args.metrics_num_workers),
        device=str(args.device),
        sqrt_device=str(args.sqrt_device),
        mmd_device=str(args.mmd_device),
        mmd_max_ref=int(args.mmd_max_ref),
        allow_tf32=bool(args.allow_metrics_tf32),
    )

    payload: Dict[str, Any] = {
        "status": "running",
        "created_at_utc": utc_now_iso(),
        "source_sample_latents": str(source_latents),
        "source_sample_latents_size_bytes": int(source_latents.stat().st_size),
        "output_dir": str(output_dir),
        "step": int(args.step),
        "scales": [float(s) for s in scales],
        "pre_decode_shift": float(args.pre_decode_shift),
        "pre_decode_stats_path": str(pre_decode_stats),
        "max_images": int(args.max_images) if args.max_images is not None else None,
        "eval_script": str(eval_script),
        "metrics_script": str(metrics_script),
        "ref_stats": str(ref_stats),
        "ref_features": str(ref_features) if ref_features is not None else "",
        "inner_metrics_template": inner_template,
        "results": [],
    }

    (output_dir / "scale_sweep_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    failures = 0
    for scale in scales:
        slug = scale_slug(scale)
        sample_dir = output_dir / slug / f"step_{int(args.step):08d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        link_mode = ensure_sample_latents_link(source_latents, sample_dir / args.latents_name)
        images_subdir = f"decoded_flux2vae_png_scale_sweep_{slug}"
        images_dir = sample_dir / images_subdir
        if args.overwrite_images and images_dir.exists():
            shutil.rmtree(images_dir)

        cmd = [
            sys.executable,
            str(eval_script),
            "--sample-dir",
            str(sample_dir),
            "--output-dir",
            str(output_dir),
            "--step",
            str(int(args.step)),
            "--latents-name",
            args.latents_name,
            "--images-subdir",
            images_subdir,
            "--device",
            str(args.device),
            "--batch-size",
            str(int(args.decode_batch_size)),
            "--model-dtype",
            str(args.model_dtype),
            "--repo-id",
            str(args.repo_id),
            "--subfolder",
            str(args.subfolder),
            "--vae-class",
            str(args.vae_class),
            "--fid-timeout-sec",
            str(int(args.fid_timeout_sec)),
            "--pre-decode-scale",
            str(float(scale)),
            "--pre-decode-shift",
            str(float(args.pre_decode_shift)),
            "--pre-decode-stats-path",
            str(pre_decode_stats),
            "--decode-manifest-mode",
            str(args.decode_manifest_mode),
            "--fid-command-template",
            inner_template,
        ]
        if args.revision:
            cmd.extend(["--revision", str(args.revision)])
        if args.local_files_only:
            cmd.append("--local-files-only")
        if args.max_images is not None:
            cmd.extend(["--max-images", str(int(args.max_images))])
        if args.save_grid:
            cmd.append("--save-grid")

        command_path = sample_dir / "scale_sweep_eval_command.json"
        command_path.write_text(
            json.dumps(
                {
                    "created_at_utc": utc_now_iso(),
                    "scale": float(scale),
                    "link_mode": link_mode,
                    "cmd": cmd,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"[scale {scale:g}] start sample_dir={sample_dir} link={link_mode}", flush=True)
        scale_started = time.perf_counter()
        proc = subprocess.run(cmd, text=True, capture_output=True)
        scale_elapsed = time.perf_counter() - scale_started
        (sample_dir / "scale_sweep_eval_stdout.txt").write_text(proc.stdout or "", encoding="utf-8")
        (sample_dir / "scale_sweep_eval_stderr.txt").write_text(proc.stderr or "", encoding="utf-8")
        record = read_json_if_exists(sample_dir / "flux2vae_imagespace_eval_record.json")
        entry = compact_record(float(scale), sample_dir, record, proc.returncode, scale_elapsed)
        entry["link_mode"] = link_mode
        payload["results"].append(entry)
        (output_dir / "scale_sweep_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (output_dir / "scale_sweep_summary.md").write_text(markdown_summary(payload), encoding="utf-8")
        if proc.returncode != 0:
            failures += 1
            print(f"[scale {scale:g}] FAILED returncode={proc.returncode} elapsed={scale_elapsed:.1f}s", flush=True)
            if proc.stderr:
                print(proc.stderr[-2000:], file=sys.stderr, flush=True)
            if args.stop_on_fail:
                break
        else:
            print(
                f"[scale {scale:g}] ok fid={entry.get('fid')} mmd2={entry.get('mmd2')} "
                f"decode/ref={entry.get('decode_space_std_over_reference_raw_flux_std')} "
                f"elapsed={scale_elapsed:.1f}s",
                flush=True,
            )

    payload["status"] = "ok" if failures == 0 else "failed"
    payload["num_failures"] = int(failures)
    payload["elapsed_sec"] = time.perf_counter() - started
    ok_rows = [r for r in payload["results"] if r.get("status") == "ok" and r.get("fid") is not None]
    if ok_rows:
        best_fid = min(ok_rows, key=lambda r: float(r["fid"]))
        best_mmd = min(ok_rows, key=lambda r: float(r.get("mmd2", float("inf"))))
        payload["best_fid"] = best_fid
        payload["best_mmd2"] = best_mmd

    (output_dir / "scale_sweep_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "scale_sweep_summary.md").write_text(markdown_summary(payload), encoding="utf-8")
    print(json.dumps({k: payload.get(k) for k in ["status", "num_failures", "elapsed_sec", "best_fid", "best_mmd2"]}, sort_keys=True), flush=True)
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
