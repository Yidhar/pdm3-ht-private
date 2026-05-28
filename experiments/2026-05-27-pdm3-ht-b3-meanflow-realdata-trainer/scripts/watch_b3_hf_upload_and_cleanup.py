#!/usr/bin/env python3
"""Upload B3 MeanFlow artifacts to Hugging Face and prune local checkpoints.

This watcher is deliberately separate from the trainer so it can be enabled for
an already-running job.  It follows the artifact routing used in this project:

* GitHub only receives code/config/docs.
* Hugging Face receives model/eval artifacts.
* Raw ImageNet/real-reference images are not uploaded by default.

Default policy for the active H100 b96 run:

* archive full trainer checkpoints only every 10k steps;
* upload those archived checkpoints to
  ``LAXMAYDAY/pdm3-ht-model-artifacts``;
* delete uploaded archived checkpoints once they are no longer the live
  ``latest.pt`` target;
* delete non-archive local checkpoints once they are no longer the
  ``latest.pt`` target;
* keep the current latest checkpoint locally for fast resume.

The script is idempotent via a JSON state file.  If an upload fails, the local
checkpoint is kept and retried on the next poll.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str, **fields: Any) -> None:
    payload = " ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in fields.items())
    print(f"[{utc_now()}] {msg}" + (f" {payload}" if payload else ""), flush=True)


def load_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return dict(default)
    except json.JSONDecodeError as exc:
        backup = path.with_suffix(path.suffix + f".bad_{int(time.time())}")
        try:
            path.replace(backup)
        except Exception:
            pass
        log("state_json_corrupt_reset", path=str(path), error=str(exc), backup=str(backup))
        return dict(default)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def step_from_path(path: Path) -> int:
    m = re.search(r"step_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def parse_step_dir(path: Path) -> int:
    m = re.search(r"step_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def latest_train_step(metrics_jsonl: Path) -> int:
    last = 0
    try:
        # Avoid reading a multi-GB jsonl in pathological long runs.  Current file
        # is small, but tailing bytes keeps this watcher cheap.
        data = metrics_jsonl.read_bytes()
        if len(data) > 16 * 1024 * 1024:
            data = data[-16 * 1024 * 1024 :]
        for raw in data.splitlines():
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if obj.get("type") == "train_step":
                last = max(last, int(obj.get("step", 0) or 0))
    except FileNotFoundError:
        pass
    return last


def pid_alive(pid_file: Path) -> Tuple[Optional[int], bool]:
    try:
        text = pid_file.read_text(encoding="utf-8").strip()
        pid = int(text) if text else None
    except Exception:
        return None, False
    if not pid:
        return None, False
    try:
        os.kill(pid, 0)
        return pid, True
    except OSError:
        return pid, False


def resolve_latest_target(ckpt_dir: Path) -> Optional[Path]:
    latest = ckpt_dir / "latest.pt"
    if not latest.exists() and not latest.is_symlink():
        return None
    try:
        resolved = latest.resolve(strict=True)
        if resolved.exists():
            return resolved
    except FileNotFoundError:
        return None
    if latest.is_file():
        return latest
    return None


def checkpoint_paths(ckpt_dir: Path) -> List[Path]:
    return sorted(ckpt_dir.glob("step_*.pt"), key=step_from_path)


def ensure_hf_api() -> Any:
    from huggingface_hub import HfApi

    return HfApi()


def commit_info_dict(info: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in ("commit_url", "oid", "commit_message", "pr_url"):
        val = getattr(info, name, None)
        if val is not None:
            out[name] = str(val)
    return out


def remote_file_exists(api: Any, *, repo_id: str, repo_type: str, path_in_repo: str) -> bool:
    try:
        return bool(api.file_exists(repo_id=repo_id, filename=path_in_repo, repo_type=repo_type))
    except Exception:
        return False


@dataclass
class UploadConfig:
    repo_id: str
    repo_type: str
    remote_prefix: str
    archive_every: int
    checkpoint_start_step: int
    eval_start_step: int
    stop_step: int
    delete_after_upload: bool
    delete_non_archive: bool
    delete_latest_after_upload: bool
    upload_eval: bool
    upload_checkpoints: bool


def init_state(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "remote_prefix": args.remote_prefix,
        "checkpoint_uploads": {},
        "eval_uploads": {},
        "deleted_local_checkpoints": {},
        "events": [],
    }


def append_event(state: Dict[str, Any], event: Dict[str, Any], limit: int = 200) -> None:
    events = state.setdefault("events", [])
    events.append({"created_at_utc": utc_now(), **event})
    if len(events) > limit:
        del events[: len(events) - limit]


def step_key(step: int) -> str:
    return f"{int(step):08d}"


def upload_checkpoint(
    *,
    api: Any,
    cfg: UploadConfig,
    state: Dict[str, Any],
    state_path: Path,
    ckpt: Path,
    latest_target: Optional[Path],
) -> bool:
    step = step_from_path(ckpt)
    key = step_key(step)
    uploads = state.setdefault("checkpoint_uploads", {})
    prev = uploads.get(key, {})
    if prev.get("status") == "ok":
        return True

    remote_path = f"{cfg.remote_prefix.rstrip('/')}/checkpoints/{ckpt.name}"
    size = file_size(ckpt)
    latest = latest_target is not None and ckpt.resolve() == latest_target.resolve()
    log("checkpoint_upload_start", step=step, path=str(ckpt), size=size, remote_path=remote_path, latest=latest)
    uploads[key] = {
        "status": "running",
        "local_path": str(ckpt),
        "path_in_repo": remote_path,
        "size_bytes": size,
        "started_at_utc": utc_now(),
    }
    append_event(state, {"type": "checkpoint_upload_start", "step": step, "path": str(ckpt), "remote_path": remote_path})
    atomic_write_json(state_path, state)

    try:
        if remote_file_exists(api, repo_id=cfg.repo_id, repo_type=cfg.repo_type, path_in_repo=remote_path):
            info = {"remote_preexisted": True}
        else:
            commit = api.upload_file(
                path_or_fileobj=str(ckpt),
                path_in_repo=remote_path,
                repo_id=cfg.repo_id,
                repo_type=cfg.repo_type,
                commit_message=f"B3 MeanFlow b96 checkpoint step {step:08d}",
            )
            info = commit_info_dict(commit)
        uploads[key] = {
            "status": "ok",
            "local_path": str(ckpt),
            "path_in_repo": remote_path,
            "size_bytes": size,
            "completed_at_utc": utc_now(),
            **info,
        }
        append_event(state, {"type": "checkpoint_upload_ok", "step": step, "remote_path": remote_path, **info})
        atomic_write_json(state_path, state)
        log("checkpoint_upload_ok", step=step, remote_path=remote_path, **info)
        return True
    except Exception as exc:  # noqa: BLE001
        uploads[key] = {
            "status": "error",
            "local_path": str(ckpt),
            "path_in_repo": remote_path,
            "size_bytes": size,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at_utc": utc_now(),
        }
        append_event(state, {"type": "checkpoint_upload_error", "step": step, "error_type": type(exc).__name__, "error": str(exc)})
        atomic_write_json(state_path, state)
        log("checkpoint_upload_error", step=step, error_type=type(exc).__name__, error=str(exc))
        return False


def maybe_delete_checkpoint(
    *,
    state: Dict[str, Any],
    state_path: Path,
    ckpt: Path,
    reason: str,
    latest_target: Optional[Path],
    allow_latest: bool,
) -> bool:
    step = step_from_path(ckpt)
    key = step_key(step)
    if not ckpt.exists():
        return False
    if latest_target is not None:
        try:
            if ckpt.resolve() == latest_target.resolve() and not allow_latest:
                log("checkpoint_delete_skip_latest", step=step, path=str(ckpt), reason=reason)
                return False
        except FileNotFoundError:
            pass
    size = file_size(ckpt)
    try:
        ckpt.unlink()
    except FileNotFoundError:
        return False
    state.setdefault("deleted_local_checkpoints", {})[key] = {
        "path": str(ckpt),
        "size_bytes": size,
        "reason": reason,
        "deleted_at_utc": utc_now(),
    }
    append_event(state, {"type": "checkpoint_deleted", "step": step, "path": str(ckpt), "size_bytes": size, "reason": reason})
    atomic_write_json(state_path, state)
    log("checkpoint_deleted", step=step, path=str(ckpt), size=size, reason=reason)
    return True


def upload_eval_dir(
    *,
    api: Any,
    cfg: UploadConfig,
    state: Dict[str, Any],
    state_path: Path,
    eval_dir: Path,
) -> bool:
    step = parse_step_dir(eval_dir)
    key = step_key(step)
    eval_uploads = state.setdefault("eval_uploads", {})
    prev = eval_uploads.get(key, {})
    if prev.get("status") == "ok":
        return True

    # Avoid uploading raw real ImageNet PNGs/NPZ arrays.  Generated grids/images,
    # sample latents, and metric JSON/MD files are sufficient for handoff.
    allow_patterns = [
        "eval_record.json",
        "sample_latents.safetensors",
        "compact_metrics.json",
        "compact_metrics.md",
        "decoded_samples.npz",
        "images/generated_grid.png",
        "images/generated/*.png",
        "inception_eval/inception_metrics.json",
        "inception_eval/inception_metrics.md",
        "inception_eval/images/generated_grid.png",
        "inception_eval/images/generated/*.png",
        "inception_eval_*/inception_metrics.json",
        "inception_eval_*/inception_metrics.md",
        "inception_eval_*/images/generated_grid.png",
        "inception_eval_*/images/generated/*.png",
        "*.log",
        "*.txt",
    ]
    ignore_patterns = [
        "**/real*",
        "**/*real*",
        "**/inception_features.npz",
        "**/decoded_and_real*.npz",
        "**/real_ref/**",
        "**/real_imagenet256/**",
    ]

    files: List[str] = []
    for p in eval_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(eval_dir).as_posix()
        if any(fnmatch.fnmatch(rel, pat) for pat in ignore_patterns):
            continue
        if any(fnmatch.fnmatch(rel, pat) for pat in allow_patterns):
            files.append(rel)
    if not files:
        return False

    remote_path = f"{cfg.remote_prefix.rstrip('/')}/step_{step:08d}"
    log("eval_upload_start", step=step, eval_dir=str(eval_dir), remote_path=remote_path, files=len(files))
    eval_uploads[key] = {
        "status": "running",
        "local_dir": str(eval_dir),
        "path_in_repo": remote_path,
        "file_count": len(files),
        "started_at_utc": utc_now(),
    }
    append_event(state, {"type": "eval_upload_start", "step": step, "local_dir": str(eval_dir), "remote_path": remote_path, "file_count": len(files)})
    atomic_write_json(state_path, state)
    try:
        commit = api.upload_folder(
            folder_path=str(eval_dir),
            path_in_repo=remote_path,
            repo_id=cfg.repo_id,
            repo_type=cfg.repo_type,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            commit_message=f"B3 MeanFlow b96 eval artifacts step {step:08d}",
        )
        info = commit_info_dict(commit)
        eval_uploads[key] = {
            "status": "ok",
            "local_dir": str(eval_dir),
            "path_in_repo": remote_path,
            "file_count": len(files),
            "completed_at_utc": utc_now(),
            **info,
        }
        append_event(state, {"type": "eval_upload_ok", "step": step, "remote_path": remote_path, **info})
        atomic_write_json(state_path, state)
        log("eval_upload_ok", step=step, remote_path=remote_path, **info)
        return True
    except Exception as exc:  # noqa: BLE001
        eval_uploads[key] = {
            "status": "error",
            "local_dir": str(eval_dir),
            "path_in_repo": remote_path,
            "file_count": len(files),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "failed_at_utc": utc_now(),
        }
        append_event(state, {"type": "eval_upload_error", "step": step, "error_type": type(exc).__name__, "error": str(exc)})
        atomic_write_json(state_path, state)
        log("eval_upload_error", step=step, error_type=type(exc).__name__, error=str(exc))
        return False


def checkpoint_upload_ok(state: Dict[str, Any], step: int) -> bool:
    return state.get("checkpoint_uploads", {}).get(step_key(step), {}).get("status") == "ok"


def process_checkpoints(
    *,
    api: Any,
    cfg: UploadConfig,
    state: Dict[str, Any],
    state_path: Path,
    ckpt_dir: Path,
) -> None:
    latest_target = resolve_latest_target(ckpt_dir)
    latest_step = step_from_path(latest_target) if latest_target is not None else -1
    ckpts = checkpoint_paths(ckpt_dir)

    # First prune superseded non-archive checkpoints.  This keeps disk pressure
    # low even if a large archive upload is still in progress for a while.
    if cfg.delete_non_archive:
        for ckpt in ckpts:
            step = step_from_path(ckpt)
            if latest_target is not None:
                try:
                    if ckpt.resolve() == latest_target.resolve():
                        continue
                except FileNotFoundError:
                    continue
            is_archive = cfg.archive_every > 0 and step >= cfg.checkpoint_start_step and step % cfg.archive_every == 0
            if (not is_archive) and (latest_step < 0 or step < latest_step):
                maybe_delete_checkpoint(
                    state=state,
                    state_path=state_path,
                    ckpt=ckpt,
                    reason="non_archive_checkpoint_superseded_by_latest",
                    latest_target=latest_target,
                    allow_latest=False,
                )

    # Then upload archive candidates, because deleting archive checkpoints is
    # only allowed after confirmed HF upload.
    if cfg.upload_checkpoints:
        for ckpt in checkpoint_paths(ckpt_dir):
            step = step_from_path(ckpt)
            if step < cfg.checkpoint_start_step:
                continue
            if cfg.archive_every <= 0 or step % cfg.archive_every != 0:
                continue
            ok = upload_checkpoint(api=api, cfg=cfg, state=state, state_path=state_path, ckpt=ckpt, latest_target=latest_target)
            if ok and cfg.delete_after_upload:
                latest_target = resolve_latest_target(ckpt_dir)
                maybe_delete_checkpoint(
                    state=state,
                    state_path=state_path,
                    ckpt=ckpt,
                    reason="uploaded_archive_checkpoint",
                    latest_target=latest_target,
                    allow_latest=cfg.delete_latest_after_upload,
                )

    # Then prune local non-archive checkpoints.  These are frequent trainer
    # safety points and do not need HF archival once a newer local latest exists.
    if cfg.delete_non_archive:
        latest_target = resolve_latest_target(ckpt_dir)
        for ckpt in checkpoint_paths(ckpt_dir):
            step = step_from_path(ckpt)
            if latest_target is not None:
                try:
                    if ckpt.resolve() == latest_target.resolve():
                        continue
                except FileNotFoundError:
                    continue
            is_archive = cfg.archive_every > 0 and step >= cfg.checkpoint_start_step and step % cfg.archive_every == 0
            if not is_archive:
                if latest_step < 0 or step < latest_step:
                    maybe_delete_checkpoint(
                        state=state,
                        state_path=state_path,
                        ckpt=ckpt,
                        reason="non_archive_checkpoint_superseded_by_latest",
                        latest_target=latest_target,
                        allow_latest=False,
                    )
            elif cfg.delete_after_upload and checkpoint_upload_ok(state, step):
                maybe_delete_checkpoint(
                    state=state,
                    state_path=state_path,
                    ckpt=ckpt,
                    reason="uploaded_archive_checkpoint_superseded_by_latest",
                    latest_target=latest_target,
                    allow_latest=cfg.delete_latest_after_upload,
                )


def process_eval_dirs(
    *,
    api: Any,
    cfg: UploadConfig,
    state: Dict[str, Any],
    state_path: Path,
    eval_root: Path,
) -> None:
    if not cfg.upload_eval:
        return
    for eval_dir in sorted(eval_root.glob("step_*"), key=parse_step_dir):
        if not eval_dir.is_dir():
            continue
        step = parse_step_dir(eval_dir)
        if step < cfg.eval_start_step:
            continue
        # Wait until the CPU Inception watcher has written metrics.  The trainer
        # sample_latents alone is less useful and can be uploaded alongside the
        # metrics a few minutes later.
        has_metrics = any(p.name == "inception_metrics.json" for p in eval_dir.rglob("inception_metrics.json"))
        has_sample = (eval_dir / "sample_latents.safetensors").exists()
        if not (has_metrics or has_sample):
            continue
        if state.get("eval_uploads", {}).get(step_key(step), {}).get("status") == "ok":
            continue
        upload_eval_dir(api=api, cfg=cfg, state=state, state_path=state_path, eval_dir=eval_dir)


def once(args: argparse.Namespace) -> None:
    state_path = args.state_path
    state = load_json(state_path, init_state(args))
    state["last_poll_at_utc"] = utc_now()
    state["repo_id"] = args.repo_id
    state["repo_type"] = args.repo_type
    state["remote_prefix"] = args.remote_prefix

    cfg = UploadConfig(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        remote_prefix=args.remote_prefix,
        archive_every=args.archive_every,
        checkpoint_start_step=args.checkpoint_start_step,
        eval_start_step=args.eval_start_step,
        stop_step=args.stop_step,
        delete_after_upload=args.delete_after_upload,
        delete_non_archive=args.delete_non_archive,
        delete_latest_after_upload=args.delete_latest_after_upload,
        upload_eval=args.upload_eval,
        upload_checkpoints=args.upload_checkpoints,
    )

    if args.dry_run:
        latest_target = resolve_latest_target(args.ckpt_dir)
        ckpts = checkpoint_paths(args.ckpt_dir)
        evals = sorted(args.eval_root.glob("step_*"), key=parse_step_dir)
        log(
            "dry_run",
            latest_target=str(latest_target) if latest_target else None,
            checkpoints=[str(p.name) for p in ckpts],
            eval_dirs=[p.name for p in evals],
            state_path=str(state_path),
            cfg=cfg.__dict__,
        )
        return

    api = ensure_hf_api()
    process_checkpoints(api=api, cfg=cfg, state=state, state_path=state_path, ckpt_dir=args.ckpt_dir)
    process_eval_dirs(api=api, cfg=cfg, state=state, state_path=state_path, eval_root=args.eval_root)
    state["last_poll_completed_at_utc"] = utc_now()
    atomic_write_json(state_path, state)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--res", type=Path, default=Path("/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/results/fullcache_realdata_singleproc_template"))
    p.add_argument("--pid-file", type=Path, default=Path("/workspace/PDM/experiments/2026-05-27-pdm3-ht-b3-meanflow-realdata-trainer/fullcache_realdata.pid"))
    p.add_argument("--repo-id", default="LAXMAYDAY/pdm3-ht-model-artifacts")
    p.add_argument("--repo-type", default="model")
    p.add_argument("--remote-prefix", default="b3_meanflow_realdata/fullcache_b96")
    p.add_argument("--archive-every", type=int, default=10000)
    p.add_argument("--checkpoint-start-step", type=int, default=20000)
    p.add_argument("--eval-start-step", type=int, default=30000)
    p.add_argument("--stop-step", type=int, default=1070000)
    p.add_argument("--interval", type=int, default=300)
    p.add_argument("--state-path", type=Path, default=None)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-upload-checkpoints", dest="upload_checkpoints", action="store_false")
    p.add_argument("--no-upload-eval", dest="upload_eval", action="store_false")
    p.add_argument("--no-delete-after-upload", dest="delete_after_upload", action="store_false")
    p.add_argument("--no-delete-non-archive", dest="delete_non_archive", action="store_false")
    p.add_argument("--delete-latest-after-upload", action="store_true")
    p.set_defaults(upload_checkpoints=True, upload_eval=True, delete_after_upload=True, delete_non_archive=True)
    args = p.parse_args(argv)

    args.res = args.res.resolve()
    args.ckpt_dir = args.res / "checkpoints"
    args.eval_root = args.res / "eval"
    args.state_path = (args.state_path or (args.res / "hf_artifact_upload_state.json")).resolve()

    log(
        "watcher_start",
        res=str(args.res),
        repo_id=args.repo_id,
        remote_prefix=args.remote_prefix,
        archive_every=args.archive_every,
        checkpoint_start_step=args.checkpoint_start_step,
        eval_start_step=args.eval_start_step,
        stop_step=args.stop_step,
        interval=args.interval,
        dry_run=args.dry_run,
        once=args.once,
    )

    while True:
        try:
            once(args)
        except KeyboardInterrupt:
            raise
        except Exception as exc:  # noqa: BLE001
            log("watcher_poll_error", error_type=type(exc).__name__, error=str(exc))

        latest_step = latest_train_step(args.res / "metrics.jsonl")
        pid, alive = pid_alive(args.pid_file)
        log("watcher_poll_done", latest_step=latest_step, train_pid=pid, train_alive=alive)

        if args.once or args.dry_run:
            return 0
        if (not alive) and latest_step >= int(args.stop_step):
            log("watcher_stop_train_complete", latest_step=latest_step, stop_step=args.stop_step)
            return 0
        time.sleep(max(5, int(args.interval)))


if __name__ == "__main__":
    raise SystemExit(main())
