#!/usr/bin/env python3
"""
Build PAE_DINOv2L_d32 latent shards from Hugging Face ImageNet-1k.

This script is intentionally smoke-first. It supports small max-sample runs and
writes safetensors shards compatible with external/PAE/pae_with_generator/
dataset/img_latent_dataset.py:

  - latents:      [N, C, H, W]
  - latents_flip: [N, C, H, W]
  - labels:       [N]

Images are center-cropped with the ADM center-crop recipe and converted to
[0,1] tensors. PAE performs encoder normalization internally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file
from torchvision import transforms
import yaml


def find_repo_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "external" / "PAE" / "pae_with_generator" / "tokenizer" / "pae.py").exists():
            return p
    # script lives under experiments/.../scripts, so fallback to known layout
    return Path("/workspace/PDM")


REPO_ROOT = find_repo_root(Path(__file__).resolve())
PAE_ROOT = REPO_ROOT / "external" / "PAE" / "pae_with_generator"
if str(PAE_ROOT) not in sys.path:
    sys.path.insert(0, str(PAE_ROOT))

# Import tokenizer.pae lazily in load_pae_model so --access-check-only works even
# before the local transformers version supports Dinov2WithRegistersModel.


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """ADM center crop used by the original PAE extraction script."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.Resampling.BOX,
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.Resampling.BICUBIC,
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def resolve_path_maybe_relative(path_value: str, base: Path) -> str:
    p = Path(path_value)
    if p.is_absolute():
        return str(p)
    candidate = base / p
    return str(candidate) if candidate.exists() else path_value


def load_pae_params(pae_config_path: Path) -> Dict[str, Any]:
    with pae_config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    params = dict(cfg["stage_1"].get("params", {}))
    # The upstream config uses repo-relative decoder paths such as configs/decoder/ViTL.
    if "decoder_config_path" in params:
        params["decoder_config_path"] = resolve_path_maybe_relative(params["decoder_config_path"], PAE_ROOT)
    return params


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model", "ema", "state_dict", "module"):
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
        # Some checkpoints are state_dicts directly.
        if all(isinstance(k, str) for k in ckpt.keys()) and any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise TypeError(f"Could not find a model state_dict in checkpoint object of type {type(ckpt)}")


def clean_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    prefixes = ("module.", "_orig_mod.", "model.")
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
                    changed = True
        cleaned[new_key] = value
    return cleaned


def dtype_from_name(name: str) -> torch.dtype:
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype: {name}")
    return table[name]


def load_pae_model(
    pae_config_path: Path,
    pae_ckpt_path: Path,
    device: torch.device,
    model_dtype: torch.dtype,
) -> torch.nn.Module:
    from tokenizer.pae import PAE

    params = load_pae_params(pae_config_path)
    print("PAE params:", json.dumps(params, indent=2, sort_keys=True))
    print(f"Instantiating PAE from config: {pae_config_path}")
    model = PAE(**params)
    print(f"Loading PAE checkpoint: {pae_ckpt_path}")
    ckpt = torch.load(str(pae_ckpt_path), map_location="cpu", weights_only=False)
    state_dict = clean_state_dict_keys(extract_state_dict(ckpt))
    msg = model.load_state_dict(state_dict, strict=False)
    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)
    print(f"load_state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("missing_first20=", missing[:20])
    if unexpected:
        print("unexpected_first20=", unexpected[:20])
    del ckpt, state_dict
    model = model.to(device=device, dtype=model_dtype).eval()
    return model


def image_to_tensor(image: Image.Image, image_size: int, hflip: bool = False) -> torch.Tensor:
    image = image.convert("RGB")
    ops: List[Any] = [
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
    ]
    if hflip:
        ops.append(transforms.functional.hflip)
    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)(image)


def encode_batch(
    model: torch.nn.Module,
    x: torch.Tensor,
    device: torch.device,
    model_dtype: torch.dtype,
    save_dtype: torch.dtype,
) -> torch.Tensor:
    x = x.to(device=device, non_blocking=True)
    use_autocast = device.type == "cuda" and model_dtype in (torch.bfloat16, torch.float16)
    autocast_dtype = model_dtype if use_autocast else torch.float32
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=use_autocast):
            z = model.encode(x)
    return z.detach().to(dtype=save_dtype, device="cpu").contiguous()


def verify_safetensors_file(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path), "ok": False, "keys": {}}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in ("latents", "latents_flip", "labels"):
            tensor_slice = f.get_slice(key)
            info["keys"][key] = {"shape": list(tensor_slice.get_shape()), "dtype": str(tensor_slice.get_dtype())}
    n = info["keys"]["labels"]["shape"][0]
    ok = n > 0 and info["keys"]["latents"]["shape"][0] == n and info["keys"]["latents_flip"]["shape"][0] == n
    ok = ok and info["keys"]["latents"]["shape"] == info["keys"]["latents_flip"]["shape"]
    info["ok"] = bool(ok)
    return info


@dataclass
class ShardStats:
    shard_index: int
    path: str
    num_samples: int
    latents_shape: List[int]
    latents_dtype: str
    labels_min: int
    labels_max: int
    save_seconds: float
    verify_ok: bool


def save_shard(
    output_dir: Path,
    shard_index: int,
    latents: List[torch.Tensor],
    latents_flip: List[torch.Tensor],
    labels: List[torch.Tensor],
    metadata: Dict[str, str],
) -> ShardStats:
    output_dir.mkdir(parents=True, exist_ok=True)
    z = torch.cat(latents, dim=0).contiguous()
    zf = torch.cat(latents_flip, dim=0).contiguous()
    y = torch.cat(labels, dim=0).to(dtype=torch.long).contiguous()
    assert z.shape == zf.shape, (z.shape, zf.shape)
    assert z.shape[0] == y.shape[0], (z.shape, y.shape)
    path = output_dir / f"latents_rank00_shard{shard_index:03d}.safetensors"
    save_dict = {"latents": z, "latents_flip": zf, "labels": y}
    meta = dict(metadata)
    meta.update({"num_samples": str(y.shape[0]), "latent_dtype": str(z.dtype), "latent_shape": str(tuple(z.shape))})
    t0 = time.time()
    save_file(save_dict, str(path), metadata=meta)
    save_seconds = time.time() - t0
    verify = verify_safetensors_file(path)
    print("saved shard:", json.dumps(verify, indent=2))
    return ShardStats(
        shard_index=shard_index,
        path=str(path),
        num_samples=int(y.shape[0]),
        latents_shape=list(z.shape),
        latents_dtype=str(z.dtype),
        labels_min=int(y.min().item()),
        labels_max=int(y.max().item()),
        save_seconds=save_seconds,
        verify_ok=bool(verify["ok"]),
    )


def resolve_hf_token(token_env: str):
    token = os.environ.get(token_env) or None
    if token:
        return token, "env"
    try:
        from huggingface_hub import get_token

        token = get_token()
    except Exception:
        token = None
    if token:
        return token, "hf_cache"
    return None, "none"


def load_hf_dataset(dataset: str, split: str, streaming: bool, token_env: str):
    from datasets import load_dataset

    token, token_source = resolve_hf_token(token_env)
    kwargs = {"split": split, "streaming": streaming}
    if token:
        kwargs["token"] = token
    print(
        f"Loading HF dataset={dataset!r} split={split!r} streaming={streaming} "
        f"token_source={token_source} token_set={bool(token)}"
    )
    return load_dataset(dataset, **kwargs)


def access_check(dataset: str, split: str, streaming: bool, token_env: str) -> None:
    ds = load_hf_dataset(dataset, split, streaming, token_env)
    sample = next(iter(ds))
    image = sample.get("image")
    label = sample.get("label")
    print("ACCESS_CHECK_OK")
    print("sample_keys", list(sample.keys()))
    print("image_type", type(image).__name__, "size", getattr(image, "size", None), "mode", getattr(image, "mode", None))
    print("label", label, type(label).__name__)


def run_random_self_test(model: torch.nn.Module, args: argparse.Namespace, device: torch.device, model_dtype: torch.dtype, save_dtype: torch.dtype) -> None:
    print("Running random encode self-test")
    x = torch.rand(args.batch_size, 3, args.image_size, args.image_size)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    z = encode_batch(model, x, device, model_dtype, save_dtype)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    print("RANDOM_SELF_TEST_OK", {"input_shape": list(x.shape), "latent_shape": list(z.shape), "latent_dtype": str(z.dtype), "cuda_peak_mb": peak_mb})


def build_latents(args: argparse.Namespace) -> Dict[str, Any]:
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA is not available; pass --allow-cpu only for debugging.")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = dtype_from_name(args.model_dtype)
    save_dtype = dtype_from_name(args.save_dtype)

    model = load_pae_model(Path(args.pae_config), Path(args.pae_ckpt), device, model_dtype)

    if args.random_self_test:
        run_random_self_test(model, args, device, model_dtype, save_dtype)
        if args.self_test_only:
            return {"mode": "self_test_only", "ok": True}

    if args.synthetic_save_test:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        latents: List[torch.Tensor] = []
        latents_flip: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        total = int(args.max_samples or args.shard_size or args.batch_size)
        done = 0
        while done < total:
            n = min(args.batch_size, total - done)
            x = torch.rand(n, 3, args.image_size, args.image_size)
            xf = torch.flip(x, dims=[-1])
            y = torch.arange(done, done + n, dtype=torch.long) % 1000
            latents.append(encode_batch(model, x, device, model_dtype, save_dtype))
            latents_flip.append(encode_batch(model, xf, device, model_dtype, save_dtype))
            labels.append(y)
            done += n
            print(f"synthetic_encoded={done}/{total}")
        meta = {
            "dataset": "synthetic_random",
            "split": "none",
            "image_size": str(args.image_size),
            "pae_config": str(args.pae_config),
            "pae_ckpt": str(args.pae_ckpt),
            "model_dtype": args.model_dtype,
            "save_dtype": args.save_dtype,
            "created_by": "build_pae_latents_from_hf_imagenet.py --synthetic-save-test",
        }
        stats = save_shard(output_dir, 0, latents, latents_flip, labels, meta)
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
        result = {
            "ok": bool(stats.verify_ok),
            "mode": "synthetic_save_test",
            "output_dir": str(output_dir),
            "encoded": done,
            "num_shards": 1,
            "shards": [asdict(stats)],
            "elapsed_seconds": time.time() - started,
            "cuda_peak_mb": peak_mb,
        }
        result_path = output_dir / "build_summary.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("SYNTHETIC_SAVE_SUMMARY", json.dumps(result, indent=2))
        print(f"summary_path={result_path}")
        return result

    ds = load_hf_dataset(args.dataset, args.split, args.streaming, args.hf_token_env)
    if args.skip_samples:
        # IterableDataset supports skip; regular datasets support select fallback.
        if hasattr(ds, "skip"):
            ds = ds.skip(args.skip_samples)
        else:
            ds = ds.select(range(args.skip_samples, len(ds)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset": args.dataset,
        "split": args.split,
        "image_size": str(args.image_size),
        "pae_config": str(args.pae_config),
        "pae_ckpt": str(args.pae_ckpt),
        "model_dtype": args.model_dtype,
        "save_dtype": args.save_dtype,
        "created_by": "build_pae_latents_from_hf_imagenet.py",
    }
    latents: List[torch.Tensor] = []
    latents_flip: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    shard_stats: List[ShardStats] = []
    batch_x: List[torch.Tensor] = []
    batch_xf: List[torch.Tensor] = []
    batch_y: List[int] = []
    seen = 0
    encoded = 0
    shard_index = 0
    started = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def flush_batch() -> None:
        nonlocal batch_x, batch_xf, batch_y, encoded, latents, latents_flip, labels, shard_index
        if not batch_x:
            return
        x = torch.stack(batch_x, dim=0)
        xf = torch.stack(batch_xf, dim=0)
        y = torch.tensor(batch_y, dtype=torch.long)
        z = encode_batch(model, x, device, model_dtype, save_dtype)
        zf = encode_batch(model, xf, device, model_dtype, save_dtype)
        latents.append(z)
        latents_flip.append(zf)
        labels.append(y)
        encoded += int(y.shape[0])
        print(f"encoded={encoded} batch={y.shape[0]} latent_shape={tuple(z.shape)} dtype={z.dtype}")
        batch_x, batch_xf, batch_y = [], [], []
        # Save as many full shards as are accumulated.
        cur_n = sum(t.shape[0] for t in labels)
        if cur_n >= args.shard_size:
            stats = save_shard(output_dir, shard_index, latents, latents_flip, labels, meta)
            shard_stats.append(stats)
            shard_index += 1
            latents, latents_flip, labels = [], [], []

    for sample in ds:
        if args.max_samples is not None and seen >= args.max_samples:
            break
        image = sample[args.image_key]
        label = int(sample[args.label_key])
        batch_x.append(image_to_tensor(image, args.image_size, hflip=False))
        batch_xf.append(image_to_tensor(image, args.image_size, hflip=True))
        batch_y.append(label)
        seen += 1
        if len(batch_x) >= args.batch_size:
            flush_batch()

    flush_batch()
    if labels:
        stats = save_shard(output_dir, shard_index, latents, latents_flip, labels, meta)
        shard_stats.append(stats)

    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    result = {
        "ok": all(s.verify_ok for s in shard_stats) and encoded > 0,
        "dataset": args.dataset,
        "split": args.split,
        "output_dir": str(output_dir),
        "seen": seen,
        "encoded": encoded,
        "num_shards": len(shard_stats),
        "shards": [asdict(s) for s in shard_stats],
        "elapsed_seconds": time.time() - started,
        "cuda_peak_mb": peak_mb,
    }
    result_path = output_dir / "build_summary.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("BUILD_SUMMARY", json.dumps(result, indent=2))
    print(f"summary_path={result_path}")
    return result


def hard_finish(exit_code: int = 0) -> None:
    """Flush logs and bypass Python finalizers that can hang/crash datasets streaming cleanup."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="ILSVRC/imagenet-1k")
    p.add_argument("--split", default="train")
    p.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hf-token-env", default="HF_TOKEN", help="Environment variable containing an HF token; no token is printed.")
    p.add_argument("--pae-config", default=str(PAE_ROOT / "configs" / "pae" / "PAE_DINOv2L_d32.yaml"))
    p.add_argument("--pae-ckpt", default="/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt")
    p.add_argument("--output-dir", default="/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/imagenet256_train_smoke64")
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--shard-size", type=int, default=64)
    p.add_argument("--max-samples", type=int, default=64)
    p.add_argument("--skip-samples", type=int, default=0)
    p.add_argument("--image-key", default="image")
    p.add_argument("--label-key", default="label")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model-dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    p.add_argument("--save-dtype", default="bf16", choices=["bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--synthetic-save-test", action="store_true", help="No HF data: encode random images and save one/more safetensors shards to validate the PAE+safetensors+ImgLatentDataset path.")
    p.add_argument("--access-check-only", action="store_true")
    p.add_argument("--random-self-test", action="store_true", help="Encode random tensors after loading PAE before using HF data.")
    p.add_argument("--self-test-only", action="store_true", help="With --random-self-test, stop before HF dataset iteration.")
    p.add_argument("--hard-exit", action="store_true", help="Call os._exit(0) after successful completion to avoid datasets streaming finalizer hangs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("ARGS", json.dumps(vars(args), indent=2, sort_keys=True))
    if args.access_check_only:
        access_check(args.dataset, args.split, args.streaming, args.hf_token_env)
        if args.hard_exit:
            hard_finish(0)
        return
    result = build_latents(args)
    if not result.get("ok"):
        if args.hard_exit:
            hard_finish(1)
        raise SystemExit(1)
    if args.hard_exit:
        hard_finish(0)


if __name__ == "__main__":
    main()
