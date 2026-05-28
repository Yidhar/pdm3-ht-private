#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path('/workspace/PDM')
SRC_SCRIPT = REPO_ROOT / 'experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-build/scripts/build_pae_latents_from_hf_imagenet.py'
spec = importlib.util.spec_from_file_location('pae_latent_builder', SRC_SCRIPT)
assert spec and spec.loader, SRC_SCRIPT
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)  # type: ignore[union-attr]


def parse_ints(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def cuda_mb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0.0


def run_pair(model: torch.nn.Module, x: torch.Tensor, device: torch.device, model_dtype: torch.dtype, save_dtype: torch.dtype) -> Dict[str, Any]:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    z = builder.encode_batch(model, x, device, model_dtype, save_dtype)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    # Reuse x for the second pass. This intentionally measures encoder saturation only;
    # the real extraction sweep measures CPU/PIL flip and input pipeline separately.
    zf = builder.encode_batch(model, x, device, model_dtype, save_dtype)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t2 = time.perf_counter()
    n = int(x.shape[0])
    out = {
        'normal_sec': t1 - t0,
        'flip_sec': t2 - t1,
        'pair_sec': t2 - t0,
        'encoded_samples': n,
        'latent_shape': list(z.shape),
        'latent_dtype': str(z.dtype),
        'samples_per_sec_pair': n / (t2 - t0) if (t2 - t0) > 0 else 0.0,
        'pae_encode_ops_per_sec': (2 * n) / (t2 - t0) if (t2 - t0) > 0 else 0.0,
    }
    # Touch outputs so transfers cannot be optimized away.
    out['latent_checksum_head'] = float(z[: min(8, n)].float().mean().item()) + float(zf[: min(8, n)].float().mean().item())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description='PAE_DINOv2L_d32 encode-only high-batch saturation probe')
    p.add_argument('--pae-config', default=str(REPO_ROOT / 'external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml'))
    p.add_argument('--pae-ckpt', default='/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt')
    p.add_argument('--batch-sizes', default='1024,2048,3072,4096')
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--model-dtype', default='bf16')
    p.add_argument('--save-dtype', default='bf16')
    p.add_argument('--seed', type=int, default=20260527)
    p.add_argument('--output-json', default='/workspace/PDM/experiments/2026-05-27-pdm3-ht-imagenet1k-pae-latent-batch-saturation/results/encode_only_saturation_probe.json')
    args = p.parse_args()
    print('ARGS', json.dumps(vars(args), indent=2, sort_keys=True), flush=True)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model_dtype = builder.dtype_from_name(args.model_dtype)
    save_dtype = builder.dtype_from_name(args.save_dtype)
    batch_sizes = parse_ints(args.batch_sizes)
    total_mem_gib = torch.cuda.get_device_properties(device).total_memory / 1024**3

    started = time.perf_counter()
    print('Loading PAE once for encode-only probe', flush=True)
    model = builder.load_pae_model(Path(args.pae_config), Path(args.pae_ckpt), device, model_dtype)

    # One small warm-up to avoid charging one-time CUDA/autocast setup to the first timed large batch.
    print('WARMUP batch_size=128', flush=True)
    warm = torch.rand(128, 3, args.image_size, args.image_size, dtype=torch.float32)
    _ = run_pair(model, warm, device, model_dtype, save_dtype)
    del warm, _
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)

    results: List[Dict[str, Any]] = []
    for bs in batch_sizes:
        print(f'===== ENCODE_ONLY_BATCH_START batch_size={bs} =====', flush=True)
        result: Dict[str, Any] = {'batch_size': bs, 'status': 'PASS'}
        try:
            alloc_t0 = time.perf_counter()
            x = torch.rand(bs, 3, args.image_size, args.image_size, dtype=torch.float32)
            alloc_sec = time.perf_counter() - alloc_t0
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            pair = run_pair(model, x, device, model_dtype, save_dtype)
            peak_mb = cuda_mb(device)
            result.update(pair)
            result.update({
                'input_alloc_sec_cpu': alloc_sec,
                'cuda_peak_mb': peak_mb,
                'cuda_peak_gib': peak_mb / 1024.0,
                'cuda_peak_fraction': (peak_mb / 1024.0) / total_mem_gib,
                'cuda_total_memory_gib': total_mem_gib,
            })
            del x
        except Exception as exc:
            result['status'] = 'FAIL'
            result['error'] = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
            print(f'ENCODE_ONLY_BATCH_FAIL batch_size={bs} error={result["error"]}', flush=True)
            traceback.print_exc()
        finally:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.synchronize(device)
        print('ENCODE_ONLY_BATCH_RESULT', json.dumps(result, indent=2), flush=True)
        results.append(result)

    pass_results = [r for r in results if r.get('status') == 'PASS']
    best_pair = max(pass_results, key=lambda r: r.get('samples_per_sec_pair', 0.0)) if pass_results else None
    largest_pass = max(pass_results, key=lambda r: r.get('batch_size', 0)) if pass_results else None
    summary = {
        'status': 'PASS' if pass_results else 'FAIL',
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'mode': 'encode_only_synthetic_preprocessed_tensor_upper_bound',
        'note': 'Uses random preprocessed-shape tensors to isolate PAE encoder throughput. It excludes HF streaming, PIL decode/crop/flip, and safetensors writes.',
        'cuda_device_name': torch.cuda.get_device_name(device),
        'cuda_total_memory_gib': total_mem_gib,
        'model_dtype': args.model_dtype,
        'save_dtype': args.save_dtype,
        'batch_sizes': batch_sizes,
        'image_size': args.image_size,
        'results': results,
        'best_by_pair_samples_per_sec': best_pair,
        'largest_passing_batch': largest_pass,
        'elapsed_sec': time.perf_counter() - started,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('ENCODE_ONLY_SUMMARY', json.dumps(summary, indent=2), flush=True)
    print(f'summary_path={out}', flush=True)
    if not pass_results:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
