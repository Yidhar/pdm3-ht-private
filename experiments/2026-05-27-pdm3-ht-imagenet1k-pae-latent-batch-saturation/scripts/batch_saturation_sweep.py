#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import asdict
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


def hard_finish(code: int = 0) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def parse_batch_sizes(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(',') if x.strip()]


def flush_batch(
    *,
    model: torch.nn.Module,
    batch_x: List[torch.Tensor],
    batch_xf: List[torch.Tensor],
    batch_y: List[int],
    device: torch.device,
    model_dtype: torch.dtype,
    save_dtype: torch.dtype,
    latents: List[torch.Tensor],
    latents_flip: List[torch.Tensor],
    labels: List[torch.Tensor],
) -> Dict[str, Any]:
    if not batch_x:
        return {'encoded': 0, 'encode_sec': 0.0, 'normal_sec': 0.0, 'flip_sec': 0.0}
    x = torch.stack(batch_x, dim=0)
    xf = torch.stack(batch_xf, dim=0)
    y = torch.tensor(batch_y, dtype=torch.long)

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    z = builder.encode_batch(model, x, device, model_dtype, save_dtype)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    zf = builder.encode_batch(model, xf, device, model_dtype, save_dtype)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    t2 = time.perf_counter()

    latents.append(z)
    latents_flip.append(zf)
    labels.append(y)
    return {
        'encoded': int(y.shape[0]),
        'encode_sec': t2 - t0,
        'normal_sec': t1 - t0,
        'flip_sec': t2 - t1,
    }


def run_segment(args: argparse.Namespace, ds_iter: Any, model: torch.nn.Module, device: torch.device, model_dtype: torch.dtype, save_dtype: torch.dtype, batch_size: int, segment_index: int, start_offset: int) -> Dict[str, Any]:
    output_dir = Path(args.output_base) / f'imagenet256_train_saturation_b{batch_size}_n{args.samples_per_batch_size}_offset{start_offset}'
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        'dataset': args.dataset,
        'split': args.split,
        'image_size': str(args.image_size),
        'pae_config': str(args.pae_config),
        'pae_ckpt': str(args.pae_ckpt),
        'model_dtype': args.model_dtype,
        'save_dtype': args.save_dtype,
        'created_by': 'batch_saturation_sweep.py',
        'batch_size': str(batch_size),
        'segment_index': str(segment_index),
        'start_offset': str(start_offset),
    }

    latents: List[torch.Tensor] = []
    latents_flip: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    batch_x: List[torch.Tensor] = []
    batch_xf: List[torch.Tensor] = []
    batch_y: List[int] = []

    seen = 0
    encoded = 0
    preprocess_sec = 0.0
    encode_sec = 0.0
    normal_sec = 0.0
    flip_sec = 0.0
    flushes = 0

    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    status = 'PASS'
    error = None
    stats = None

    try:
        while seen < args.samples_per_batch_size:
            sample = next(ds_iter)
            t_pre0 = time.perf_counter()
            image = sample[args.image_key]
            label = int(sample[args.label_key])
            batch_x.append(builder.image_to_tensor(image, args.image_size, hflip=False))
            batch_xf.append(builder.image_to_tensor(image, args.image_size, hflip=True))
            batch_y.append(label)
            preprocess_sec += time.perf_counter() - t_pre0
            seen += 1
            if len(batch_x) >= batch_size:
                r = flush_batch(
                    model=model,
                    batch_x=batch_x,
                    batch_xf=batch_xf,
                    batch_y=batch_y,
                    device=device,
                    model_dtype=model_dtype,
                    save_dtype=save_dtype,
                    latents=latents,
                    latents_flip=latents_flip,
                    labels=labels,
                )
                encoded += r['encoded']
                encode_sec += r['encode_sec']
                normal_sec += r['normal_sec']
                flip_sec += r['flip_sec']
                flushes += 1
                print(f'SEGMENT_PROGRESS batch_size={batch_size} encoded={encoded}/{args.samples_per_batch_size} encode_sec_last={r["encode_sec"]:.6f}', flush=True)
                batch_x, batch_xf, batch_y = [], [], []
        if batch_x:
            r = flush_batch(
                model=model,
                batch_x=batch_x,
                batch_xf=batch_xf,
                batch_y=batch_y,
                device=device,
                model_dtype=model_dtype,
                save_dtype=save_dtype,
                latents=latents,
                latents_flip=latents_flip,
                labels=labels,
            )
            encoded += r['encoded']
            encode_sec += r['encode_sec']
            normal_sec += r['normal_sec']
            flip_sec += r['flip_sec']
            flushes += 1
        save_started = time.perf_counter()
        stats = builder.save_shard(output_dir, 0, latents, latents_flip, labels, meta)
        save_wall_sec = time.perf_counter() - save_started
    except Exception as exc:  # continue sweep after OOM/errors where possible
        status = 'FAIL'
        error = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
        save_wall_sec = 0.0
        print(f'SEGMENT_FAIL batch_size={batch_size} error={error}', flush=True)
        traceback.print_exc()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elapsed_sec = time.perf_counter() - started
    cuda_peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else 0.0
    result = {
        'status': status,
        'batch_size': batch_size,
        'segment_index': segment_index,
        'start_offset': start_offset,
        'output_dir': str(output_dir),
        'seen': seen,
        'encoded': encoded,
        'flushes': flushes,
        'elapsed_sec': elapsed_sec,
        'preprocess_sec': preprocess_sec,
        'encode_sec': encode_sec,
        'normal_encode_sec': normal_sec,
        'flip_encode_sec': flip_sec,
        'save_wall_sec': save_wall_sec,
        'cuda_peak_mb': cuda_peak_mb,
        'samples_per_sec_total': encoded / elapsed_sec if elapsed_sec > 0 else 0.0,
        'samples_per_sec_encode_only': encoded / encode_sec if encode_sec > 0 else 0.0,
        'pae_encode_ops_per_sec_encode_only': (2 * encoded) / encode_sec if encode_sec > 0 else 0.0,
        'error': error,
        'shards': [asdict(stats)] if stats is not None else [],
    }
    (output_dir / 'build_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('SEGMENT_SUMMARY', json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='ILSVRC/imagenet-1k')
    p.add_argument('--split', default='train')
    p.add_argument('--streaming', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--hf-token-env', default='HF_TOKEN')
    p.add_argument('--pae-config', default=str(REPO_ROOT / 'external/PAE/pae_with_generator/configs/pae/PAE_DINOv2L_d32.yaml'))
    p.add_argument('--pae-ckpt', default='/workspace/PDM/data/pae_ckpts/PAE_DINOv2L_d32/AE-models/dinov2-large.pt')
    p.add_argument('--output-base', default='/workspace/PDM/data/pae_latents/PAE_DINOv2L_d32/batch_saturation_2026-05-27')
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--batch-sizes', default='64,128,256,512,1024')
    p.add_argument('--samples-per-batch-size', type=int, default=2048)
    p.add_argument('--initial-skip', type=int, default=1024)
    p.add_argument('--image-key', default='image')
    p.add_argument('--label-key', default='label')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--model-dtype', default='bf16')
    p.add_argument('--save-dtype', default='bf16')
    p.add_argument('--hard-exit', action='store_true')
    args = p.parse_args()

    print('ARGS', json.dumps(vars(args), indent=2, sort_keys=True), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this sweep')
    device = torch.device(args.device)
    model_dtype = builder.dtype_from_name(args.model_dtype)
    save_dtype = builder.dtype_from_name(args.save_dtype)
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    shell_started = time.perf_counter()
    print('Loading PAE once for sweep', flush=True)
    model = builder.load_pae_model(Path(args.pae_config), Path(args.pae_ckpt), device, model_dtype)
    print('Loading HF dataset once for sweep', flush=True)
    ds = builder.load_hf_dataset(args.dataset, args.split, args.streaming, args.hf_token_env)
    ds_iter = iter(ds)

    print(f'Pre-skipping {args.initial_skip} samples before timed segments', flush=True)
    skip_started = time.perf_counter()
    for i in range(args.initial_skip):
        _ = next(ds_iter)
        if (i + 1) % 512 == 0:
            print(f'PRE_SKIP_PROGRESS {i+1}/{args.initial_skip}', flush=True)
    skip_sec = time.perf_counter() - skip_started
    print(f'PRE_SKIP_DONE seconds={skip_sec}', flush=True)

    results = []
    offset = args.initial_skip
    for idx, bs in enumerate(batch_sizes):
        print(f'===== START_SEGMENT idx={idx} batch_size={bs} offset={offset} samples={args.samples_per_batch_size} =====', flush=True)
        result = run_segment(args, ds_iter, model, device, model_dtype, save_dtype, bs, idx, offset)
        results.append(result)
        offset += result.get('seen', 0)

    ok_results = [r for r in results if r['status'] == 'PASS' and r['encoded'] > 0]
    best = max(ok_results, key=lambda r: r['samples_per_sec_total']) if ok_results else None
    summary = {
        'status': 'PASS' if ok_results else 'FAIL',
        'created_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'dataset': args.dataset,
        'split': args.split,
        'initial_skip': args.initial_skip,
        'skip_sec': skip_sec,
        'batch_sizes': batch_sizes,
        'samples_per_batch_size': args.samples_per_batch_size,
        'output_base': args.output_base,
        'cuda_device_name': torch.cuda.get_device_name(device),
        'cuda_total_memory_gib': torch.cuda.get_device_properties(device).total_memory / 1024**3,
        'model_dtype': args.model_dtype,
        'save_dtype': args.save_dtype,
        'shell_total_sec': time.perf_counter() - shell_started,
        'results': results,
        'best_by_total_samples_per_sec': best,
    }
    out_base = Path(args.output_base)
    out_base.mkdir(parents=True, exist_ok=True)
    summary_path = out_base / 'batch_saturation_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SWEEP_SUMMARY', json.dumps(summary, indent=2), flush=True)
    print(f'summary_path={summary_path}', flush=True)
    if args.hard_exit:
        hard_finish(0 if ok_results else 1)


if __name__ == '__main__':
    main()
