#!/usr/bin/env python3
"""Verify PAE latent shards are readable by ImgLatentDataset."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from safetensors import safe_open

REPO_ROOT = Path('/workspace/PDM')
PAE_ROOT = REPO_ROOT / 'external' / 'PAE' / 'pae_with_generator'
sys.path.insert(0, str(PAE_ROOT))
from dataset.img_latent_dataset import ImgLatentDataset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_dir')
    ap.add_argument('--latent-norm', action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob('*.safetensors'))
    print('files', [str(f) for f in files])
    shards = []
    for f in files:
        with safe_open(str(f), framework='pt', device='cpu') as sf:
            item = {'file': str(f), 'keys': {}}
            for k in ('latents','latents_flip','labels'):
                sl=sf.get_slice(k)
                item['keys'][k]={'shape': list(sl.get_shape()), 'dtype': str(sl.get_dtype())}
            shards.append(item)
    ds = ImgLatentDataset(str(data_dir), latent_norm=args.latent_norm)
    first = ds[0] if len(ds) else None
    out = {
        'ok': len(ds) > 0 and bool(files),
        'data_dir': str(data_dir),
        'num_files': len(files),
        'len': len(ds),
        'latent_norm': args.latent_norm,
        'shards': shards,
        'first_shape': list(first[0].shape) if first is not None else None,
        'first_dtype': str(first[0].dtype) if first is not None else None,
        'first_label': int(first[1].item()) if first is not None else None,
    }
    print(json.dumps(out, indent=2))
    if not out['ok']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
