"""
Latent extraction script for Scale-PAE (Prior-Aligned Autoencoder).

Extracts and caches PAE latents from an image dataset for efficient
downstream DiT training.

by Zhengrong Yue
from SJTU
"""

import os
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "72000"
import sys
import time
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import argparse
from safetensors.torch import save_file
from safetensors import safe_open
from datetime import datetime
from omegaconf import OmegaConf
from glob import glob

from dataset.img_latent_dataset import ImgLatentDataset

from PIL import Image
import numpy as np

from datetime import timedelta

from tokenizer.pae import PAE


def verify_safetensors_file(filepath):
    """Verify a safetensors file is readable and not corrupted.
    Returns True if valid, False otherwise."""
    try:
        with safe_open(filepath, framework="pt", device="cpu") as f:
            # Check all expected keys exist and are readable
            for key in ('latents', 'latents_flip', 'labels'):
                tensor_slice = f.get_slice(key)
                shape = tensor_slice.get_shape()
                if len(shape) == 0 or shape[0] == 0:
                    return False
        return True
    except Exception:
        return False


def save_and_verify_shard(save_dict, filepath, max_retries=3):
    """Save a safetensors shard and verify it, retrying on failure."""
    metadata = {
        'total_size': f'{save_dict["latents"].shape[0]}',
        'dtype': f'{save_dict["latents"].dtype}',
        'device': f'{save_dict["latents"].device}',
    }
    for attempt in range(max_retries):
        save_file(save_dict, filepath, metadata=metadata)
        if verify_safetensors_file(filepath):
            return True
        print(f"[WARNING] Shard verification failed (attempt {attempt + 1}/{max_retries}): {filepath}")
        # Remove corrupted file before retry
        if os.path.exists(filepath):
            os.remove(filepath)
        time.sleep(1)
    print(f"[ERROR] Failed to save valid shard after {max_retries} attempts: {filepath}")
    return False


def verify_all_shards(output_dir):
    """Verify all safetensors files in the output directory.
    Returns list of corrupted file paths."""
    all_files = sorted(glob(os.path.join(output_dir, "*.safetensors")))
    corrupted = []
    for filepath in all_files:
        if not verify_safetensors_file(filepath):
            corrupted.append(filepath)
    return corrupted

def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126

    Args:
        pil_image: Input PIL Image
        image_size: Target size for both dimensions

    Returns:
        Center-cropped PIL Image of size (image_size, image_size)
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])



def main(args):
    assert torch.cuda.is_available(), "Requires at least one GPU"
    try:
        # dist.init_process_group("nccl")
        dist.init_process_group(
        "nccl", 
        timeout=timedelta(hours=5)
    )
        rank, world_size = dist.get_rank(), dist.get_world_size()
        device = rank % torch.cuda.device_count()
        seed = args.seed + rank
        if rank == 0:
            print(f"rank={rank}, seed={seed}, world_size={world_size}")
    except:
        rank, device, world_size, seed = 0, 0, 1, args.seed

    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # Determine output directory based on model name
    config = OmegaConf.load(args.config)
    output_dir = os.path.join(args.output_path, 'latents', config.vae.model_name, f'imgnet{args.image_size}_norm{args.normalize_type}')
    if rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        else:
            print(f"dir exist: {output_dir}")
    if dist.is_initialized():
        dist.barrier()
    
    if rank == 0:
        print(f"Loading VAE model from: {config.vae.vae_ckpt}")
    
    # Load PAE model
    if rank == 0:
        print(20*'=')
        print("PAE config: ", config.vae.vae_config)
        print(20*'=')
    pae_ckpt_path = config.vae.vae_ckpt
    pae_train_config = OmegaConf.load(config.vae.vae_config)
    pae_config = pae_train_config.stage_1
    pae_params = OmegaConf.to_container(pae_config.get("params", {}), resolve=True)
    vae = PAE(**pae_params)
    # load model ckpt (don't use ema ckpt)
    state_dict = torch.load(pae_ckpt_path, map_location="cpu", weights_only=False)["model"]
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for prefix in ("module.", "_orig_mod."):
                while new_key.startswith(prefix):  new_key = new_key[len(prefix):]
            cleaned_state_dict[new_key] = value
        msg = vae.load_state_dict(cleaned_state_dict, strict=False)
        if rank == 0:
            missing, unexpected = msg
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
        vae = vae.to(device).to(torch.bfloat16).eval()
        del cleaned_state_dict, state_dict
    if rank == 0:
        print(f"VAE model loaded!")
    
    def img_transform(p_hflip=0, img_size=None):
        img_size = img_size if img_size is not None else 256
        return transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, img_size)),
            transforms.RandomHorizontalFlip(p=p_hflip),
            transforms.ToTensor(),
            nn.Identity()  # PAE handles normalization internally
        ])
    if rank == 0:
        print('Image Transform:', img_transform())

    if rank == 0:
        print(f"Loading data from: {args.data_path}")
    datasets = [
        ImageFolder(args.data_path, transform=img_transform(p_hflip=p)) for p in [0.0, 1.0]
    ]
    samplers = [
        DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False, seed=args.seed)
        for ds in datasets
    ]
    loaders = [
        DataLoader(ds, batch_size=args.batch_size, shuffle=False, sampler=s,
                   num_workers=args.num_workers, pin_memory=True, drop_last=False)
        for ds, s in zip(datasets, samplers)
    ]
    if rank == 0:
        print(f"Total data: {len(loaders[0].dataset)}")

    run_images = saved_files = 0
    latents, latents_flip, labels = [], [], []

    for batch_idx, batch_data in enumerate(zip(*loaders)):
        run_images += batch_data[0][0].shape[0]
        if run_images % 100 == 0 and rank == 0:
            print(f'{datetime.now()} processing {run_images}/{len(loaders[0].dataset)}')

        for loader_idx, (x, y) in enumerate(batch_data):
            x = x.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                with torch.no_grad():
                    z = vae.encode(x).detach().cpu() # (N, C, H, W)

            if batch_idx == 0 and rank == 0:
                print('latent shape', z.shape, 'dtype', z.dtype)
            (latents if loader_idx == 0 else latents_flip).append(z)
            if loader_idx == 0:
                labels.append(y)

        if len(latents) == 10000 // args.batch_size:
            save_dict = {
                'latents': torch.cat(latents, dim=0).contiguous(),
                'latents_flip': torch.cat(latents_flip, dim=0).contiguous(),
                'labels': torch.cat(labels, dim=0).contiguous()
            }
            if rank == 0:
                for k, v in save_dict.items():
                    print(k, v.shape)
            shard_path = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
            success = save_and_verify_shard(save_dict, shard_path)
            if rank == 0:
                status = "OK" if success else "FAILED"
                print(f'Saved shard {saved_files} [{status}]')
            latents, latents_flip, labels = [], [], []
            saved_files += 1

    if len(latents) > 0:
        save_dict = {
            'latents': torch.cat(latents, dim=0).contiguous(),
            'latents_flip': torch.cat(latents_flip, dim=0).contiguous(),
            'labels': torch.cat(labels, dim=0).contiguous()
        }
        if rank == 0:
            for k, v in save_dict.items():
                print(k, v.shape)
        shard_path = os.path.join(output_dir, f'latents_rank{rank:02d}_shard{saved_files:03d}.safetensors')
        success = save_and_verify_shard(save_dict, shard_path)
        if rank == 0:
            status = "OK" if success else "FAILED"
            print(f'Saved shard {saved_files} to {output_dir} [{status}]')

    dist.barrier()
    if rank == 0:
        # Ensure all files are fully flushed to disk (important for NFS/network FS)
        import subprocess as sp
        try:
            sp.run(["sync"], timeout=60)
        except Exception:
            pass
        time.sleep(3)

        # Final integrity check: verify all shards
        print(f"\n[VERIFY] Checking all safetensors files in: {output_dir}")
        corrupted_files = verify_all_shards(output_dir)
        if corrupted_files:
            print(f"[ERROR] Found {len(corrupted_files)} corrupted file(s):")
            for corrupted_path in corrupted_files:
                print(f"  - {corrupted_path}")
                os.remove(corrupted_path)
                print(f"    -> Deleted")
            print("[ERROR] Corrupted shards have been removed. Please re-run extraction.")
            dist.barrier()
            dist.destroy_process_group()
            sys.exit(1)

        print(f"[VERIFY] All shards verified OK!")
        print(f"Computing latent stats from: {output_dir}")
        ImgLatentDataset(output_dir, latent_norm=True)
        print(f"Latent stats saved to {output_dir}/latents_stats.pt")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_pae_dit_xl.yaml", required=False)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    args.data_path = config.data.raw_image_path
    args.output_path = config.train.output_dir
    args.image_size = config.data.image_size
    args.batch_size = config.vae.per_proc_batch_size
    args.seed = 42
    args.num_workers = 4

    args.normalize_type = config.vae.get('normalize_type', 'imagenet')

    main(args)