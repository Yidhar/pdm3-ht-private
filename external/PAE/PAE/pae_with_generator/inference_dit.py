import os
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_TIMEOUT"] = "72000"
import os
import sys
import random
import shutil
from tqdm import tqdm
import importlib.util
import argparse
import logging
from datetime import datetime
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from inference_sample import do_sample, load_config # category balance
from accelerate import Accelerator
from models.lightningdit import LightningDiT_models
import torch.distributed as dist
import numpy as np
from PIL import Image

from tokenizer.pae import PAE
import tensorflow.compat.v1 as tf
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'guided-diffusion', 'evaluations'))
from evaluator import Evaluator


def create_npz_from_sample_folder(sample_dir, num=50000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

def copy_single_file(src_path, dst_path):
    """Copy a single file from src to dst.

    Use shutil.copyfile (content only) instead of shutil.copy to avoid
    calling chmod on the destination. Some mounted filesystems
    (e.g. NFS / SMB / object-storage mounts) do not allow chmod and will
    raise PermissionError: [Errno 1] Operation not permitted.

    Returns True on success, False if source disappeared (e.g. another
    process deleted the source dir concurrently).
    """
    if not os.path.exists(src_path):
        return False
    try:
        shutil.copyfile(src_path, dst_path)
        return True
    except FileNotFoundError:
        # Source vanished mid-copy (concurrent deletion). Swallow and report.
        return False

def _count_images_in_dir(directory):
    """Count image files (png/jpg/jpeg) in a directory (non-recursive top-level only)."""
    if not os.path.isdir(directory):
        return 0
    return sum(
        1 for f in os.listdir(directory)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    )

def shuffle_images(src_dir, seed=42, num_workers=16, expected_num=None):
    """Randomly rename images to shuffle their order using multi-threading.

    Args:
        src_dir: Source image directory.
        seed: Random seed for shuffling.
        num_workers: Number of threads for copying.
        expected_num: Expected number of shuffled images. If the destination
            directory already contains >= expected_num images, skip shuffling
            entirely (assume a previous successful run).

    Returns:
        dst_dir (str): The shuffle directory path.
    """

    # Target directory: original directory + "-shuffle"
    dst_dir = src_dir.rstrip('/') + '-shuffle'

    # Fast path: skip shuffling if dst already has enough images.
    if expected_num is not None:
        existing = _count_images_in_dir(dst_dir)
        if existing >= expected_num:
            print(f"[shuffle_images] {dst_dir} already has {existing} images "
                  f"(>= expected {expected_num}), skip shuffling.")
            return dst_dir

    # Clean up any leftover files from previous (possibly failed) runs,
    # otherwise old files with the same names will be overwritten and
    # may cause permission errors on certain filesystems.
    if os.path.exists(dst_dir):
        print(f"Removing existing shuffle dir: {dst_dir}")
        shutil.rmtree(dst_dir)

    # Collect all images
    image_files = []
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG')):
                full_path = os.path.join(root, f)
                image_files.append(full_path)

    print(f"Found {len(image_files)} images")

    # Shuffle randomly
    random.seed(seed)
    random.shuffle(image_files)
    print(f"Shuffled with seed={seed}")

    # Create target directory
    os.makedirs(dst_dir, exist_ok=True)

    # Build copy task list: (src, dst) pairs
    copy_tasks = []
    for i, src_path in enumerate(image_files):
        ext = os.path.splitext(src_path)[1]
        dst_path = os.path.join(dst_dir, f"{i:06d}{ext}")
        copy_tasks.append((src_path, dst_path))

    # Multi-threaded copy
    print(f"Copying {len(image_files)} images to {dst_dir} with {num_workers} threads...")
    missing_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(copy_single_file, src, dst): (src, dst)
            for src, dst in copy_tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            ok = future.result()
            if not ok:
                missing_count += 1

    if missing_count > 0:
        print(f"⚠️  {missing_count}/{len(image_files)} source files were missing during copy "
              f"(likely concurrent deletion). They were skipped.")
    print(f"✅ Done! {len(image_files) - missing_count} images shuffled to {dst_dir}")
    return dst_dir

def setup_logger(log_dir, rank):
    """Setup logger that outputs to both console and file (rank 0 only)."""
    logger = logging.getLogger('inference')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler only for rank 0
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'inference_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f'Log file: {log_file}')

    return logger

def evaluate_single_sample(sample_folder_dir, sample_cfg, evaluator, ref_acts, ref_stats, ref_stats_spatial, seed):
    """
    Evaluate a single sample folder and return metrics.
    Returns: dict with metrics and paths
    """
    fid_num = sample_cfg.get('fid_num', 50000)
    
    # Shuffle images to new directory
    shuffle_images(sample_folder_dir, seed=seed)
    shuffle_dir = sample_folder_dir.rstrip('/') + '-shuffle'
    
    # Pack samples into .npz
    create_npz_from_sample_folder(shuffle_dir, fid_num)
    npz_path = shuffle_dir.rstrip('/') + '.npz'
    
    # Evaluate FID
    guidance_acts = evaluator.read_activations(npz_path)
    guidance_stats, guidance_stats_spatial = evaluator.read_statistics(npz_path, guidance_acts)
    guidance_is = evaluator.compute_inception_score(guidance_acts[0])
    guidance_fid = guidance_stats.frechet_distance(ref_stats)
    guidance_sfid = guidance_stats_spatial.frechet_distance(ref_stats_spatial)
    guidance_prec, guidance_recall = evaluator.compute_prec_recall(ref_acts[0], guidance_acts[0])
    
    return {
        'fid': guidance_fid,
        'is': guidance_is,
        'sfid': guidance_sfid,
        'precision': guidance_prec,
        'recall': guidance_recall,
        'shuffle_dir': shuffle_dir,
        'npz_path': npz_path,
        'original_dir': sample_folder_dir
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./", required=False)
    parser.add_argument("--vae_config", type=str, default="./",required=False, help="pae yaml config")
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--batch_demo', action='store_true', default=False,
                        help='Generate demo_batch_size images per class in demo_class_list, saved individually')
    parser.add_argument('--use_ema', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help="Whether to use EMA weights (True/False)")
    parser.add_argument('--search_best_cfg', action='store_true', default=False,
                        help='Search for best cfg_scale and cfg_interval_start for sample_wcfg')
    parser.add_argument('--cfg_search_scales', type=float, nargs='+', 
                        default=[2.0, 2.5, 3.0, 3.3, 3.5, 4.0, 5.0],
                        help='CFG scales to search (space-separated)')
    parser.add_argument('--cfg_search_intervals', type=float, nargs='+', 
                        default=[0.11, 0.2, 0.25, 0.3, 0.35],
                        help='CFG intervals to search (space-separated)')
    args = parser.parse_args()

    accelerator = Accelerator()
    train_config = load_config(args.config)

    # Setup logger - log to sample output directory
    exp_name = train_config['train']['exp_name']
    output_dir = train_config['train']['output_dir']
    log_dir = os.path.join(output_dir, exp_name, 'logs')
    logger = setup_logger(log_dir, accelerator.process_index)

    # Log all sample configs
    if accelerator.process_index == 0:
        sample_keys = [key for key in train_config if key.startswith('sample')]
        logger.info(f"Found {len(sample_keys)} sample config(s): {sample_keys}")
    
    pae_config = OmegaConf.load(train_config['vae']['vae_config'])
    patch_size = train_config['vae']['downsample_ratio']
    in_chans = train_config['vae']['latent_dim']
    train_config['vae']['in_chans'] = in_chans
    ckpt_path = train_config.get('ckpt_path')
    if ckpt_path is None:
        raise ValueError("ckpt_path must be specified in config")
    latent_size = train_config['data']['image_size'] // patch_size
    if accelerator.process_index == 0:
        logger.info(f'Using ckpt: {ckpt_path}')
    
    model = LightningDiT_models[train_config['model']['model_type']](
        input_size=latent_size,
        num_classes=train_config['data']['num_classes'],
        use_qknorm=train_config['model']['use_qknorm'],
        use_swiglu=train_config['model'].get('use_swiglu', False),
        use_rope=train_config['model'].get('use_rope', False),
        use_rmsnorm=train_config['model'].get('use_rmsnorm', False),
        wo_shift=train_config['model'].get('wo_shift', False),
        in_channels=train_config['model'].get('in_chans', in_chans),
        learn_sigma=train_config['model'].get('learn_sigma', False),
        use_abs_pos=train_config['model'].get('use_abs_pos', True),
    )

    rank, world_size = dist.get_rank(), dist.get_world_size()
    device = rank % torch.cuda.device_count()
    config = OmegaConf.load(args.config)
    
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
    
    # Collect all sample configs (keys starting with "sample")
    sample_configs = {key: train_config[key] for key in train_config if key.startswith('sample')}
    if not sample_configs:
        print("No sample config found in config file (expected keys like 'sample_wcfg', 'sample_wocfg', or 'sample').")
        return

    # ========== Phase 1: All ranks participate in sampling ==========
    sample_results = {}  # sample_key -> sample_folder_dir

    if args.search_best_cfg:
        # ---- CFG Search mode: only do CFG search sampling, skip normal wcfg/wocfg ----
        if rank == 0:
            print("\n" + "=" * 70)
            print("[CFG Search] Starting CFG search sampling (all ranks)...")
            print("=" * 70)
        
        cfg_scales = args.cfg_search_scales
        cfg_intervals = args.cfg_search_intervals
        combinations = [(s, i) for s in cfg_scales for i in cfg_intervals]
        
        if rank == 0:
            print(f"CFG scales: {cfg_scales}")
            print(f"CFG intervals: {cfg_intervals}")
            print(f"Total combinations: {len(combinations)}")
        
        # All ranks participate in sampling for each CFG combination
        cfg_sample_results = {}  # (cfg_scale, cfg_interval) -> sample_folder_dir
        
        for idx, (cfg_scale, cfg_interval) in enumerate(combinations):
            if rank == 0:
                print(f"\n[Search {idx+1}/{len(combinations)}] cfg_scale={cfg_scale:.1f}, cfg_interval_start={cfg_interval:.2f}")
            
            # Create temporary config
            temp_cfg = train_config['sample_wcfg'].copy()
            temp_cfg['cfg_scale'] = cfg_scale
            temp_cfg['cfg_interval_start'] = cfg_interval
            train_config['sample'] = temp_cfg
            
            # All ranks participate in sampling
            sample_folder_dir = do_sample(train_config, accelerator, ckpt_path=ckpt_path, model=model, vae=vae, demo_sample_mode=args.demo, batch_demo_mode=args.batch_demo, use_ema=args.use_ema)
            dist.barrier()
            
            if sample_folder_dir is not None:
                cfg_sample_results[(cfg_scale, cfg_interval)] = sample_folder_dir
            elif rank == 0:
                print(f"    [Search] Sampling returned None, skipping.")
        
        if rank == 0:
            print(f"\n[CFG Search] All {len(combinations)} combinations sampled.")
    else:
        # ---- Normal mode: sample all configs ----
        cfg_sample_results = None
        
        for sample_key, sample_cfg in sample_configs.items():
            if rank == 0:
                print("\n" + "=" * 70)
                print(f"[{sample_key}] Starting sampling...")
                print("=" * 70)

            # Inject current sample config as train_config['sample'] for do_sample compatibility
            train_config['sample'] = sample_cfg

            if rank == 0:
                logger.info(f"[{sample_key}] Sampling config: "
                    f"method={sample_cfg.get('sampling_method')}, "
                    f"steps={sample_cfg.get('num_sampling_steps')}, "
                    f"shift={sample_cfg.get('timestep_shift')}, "
                    f"cfg={sample_cfg.get('cfg_scale')}, "
                    f"mode={sample_cfg.get('mode')}, "
                    f"fid_num={sample_cfg.get('fid_num')}, "
                    f"seed={train_config.get('train', {}).get('global_seed')}"
                )

            # All ranks participate in sampling
            sample_folder_dir = do_sample(train_config, accelerator, ckpt_path=ckpt_path, model=model, vae=vae, demo_sample_mode=args.demo, batch_demo_mode=args.batch_demo, use_ema=args.use_ema)
            dist.barrier()

            if sample_folder_dir is not None:
                sample_results[sample_key] = sample_folder_dir
            elif rank == 0:
                print(f"[{sample_key}] Sampling returned None, will skip evaluation.")

    # ========== Phase 2: Non-rank-0 processes exit early ==========
    if rank != 0:
        print(f"[Rank {rank}] All sampling done. Exiting.")
        dist.destroy_process_group()
        return
    if args.demo or args.batch_demo:
        print("Demo mode, exiting.")
        return

    # ========== Phase 3: Rank 0 does packing + FID evaluation ==========
    print("\n" + "=" * 70)
    print("All sampling complete. Starting evaluation (rank 0 only)...")
    print("=" * 70)

    # Initialize TF evaluator and reference activations
    ref_npz = train_config['data']['fid_reference_file']
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=tf_config))
    print("Warming up TensorFlow...")
    evaluator.warmup()
    print("Computing reference batch activations...")
    ref_acts = evaluator.read_activations(ref_npz)
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_npz, ref_acts)

    all_results = []
    seed = train_config.get('train', {}).get('global_seed', 42)

    # ========== CFG Search evaluation (if enabled) ==========
    if args.search_best_cfg and cfg_sample_results:
        print("\n" + "=" * 70)
        print("[CFG Search] Starting evaluation of all CFG combinations...")
        print("=" * 70)
        
        cfg_results = []
        
        for (cfg_scale, cfg_interval), sample_folder_dir in cfg_sample_results.items():
            print(f"\n[Search Eval] cfg_scale={cfg_scale:.1f}, cfg_interval_start={cfg_interval:.2f}")
            
            temp_cfg = train_config['sample_wcfg'].copy()
            temp_cfg['cfg_scale'] = cfg_scale
            temp_cfg['cfg_interval_start'] = cfg_interval
            
            metrics = evaluate_single_sample(
                sample_folder_dir, temp_cfg, evaluator, ref_acts, ref_stats, ref_stats_spatial, seed
            )
            
            cfg_results.append({
                'cfg_scale': cfg_scale,
                'cfg_interval_start': cfg_interval,
                **metrics
            })
        
        if cfg_results:
            # Find best result (lowest FID)
            best_result = min(cfg_results, key=lambda x: x['fid'])
            
            # Print all results sorted by FID
            print("\n" + "=" * 70)
            print("CFG Search Results:")
            print("=" * 70)
            for r in sorted(cfg_results, key=lambda x: x['fid']):
                marker = " ★ BEST" if r is best_result else ""
                print(f"cfg_scale={r['cfg_scale']:.1f}, interval={r['cfg_interval_start']:.2f} -> "
                      f"FID: {r['fid']:.4f}, IS: {r['is']:.4f}, sFID: {r['sfid']:.4f}, "
                      f"Prec: {r['precision']:.4f}, Recall: {r['recall']:.4f}{marker}")
            
            print("\n" + "=" * 70)
            print(f"Best: cfg_scale={best_result['cfg_scale']:.1f}, "
                  f"interval={best_result['cfg_interval_start']:.2f}, "
                  f"FID: {best_result['fid']:.4f}")
            print("=" * 70)
            
            # Clean up non-best results
            print("\nCleaning up non-best results...")
            for r in cfg_results:
                if r is not best_result:
                    if os.path.exists(r['shuffle_dir']):
                        shutil.rmtree(r['shuffle_dir'])
                        print(f"  Deleted dir: {r['shuffle_dir']}")
                    if os.path.exists(r['npz_path']):
                        os.remove(r['npz_path'])
                        print(f"  Deleted npz: {r['npz_path']}")
            
            print(f"Kept best result: {best_result['shuffle_dir']}")
            
            # Add best result to summary
            result_line = (f"[sample_wcfg_best] cfg_scale={best_result['cfg_scale']:.1f}, "
                          f"interval={best_result['cfg_interval_start']:.2f}, "
                          f"IS: {best_result['is']:.4f}, FID: {best_result['fid']:.4f}, "
                          f"sFID: {best_result['sfid']:.4f}, Precision: {best_result['precision']:.4f}, "
                          f"Recall: {best_result['recall']:.4f}")
            all_results.append(result_line)
            logger.info(result_line)
    
    # ========== Evaluate other sample configs (normal mode) ==========
    for sample_key, sample_folder_dir in sample_results.items():
        sample_cfg = sample_configs[sample_key]
        fid_num = sample_cfg.get('fid_num', 50000)

        # Shuffle images to new directory (skip if dst already has enough images)
        shuffle_dir = sample_folder_dir.rstrip('/') + '-shuffle'
        already_done = _count_images_in_dir(shuffle_dir) >= fid_num
        shuffle_images(sample_folder_dir, seed=seed, expected_num=fid_num)
        # Only remove the source dir when shuffle actually produced new outputs;
        # if we skipped because the shuffle dir was already complete, the source
        # dir may have been deleted in a previous run already.
        if not already_done and os.path.exists(sample_folder_dir):
            shutil.rmtree(sample_folder_dir)  # 释放空间
        sample_folder_dir = shuffle_dir

        # Pack samples into .npz
        create_npz_from_sample_folder(sample_folder_dir, fid_num)

        # Evaluate FID
        sample_folder_dir_npz = sample_folder_dir.rstrip('/') + '.npz'
        print(f"\n[{sample_key}] Evaluating samples...")
        guidance_acts = evaluator.read_activations(sample_folder_dir_npz)
        guidance_stats, guidance_stats_spatial = evaluator.read_statistics(sample_folder_dir_npz, guidance_acts)
        guidance_is = evaluator.compute_inception_score(guidance_acts[0])
        guidance_fid = guidance_stats.frechet_distance(ref_stats)
        guidance_sfid = guidance_stats_spatial.frechet_distance(ref_stats_spatial)
        guidance_prec, guidance_recall = evaluator.compute_prec_recall(ref_acts[0], guidance_acts[0])

        result_line = (f"[{sample_key}] IS: {guidance_is:.4f}, FID: {guidance_fid:.4f}, sFID: {guidance_sfid:.4f}, "
                       f"Precision: {guidance_prec:.4f}, Recall: {guidance_recall:.4f}")
        all_results.append(result_line)

        print("=" * 50)
        print(result_line)
        print("=" * 50)
        logger.info(result_line)

    # Print summary
    print("\n" + "=" * 70)
    print("All Evaluation Results Summary")
    print("=" * 70)
    for result in all_results:
        print(result)
    print("=" * 70)
    print("Evaluation complete!")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()

