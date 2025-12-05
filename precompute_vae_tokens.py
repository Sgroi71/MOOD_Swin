#!/usr/bin/env python3
"""
Pre-computa i token del VAE per tutto il dataset ImageNet.
Salva i token in file .pt per ogni immagine, da usare durante il training.

Uso:
    python precompute_vae_tokens.py --data_path /path/to/imagenet --output_dir /path/to/tokens --vae_path /path/to/dall_e_weights

Con più GPU:
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 precompute_vae_tokens.py ...
"""

import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path
import numpy as np

import utils


def get_args():
    parser = argparse.ArgumentParser('Pre-compute VAE tokens')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to ImageNet dataset (should contain train/ folder)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save precomputed tokens')
    parser.add_argument('--discrete_vae_weight_path', type=str, required=True,
                        help='Path to DALL-E VAE weights')
    parser.add_argument('--discrete_vae_type', type=str, default='dall-e',
                        choices=['dall-e', 'customized'])
    parser.add_argument('--second_input_size', type=int, default=112,
                        help='Image size for VAE input (default 112 for dall-e)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for VAE inference')
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Initialize distributed if available
    utils.init_distributed_mode(args)
    device = torch.device(args.device)
    
    print(f"Loading VAE from {args.discrete_vae_weight_path}")
    d_vae = utils.create_d_vae(
        weight_path=args.discrete_vae_weight_path,
        d_vae_type=args.discrete_vae_type,
        device=device,
        image_size=args.second_input_size
    )
    d_vae.eval()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup transforms for VAE (same as in datasets.py)
    if args.discrete_vae_type == "dall-e":
        from dall_e.utils import map_pixels
        vae_transform = transforms.Compose([
            transforms.Resize(args.second_input_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(args.second_input_size),
            transforms.ToTensor(),
            map_pixels,
        ])
    else:
        from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
        vae_transform = transforms.Compose([
            transforms.Resize(args.second_input_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(args.second_input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
        ])
    
    # Load dataset
    train_dir = os.path.join(args.data_path, 'train')
    dataset = datasets.ImageFolder(train_dir, transform=vae_transform)
    
    print(f"Dataset size: {len(dataset)} images")
    
    # Distributed sampler
    if args.distributed:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    # Get image paths
    all_paths = [s[0] for s in dataset.samples]
    
    # Process in batches
    rank = utils.get_rank() if args.distributed else 0
    world_size = utils.get_world_size() if args.distributed else 1
    
    print(f"[Rank {rank}] Processing {len(dataloader)} batches...")
    
    all_tokens = []
    all_indices = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"[Rank {rank}] Computing VAE tokens", disable=(rank != 0))
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            
            # Get VAE tokens
            tokens = d_vae.get_codebook_indices(images)  # [B, H, W] or [B, L]
            tokens = tokens.flatten(1).cpu()  # [B, num_tokens]
            
            all_tokens.append(tokens)
            
            # Calculate global indices for this batch
            if args.distributed:
                start_idx = batch_idx * args.batch_size * world_size + rank * args.batch_size
            else:
                start_idx = batch_idx * args.batch_size
            
            batch_indices = list(range(start_idx, start_idx + tokens.shape[0]))
            all_indices.extend(batch_indices)
    
    # Concatenate all tokens
    all_tokens = torch.cat(all_tokens, dim=0)
    
    print(f"[Rank {rank}] Processed {all_tokens.shape[0]} images, token shape: {all_tokens.shape}")
    
    # Save tokens
    if args.distributed:
        # Each rank saves its own shard
        shard_path = output_dir / f"tokens_shard_{rank}.pt"
        indices_path = output_dir / f"indices_shard_{rank}.pt"
        torch.save(all_tokens, shard_path)
        torch.save(torch.tensor(all_indices), indices_path)
        print(f"[Rank {rank}] Saved shard to {shard_path}")
        
        # Wait for all ranks
        if dist.is_initialized():
            dist.barrier()
        
        # Rank 0 merges all shards
        if rank == 0:
            print("Merging shards...")
            merged_tokens = {}
            for r in range(world_size):
                shard = torch.load(output_dir / f"tokens_shard_{r}.pt")
                indices = torch.load(output_dir / f"indices_shard_{r}.pt")
                for i, idx in enumerate(indices.tolist()):
                    merged_tokens[idx] = shard[i]
            
            # Sort by index and stack
            sorted_indices = sorted(merged_tokens.keys())
            final_tokens = torch.stack([merged_tokens[i] for i in sorted_indices])
            
            # Save final merged file
            final_path = output_dir / "vae_tokens.pt"
            torch.save({
                'tokens': final_tokens,
                'num_images': len(final_tokens),
                'token_shape': final_tokens.shape,
                'vae_type': args.discrete_vae_type,
                'image_size': args.second_input_size,
            }, final_path)
            print(f"Saved merged tokens to {final_path}")
            print(f"Final shape: {final_tokens.shape}")
            
            # Cleanup shards
            for r in range(world_size):
                os.remove(output_dir / f"tokens_shard_{r}.pt")
                os.remove(output_dir / f"indices_shard_{r}.pt")
    else:
        # Single GPU: save directly
        final_path = output_dir / "vae_tokens.pt"
        torch.save({
            'tokens': all_tokens,
            'num_images': len(all_tokens),
            'token_shape': all_tokens.shape,
            'vae_type': args.discrete_vae_type,
            'image_size': args.second_input_size,
        }, final_path)
        print(f"Saved tokens to {final_path}")
        print(f"Final shape: {all_tokens.shape}")


if __name__ == '__main__':
    main()
