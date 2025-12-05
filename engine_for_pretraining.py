# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
from tqdm import tqdm

import utils
print_freq = 1000


def _model_supports_masked_pos(model):
    """Check if model forward accepts bool_masked_pos argument."""
    import inspect
    # Get the actual model (unwrap DDP if needed)
    m = model.module if hasattr(model, 'module') else model
    sig = inspect.signature(m.forward)
    supports = 'bool_masked_pos' in sig.parameters
    print(f"[DEBUG] Model type: {type(m).__name__}, supports_masked_pos: {supports}")
    return supports


def train_one_epoch(model: torch.nn.Module, d_vae: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    
    # Check if model supports masked position argument
    supports_masked_pos = _model_supports_masked_pos(model)
    
    # Wrap data_loader with tqdm for progress bar
    total_steps = len(data_loader)
    pbar = tqdm(enumerate(metric_logger.log_every(data_loader, print_freq, header)), 
                total=total_steps, desc=f"Epoch {epoch}", leave=True)
    
    for step, (batch, _) in pbar:
        # assign learning rate & weight decay for each step
        it = start_steps + step  # global training iteration
        if lr_schedule_values is not None or wd_schedule_values is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples, images, bool_masked_pos = batch
        images = images.to(device, non_blocking=True)
        samples = samples.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)

        with torch.no_grad():
            input_ids = d_vae.get_codebook_indices(images).flatten(1)
            bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
            labels = input_ids[bool_masked_pos]

        with torch.cuda.amp.autocast():
            if supports_masked_pos:
                # BEiT-style model with native masked position support
                outputs = model(samples, bool_masked_pos=bool_masked_pos, return_all_tokens=False)
            else:
                # SWIN or other models: get all features then select masked positions
                # For SWIN, we need to get features and project to vocab size
                features = model.forward_features(samples)  # Could be [B, num_patches, embed_dim] or [B, H, W, C]
                
                # Handle 4D features (B, H, W, C) -> (B, L, C)
                if features.dim() == 4:
                    B, H, W, C = features.shape
                    features = features.reshape(B, H * W, C)
                
                # If model has a head for MIM prediction, use it; otherwise need a projection
                if hasattr(model, 'lm_head'):
                    all_outputs = model.lm_head(features)
                elif hasattr(model, 'head'):
                    # Flatten and apply head
                    B, N, C = features.shape
                    all_outputs = model.head(features.reshape(B * N, C)).reshape(B, N, -1)
                else:
                    # Fallback: need to add a projection layer (this case shouldn't happen ideally)
                    raise ValueError("Model does not have lm_head or head for MIM prediction. "
                                     "Consider using a BEiT model or adding a prediction head.")
                # Select outputs at masked positions
                outputs = all_outputs[bool_masked_pos]
            loss = nn.CrossEntropyLoss()(input=outputs, target=labels)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        mlm_acc = (outputs.max(-1)[1] == labels).float().mean().item()

        metric_logger.update(mlm_acc=mlm_acc)
        if log_writer is not None:
            log_writer.update(mlm_acc=mlm_acc, head="loss")

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)

        # Update tqdm progress bar
        pbar.set_postfix({'loss': f'{loss_value:.4f}', 'acc': f'{mlm_acc:.4f}', 'lr': f'{max_lr:.2e}'})

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
