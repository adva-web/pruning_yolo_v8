#!/usr/bin/env python3
"""
C2f Activation Collection via Forward Hooks
Implements GT-aligned patch extraction from C2f after-concat conv layers.
Uses forward hooks to capture feature maps during full model forward pass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional
import yaml
import cv2
import os

from yolov8_utils import get_all_conv2d_layers, convert_label_to_xyxy, compute_patch_indices
from c2f_utils import is_c2f_block

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_c2f_conv_activations(
    model: YOLO,
    target_conv: nn.Conv2d,
    train_data: List,
    classes: List[int],
    max_patches: Optional[int] = 100000,
    sampling: str = 'gt'
) -> Dict[int, Dict[int, List[float]]]:
    """
    Collect activations from a C2f after-concat conv layer using forward hooks.
    
    Uses GT-aligned sampling: extracts patch vectors at ground truth object centers
    mapped to the feature map spatial coordinates.
    
    Args:
        model: YOLO model
        target_conv: Target Conv2d layer (after concat in C2f block)
        train_data: List of training samples (dicts with 'image', 'labels', etc.)
        classes: List of class indices
        max_patches: Maximum number of patches to collect (default: 100k)
        sampling: Sampling strategy ('gt' for GT-aligned, 'random' for random)
    
    Returns:
        Dictionary: {channel_idx: {class_id: [activation_values]}}
    """
    num_channels = target_conv.weight.shape[0]
    activations = {ch: {cls: [] for cls in classes} for ch in range(num_channels)}
    
    # Storage for captured activations during forward pass
    captured_output = None
    
    def hook_fn(module, input, output):
        """Forward hook to capture conv layer output."""
        nonlocal captured_output
        # Detach and move to CPU immediately to save memory
        captured_output = output.detach().cpu()
    
    # Register forward hook
    hook_handle = target_conv.register_forward_hook(hook_fn)
    
    try:
        model.model.eval()
        device = next(target_conv.parameters()).device
        
        # Clear CUDA cache before processing
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        patches_collected = 0
        
        with torch.no_grad():
            for sample_idx, sample in enumerate(train_data):
                # Clear cache periodically to prevent memory buildup
                if sample_idx > 0 and sample_idx % 10 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                if patches_collected >= max_patches:
                    break
                
                try:
                    # Get image data
                    if 'image' in sample:
                        image = sample['image']  # HWC numpy array
                        h, w = image.shape[:2]
                        x = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                    elif 'image_path' in sample:
                        image_path = sample['image_path']
                        if not os.path.exists(image_path):
                            continue
                        image = cv2.imread(image_path)
                        if image is None:
                            continue
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        h, w = image.shape[:2]
                        x = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                    else:
                        continue
                    
                    # Ensure H, W are divisible by model stride
                    stride = int(model.model.stride.max() if hasattr(model.model, 'stride') else 32)
                    _, _, H, W = x.shape
                    new_H = math.ceil(H / stride) * stride
                    new_W = math.ceil(W / stride) * stride
                    if new_H != H or new_W != W:
                        x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
                    
                    # Get ground truth labels
                    gt_labels = sample.get('labels', []) or sample.get('label', [])
                    if not gt_labels:
                        continue
                    
                    # Get actual input tensor dimensions before forward pass
                    _, _, input_h, input_w = x.shape
                    
                    # Run model forward pass (hook will capture conv output)
                    captured_output = None
                    predictions = model(x)[0]
                    
                    # Delete input tensor immediately
                    del x
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Extract activations from captured output
                    if captured_output is None:
                        continue
                    
                    # captured_output shape: [1, channels, H_fm, W_fm] (single batch)
                    _, channels, fm_h, fm_w = captured_output.shape
                    
                    # Calculate stride from input to feature map
                    stride_h = input_h / fm_h
                    stride_w = input_w / fm_w
                    
                    # Process each GT object
                    for gt in gt_labels:
                        if patches_collected >= max_patches:
                            break
                        
                        # Handle both 'class_id' and 'class' keys
                        gt_class = gt.get('class_id', gt.get('class', None))
                        if gt_class is None or gt_class not in classes:
                            continue
                        
                        # Convert GT label to image coordinates
                        # Labels are in YOLO format (normalized x_center, y_center, width, height)
                        if 'x_center' in gt and 'y_center' in gt:
                            # YOLO format (normalized)
                            xc = gt['x_center'] * input_w
                            yc = gt['y_center'] * input_h
                        else:
                            # Fallback: try to convert from bbox format
                            gt_bbox = convert_label_to_xyxy(gt, input_w, input_h)
                            xc = (gt_bbox[0] + gt_bbox[2]) / 2.0
                            yc = (gt_bbox[1] + gt_bbox[3]) / 2.0
                        
                        # Map to feature map coordinates
                        patch_col = int(xc / stride_w)
                        patch_row = int(yc / stride_h)
                        
                        # Clamp to valid feature map bounds
                        patch_row = max(0, min(patch_row, fm_h - 1))
                        patch_col = max(0, min(patch_col, fm_w - 1))
                        
                        # Extract patch vector: [C] - one pixel across all channels
                        patch_vector = captured_output[0, :, patch_row, patch_col].numpy()
                        
                        # Store activations per channel and class
                        for ch_idx in range(channels):
                            activations[ch_idx][gt_class].append(float(patch_vector[ch_idx]))
                        
                        patches_collected += 1
                    
                    # Clear captured output
                    captured_output = None
                    
                except Exception as e:
                    logger.warning(f"Error processing sample {sample_idx}: {e}")
                    continue
        
        logger.info(f"Collected {patches_collected} patches from {sample_idx + 1} samples")
        
    finally:
        # Always remove hook
        hook_handle.remove()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return activations

