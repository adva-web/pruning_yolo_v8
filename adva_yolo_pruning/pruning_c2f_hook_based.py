#!/usr/bin/env python3
"""
C2f Hook-Based Activation Pruning
Implements Strategy 1: Full Model Hook-Based Activation Pruning for C2f blocks.
Uses forward hooks on the full model to extract activations directly from target
conv layers, bypassing sliced_block construction completely.
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

from yolov8_utils import get_all_conv2d_layers, convert_label_to_xyxy, compute_patch_indices, match_prediction_for_gt
from yolo_layer_pruner import YoloLayerPruner
from clustering import select_optimal_components
from c2f_utils import is_c2f_block

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_activations_with_hook(model: YOLO, target_conv: nn.Conv2d, train_data: List, classes: List[int]) -> Dict[int, Dict[int, List[float]]]:
    """
    Extract activations using forward hook on target conv layer.
    
    This function registers a forward hook on the target conv layer,
    runs the full model forward pass on all training samples, and collects
    activations directly from the conv layer output.
    
    Args:
        model: YOLO model
        target_conv: Target Conv2d layer to extract activations from
        train_data: List of training samples (dicts with 'image', 'labels', etc.)
        classes: List of class indices
    
    Returns:
        Dictionary: {channel_idx: {class_id: [activation_values]}}
    """
    num_channels = target_conv.weight.shape[0]
    activations = {ch: {cls: [] for cls in classes} for ch in range(num_channels)}
    
    # Storage for captured activations during forward pass
    captured_output = None
    
    def hook_fn(module, input, output):
        """Forward hook to capture conv layer output."""
        # output shape: [batch, channels, H, W]
        nonlocal captured_output
        # Detach and move to CPU to avoid memory issues
        captured_output = output.detach().cpu()
    
    # Register forward hook
    hook_handle = target_conv.register_forward_hook(hook_fn)
    
    try:
        model.model.eval()
        device = next(target_conv.parameters()).device
        
        # Clear CUDA cache before processing
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for sample_idx, sample in enumerate(train_data):
                # Clear cache periodically to prevent memory buildup
                if sample_idx > 0 and sample_idx % 10 == 0 and device.type == 'cuda':
                    torch.cuda.empty_cache()
                try:
                    # Get image data - could be path or numpy array
                    if 'image' in sample:
                        # image is numpy array (from load_training_data)
                        image = sample['image']  # HWC numpy array
                        h, w = image.shape[:2]
                        
                        # Preprocess image: convert to tensor, normalize, resize
                        x = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0  # 1×3×H×W
                        
                        # Ensure H, W are divisible by model stride
                        stride = int(model.model.stride.max() if hasattr(model.model, 'stride') else 32)  # typically 32
                        _, _, H, W = x.shape
                        new_H = math.ceil(H / stride) * stride
                        new_W = math.ceil(W / stride) * stride
                        if new_H != H or new_W != W:
                            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
                        
                    elif 'image_path' in sample:
                        # image_path is string (need to load)
                        image_path = sample['image_path']
                        if not os.path.exists(image_path):
                            continue
                        
                        image = cv2.imread(image_path)
                        if image is None:
                            continue
                        
                        # Convert BGR to RGB
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        h, w = image.shape[:2]
                        
                        # Preprocess
                        x = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
                        stride = int(model.model.stride.max() if hasattr(model.model, 'stride') else 32)
                        _, _, H, W = x.shape
                        new_H = math.ceil(H / stride) * stride
                        new_W = math.ceil(W / stride) * stride
                        if new_H != H or new_W != W:
                            x = F.interpolate(x, size=(new_H, new_W), mode='bilinear', align_corners=False)
                    else:
                        continue
                    
                    # Get ground truth labels
                    gt_labels = sample.get('labels', []) or sample.get('label', [])
                    if not gt_labels:
                        continue
                    
                    # Run model forward pass (hook will capture conv output during forward)
                    # Clear captured_output before forward pass
                    captured_output = None
                    predictions = model(x)[0]
                    
                    # Get predictions and store on CPU before clearing x
                    pred_boxes = []
                    if predictions.boxes is not None:
                        for box in predictions.boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            pred_boxes.append({'bbox': xyxy.tolist(), 'class': cls})
                    
                    # Get actual input tensor dimensions (after resize) before deleting
                    _, _, input_h, input_w = x.shape
                    
                    # Clear input tensor from GPU immediately after extracting dimensions
                    del x
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                    # Extract activations from captured output
                    if captured_output is None:
                        continue
                    
                    # captured_output shape: [1, channels, H, W] (single batch)
                    if len(captured_output.shape) == 4:
                        fm = captured_output[0]  # [channels, H, W]
                        fm_h, fm_w = fm.shape[1], fm.shape[2]
                        
                        # Calculate stride from input tensor size to feature map size
                        stride_h = input_h / fm_h
                        stride_w = input_w / fm_w
                        
                        # Match GT boxes with predictions and extract activations
                        for gt in gt_labels:
                            gt_class = gt['class_id']
                            gt_bbox = convert_label_to_xyxy(gt, w, h)
                            
                            # Find patch location in feature map
                            patch_row, patch_col = compute_patch_indices(
                                gt_bbox, stride_h, stride_w, fm_h, fm_w
                            )
                            
                            # Match with prediction
                            best_pred, max_iou = match_prediction_for_gt(
                                gt_bbox, gt_class, pred_boxes
                            )
                            
                            # Only collect activations if IOU > 0.5 (matched detection)
                            if max_iou > 0.5:
                                # Extract activation values at this patch location
                                patch_activations = fm[:, patch_row, patch_col].numpy().tolist()
                                
                                # Aggregate by channel and class
                                for ch in range(num_channels):
                                    activations[ch][gt_class].append(patch_activations[ch])
                    
                    # Clear captured output and predictions for next iteration
                    del captured_output
                    del predictions
                    captured_output = None
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    
                except Exception as e:
                    logger.warning(f"Error processing sample {sample_idx}: {e}")
                    captured_output = None
                    # Clear CUDA cache on error
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
    
    finally:
        # Always remove hook
        hook_handle.remove()
    
    # Log statistics
    total_samples = sum(len(acts) for ch_acts in activations.values() for acts in ch_acts.values())
    logger.info(f"✅ Hook-based activation extraction complete:")
    logger.info(f"   Total activation samples collected: {total_samples}")
    for ch in range(num_channels):
        ch_total = sum(len(activations[ch][cls]) for cls in classes)
        logger.debug(f"   Channel {ch}: {ch_total} samples")
    
    return activations


def prune_c2f_conv_with_hook_activations(
    model_path: str,
    block_idx: int,
    conv_in_block_idx: int,
    train_data: List,
    valid_data: List,
    classes: List[int],
    data_yaml: str,
    log_file: Optional[str] = None
) -> YOLO:
    """
    Prune a specific Conv2d layer in a C2f block using hook-based activation extraction.
    
    This function uses forward hooks to extract activations directly from the target
    conv layer, completely avoiding sliced_block construction. This eliminates channel
    mismatch issues in C2f blocks.
    
    Args:
        model_path: Path to YOLO model
        block_idx: Index of the C2f block
        conv_in_block_idx: Index of Conv within the block
        train_data: Training data for activation extraction
        valid_data: Validation data (not currently used but kept for compatibility)
        classes: List of class indices
        data_yaml: Path to data YAML file
        log_file: Optional log file path
    
    Returns:
        Pruned YOLO model
    """
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Verify block is C2f
    if block_idx >= len(detection_model):
        logger.error(f"Block index {block_idx} out of range")
        return model
    
    block = detection_model[block_idx]
    if not is_c2f_block(block):
        logger.warning(f"Block {block_idx} is not a C2f block")
        return model
    
    # Get target conv layer
    conv_layers_in_block = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(conv_layers_in_block):
        logger.warning(f"conv_in_block_idx {conv_in_block_idx} out of range for block {block_idx}")
        return model
    
    target_conv_layer = conv_layers_in_block[conv_in_block_idx]
    num_channels = target_conv_layer.weight.shape[0]
    
    logger.info(f"🎯 Hook-based activation pruning:")
    logger.info(f"   Block: {block_idx} (C2f)")
    logger.info(f"   Conv in block: {conv_in_block_idx}")
    logger.info(f"   Total channels: {num_channels}")
    
    # Extract activations using hook
    logger.info(f"🔍 Extracting activations using forward hook...")
    try:
        train_activations = extract_activations_with_hook(model, target_conv_layer, train_data, classes)
    except Exception as e:
        logger.error(f"Activation extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return model
    
    # Check if we got any activations
    if not train_activations or all(len(v) == 0 for ch_acts in train_activations.values() for v in ch_acts.values()):
        logger.warning("No activations found, skipping pruning.")
        return model
    
    # Create layer space and select optimal components
    logger.info(f"📊 Analyzing activations and selecting optimal channels...")
    try:
        graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
        layer_weights = target_conv_layer.weight.data.detach().cpu().numpy()
        
        # Use aggressive pruning approach (keep 25-50% of channels)
        target_channels = max(num_channels // 2, num_channels // 4)
        optimal_components = select_optimal_components(
            graph_space, 
            layer_weights, 
            num_channels, 
            target_channels
        )
        
        channels_to_keep = len(optimal_components)
        channels_to_remove = num_channels - channels_to_keep
        
        logger.info(f"📊 Activation analysis complete:")
        logger.info(f"  - Total channels: {num_channels}")
        logger.info(f"  - Channels to keep: {channels_to_keep}")
        logger.info(f"  - Channels to remove: {channels_to_remove}")
        logger.info(f"  - Pruning ratio: {(channels_to_remove/num_channels*100):.1f}%")
        
    except Exception as e:
        logger.error(f"Channel selection failed: {e}")
        import traceback
        traceback.print_exc()
        return model
    
    # Apply pruning (zero out non-selected channels)
    logger.info(f"✂️  Applying pruning...")
    with torch.no_grad():
        all_indices = list(range(num_channels))
        indices_to_remove = [i for i in all_indices if i not in optimal_components]
        
        target_conv_layer.weight[indices_to_remove] = 0
        if target_conv_layer.bias is not None:
            target_conv_layer.bias[indices_to_remove] = 0
        
        # Also zero corresponding BN parameters if present
        # Find BN layer with matching num_features
        for module in block.modules():
            if isinstance(module, nn.BatchNorm2d) and module.num_features == num_channels:
                module.weight[indices_to_remove] = 0
                module.bias[indices_to_remove] = 0
                break
    
    logger.info(f"✅ Hook-based activation pruning applied successfully!")
    return model

