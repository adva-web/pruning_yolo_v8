#!/usr/bin/env python3
"""
C2f Hybrid Conv Pruning
Implements hybrid pruning for C2f blocks:
- Conv0: Gamma pruning (BN gamma-based channel selection)
- After-concat convs: Activation pruning (hook-based + clustering)
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from ultralytics import YOLO
from typing import List, Dict, Tuple, Optional

from yolov8_utils import get_all_conv2d_layers
from c2f_utils import is_c2f_block
from c2f_hybrid_utils import find_following_bn, analyze_c2f_block_structure
from c2f_activation_hook import collect_c2f_conv_activations
from yolo_layer_pruner import YoloLayerPruner
from clustering import select_optimal_components

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prune_c2f_conv0_gamma(
    model: YOLO,
    block_idx: int,
    conv0_idx: int = 0,
    pruning_ratio: float = 0.5
) -> Tuple[YOLO, Optional[Dict]]:
    """
    Prune Conv0 in a C2f block using gamma values.
    
    Args:
        model: YOLO model
        block_idx: Index of the C2f block
        conv0_idx: Index of Conv0 within the block (typically 0)
        pruning_ratio: Ratio of channels to remove (default: 0.5 = 50%)
    
    Returns:
        Tuple of (pruned_model, pruning_details)
    """
    torch_model = model.model
    detection_model = torch_model.model
    
    if block_idx >= len(detection_model):
        logger.warning(f"Block {block_idx} out of range")
        return model, None
    
    block = detection_model[block_idx]
    if not is_c2f_block(block):
        logger.warning(f"Block {block_idx} is not a C2f block")
        return model, None
    
    all_convs = get_all_conv2d_layers(block)
    if conv0_idx >= len(all_convs):
        logger.warning(f"Conv {conv0_idx} out of range for block {block_idx}")
        return model, None
    
    target_conv = all_convs[conv0_idx]
    num_channels = target_conv.weight.shape[0]
    channels_to_keep = int(num_channels * (1 - pruning_ratio))
    channels_to_remove = num_channels - channels_to_keep
    
    # Find corresponding BN
    bn_layer = find_following_bn(block, conv0_idx)
    if bn_layer is None:
        logger.warning(f"No BatchNorm found for Block {block_idx}, Conv {conv0_idx}")
        return model, None
    
    # Get gamma values and select channels to keep
    gamma_values = bn_layer.weight.data.abs().cpu().numpy()
    indices_sorted = np.argsort(gamma_values)  # ascending: lowest gamma first
    indices_to_keep = indices_sorted[channels_to_remove:]  # keep highest gamma
    indices_to_remove = indices_sorted[:channels_to_remove]
    
    logger.info(f"Gamma pruning Block {block_idx}, Conv {conv0_idx}: {num_channels} -> {channels_to_keep} channels")
    
    # Apply pruning
    with torch.no_grad():
        target_conv.weight[indices_to_remove] = 0
        if target_conv.bias is not None:
            target_conv.bias[indices_to_remove] = 0
        
        bn_layer.weight[indices_to_remove] = 0
        bn_layer.bias[indices_to_remove] = 0
        bn_layer.running_mean[indices_to_remove] = 0
        bn_layer.running_var[indices_to_remove] = 1
    
    pruning_details = {
        'block_idx': block_idx,
        'conv_in_block_idx': conv0_idx,
        'original_channels': num_channels,
        'remaining_channels': channels_to_keep,
        'pruned_channels': channels_to_remove,
        'pruning_ratio': channels_to_remove / num_channels,
        'method': 'gamma'
    }
    
    return model, pruning_details


def prune_c2f_after_concat_activation(
    model: YOLO,
    block_idx: int,
    conv_idx: int,
    train_data: List,
    classes: List[int],
    data_yaml: str,
    max_patches: Optional[int] = 100000
) -> Tuple[YOLO, Optional[Dict]]:
    """
    Prune an after-concat Conv in a C2f block using activation-based method with clustering.
    
    Args:
        model: YOLO model
        block_idx: Index of the C2f block
        conv_idx: Index of Conv within the block (after concat)
        train_data: Training data for activation extraction
        classes: List of class indices
        data_yaml: Path to data YAML file
        max_patches: Maximum number of patches to collect (default: 100k)
    
    Returns:
        Tuple of (pruned_model, pruning_details)
    """
    torch_model = model.model
    detection_model = torch_model.model
    
    if block_idx >= len(detection_model):
        logger.warning(f"Block {block_idx} out of range")
        return model, None
    
    block = detection_model[block_idx]
    if not is_c2f_block(block):
        logger.warning(f"Block {block_idx} is not a C2f block")
        return model, None
    
    all_convs = get_all_conv2d_layers(block)
    if conv_idx >= len(all_convs):
        logger.warning(f"Conv {conv_idx} out of range for block {block_idx}")
        return model, None
    
    target_conv = all_convs[conv_idx]
    num_channels = target_conv.weight.shape[0]
    
    # Collect activations using hook
    logger.info(f"Collecting activations for Block {block_idx}, Conv {conv_idx}...")
    try:
        train_activations = collect_c2f_conv_activations(
            model=model,
            target_conv=target_conv,
            train_data=train_data,
            classes=classes,
            max_patches=max_patches,
            sampling='gt'
        )
    except Exception as e:
        logger.error(f"Activation collection failed: {e}")
        return model, None
    
    if not train_activations or all(len(v) == 0 for v in train_activations.values()):
        logger.warning("No activations found, skipping pruning")
        return model, None
    
    # Create layer space and select optimal components using clustering
    logger.info("Building graph space and selecting optimal channels...")
    try:
        graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
        
        # Get layer weights (flattened per channel for weight-based selection)
        layer_weights = target_conv.weight.data.detach().cpu().numpy()
        layer_weights_flat = layer_weights.reshape(num_channels, -1).mean(axis=1)  # Average weight per channel
        
        # Select optimal components using clustering (k-medoids + weighted selection)
        # The function will automatically determine optimal k via knee detection
        optimal_components = select_optimal_components(
            graph_space,
            layer_weights_flat,
            num_channels,
            num_channels,  # k_value parameter (not directly used, but passed for compatibility)
            weight_form=True
        )
        
        channels_to_keep = len(optimal_components)
        channels_to_remove = num_channels - channels_to_keep
        
        logger.info(f"Activation analysis: {num_channels} -> {channels_to_keep} channels (remove {channels_to_remove})")
        
    except Exception as e:
        logger.error(f"Channel selection failed: {e}")
        return model, None
    
    # Apply pruning
    with torch.no_grad():
        all_indices = list(range(num_channels))
        indices_to_remove = [i for i in all_indices if i not in optimal_components]
        
        target_conv.weight[indices_to_remove] = 0
        if target_conv.bias is not None:
            target_conv.bias[indices_to_remove] = 0
        
        # Find and zero corresponding BatchNorm channels
        bn_layer = find_following_bn(block, conv_idx)
        if bn_layer is not None:
            bn_layer.weight[indices_to_remove] = 0
            bn_layer.bias[indices_to_remove] = 0
            bn_layer.running_mean[indices_to_remove] = 0
            bn_layer.running_var[indices_to_remove] = 1
    
    pruning_details = {
        'block_idx': block_idx,
        'conv_in_block_idx': conv_idx,
        'original_channels': num_channels,
        'remaining_channels': channels_to_keep,
        'pruned_channels': channels_to_remove,
        'pruning_ratio': channels_to_remove / num_channels,
        'method': 'activation'
    }
    
    logger.info(f"Activation pruning applied successfully!")
    return model, pruning_details

