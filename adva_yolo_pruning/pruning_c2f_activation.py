#!/usr/bin/env python3
"""
C2f-Aware Activation Pruning for YOLOv8
This module provides pruning functions specifically designed for C2f blocks,
using C2f-aware mini-net construction to handle the internal split-concat structure.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from ultralytics import YOLO
from typing import List, Dict, Tuple, Any
import os

from yolov8_utils import build_mini_net, extract_conv_weights_norm, get_all_conv2d_layers, get_raw_objects_debug_v8, aggregate_activations_from_matches
from yolo_layer_pruner import YoloLayerPruner
from clustering import select_optimal_components, kmedoids_fasterpam
from c2f_utils import is_c2f_block, build_c2f_aware_mini_net

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prune_conv2d_in_c2f_with_activations(
    model_path, 
    train_data, 
    valid_data, 
    classes, 
    block_idx=2, 
    conv_in_block_idx=0, 
    log_file="pruning_c2f_block_conv.txt", 
    data_yaml="data/VOC_adva.yaml"
):
    """
    Prune a specific Conv2d layer inside a block with C2f-aware activation extraction.
    
    This function is similar to prune_conv2d_in_block_with_activations but handles
    C2f blocks specially by using C2f-aware mini-net construction for Convs after
    the internal concatenation.
    
    Args:
        model_path: Path to YOLO model
        train_data: Training data for activation extraction
        valid_data: Validation data (not currently used but kept for compatibility)
        classes: List of class indices
        block_idx: Index of the block (C2f or regular)
        conv_in_block_idx: Index of Conv within the block
        log_file: Log file path (not currently used)
        data_yaml: Path to data YAML file
    
    Returns:
        Pruned YOLO model
    """
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Get the target block and its Conv2d layers
    block = detection_model[block_idx]
    conv_layers_in_block = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(conv_layers_in_block):
        logger.warning(f"conv_in_block_idx {conv_in_block_idx} out of range for block {block_idx}.")
        return model
    
    target_conv_layer = conv_layers_in_block[conv_in_block_idx]
    
    # Determine if this is a C2f block and build appropriate mini-net
    if is_c2f_block(block):
        print(f"   ℹ️  C2f block detected: using C2f-aware mini-net construction")
        try:
            mini_net = build_c2f_aware_mini_net(
                detection_model=detection_model,
                block_idx=block_idx,
                conv_in_block_idx=conv_in_block_idx,
                target_conv=target_conv_layer,
                get_all_conv2d_layers=get_all_conv2d_layers,
                build_mini_net_standard=build_mini_net
            )
        except Exception as e:
            logger.error(f"Failed to build C2f-aware mini-net: {e}")
            return model
    else:
        # Regular block: use standard sliced_block construction
        print(f"   ℹ️  Regular Conv block detected: using standard sliced_block")
        # Build sliced_block: all blocks before, plus partial block up to target Conv2d
        blocks_up_to = list(detection_model[:block_idx])
        submodules = []
        conv_count = 0
        for sublayer in block.children():
            submodules.append(sublayer)
            if isinstance(sublayer, nn.Conv2d):
                if conv_count == conv_in_block_idx:
                    break
                conv_count += 1
        partial_block = nn.Sequential(*submodules)
        sliced_block = nn.Sequential(*(blocks_up_to + [partial_block]))
        mini_net = build_mini_net(sliced_block, target_conv_layer)
    
    # Extract activations using the mini-net
    try:
        train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
        train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
    except Exception as e:
        logger.error(f"Activation extraction failed: {e}")
        return model
    
    if not train_activations or all(len(v) == 0 for v in train_activations.values()):
        logger.warning("No activations found, skipping pruning.")
        return model
    
    # Create layer space and select optimal components
    graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
    layer_weights = target_conv_layer.weight.data.detach().cpu().numpy()
    
    # Use aggressive pruning approach
    target_channels = max(target_conv_layer.weight.shape[0] // 2, target_conv_layer.weight.shape[0] // 4)
    optimal_components = select_optimal_components(
        graph_space, 
        layer_weights, 
        target_conv_layer.weight.shape[0], 
        target_channels
    )
    
    channels_to_keep = len(optimal_components)
    channels_to_remove = target_conv_layer.weight.shape[0] - channels_to_keep
    
    print(f"📊 Activation analysis complete:")
    print(f"  - Total channels: {target_conv_layer.weight.shape[0]}")
    print(f"  - Channels to keep: {channels_to_keep}")
    print(f"  - Channels to remove: {channels_to_remove}")
    print(f"  - Pruning ratio: {(channels_to_remove/target_conv_layer.weight.shape[0]*100):.1f}%")
    
    # Apply pruning
    with torch.no_grad():
        all_indices = list(range(target_conv_layer.weight.shape[0]))
        indices_to_remove = [i for i in all_indices if i not in optimal_components]
        
        target_conv_layer.weight[indices_to_remove] = 0
        if target_conv_layer.bias is not None:
            target_conv_layer.bias[indices_to_remove] = 0
    
    print(f"✅ C2f-aware activation-based pruning applied successfully!")
    return model

