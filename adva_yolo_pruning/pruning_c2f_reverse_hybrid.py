#!/usr/bin/env python3
"""
C2f Reverse Hybrid Pruning
Implements Strategy 3: Reverse Hybrid Pruning for C2f blocks.
- Gamma pruning for Conv 0 (before concat)
- Activation pruning for Conv 1+ (after concat)
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from typing import List, Optional
import logging
import os

from yolov8_utils import get_all_conv2d_layers
from c2f_hybrid_utils import (
    analyze_c2f_block_structure,
    get_c2f_conv_categories,
    find_following_bn,
    print_c2f_structure
)
from pruning_c2f_activation import prune_conv2d_in_c2f_with_activations
from pruning_yolo_v8_sequential_fix import count_active_channels

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def gamma_prune_conv_and_bn(conv: nn.Conv2d, bn: Optional[nn.BatchNorm2d], 
                            keep_ratio: float = 0.5) -> int:
    """
    Soft-prune output channels using BN gamma magnitudes (abs). 
    Returns number of channels removed.
    """
    out_ch = conv.weight.shape[0]
    k_keep = max(int(out_ch * keep_ratio), max(1, out_ch // 4))
    
    if bn is None:
        # Fallback to conv weight magnitude if no BN
        scores = conv.weight.detach().abs().mean(dim=(1, 2, 3))
    else:
        scores = bn.weight.detach().abs()
    
    order = torch.argsort(scores, descending=True)
    keep_idx = set(order[:k_keep].tolist())
    remove_idx = [i for i in range(out_ch) if i not in keep_idx]
    
    if not remove_idx:
        return 0
    
    with torch.no_grad():
        conv.weight[remove_idx] = 0
        if conv.bias is not None:
            conv.bias[remove_idx] = 0
        if bn is not None:
            bn.weight[remove_idx] = 0
            bn.bias[remove_idx] = 0
    
    return len(remove_idx)


def prune_c2f_block_reverse_hybrid(
    model_path: str,
    block_idx: int,
    train_data: List,
    valid_data: List,
    classes: List[int],
    data_yaml: str,
    gamma_pruning_ratio: float = 0.5,
    fine_tune_epochs: int = 5
) -> YOLO:
    """
    Reverse hybrid pruning for C2f block:
    - Gamma pruning on Conv 0 (before concat)
    - Activation pruning on Conv 1+ (after concat)
    
    Args:
        model_path: Path to YOLO model
        block_idx: Index of the C2f block
        train_data: Training data for activation extraction
        valid_data: Validation data
        classes: List of class indices
        data_yaml: Path to data YAML file
        gamma_pruning_ratio: Ratio of channels to keep for gamma pruning (default: 0.5)
        fine_tune_epochs: Number of epochs to fine-tune after each pruning step (default: 5)
    
    Returns:
        Pruned YOLO model
    """
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Get C2f block
    if block_idx >= len(detection_model):
        logger.error(f"Block index {block_idx} out of range")
        return model
    
    block = detection_model[block_idx]
    
    # Analyze C2f structure
    structure = analyze_c2f_block_structure(block)
    if structure is None:
        logger.error(f"Block {block_idx} is not a C2f block or structure analysis failed")
        return model
    
    categories = get_c2f_conv_categories(block)
    
    # Print structure
    print_c2f_structure(block, block_idx)
    
    pruning_results = []
    
    # Step 1: Prune Conv 0 (before concat) with GAMMA
    if len(categories['before_concat']) > 0:
        conv0_info = categories['before_concat'][0]  # First conv (Conv 0)
        conv0_idx = conv0_info['idx']
        conv0_layer = conv0_info['conv']
        
        print(f"\n{'='*70}")
        print(f"STEP 1: GAMMA PRUNING - Conv {conv0_idx} (BEFORE concat)")
        print(f"{'='*70}")
        print(f"Channels: {conv0_info['in_channels']} → {conv0_info['out_channels']}")
        print(f"Gamma pruning ratio: {gamma_pruning_ratio} (keep {int(conv0_info['out_channels'] * gamma_pruning_ratio)} channels)")
        
        # Find corresponding BN layer
        bn_layer = find_following_bn(block, conv0_idx)
        
        if bn_layer is None:
            print(f"⚠️  No BatchNorm found for Conv {conv0_idx}, using weight magnitude")
        
        try:
            # Apply gamma pruning
            removed = gamma_prune_conv_and_bn(conv0_layer, bn_layer, keep_ratio=gamma_pruning_ratio)
            remaining = count_active_channels(conv0_layer)
            
            print(f"✅ Gamma pruning complete:")
            print(f"   Original: {conv0_info['out_channels']} channels")
            print(f"   Remaining: {remaining} channels")
            print(f"   Removed: {removed} channels ({removed/conv0_info['out_channels']*100:.1f}%)")
            
            pruning_results.append({
                'conv_idx': conv0_idx,
                'method': 'gamma',
                'original': conv0_info['out_channels'],
                'remaining': remaining,
                'removed': removed
            })
            
            # Fine-tune after Conv 0 pruning
            if fine_tune_epochs > 0:
                print(f"\n🔄 Fine-tuning for {fine_tune_epochs} epochs after Conv {conv0_idx} pruning...")
                try:
                    model.train(data=data_yaml, epochs=fine_tune_epochs, verbose=True)
                    print(f"✅ Fine-tuning completed")
                except Exception as e:
                    logger.warning(f"Fine-tuning failed: {e}")
        
        except Exception as e:
            logger.error(f"Gamma pruning failed for Conv {conv0_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    # Step 2: Prune Conv 1+ (after concat) with ACTIVATION
    if len(categories['after_concat']) > 0:
        print(f"\n{'='*70}")
        print(f"STEP 2: ACTIVATION PRUNING - Convs AFTER concat")
        print(f"{'='*70}")
        
        for conv_info in categories['after_concat']:
            conv_idx = conv_info['idx']
            
            print(f"\n--- Pruning Conv {conv_idx} (AFTER concat) ---")
            print(f"Channels: {conv_info['in_channels']} → {conv_info['out_channels']}")
            
            # Save model state
            temp_path = f"temp_c2f_reverse_hybrid_conv{conv_idx}.pt"
            model.save(temp_path)
            
            try:
                # Apply activation pruning (from pruned model state)
                updated_model = prune_conv2d_in_c2f_with_activations(
                    model_path=temp_path,
                    train_data=train_data,
                    valid_data=valid_data,
                    classes=classes,
                    block_idx=block_idx,
                    conv_in_block_idx=conv_idx,
                    log_file=None,
                    data_yaml=data_yaml
                )
                
                if updated_model is None:
                    logger.warning(f"Activation pruning failed for Conv {conv_idx}, falling back to gamma")
                    # Fallback to gamma pruning
                    conv_layer = conv_info['conv']
                    bn_layer = find_following_bn(block, conv_idx)
                    removed = gamma_prune_conv_and_bn(conv_layer, bn_layer, keep_ratio=gamma_pruning_ratio)
                    remaining = count_active_channels(conv_layer)
                    method_used = 'gamma (fallback)'
                else:
                    # Copy pruned weights back to main model
                    updated_block = updated_model.model.model[block_idx]
                    updated_convs = get_all_conv2d_layers(updated_block)
                    updated_conv = updated_convs[conv_idx]
                    
                    main_convs = get_all_conv2d_layers(block)
                    main_conv = main_convs[conv_idx]
                    
                    with torch.no_grad():
                        main_conv.weight.copy_(updated_conv.weight)
                        if main_conv.bias is not None and updated_conv.bias is not None:
                            main_conv.bias.copy_(updated_conv.bias)
                    
                    # Copy BN if present
                    bn_main = find_following_bn(block, conv_idx)
                    bn_updated = find_following_bn(updated_block, conv_idx)
                    if bn_main is not None and bn_updated is not None:
                        with torch.no_grad():
                            bn_main.weight.copy_(bn_updated.weight)
                            bn_main.bias.copy_(bn_updated.bias)
                    
                    remaining = count_active_channels(main_conv)
                    removed = conv_info['out_channels'] - remaining
                    method_used = 'activation'
                
                print(f"✅ Pruning complete ({method_used}):")
                print(f"   Original: {conv_info['out_channels']} channels")
                print(f"   Remaining: {remaining} channels")
                print(f"   Removed: {removed} channels ({removed/conv_info['out_channels']*100:.1f}%)")
                
                pruning_results.append({
                    'conv_idx': conv_idx,
                    'method': method_used,
                    'original': conv_info['out_channels'],
                    'remaining': remaining,
                    'removed': removed
                })
                
                # Fine-tune after each conv (or after all)
                if fine_tune_epochs > 0:
                    print(f"\n🔄 Fine-tuning for {fine_tune_epochs} epochs after Conv {conv_idx} pruning...")
                    try:
                        model.train(data=data_yaml, epochs=fine_tune_epochs, verbose=True)
                        print(f"✅ Fine-tuning completed")
                    except Exception as e:
                        logger.warning(f"Fine-tuning failed: {e}")
            
            except Exception as e:
                logger.error(f"Activation pruning failed for Conv {conv_idx}: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback to gamma pruning
                print(f"⚠️  Falling back to gamma pruning for Conv {conv_idx}")
                try:
                    conv_layer = conv_info['conv']
                    bn_layer = find_following_bn(block, conv_idx)
                    removed = gamma_prune_conv_and_bn(conv_layer, bn_layer, keep_ratio=gamma_pruning_ratio)
                    remaining = count_active_channels(conv_layer)
                    
                    pruning_results.append({
                        'conv_idx': conv_idx,
                        'method': 'gamma (fallback)',
                        'original': conv_info['out_channels'],
                        'remaining': remaining,
                        'removed': removed
                    })
                except Exception as e2:
                    logger.error(f"Gamma fallback also failed: {e2}")
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Final fine-tune after all activation pruning
        if fine_tune_epochs > 0 and len(categories['after_concat']) > 0:
            print(f"\n🔄 Final fine-tuning for {fine_tune_epochs} epochs after all pruning...")
            try:
                model.train(data=data_yaml, epochs=fine_tune_epochs, verbose=True)
                print(f"✅ Fine-tuning completed")
            except Exception as e:
                logger.warning(f"Fine-tuning failed: {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"REVERSE HYBRID PRUNING SUMMARY")
    print(f"{'='*70}")
    print(f"Block {block_idx} (C2f block):")
    for result in pruning_results:
        method = result['method'].upper()
        print(f"  Conv {result['conv_idx']} ({method}): {result['original']} → {result['remaining']} "
              f"(removed {result['removed']}, {result['removed']/result['original']*100:.1f}%)")
    
    total_original = sum(r['original'] for r in pruning_results)
    total_remaining = sum(r['remaining'] for r in pruning_results)
    total_removed = sum(r['removed'] for r in pruning_results)
    
    if total_original > 0:
        print(f"\nTotal: {total_original} → {total_remaining} channels "
              f"(removed {total_removed}, {total_removed/total_original*100:.1f}%)")
    
    print(f"{'='*70}\n")
    
    return model

