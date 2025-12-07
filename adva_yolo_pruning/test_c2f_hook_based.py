#!/usr/bin/env python3
"""
Test script for Strategy 1: Hook-Based Activation Pruning on C2f blocks.
Tests hook-based activation pruning on Block 2 (C2f block).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from pruning_c2f_hook_based import prune_c2f_conv_with_hook_activations
from pruning_yolo_v8_sequential_fix import (
    load_training_data,
    load_validation_data,
    count_active_channels
)
from yolov8_utils import get_all_conv2d_layers
from c2f_utils import is_c2f_block
import yaml


def test_hook_based_pruning():
    """Test hook-based activation pruning on C2f block (Block 2)."""
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    block_idx = 2  # C2f block
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False
    
    # Load data (use fewer samples to reduce memory usage)
    print("📥 Loading training data...")
    train_data = load_training_data(data_yaml, max_samples=20)  # Reduced from 50 to 20
    if len(train_data) == 0:
        print("❌ No training data loaded.")
        return False
    
    print("📥 Loading validation data...")
    try:
        valid_data = load_validation_data(data_yaml, max_samples=30)
    except Exception:
        valid_data = train_data[:20]
    
    # Load classes
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        classes = list(range(len(data_config['names'])))
    
    # Load model
    print("\n🚀 Loading model...")
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Verify block is C2f
    if block_idx >= len(detection_model):
        print(f"❌ Block index {block_idx} out of range")
        return False
    
    block = detection_model[block_idx]
    if not is_c2f_block(block):
        print(f"❌ Block {block_idx} is not a C2f block")
        return False
    
    # Get all convs in block
    convs = get_all_conv2d_layers(block)
    print(f"\n✅ C2f Block {block_idx} found with {len(convs)} Conv layers")
    
    # Print initial state
    print(f"\n📊 Initial Conv Layer Status:")
    for idx, conv in enumerate(convs):
        print(f"   Conv {idx}: {conv.weight.shape[1]} → {conv.weight.shape[0]} channels")
    
    # Prune all convs sequentially, accumulating changes
    pruned_details = []
    fine_tune_epochs = 3  # Fine-tune after each layer
    
    for conv_idx in range(len(convs)):
        print(f"\n{'='*70}")
        print(f"ITERATION {conv_idx + 1}/{len(convs)}: Pruning Conv {conv_idx}")
        print(f"{'='*70}")
        
        # Get current conv state
        current_block = detection_model[block_idx]
        current_convs = get_all_conv2d_layers(current_block)
        current_conv = current_convs[conv_idx]
        original_channels = current_conv.weight.shape[0]
        
        print(f"Original channels: {original_channels}")
        
        try:
            # Save current model state
            temp_path = f"temp_c2f_hook_test_conv{conv_idx}.pt"
            model.save(temp_path)
            
            # Prune using hook-based activation
            pruned_model = prune_c2f_conv_with_hook_activations(
                model_path=temp_path,
                block_idx=block_idx,
                conv_in_block_idx=conv_idx,
                train_data=train_data,
                valid_data=valid_data,
                classes=classes,
                data_yaml=data_yaml
            )
            
            if pruned_model is None:
                print(f"⚠️  Pruning returned None for Conv {conv_idx}, skipping...")
                pruned_details.append({
                    'conv_idx': conv_idx,
                    'status': 'failed',
                    'original': original_channels,
                    'remaining': original_channels,
                    'removed': 0
                })
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                continue
            
            # Copy pruned weights back to main model
            pruned_block = pruned_model.model.model[block_idx]
            pruned_convs = get_all_conv2d_layers(pruned_block)
            pruned_conv = pruned_convs[conv_idx]
            
            main_conv = current_convs[conv_idx]
            
            import torch
            with torch.no_grad():
                main_conv.weight.copy_(pruned_conv.weight)
                if main_conv.bias is not None and pruned_conv.bias is not None:
                    main_conv.bias.copy_(pruned_conv.bias)
            
            # Copy BN if present
            def find_bn_for_conv(block, conv_idx):
                """Find BN layer corresponding to conv at conv_idx."""
                all_convs = get_all_conv2d_layers(block)
                if conv_idx >= len(all_convs):
                    return None
                target_conv = all_convs[conv_idx]
                target_out_ch = target_conv.weight.shape[0]
                
                # Find BN with matching num_features
                for module in block.modules():
                    if isinstance(module, torch.nn.BatchNorm2d) and module.num_features == target_out_ch:
                        return module
                return None
            
            bn_main = find_bn_for_conv(current_block, conv_idx)
            bn_pruned = find_bn_for_conv(pruned_block, conv_idx)
            if bn_main is not None and bn_pruned is not None:
                with torch.no_grad():
                    bn_main.weight.copy_(bn_pruned.weight)
                    bn_main.bias.copy_(bn_pruned.bias)
            
            # Check remaining channels
            remaining_channels = count_active_channels(main_conv)
            removed = original_channels - remaining_channels
            
            print(f"\n✅ Hook-based pruning complete:")
            print(f"   Original: {original_channels} channels")
            print(f"   Remaining: {remaining_channels} channels")
            print(f"   Removed: {removed} channels ({removed/original_channels*100:.1f}%)")
            
            pruned_details.append({
                'conv_idx': conv_idx,
                'status': 'success',
                'original': original_channels,
                'remaining': remaining_channels,
                'removed': removed
            })
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Fine-tune after each layer
            if fine_tune_epochs > 0:
                print(f"\n🔄 Fine-tuning for {fine_tune_epochs} epochs after Conv {conv_idx} pruning...")
                try:
                    model.train(data=data_yaml, epochs=fine_tune_epochs, verbose=True)
                    print(f"✅ Fine-tuning completed")
                except Exception as e:
                    print(f"⚠️  Fine-tuning failed: {e}")
        
        except Exception as e:
            print(f"❌ Pruning failed for Conv {conv_idx}: {e}")
            import traceback
            traceback.print_exc()
            pruned_details.append({
                'conv_idx': conv_idx,
                'status': 'failed',
                'original': original_channels,
                'remaining': original_channels,
                'removed': 0
            })
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Continue to next conv instead of returning
            continue
    
    # Print summary
    print(f"\n{'='*70}")
    print("PRUNING SUMMARY")
    print(f"{'='*70}")
    
    successful = sum(1 for d in pruned_details if d['status'] == 'success')
    total_original = sum(d['original'] for d in pruned_details)
    total_remaining = sum(d['remaining'] for d in pruned_details)
    total_removed = sum(d['removed'] for d in pruned_details)
    overall_pruning_ratio = (total_removed / total_original * 100) if total_original > 0 else 0
    
    print(f"\n✅ Successfully pruned: {successful}/{len(pruned_details)} layers")
    print(f"\n📊 Pruning Statistics:")
    print(f"   Total original channels: {total_original}")
    print(f"   Total remaining channels: {total_remaining}")
    print(f"   Total removed channels: {total_removed}")
    print(f"   Overall pruning ratio: {overall_pruning_ratio:.1f}%")
    
    print(f"\n📋 Layer Details:")
    for d in pruned_details:
        status_icon = "✅" if d['status'] == 'success' else "❌"
        removed = d.get('removed', 0)
        pruning_ratio = (removed / d['original'] * 100) if d['original'] > 0 else 0
        print(f"   {status_icon} Conv {d['conv_idx']}: {d['original']} → {d['remaining']} channels "
              f"(removed {removed}, {pruning_ratio:.1f}%)")
    
    print(f"\n{'='*70}")
    print("✅ All tests completed!")
    print(f"{'='*70}")
    
    return successful > 0


if __name__ == "__main__":
    success = test_hook_based_pruning()
    sys.exit(0 if success else 1)

