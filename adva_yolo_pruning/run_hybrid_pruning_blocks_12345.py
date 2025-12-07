#!/usr/bin/env python3
"""
Hybrid Pruning Experiment: Blocks 1, 2, 3, 4, 5
- Blocks 1, 3, 5: Activation pruning (regular Conv blocks)
- Blocks 2, 4: Gamma pruning (C2f blocks)

Includes inference time measurement before and after pruning.
Intermediate fine-tuning: 5 epochs after each layer
Final fine-tuning: 20 epochs
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruning_yolo_v8_sequential_fix import (
    load_training_data,
    load_validation_data,
    count_active_channels
)
from yolov8_utils import (
    get_all_conv2d_layers,
    build_mini_net,
    get_raw_objects_debug_v8,
    aggregate_activations_from_matches
)
from yolo_layer_pruner import YoloLayerPruner
from pruning_yolo_v8 import (
    prune_conv2d_in_block_with_activations,
    select_optimal_components
)
from c2f_utils import is_c2f_block


def measure_inference_time(model, data_yaml, num_runs=3):
    """Measure inference time for a model."""
    inference_times = []
    
    for i in range(num_runs):
        try:
            metrics = model.val(data=data_yaml, verbose=False)
            if hasattr(metrics, 'speed') and metrics.speed is not None:
                if isinstance(metrics.speed, dict):
                    inference_ms = metrics.speed.get('inference', None)
                    if inference_ms is not None:
                        inference_times.append(inference_ms)
                elif isinstance(metrics.speed, (int, float)):
                    inference_times.append(metrics.speed)
        except Exception as e:
            print(f"⚠️  Inference time measurement run {i+1} failed: {e}")
            continue
    
    if len(inference_times) > 0:
        avg_inference_time = sum(inference_times) / len(inference_times)
        return avg_inference_time
    else:
        return None


def find_following_bn(block: nn.Module, conv_in_block_idx: int):
    """Find the BatchNorm layer following the specified Conv2d in a block."""
    all_convs = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(all_convs):
        return None
    
    target_conv = all_convs[conv_in_block_idx]
    target_out_channels = target_conv.weight.shape[0]
    
    # Find BN by matching channel count
    matching_bns = []
    for module in block.modules():
        if isinstance(module, nn.BatchNorm2d):
            if module.num_features == target_out_channels:
                matching_bns.append(module)
    
    if len(matching_bns) > 0:
        return matching_bns[0]
    return None


def gamma_prune_c2f_block(model, block_idx, conv_in_block_idx, pruning_ratio=0.5, data_yaml=None):
    """
    Prune a Conv layer in a C2f block using gamma values.
    
    Args:
        model: YOLO model
        block_idx: Index of the C2f block
        conv_in_block_idx: Index of Conv within the block
        pruning_ratio: Ratio of channels to remove (default: 0.5 = 50%)
        data_yaml: Path to data YAML (for potential future use)
    
    Returns:
        Tuple of (pruned_model, pruning_details)
    """
    torch_model = model.model
    detection_model = torch_model.model
    
    block = detection_model[block_idx]
    all_convs = get_all_conv2d_layers(block)
    
    if conv_in_block_idx >= len(all_convs):
        print(f"⚠️  conv_in_block_idx {conv_in_block_idx} out of range for block {block_idx}")
        return model, None
    
    target_conv = all_convs[conv_in_block_idx]
    num_channels = target_conv.weight.shape[0]
    channels_to_keep = int(num_channels * (1 - pruning_ratio))
    channels_to_remove = num_channels - channels_to_keep
    
    # Find corresponding BN
    bn_layer = find_following_bn(block, conv_in_block_idx)
    
    if bn_layer is None:
        print(f"⚠️  No BatchNorm found for Block {block_idx}, Conv {conv_in_block_idx}")
        return model, None
    
    # Get gamma values and select channels to keep
    gamma_values = bn_layer.weight.data.abs().cpu().numpy()
    indices_sorted = np.argsort(gamma_values)  # ascending: lowest gamma first
    indices_to_keep = indices_sorted[channels_to_remove:]  # keep highest gamma
    indices_to_remove = indices_sorted[:channels_to_remove]
    
    print(f"📊 Gamma pruning for Block {block_idx}, Conv {conv_in_block_idx}:")
    print(f"   Original channels: {num_channels}")
    print(f"   Channels to keep: {channels_to_keep}")
    print(f"   Channels to remove: {channels_to_remove}")
    print(f"   Pruning ratio: {(channels_to_remove/num_channels*100):.1f}%")
    
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
        'conv_in_block_idx': conv_in_block_idx,
        'original_channels': num_channels,
        'remaining_channels': channels_to_keep,
        'pruned_channels': channels_to_remove,
        'pruning_ratio': channels_to_remove / num_channels,
        'method': 'gamma'
    }
    
    print(f"   ✅ Gamma pruning applied successfully!")
    return model, pruning_details


def activation_prune_regular_block(model, block_idx, conv_in_block_idx, train_data, classes, data_yaml):
    """
    Prune a Conv layer in a regular block using activation-based method.
    
    Args:
        model: YOLO model
        block_idx: Index of the block
        conv_in_block_idx: Index of Conv within the block
        train_data: Training data for activation extraction
        classes: List of class indices
        data_yaml: Path to data YAML
    
    Returns:
        Tuple of (pruned_model, pruning_details)
    """
    model_path = "temp_activation_pruning.pt"
    model.save(model_path)
    
    try:
        # Use existing activation pruning function
        pruned_model = prune_conv2d_in_block_with_activations(
            model_path=model_path,
            train_data=train_data,
            valid_data=[],  # Not needed for pruning
            classes=classes,
            block_idx=block_idx,
            conv_in_block_idx=conv_in_block_idx,
            log_file=None,
            data_yaml=data_yaml
        )
        
        # Count active channels
        torch_model = pruned_model.model
        detection_model = torch_model.model
        block = detection_model[block_idx]
        all_convs = get_all_conv2d_layers(block)
        target_conv = all_convs[conv_in_block_idx]
        
        original_channels = target_conv.weight.shape[0]
        active_channels = count_active_channels(target_conv)
        pruned_channels = original_channels - active_channels
        
        pruning_details = {
            'block_idx': block_idx,
            'conv_in_block_idx': conv_in_block_idx,
            'original_channels': original_channels,
            'remaining_channels': active_channels,
            'pruned_channels': pruned_channels,
            'pruning_ratio': pruned_channels / original_channels if original_channels > 0 else 0,
            'method': 'activation'
        }
        
        return pruned_model, pruning_details
    except Exception as e:
        print(f"⚠️  Activation pruning failed: {e}")
        return model, None
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)


def main():
    """Main experiment function."""
    print("=" * 80)
    print("HYBRID PRUNING EXPERIMENT: BLOCKS 1, 2, 3, 4, 5")
    print("=" * 80)
    print("Strategy:")
    print("  - Blocks 1, 3, 5: Activation pruning (regular Conv blocks)")
    print("  - Blocks 2, 4: Gamma pruning (C2f blocks)")
    print("=" * 80)
    
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    
    # Load model
    print("\n📥 Loading model...")
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Verify block types
    print("\n🔍 Verifying block types:")
    target_blocks = [1, 2, 3, 4, 5]
    for block_idx in target_blocks:
        if block_idx < len(detection_model):
            block = detection_model[block_idx]
            is_c2f = is_c2f_block(block)
            block_type = "C2f" if is_c2f else "Regular Conv"
            expected_method = "gamma" if block_idx in [2, 4] else "activation"
            print(f"   Block {block_idx}: {block_type} → {expected_method.upper()} pruning")
            
            if block_idx in [2, 4] and not is_c2f:
                print(f"      ⚠️  WARNING: Block {block_idx} is not C2f, but we'll use gamma pruning anyway")
            if block_idx in [1, 3, 5] and is_c2f:
                print(f"      ⚠️  WARNING: Block {block_idx} is C2f, but we'll use activation pruning")
        else:
            print(f"   Block {block_idx}: ⚠️  Out of range")
    
    # Measure original model metrics (inference time + mAP)
    print("\n" + "=" * 80)
    print("MEASURING ORIGINAL MODEL METRICS")
    print("=" * 80)
    original_model = YOLO(model_path)
    original_inference_time = None
    original_metrics_dict = None
    try:
        print("   Warm-up run (first validation)...")
        _ = original_model.val(data=data_yaml, verbose=False)  # Warm-up run
        
        print("   Running multiple validation runs for accurate measurement...")
        original_inference_time = measure_inference_time(original_model, data_yaml, num_runs=3)
        
        if original_inference_time is not None:
            print(f"   ✅ Original model inference time (averaged over 3 runs): {original_inference_time:.2f} ms/image")
        else:
            # Fallback: try single run
            print("   ⚠️  Fallback to single run...")
            original_metrics = original_model.val(data=data_yaml, verbose=False)
            if hasattr(original_metrics, 'speed') and original_metrics.speed is not None:
                if isinstance(original_metrics.speed, dict):
                    original_inference_time = original_metrics.speed.get('inference', None)
                elif isinstance(original_metrics.speed, (int, float)):
                    original_inference_time = original_metrics.speed
        
        # Get mAP metrics from final validation run
        print("   Getting mAP metrics...")
        original_metrics = original_model.val(data=data_yaml, verbose=False)
        original_metrics_dict = {
            'mAP50-95': original_metrics.results_dict.get('metrics/mAP50-95(B)', 0),
            'mAP50': original_metrics.results_dict.get('metrics/mAP50(B)', 0),
            'precision': original_metrics.results_dict.get('metrics/precision(B)', 0),
            'recall': original_metrics.results_dict.get('metrics/recall(B)', 0)
        }
        print(f"   ✅ Original model mAP@0.5: {original_metrics_dict['mAP50']:.4f}")
        print(f"   ✅ Original model mAP@0.5:0.95: {original_metrics_dict['mAP50-95']:.4f}")
        print(f"   ✅ Original model Precision: {original_metrics_dict['precision']:.4f}")
        print(f"   ✅ Original model Recall: {original_metrics_dict['recall']:.4f}")
        
        if original_inference_time is None:
            print(f"   ⚠️  Could not measure original model inference time")
    except Exception as e:
        print(f"   ⚠️  Error measuring original metrics: {e}")
    
    # Load data - ALL available data for activation extraction
    print("\n📥 Loading ALL training data for activation extraction...")
    train_data = load_training_data(data_yaml, max_samples=None)
    if len(train_data) == 0:
        print("❌ No training data loaded.")
        return
    
    print(f"   ✅ Loaded {len(train_data)} training samples")
    
    print("📥 Loading ALL validation data for activation extraction...")
    try:
        valid_data = load_validation_data(data_yaml, max_samples=None)
        all_activation_data = train_data + valid_data
        print(f"   ✅ Total samples for activation extraction: {len(all_activation_data)}")
        print(f"      - Training: {len(train_data)} samples")
        print(f"      - Validation: {len(valid_data)} samples")
    except Exception as e:
        print(f"   ⚠️  Could not load validation data: {e}, using training data only")
        all_activation_data = train_data
        valid_data = []
        print(f"   ✅ Using only training data: {len(all_activation_data)} samples")
    
    # Load classes
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        classes = list(range(len(data_config['names'])))
    
    # Pruning configuration
    pruning_config = [
        {'block_idx': 1, 'conv_idx': 0, 'method': 'activation'},
        {'block_idx': 2, 'conv_idx': 0, 'method': 'gamma'},
        {'block_idx': 3, 'conv_idx': 0, 'method': 'activation'},
        {'block_idx': 4, 'conv_idx': 0, 'method': 'gamma'},
        {'block_idx': 5, 'conv_idx': 0, 'method': 'activation'}
    ]
    
    # Store pruning results
    all_pruning_details = []
    
    print("\n" + "=" * 80)
    print("PRUNING PROCESS")
    print("=" * 80)
    
    # Prune each block
    for i, config in enumerate(pruning_config):
        block_idx = config['block_idx']
        conv_idx = config['conv_idx']
        method = config['method']
        
        print(f"\n{'='*80}")
        print(f"PRUNING LAYER {i+1}/{len(pruning_config)}: Block {block_idx}, Conv {conv_idx} ({method.upper()})")
        print(f"{'='*80}")
        
        if block_idx >= len(detection_model):
            print(f"   ⚠️  Block {block_idx} out of range, skipping")
            continue
        
        block = detection_model[block_idx]
        all_convs = get_all_conv2d_layers(block)
        
        if conv_idx >= len(all_convs):
            print(f"   ⚠️  Conv {conv_idx} out of range for block {block_idx}, skipping")
            continue
        
        if method == 'gamma':
            # Gamma pruning for C2f blocks
            model, details = gamma_prune_c2f_block(
                model=model,
                block_idx=block_idx,
                conv_in_block_idx=conv_idx,
                pruning_ratio=0.5,
                data_yaml=data_yaml
            )
        else:
            # Activation pruning for regular blocks
            model, details = activation_prune_regular_block(
                model=model,
                block_idx=block_idx,
                conv_in_block_idx=conv_idx,
                train_data=all_activation_data,
                classes=classes,
                data_yaml=data_yaml
            )
        
        if details:
            all_pruning_details.append(details)
        
        # Fine-tune after each layer
        print(f"\n🔄 Fine-tuning for 5 epochs after Block {block_idx} pruning...")
        try:
            model.train(data=data_yaml, epochs=5, verbose=True)
            print(f"   ✅ Fine-tuning completed")
        except Exception as e:
            print(f"   ⚠️  Fine-tuning failed: {e}")
    
    # Final fine-tuning
    print("\n" + "=" * 80)
    print("FINAL FINE-TUNING (20 EPOCHS)")
    print("=" * 80)
    try:
        model.train(data=data_yaml, epochs=20, verbose=True)
        print("✅ Final fine-tuning completed")
    except Exception as e:
        print(f"⚠️  Final fine-tuning failed: {e}")
    
    # Measure pruned model metrics (inference time + mAP)
    print("\n" + "=" * 80)
    print("MEASURING PRUNED MODEL METRICS")
    print("=" * 80)
    pruned_inference_time = None
    pruned_metrics_dict = None
    try:
        print("   Warm-up run (first validation)...")
        _ = model.val(data=data_yaml, verbose=False)  # Warm-up run
        
        print("   Running multiple validation runs for accurate measurement...")
        pruned_inference_time = measure_inference_time(model, data_yaml, num_runs=3)
        
        if pruned_inference_time is not None:
            print(f"   ✅ Pruned model inference time (averaged over 3 runs): {pruned_inference_time:.2f} ms/image")
        else:
            # Fallback: try single run
            print("   ⚠️  Fallback to single run...")
            pruned_metrics_temp = model.val(data=data_yaml, verbose=False)
            if hasattr(pruned_metrics_temp, 'speed') and pruned_metrics_temp.speed is not None:
                if isinstance(pruned_metrics_temp.speed, dict):
                    pruned_inference_time = pruned_metrics_temp.speed.get('inference', None)
                elif isinstance(pruned_metrics_temp.speed, (int, float)):
                    pruned_inference_time = pruned_metrics_temp.speed
        
        # Get mAP metrics from final validation run
        print("   Getting mAP metrics...")
        pruned_metrics = model.val(data=data_yaml, verbose=False)
        pruned_metrics_dict = {
            'mAP50-95': pruned_metrics.results_dict.get('metrics/mAP50-95(B)', 0),
            'mAP50': pruned_metrics.results_dict.get('metrics/mAP50(B)', 0),
            'precision': pruned_metrics.results_dict.get('metrics/precision(B)', 0),
            'recall': pruned_metrics.results_dict.get('metrics/recall(B)', 0)
        }
        print(f"   ✅ Pruned model mAP@0.5: {pruned_metrics_dict['mAP50']:.4f}")
        print(f"   ✅ Pruned model mAP@0.5:0.95: {pruned_metrics_dict['mAP50-95']:.4f}")
        print(f"   ✅ Pruned model Precision: {pruned_metrics_dict['precision']:.4f}")
        print(f"   ✅ Pruned model Recall: {pruned_metrics_dict['recall']:.4f}")
        
        if pruned_inference_time is None:
            print(f"   ⚠️  Could not measure pruned model inference time")
    except Exception as e:
        print(f"   ⚠️  Error measuring pruned metrics: {e}")
    
    # Calculate speedup
    speedup = None
    if original_inference_time is not None and pruned_inference_time is not None:
        if pruned_inference_time > 0:
            speedup = original_inference_time / pruned_inference_time
    
    # Final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    
    print("\n📊 Pruning Results:")
    total_original = 0
    total_remaining = 0
    total_pruned = 0
    
    for details in all_pruning_details:
        method_name = details['method'].upper()
        print(f"   Block {details['block_idx']}, Conv {details['conv_in_block_idx']} ({method_name}):")
        print(f"      Original: {details['original_channels']} channels")
        print(f"      Remaining: {details['remaining_channels']} channels")
        print(f"      Pruned: {details['pruned_channels']} channels ({details['pruning_ratio']*100:.1f}%)")
        total_original += details['original_channels']
        total_remaining += details['remaining_channels']
        total_pruned += details['pruned_channels']
    
    print(f"\n   Total:")
    print(f"      Original: {total_original} channels")
    print(f"      Remaining: {total_remaining} channels")
    print(f"      Pruned: {total_pruned} channels")
    if total_original > 0:
        print(f"      Overall pruning ratio: {(total_pruned/total_original*100):.1f}%")
    
    print("\n⏱️  Inference Time Comparison:")
    print(f"   Original model: {original_inference_time:.2f} ms/image" if original_inference_time else "   Original model: N/A")
    print(f"   Pruned model: {pruned_inference_time:.2f} ms/image" if pruned_inference_time else "   Pruned model: N/A")
    if speedup is not None:
        print(f"   Speedup: {speedup:.2f}x")
    
    print("\n📊 mAP Metrics Comparison:")
    if original_metrics_dict and pruned_metrics_dict:
        print(f"   Original model:")
        print(f"      mAP@0.5: {original_metrics_dict['mAP50']:.4f}")
        print(f"      mAP@0.5:0.95: {original_metrics_dict['mAP50-95']:.4f}")
        print(f"      Precision: {original_metrics_dict['precision']:.4f}")
        print(f"      Recall: {original_metrics_dict['recall']:.4f}")
        print(f"   Pruned model:")
        print(f"      mAP@0.5: {pruned_metrics_dict['mAP50']:.4f} ({pruned_metrics_dict['mAP50'] - original_metrics_dict['mAP50']:+.4f})")
        print(f"      mAP@0.5:0.95: {pruned_metrics_dict['mAP50-95']:.4f} ({pruned_metrics_dict['mAP50-95'] - original_metrics_dict['mAP50-95']:+.4f})")
        print(f"      Precision: {pruned_metrics_dict['precision']:.4f} ({pruned_metrics_dict['precision'] - original_metrics_dict['precision']:+.4f})")
        print(f"      Recall: {pruned_metrics_dict['recall']:.4f} ({pruned_metrics_dict['recall'] - original_metrics_dict['recall']:+.4f})")
    else:
        print(f"   ⚠️  Could not retrieve mAP metrics")
    
    # Save pruned model
    output_path = "runs/detect/hybrid_pruning_blocks_12345.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    print(f"\n💾 Pruned model saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

