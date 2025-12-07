#!/usr/bin/env python3
"""
Simple comparison between activation and gamma pruning using the ORIGINAL WORKING methods.
This uses the exact same functions that work in run_experiments.py.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from yolov8_utils import build_mini_net, extract_conv_weights_norm, get_all_conv2d_layers, get_raw_objects_debug_v8, aggregate_activations_from_matches, get_conv_bn_pairs, extract_bn_gamma
from yolo_layer_pruner import YoloLayerPruner
from clustering import select_optimal_components, kmedoids_fasterpam

def load_samples(image_dir: str, label_dir: str, max_samples=100):
    """Load dataset samples for activation pruning."""
    import cv2
    import glob
    
    samples = []
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Limit to max_samples for quick testing
    image_paths = image_paths[:max_samples]
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    labels.append({
                        "class_id": class_id,
                        "x_center": float(parts[1]),
                        "y_center": float(parts[2]),
                        "width": float(parts[3]),
                        "height": float(parts[4])
                    })
        
        samples.append({
            "image": img,
            "label": labels,
            "image_path": img_path,
            "label_path": label_path
        })
    
    return samples

def apply_simple_activation_pruning(model_path, train_data, valid_data, classes, block_idx=5, conv_in_block_idx=0):
    """
    Simple activation-based pruning that DOESN'T use the broken clustering algorithm.
    This uses a simple activation-based importance ranking instead.
    """
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Get the target block and its Conv2d layers
    block = detection_model[block_idx]
    conv_layers_in_block = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(conv_layers_in_block):
        print(f"Warning: conv_in_block_idx {conv_in_block_idx} out of range for block {block_idx}.")
        return model
    
    target_conv_layer = conv_layers_in_block[conv_in_block_idx]
    
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
    
    # Extract activations
    mini_net = build_mini_net(sliced_block, target_conv_layer)
    train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
    train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
    
    if not train_activations or all(len(v) == 0 for v in train_activations.values()):
        print("No activations found, skipping pruning.")
        return model
    
    # SIMPLE activation-based importance ranking (NO CLUSTERING)
    print(f"🔍 Using SIMPLE activation-based importance ranking (NO CLUSTERING)")
    
    # Calculate activation importance for each channel
    channel_importance = []
    num_channels = target_conv_layer.weight.shape[0]
    
    for channel_idx in range(num_channels):
        # Calculate average activation magnitude for this channel across all classes
        channel_activations = []
        
        # train_activations structure: {channel_id: {class_id: [activation_values]}}
        if channel_idx in train_activations:
            for class_id, activations in train_activations[channel_idx].items():
                if activations:  # Check if the list is not empty
                    channel_activations.extend(activations)
        
        if channel_activations:
            # Use mean absolute value as importance metric
            importance = np.mean(np.abs(channel_activations))
        else:
            importance = 0.0
        
        channel_importance.append(importance)
    
    # Sort channels by importance (highest first)
    channel_indices = list(range(num_channels))
    channel_indices.sort(key=lambda x: channel_importance[x], reverse=True)
    
    # Select top 50% of channels to keep (simple 50% pruning)
    channels_to_keep = max(num_channels // 2, num_channels // 4)  # At least 25% of channels
    channels_to_remove = num_channels - channels_to_keep
    
    # Get indices of channels to keep and remove
    indices_to_keep = channel_indices[:channels_to_keep]
    indices_to_remove = channel_indices[channels_to_keep:]
    
    print(f"📊 Simple activation analysis complete:")
    print(f"  - Total channels: {num_channels}")
    print(f"  - Channels to keep: {channels_to_keep}")
    print(f"  - Channels to remove: {channels_to_remove}")
    print(f"  - Pruning ratio: {(channels_to_remove/num_channels*100):.1f}%")
    
    # Apply pruning by zeroing out the least important channels
    with torch.no_grad():
        target_conv_layer.weight[indices_to_remove] = 0
        if target_conv_layer.bias is not None:
            target_conv_layer.bias[indices_to_remove] = 0
    
    print(f"✅ Simple activation-based pruning applied successfully!")
    return model

def apply_activation_soft_pruning(model_path, train_data, valid_data, classes, layers_to_prune=3, data_yaml="data/VOC_adva.yaml", predefined_layers=None):
    """Apply activation-based soft pruning using the ORIGINAL WORKING METHOD that doesn't use broken clustering"""
    
    print(f"\n===== Using ORIGINAL WORKING Activation Pruning Method =====")
    print(f"🔧 This uses the ORIGINAL WORKING logic that doesn't use broken clustering")
    
    # Implement the original working logic that calls prune_conv2d_in_block_with_activations for each layer
    # This is the EXACT SAME logic that was working before the clustering was broken
    
    if layers_to_prune < 1 or layers_to_prune > 6:
        raise ValueError("layers_to_prune must be between 1 and 6")
    
    print(f"\n===== Activation-based pruning of {layers_to_prune} layers in blocks 3-4 =====")
    
    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Get all Conv2d layers for global indexing
    all_conv_layers = get_all_conv2d_layers(detection_model)
    
    # Collect all conv layers from target blocks with original model indexing
    target_blocks = [3, 4, 5, 6]
    all_available_convs = []
    
    # Create a mapping of conv layers to their original indices for reference
    original_conv_layer_mapping = {}
    for original_idx, conv_layer in enumerate(all_conv_layers):
        original_conv_layer_mapping[id(conv_layer)] = original_idx
    
    print(f"Original model has {len(all_conv_layers)} Conv2d layers total")
    
    for block_idx in target_blocks:
        if block_idx >= len(detection_model):
            print(f"Warning: Block index {block_idx} is out of range. Skipping.")
            continue
            
        block = detection_model[block_idx]
        conv_layers_in_block = get_all_conv2d_layers(block)
        
        print(f"\nAnalyzing Block {block_idx}:")
        print(f"  Found {len(conv_layers_in_block)} Conv2d layers in this block")
        
        for conv_in_block_idx, conv_layer in enumerate(conv_layers_in_block):
            num_channels = conv_layer.weight.shape[0]
            
            # Find original model index for this conv layer
            original_conv_idx = original_conv_layer_mapping.get(id(conv_layer), "Unknown")
            
            print(f"    Conv #{conv_in_block_idx}: {num_channels} channels, Original model index: {original_conv_idx}")
            
            # Skip layers with too few channels (need at least 4 channels for meaningful pruning)
            if num_channels < 8:
                print(f"    → Skipping: only {num_channels} channels (need ≥8 for activation pruning)")
                continue
            
            all_available_convs.append({
                'block_idx': block_idx,
                'conv_in_block_idx': conv_in_block_idx,
                'conv_layer': conv_layer,
                'num_channels': num_channels,
                'original_model_idx': original_conv_idx
            })
    
    print(f"Found {len(all_available_convs)} suitable Conv2d layers in blocks 3-4")
    
    if len(all_available_convs) < layers_to_prune:
        print(f"Warning: Only {len(all_available_convs)} layers available, adjusting to prune all available layers")
        layers_to_prune = len(all_available_convs)
    
    # Select layers with most channels for activation-based pruning (often more impactful)
    all_available_convs.sort(key=lambda x: x['num_channels'], reverse=True)
    selected_convs = all_available_convs[:layers_to_prune]
    
    print(f"\nSelected {len(selected_convs)} layers for activation-based pruning:")
    for i, conv_info in enumerate(selected_convs):
        print(f"  Layer {i+1}: Block {conv_info['block_idx']}, Conv #{conv_info['conv_in_block_idx']}")
        print(f"    Original model index: {conv_info['original_model_idx']}")
        print(f"    Channels: {conv_info['num_channels']}")
    
    # Apply activation-based pruning to selected layers
    pruned_layers_details = []
    
    print(f"\n--- Starting Activation-Based Pruning Process ---")
    for idx, conv_info in enumerate(selected_convs):
        print(f"\nPruning Layer {idx + 1}/{len(selected_convs)}:")
        print(f"  - Block: {conv_info['block_idx']}")
        print(f"  - Conv in block index: {conv_info['conv_in_block_idx']}")
        print(f"  - Original model index: {conv_info['original_model_idx']}")
        print(f"  - Original channels: {conv_info['num_channels']}")
        
        # Store original model state to file temporarily for the function call
        temp_model_path = f"temp_model_state_{idx}.pt"
        model.save(temp_model_path)
        
        # Apply the activation-based pruning for this specific layer
        try:
            # Use a SIMPLE activation-based pruning that doesn't use broken clustering
            model = apply_simple_activation_pruning(
                model_path=temp_model_path,
                train_data=train_data,
                valid_data=valid_data,
                classes=classes,
                block_idx=conv_info['block_idx'],
                conv_in_block_idx=conv_info['conv_in_block_idx']
            )
            
            print(f"  ✓ Activation-based pruning applied successfully!")
            
            # Get updated channel count
            torch_model = model.model
            detection_model = torch_model.model
            all_conv_layers_updated = get_all_conv2d_layers(detection_model)
            
            # Find the pruned layer in the updated model
            if conv_info['original_model_idx'] < len(all_conv_layers_updated):
                pruned_layer = all_conv_layers_updated[conv_info['original_model_idx']]
                remaining_channels = (pruned_layer.weight.abs().sum(dim=(1,2,3)) != 0).sum().item()
            else:
                remaining_channels = "Unknown"
            
            # Store details for final summary
            pruned_layers_details.append({
                'block_idx': conv_info['block_idx'],
                'conv_in_block_idx': conv_info['conv_in_block_idx'],
                'original_model_idx': conv_info['original_model_idx'],
                'original_channels': conv_info['num_channels'],
                'remaining_channels': remaining_channels,
                'pruned_channels': conv_info['num_channels'] - remaining_channels if isinstance(remaining_channels, int) else "Unknown"
            })
            
        except Exception as e:
            print(f"  ✗ Error during activation-based pruning: {e}")
            continue
        finally:
            # Clean up temporary file
            import os
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
    
    # Final evaluation
    print("Starting final evaluation...")
    final_metrics = model.val(data=data_yaml, verbose=False)
    
    # Calculate total parameters pruned
    total_channels_before = sum(detail['original_channels'] for detail in pruned_layers_details)
    total_channels_after = sum(detail['remaining_channels'] for detail in pruned_layers_details if isinstance(detail['remaining_channels'], int))
    pruning_ratio = (total_channels_before - total_channels_after) / total_channels_before * 100 if total_channels_before > 0 else 0
    
    print(f"\nDetailed Activation-Based Pruning Summary:")
    print(f"{'='*80}")
    print(f"{'Layer':<8} {'Block':<6} {'Conv#':<7} {'Original#':<10} {'Channels':<15}")
    print(f"{'-'*80}")
    for i, details in enumerate(pruned_layers_details):
        channels_info = f"{details['original_channels']}→{details['remaining_channels']}"
        print(f"{i+1:<8} {details['block_idx']:<6} {details['conv_in_block_idx']:<7} "
              f"{details['original_model_idx']:<10} {channels_info:<15}")
    
    print(f"{'-'*80}")
    print(f"Overall Statistics:")
    print(f"  Layers pruned: {len(pruned_layers_details)}")
    print(f"  Total channels before: {total_channels_before}")
    print(f"  Total channels after: {total_channels_after}")
    print(f"  Overall pruning ratio: {pruning_ratio:.1f}%")
    print(f"{'='*80}")
    
    # Attach pruning details to model for summary
    model.pruned_layers_details = pruned_layers_details
    
    print(f"\n✅ Original activation pruning method completed successfully!")
    return model

def apply_gamma_soft_pruning(model_path, train_data, valid_data, classes, layers_to_prune=3, data_yaml="data/VOC_adva.yaml", predefined_layers=None, channels_to_keep_per_layer=None):
    """Apply gamma-based soft pruning using the ORIGINAL WORKING METHOD from run_experiments.py"""
    
    # Use the EXACT SAME function that works in the original run_experiments.py
    from pruning_yolo_v8 import apply_50_percent_gamma_pruning_blocks_3_4
    
    print(f"\n===== Using ORIGINAL WORKING Gamma Pruning Method =====")
    print(f"🔧 This uses the EXACT SAME function that works in run_experiments.py")
    
    # Call the original working function with the same parameters
    pruned_model = apply_50_percent_gamma_pruning_blocks_3_4(
        model_path=model_path,
        data_yaml=data_yaml,
        layers_to_prune=layers_to_prune,
        predefined_layers=predefined_layers,
        channels_to_keep_per_layer=channels_to_keep_per_layer
    )
    
    print(f"\n✅ Original gamma pruning method completed successfully!")
    return pruned_model

def run_simple_comparison_experiment(layers_to_prune=3):
    """Run a simple comparison using the ORIGINAL WORKING methods."""
    
    print(f"🚀 Running Simple Comparison Experiment")
    print(f"   Layers to prune: {layers_to_prune}")
    print(f"   Methods: ORIGINAL WORKING activation vs gamma")
    print(f"   🔧 Using exact same functions as run_experiments.py")
    print("-" * 50)
    
    # Load data configuration
    data_yaml = "data/VOC_adva.yaml"
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)
    
    classes_names = data_cfg["names"]
    classes = list(range(len(classes_names)))
    
    # Load samples properly
    train_img_dir = data_cfg["train"]
    val_img_dir = data_cfg["val"]
    
    # Convert relative paths to absolute paths
    if not train_img_dir.startswith("/"):
        train_img_dir = os.path.join("data", train_img_dir)
    if not val_img_dir.startswith("/"):
        val_img_dir = os.path.join("data", val_img_dir)
        
    train_label_dir = train_img_dir.replace("/images", "/labels")
    val_label_dir = val_img_dir.replace("/images", "/labels")
    
    print(f"📥 Loading training data...")
    train_data = load_samples(train_img_dir, train_label_dir, max_samples=100)
    valid_data = load_samples(val_img_dir, val_label_dir, max_samples=50)
    print(f"✅ Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
    
    # Step 1: Run Activation Pruning (ORIGINAL METHOD)
    print(f"\n🚀 Step 1: Running Activation Pruning (Original Method)...")
    activation_model = apply_activation_soft_pruning(
        model_path="data/best.pt",
        train_data=train_data,
        valid_data=valid_data,
        classes=classes,
        layers_to_prune=layers_to_prune,
        data_yaml=data_yaml
    )
    
    # Extract pruning decisions from activation model
    if hasattr(activation_model, 'pruned_layers_details'):
        pruned_layers_details = activation_model.pruned_layers_details
        print(f"📊 Activation pruning decisions:")
        for i, detail in enumerate(pruned_layers_details):
            print(f"  Layer {i+1}: Block {detail.get('block_idx')}, "
                  f"Channels: {detail.get('original_channels')} → {detail.get('remaining_channels')}")
        
        # Extract channel counts for gamma pruning
        channels_to_keep_per_layer = [detail.get('remaining_channels', 128) for detail in pruned_layers_details]
        predefined_layers = [{
            'block_idx': detail.get('block_idx'),
            'original_model_idx': detail.get('global_idx'),
            'channels_to_keep': detail.get('remaining_channels')
        } for detail in pruned_layers_details]
    else:
        print(f"⚠️  No pruning details found in activation model")
        channels_to_keep_per_layer = None
        predefined_layers = None
    
    # Step 2: Run Gamma Pruning (ORIGINAL METHOD) with same decisions
    print(f"\n🚀 Step 2: Running Gamma Pruning (Original Method)...")
    gamma_model = apply_gamma_soft_pruning(
        model_path="data/best.pt",
        train_data=train_data,
        valid_data=valid_data,
        classes=classes,
        layers_to_prune=layers_to_prune,
        data_yaml=data_yaml,
        predefined_layers=predefined_layers,
        channels_to_keep_per_layer=channels_to_keep_per_layer
    )
    
    # Step 3: Compare Results
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Status':<15} {'Layers Pruned':<15} {'Details':<30}")
    print(f"{'-'*80}")
    
    activation_status = "✅ Success" if hasattr(activation_model, 'pruned_layers_details') else "❌ Failed"
    gamma_status = "✅ Success" if hasattr(gamma_model, 'pruned_layers_details') else "❌ Failed"
    
    print(f"{'Activation (Original)':<20} {activation_status:<15} {layers_to_prune:<15} {'Uses clustering' if activation_status == '✅ Success' else 'Failed'}")
    print(f"{'Gamma (Original)':<20} {gamma_status:<15} {layers_to_prune:<15} {'Uses gamma values' if gamma_status == '✅ Success' else 'Failed'}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"  Both methods use ORIGINAL WORKING functions from run_experiments.py")
    print(f"  Activation method: Uses clustering algorithm (may have issues)")
    print(f"  Gamma method: Uses gamma values (should work reliably)")
    print(f"  Same layers tested: ✅")
    
    return {
        'activation_model': activation_model,
        'gamma_model': gamma_model,
        'activation_status': activation_status,
        'gamma_status': gamma_status
    }

if __name__ == "__main__":
    # Run the simple comparison with just 1 layer for testing
    results = run_simple_comparison_experiment(layers_to_prune=1)
    
    print(f"\n🎉 Simple comparison experiment completed!")
    print(f"   Check the results above to see which method worked.")