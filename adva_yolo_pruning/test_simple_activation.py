#!/usr/bin/env python3
"""
Simple test for activation pruning WITHOUT the broken clustering algorithm.
This uses a simple activation-based importance ranking instead.
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

from yolov8_utils import build_mini_net, get_all_conv2d_layers, get_raw_objects_debug_v8, aggregate_activations_from_matches

def load_samples_for_test(image_dir: str, label_dir: str, max_samples=50):
    """Load dataset samples for testing."""
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

def simple_activation_pruning_no_clustering(model_path, train_data, valid_data, classes, block_idx=5, conv_in_block_idx=0):
    """
    Simple activation-based pruning that DOESN'T use the broken clustering algorithm.
    This uses a simple activation-based importance ranking instead.
    """
    print(f"🔧 Using SIMPLE activation-based importance ranking (NO CLUSTERING)")
    
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

def test_simple_activation_pruning():
    """Test simple activation pruning without clustering."""
    
    print("🧪 Testing Simple Activation Pruning (NO CLUSTERING)")
    print("=" * 60)
    print("🔧 Features:")
    print("  ✅ Simple activation-based importance ranking")
    print("  ✅ NO clustering algorithm (avoids the broken select_optimal_components)")
    print("  ✅ Fast and reliable")
    print("=" * 60)
    
    # Load data configuration
    data_yaml = "data/VOC_adva.yaml"
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)
    
    classes_names = data_cfg["names"]
    classes = list(range(len(classes_names)))
    
    # Load samples (smaller sample size for faster testing)
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
    train_data = load_samples_for_test(train_img_dir, train_label_dir, max_samples=50)
    valid_data = load_samples_for_test(val_img_dir, val_label_dir, max_samples=25)
    print(f"✅ Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
    
    try:
        # Test with 1 layer first
        print(f"\n🚀 Testing simple activation pruning on 1 layer...")
        
        # Run simple activation pruning (no clustering)
        pruned_model = simple_activation_pruning_no_clustering(
            model_path="data/best.pt",
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            block_idx=5,  # Test on block 5
            conv_in_block_idx=0  # First conv in block
        )
        
        print(f"\n✅ Simple activation pruning completed successfully!")
        
        # Final evaluation
        print(f"\n🔍 Running final evaluation...")
        final_metrics = pruned_model.val(data=data_yaml, verbose=False)
        
        print(f"\n📈 Final Results:")
        print(f"  mAP@0.5: {final_metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@0.5:0.95: {final_metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {final_metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall: {final_metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
        
        return pruned_model
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧪 Starting Simple Activation Pruning Test (NO CLUSTERING)")
    print("This will test a simple activation-based pruning that doesn't use clustering")
    print()
    
    result = test_simple_activation_pruning()
    
    if result is not None:
        print(f"\n🎉 Test completed successfully!")
        print(f"   Simple activation pruning works without clustering.")
    else:
        print(f"\n❌ Test failed!")
        print(f"   There was an error with the simple activation pruning.")
