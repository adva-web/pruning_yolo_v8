#!/usr/bin/env python3
"""
Quick test script to verify pruning and channel counting before full experiment.
This will test a single layer pruning to check if the debugging and fixes work.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pruning_yolo_v8 import (
    apply_enhanced_activation_pruning_blocks_3_4,
    apply_50_percent_gamma_pruning_blocks_3_4,
    get_layer_selection_info
)
from pruning_experiments import PruningEvaluator
import yaml

def quick_test():
    """Run a quick test with just 1 layer to verify everything works."""
    print("🧪 QUICK TEST: Single Layer Pruning Verification")
    print("=" * 60)
    
    # Load data configuration
    with open("data/VOC_adva.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)
    
    classes_names = data_cfg["names"]
    classes = list(range(len(classes_names)))
    
    # Load a small sample of data for testing
    print("📊 Loading test data...")
    train_img_dir = data_cfg["train"]
    val_img_dir = data_cfg["val"]
    
    # Convert relative paths to absolute paths
    if not train_img_dir.startswith("/"):
        train_img_dir = os.path.join("data", train_img_dir)
    if not val_img_dir.startswith("/"):
        val_img_dir = os.path.join("data", val_img_dir)
    
    print(f"  Train dir: {train_img_dir}")
    print(f"  Val dir: {val_img_dir}")
    
    # Create dummy data for quick testing (just to test the pruning functions)
    print("  Using dummy data for quick testing...")
    train_data = [{"image": "dummy.jpg", "labels": []}]
    valid_data = [{"image": "dummy.jpg", "labels": []}]
    print(f"  Using {len(train_data)} training samples, {len(valid_data)} validation samples")
    
    # Test 1: Check layer selection
    print(f"\n🔍 Test 1: Layer Selection")
    print("-" * 40)
    activation_layers = get_layer_selection_info("data/best.pt", layers_to_prune=1, method="activation")
    print(f"Selected activation layers: {len(activation_layers)}")
    for layer in activation_layers:
        print(f"  Layer: Block {layer['block_idx']}, Original idx {layer['original_model_idx']}, Channels {layer['num_channels']}")
    
    # Test 2: Quick activation pruning (1 layer only)
    print(f"\n🧪 Test 2: Activation Pruning (1 layer)")
    print("-" * 40)
    try:
        activation_model = apply_enhanced_activation_pruning_blocks_3_4(
            model_path="data/best.pt",
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=1,  # Just 1 layer for quick test
            data_yaml="data/VOC_adva.yaml",
            fine_tune_epochs_per_step=5,
            predefined_layers=activation_layers
        )
        
        activation_details = getattr(activation_model, 'pruned_layers_details', [])
        print(f"✅ Activation pruning completed!")
        print(f"Pruned layers: {len(activation_details)}")
        for detail in activation_details:
            print(f"  Block {detail['block_idx']}: {detail['original_channels']}→{detail['remaining_channels']} channels")
            
    except Exception as e:
        print(f"❌ Activation pruning failed: {e}")
        return False
    
    # Test 3: Extract channel counts for gamma pruning
    print(f"\n📊 Test 3: Channel Count Extraction")
    print("-" * 40)
    channels_to_keep = []
    for detail in activation_details:
        if detail.get('status') == 'success' and 'remaining_channels' in detail:
            remaining_channels = detail['remaining_channels']
            if isinstance(remaining_channels, int):
                channels_to_keep.append(remaining_channels)
            else:
                channels_to_keep.append(detail['original_channels'])
    
    print(f"Channels to keep for gamma pruning: {channels_to_keep}")
    
    # Test 4: Quick gamma pruning (1 layer only)
    print(f"\n🧪 Test 4: Gamma Pruning (1 layer)")
    print("-" * 40)
    try:
        gamma_model = apply_50_percent_gamma_pruning_blocks_3_4(
            model_path="data/best.pt",
            data_yaml="data/VOC_adva.yaml",
            layers_to_prune=1,
            predefined_layers=activation_layers,
            channels_to_keep_per_layer=channels_to_keep
        )
        
        gamma_details = getattr(gamma_model, 'pruned_layers_details', [])
        print(f"✅ Gamma pruning completed!")
        print(f"Pruned layers: {len(gamma_details)}")
        for detail in gamma_details:
            print(f"  Block {detail['block_idx']}: {detail['original_channels']}→{detail['remaining_channels']} channels")
            
    except Exception as e:
        print(f"❌ Gamma pruning failed: {e}")
        return False
    
    # Test 5: Compare results
    print(f"\n📊 Test 5: Results Comparison")
    print("-" * 40)
    print(f"{'Method':<15} {'Block':<6} {'Channels':<15}")
    print("-" * 40)
    
    for detail in activation_details:
        channels_info = f"{detail['original_channels']}→{detail['remaining_channels']}"
        print(f"{'Activation':<15} {detail['block_idx']:<6} {channels_info:<15}")
    
    for detail in gamma_details:
        channels_info = f"{detail['original_channels']}→{detail['remaining_channels']}"
        print(f"{'Gamma':<15} {detail['block_idx']:<6} {channels_info:<15}")
    
    # Check if channel counts are correct
    print(f"\n🎯 Test 6: Channel Count Verification")
    print("-" * 40)
    success = True
    for i, (act_detail, gamma_detail) in enumerate(zip(activation_details, gamma_details)):
        act_remaining = act_detail.get('remaining_channels', 0)
        gamma_remaining = gamma_detail.get('remaining_channels', 0)
        original = act_detail.get('original_channels', 0)
        
        print(f"Layer {i+1}:")
        print(f"  Original: {original} channels")
        print(f"  Activation: {original}→{act_remaining} ({original - act_remaining} pruned)")
        print(f"  Gamma: {original}→{gamma_remaining} ({original - gamma_remaining} pruned)")
        
        if act_remaining == original:
            print(f"  ⚠️  Activation pruning shows no pruning!")
            success = False
        if gamma_remaining == original:
            print(f"  ⚠️  Gamma pruning shows no pruning!")
            success = False
        if act_remaining != gamma_remaining:
            print(f"  ⚠️  Channel counts don't match!")
            success = False
        else:
            print(f"  ✅ Channel counts match and show actual pruning!")
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 QUICK TEST PASSED! All systems working correctly.")
        print("✅ Ready for full comparison experiment!")
    else:
        print("❌ QUICK TEST FAILED! Issues detected.")
        print("🔧 Need to fix issues before running full experiment.")
    
    return success

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
