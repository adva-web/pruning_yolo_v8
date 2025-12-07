#!/usr/bin/env python3
"""
Quick test to verify if intended pruning results are being captured correctly.
This script will run a single layer activation pruning and check if the intended results
are properly stored and retrieved.
"""

import os
import sys
import torch
import numpy as np
from ultralytics import YOLO

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pruning_yolo_v8 import apply_enhanced_activation_pruning_blocks_3_4

def create_dummy_data():
    """Create dummy training and validation data for testing."""
    print("📊 Creating dummy data for testing...")
    
    # Create dummy training data in the correct format
    train_data = []
    for i in range(10):  # Small dataset for quick testing
        # Create a dummy image (640x640x3)
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Create dummy labels (class_id, x_center, y_center, width, height)
        # Normalized coordinates
        dummy_labels = np.array([
            [0, 0.5, 0.5, 0.1, 0.1],  # One object in center
        ])
        
        # Format as dictionary (required by the processing functions)
        sample = {
            'image': dummy_img,
            'label': dummy_labels,
            'image_path': f'dummy_train_{i}.jpg',
            'label_path': f'dummy_train_{i}.txt'
        }
        train_data.append(sample)
    
    # Create dummy validation data
    valid_data = []
    for i in range(5):
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        dummy_labels = np.array([
            [0, 0.3, 0.3, 0.1, 0.1],
        ])
        
        sample = {
            'image': dummy_img,
            'label': dummy_labels,
            'image_path': f'dummy_val_{i}.jpg',
            'label_path': f'dummy_val_{i}.txt'
        }
        valid_data.append(sample)
    
    classes = ['person', 'bicycle', 'car']  # Dummy class names
    
    print(f"✅ Created {len(train_data)} training samples and {len(valid_data)} validation samples")
    return train_data, valid_data, classes

def test_intended_results():
    """Test if intended pruning results are captured correctly."""
    print("🧪 TESTING INTENDED PRUNING RESULTS CAPTURE")
    print("=" * 60)
    
    # Check if required files exist
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML not found: {data_yaml}")
        return False
    
    print(f"✅ Found model: {model_path}")
    print(f"✅ Found data config: {data_yaml}")
    
    # Create dummy data
    train_data, valid_data, classes = create_dummy_data()
    
    try:
        print(f"\n🔬 Running activation pruning test (1 layer)...")
        
        # Run enhanced activation pruning for just 1 layer
        pruned_model = apply_enhanced_activation_pruning_blocks_3_4(
            model_path=model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=1,  # Only test 1 layer for speed
            data_yaml=data_yaml,
            fine_tune_epochs_per_step=1,  # Minimal fine-tuning for speed
        )
        
        print(f"\n📊 CHECKING RESULTS:")
        print("=" * 40)
        
        # Check if the model has pruned_layers_details
        if hasattr(pruned_model, 'pruned_layers_details'):
            details = pruned_model.pruned_layers_details
            print(f"✅ Model has pruned_layers_details: {len(details)} layers")
            
            for i, detail in enumerate(details):
                print(f"\n📋 Layer {i+1} Details:")
                print(f"  Block: {detail.get('block_idx', 'N/A')}")
                print(f"  Conv#: {detail.get('conv_in_block_idx', 'N/A')}")
                print(f"  Global#: {detail.get('global_conv_idx', 'N/A')}")
                print(f"  Original channels: {detail.get('original_channels', 'N/A')}")
                print(f"  Remaining channels: {detail.get('remaining_channels', 'N/A')}")
                print(f"  Pruned channels: {detail.get('pruned_channels', 'N/A')}")
                print(f"  Status: {detail.get('status', 'N/A')}")
                
                # Calculate pruning percentage
                original = detail.get('original_channels', 0)
                remaining = detail.get('remaining_channels', 0)
                if isinstance(original, int) and isinstance(remaining, int) and original > 0:
                    pruning_pct = ((original - remaining) / original) * 100
                    print(f"  Pruning%: {pruning_pct:.1f}%")
                    
                    # Check if pruning actually occurred
                    if remaining < original:
                        print(f"  ✅ SUCCESS: Actual pruning occurred ({original}→{remaining})")
                        return True
                    else:
                        print(f"  ❌ PROBLEM: No pruning occurred ({original}→{remaining})")
                        return False
                else:
                    print(f"  ❌ PROBLEM: Invalid channel counts")
                    return False
        else:
            print(f"❌ Model does not have pruned_layers_details attribute")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Quick Test: Intended Pruning Results Capture")
    print("=" * 60)
    
    success = test_intended_results()
    
    print(f"\n{'='*60}")
    if success:
        print("🎉 TEST PASSED: Intended pruning results are being captured correctly!")
        print("✅ The summary should now show the correct pruning results.")
    else:
        print("❌ TEST FAILED: Intended pruning results are NOT being captured correctly.")
        print("🔧 The summary will still show incorrect results.")
    print(f"{'='*60}")
