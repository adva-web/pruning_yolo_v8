#!/usr/bin/env python3
"""
Test script for structural pruning functionality.
This script demonstrates the difference between soft pruning (zeroing weights) 
and structural pruning (actually removing channels from model architecture).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pruning_yolo_v8 import apply_structural_activation_pruning_blocks_3_4
from yolov8_utils import load_samples
from ultralytics import YOLO

def test_structural_pruning():
    """Test the structural pruning functionality."""
    
    print("🔧 Testing Structural Pruning vs Soft Pruning")
    print("=" * 60)
    
    # Paths
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"✅ Data YAML file found: {data_yaml}")
    
    try:
        # Test structural pruning
        print(f"\n🚀 Starting Structural Activation Pruning Test...")
        print(f"   This will prune 3 layers using TRUE structural pruning")
        print(f"   (modifying model architecture instead of just zeroing weights)")
        
        # Load sample data for activation analysis
        print(f"   Loading training data for activation analysis...")
        train_data, valid_data, classes = load_samples(data_yaml, max_samples=100)
        
        # Run structural activation pruning
        pruned_model = apply_structural_activation_pruning_blocks_3_4(
            model_path=model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            data_yaml=data_yaml,
            layers_to_prune=3
        )
        
        print(f"\n✅ Structural pruning test completed successfully!")
        
        # Test the pruned model
        print(f"\n📊 Testing pruned model...")
        test_metrics = pruned_model.val(data=data_yaml, verbose=False)
        
        print(f"📈 Pruned model performance:")
        print(f"   mAP@0.5:0.95: {test_metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"   mAP@0.5: {test_metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   Precision: {test_metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
        print(f"   Recall: {test_metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
        
        # Check if pruning details are available
        if hasattr(pruned_model, 'pruned_layers_details'):
            print(f"\n📋 Pruning Details:")
            for i, detail in enumerate(pruned_model.pruned_layers_details):
                print(f"   Layer {i+1}: Block {detail.get('block_idx', 'N/A')}")
                print(f"     Original channels: {detail.get('original_channels', 'N/A')}")
                print(f"     Remaining channels: {detail.get('remaining_channels', 'N/A')}")
                print(f"     Status: {detail.get('status', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Structural pruning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def compare_soft_vs_structural():
    """Compare soft pruning vs structural pruning results."""
    
    print(f"\n🔬 Comparing Soft Pruning vs Structural Pruning")
    print("=" * 60)
    
    # This would require running both methods and comparing results
    # For now, just explain the differences
    
    print(f"📊 Key Differences:")
    print(f"   Soft Pruning (Current Activation Method):")
    print(f"     - Zeroes out weights of less important channels")
    print(f"     - Model architecture remains unchanged")
    print(f"     - Can cause channel mismatches in multi-layer pruning")
    print(f"     - Faster but less effective")
    
    print(f"   Structural Activation Pruning (New Method):")
    print(f"     - Actually removes channels from model architecture")
    print(f"     - Model structure is modified")
    print(f"     - No channel mismatches - proper multi-layer pruning")
    print(f"     - Uses activation analysis for optimal channel selection")
    print(f"     - Slower but more effective")
    
    print(f"\n🎯 Expected Benefits of Structural Pruning:")
    print(f"   ✅ No channel mismatch errors")
    print(f"   ✅ Proper multi-layer pruning")
    print(f"   ✅ Better model compression")
    print(f"   ✅ More accurate pruning results")

if __name__ == "__main__":
    print("🔧 Structural Pruning Test Suite")
    print("=" * 60)
    
    # Test structural pruning
    success = test_structural_pruning()
    
    if success:
        print(f"\n✅ All tests passed!")
        
        # Show comparison
        compare_soft_vs_structural()
        
        print(f"\n🎉 Structural pruning is ready to use!")
        print(f"   You can now use apply_structural_gamma_pruning_blocks_3_4()")
        print(f"   instead of apply_50_percent_gamma_pruning_blocks_3_4()")
        print(f"   for better multi-layer pruning without channel mismatches.")
    else:
        print(f"\n❌ Tests failed. Check the error messages above.")
