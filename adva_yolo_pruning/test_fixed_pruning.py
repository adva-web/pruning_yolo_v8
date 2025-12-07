#!/usr/bin/env python3
"""
Test script for the fixed YOLOv8 pruning implementation.
This script demonstrates how to use the fixed pruning methods.
"""

import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruning_yolo_v8_fixed import apply_activation_pruning_blocks_3_4_fixed, run_fixed_comparison_experiment

def test_fixed_pruning():
    """Test the fixed pruning implementation."""
    
    # Configuration
    model_path = "data/best.pt"  # Adjust path as needed
    data_yaml = "data/VOC_adva.yaml"
    layers_to_prune = 3
    
    print("🧪 Testing Fixed YOLOv8 Pruning Implementation")
    print("=" * 60)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the model file exists or update the path in the script.")
        return False
    
    # Check if data yaml exists
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        print("Please ensure the data YAML file exists or update the path in the script.")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"✅ Data YAML found: {data_yaml}")
    print(f"🎯 Layers to prune: {layers_to_prune}")
    
    try:
        # Load real training and validation data
        from pruning_yolo_v8_fixed import load_training_data, load_validation_data
        
        print(f"\n📥 Loading training data...")
        train_data = load_training_data(data_yaml, max_samples=50)
        
        if len(train_data) == 0:
            print(f"❌ No training data loaded. Cannot proceed.")
            return False
        
        print(f"\n📥 Loading validation data...")
        valid_data = load_validation_data(data_yaml, max_samples=30)
        
        if len(valid_data) == 0:
            print(f"⚠️  No validation data loaded, using training data for validation")
            valid_data = train_data[:20]  # Use subset of training data
        
        # Load classes from YAML
        import yaml
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        classes = list(range(len(data_cfg['names'])))
        
        print(f"\n🚀 Starting Fixed Activation Pruning...")
        print(f"   Method: Fixed activation-based pruning")
        print(f"   Target blocks: 3-4")
        print(f"   Layers to prune: {layers_to_prune}")
        
        # Run the fixed pruning
        pruned_model = apply_activation_pruning_blocks_3_4_fixed(
            model_path=model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=layers_to_prune
        )
        
        print(f"\n✅ Fixed pruning completed successfully!")
        print(f"📊 Model has been pruned and is ready for evaluation")
        
        return True
        
    except Exception as e:
        print(f"❌ Fixed pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comparison():
    """Test the comparison between original and fixed methods."""
    
    print(f"\n🔬 Testing Fixed vs Original Comparison")
    print("=" * 60)
    
    # Configuration
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    layers_to_prune = 3
    
    try:
        # Load real training and validation data
        from pruning_yolo_v8_fixed import load_training_data, load_validation_data
        
        print(f"\n📥 Loading training data...")
        train_data = load_training_data(data_yaml, max_samples=50)
        
        if len(train_data) == 0:
            print(f"❌ No training data loaded. Cannot proceed.")
            return False
        
        print(f"\n📥 Loading validation data...")
        valid_data = load_validation_data(data_yaml, max_samples=30)
        
        if len(valid_data) == 0:
            print(f"⚠️  No validation data loaded, using training data for validation")
            valid_data = train_data[:20]  # Use subset of training data
        
        # Load classes from YAML
        import yaml
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        classes = list(range(len(data_cfg['names'])))
        
        print(f"🚀 Running Fixed Comparison Experiment...")
        
        # Run comparison
        fixed_model = run_fixed_comparison_experiment(
            model_path=model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=layers_to_prune,
            data_yaml=data_yaml
        )
        
        print(f"✅ Comparison completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("YOLOv8 Fixed Pruning Test Suite")
    print("=" * 60)
    
    # Test 1: Basic fixed pruning
    print("\n🧪 Test 1: Basic Fixed Pruning")
    success1 = test_fixed_pruning()
    
    # Test 2: Comparison test
    print("\n🧪 Test 2: Comparison Test")
    success2 = test_comparison()
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"   Basic Fixed Pruning: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Comparison Test: {'✅ PASS' if success2 else '❌ FAIL'}")
    
    if success1 and success2:
        print(f"\n🎉 All tests passed! The fixed pruning implementation is working correctly.")
    else:
        print(f"\n⚠️  Some tests failed. Please check the error messages above.")
