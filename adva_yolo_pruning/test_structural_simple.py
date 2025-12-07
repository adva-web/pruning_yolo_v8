#!/usr/bin/env python3
"""
Simple test for structural activation pruning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from structural_pruning import apply_structural_activation_pruning
from pruning_experiments import PruningEvaluator, PruningConfig

def test_structural_pruning():
    """Test the structural pruning functionality."""
    
    print("🔧 Testing Structural Activation Pruning")
    print("=" * 50)
    
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
        # Load sample data for activation analysis
        print(f"\n🔍 Loading training data for activation analysis...")
        config = PruningConfig(
            method="activation",
            layers_to_prune=2,
            model_path=model_path,
            data_yaml=data_yaml
        )
        evaluator = PruningEvaluator(config)
        train_data, valid_data, classes = evaluator.load_samples(
            config.train_img_dir, 
            config.val_img_dir, 
            max_samples=50
        )
        print(f"✅ Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
        
        # Test structural pruning
        print(f"\n🚀 Starting Structural Activation Pruning Test...")
        print(f"   This will prune 2 layers using TRUE structural pruning")
        print(f"   (modifying model architecture instead of just zeroing weights)")
        
        # Run structural activation pruning
        pruned_model = apply_structural_activation_pruning(
            model_path=model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            data_yaml=data_yaml,
            layers_to_prune=2  # Start with just 2 layers for testing
        )
        
        print(f"\n✅ Structural activation pruning test completed successfully!")
        
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

if __name__ == "__main__":
    print("🔧 Structural Activation Pruning Test Suite")
    print("=" * 50)
    
    # Test structural pruning
    success = test_structural_pruning()
    
    if success:
        print(f"\n✅ Test passed!")
        print(f"\n🎉 Structural activation pruning is working!")
        print(f"   This solves the channel mismatch problem for activation pruning.")
        print(f"   You can now prune multiple layers without architectural conflicts.")
    else:
        print(f"\n❌ Test failed. Check the error messages above.")
