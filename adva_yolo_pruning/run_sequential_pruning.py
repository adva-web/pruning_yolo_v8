#!/usr/bin/env python3
"""
Test runner for sequential activation pruning fix.
This script tests the fixed implementation without modifying the original files.
"""

import os
import sys
import yaml

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruning_yolo_v8_sequential_fix import (
    apply_activation_pruning_blocks_3_4_sequential,
    load_training_data,
    load_validation_data
)

def main():
    print("=" * 80)
    print("Sequential Activation Pruning - Fixed Implementation Test")
    print("=" * 80)
    
    # Configuration
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    layers_to_prune = 3
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"✅ Data YAML found: {data_yaml}")
    print(f"🎯 Layers to prune: {layers_to_prune}")
    
    try:
        # Load real training and validation data
        train_data = load_training_data(data_yaml, max_samples=50)
        
        if len(train_data) == 0:
            print(f"❌ No training data loaded. Cannot proceed.")
            return False
        
        valid_data = load_validation_data(data_yaml, max_samples=30)
        
        if len(valid_data) == 0:
            print(f"⚠️  No validation data loaded, using training data for validation")
            valid_data = train_data[:20]
        
        # Get classes from YAML
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        classes = list(range(len(data_cfg['names'])))
        
        print(f"\n📊 Data loaded:")
        print(f"   - Training samples: {len(train_data)}")
        print(f"   - Validation samples: {len(valid_data)}")
        print(f"   - Classes: {len(classes)}")
        
        # Run fixed pruning
        print(f"\n🚀 Starting sequential activation pruning...")
        pruned_model = apply_activation_pruning_blocks_3_4_sequential(
            model_path=model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=layers_to_prune,
            data_yaml=data_yaml
        )
        
        print(f"\n✅ Pruning completed successfully!")
        
        # Test the pruned model
        print(f"\n📊 Testing pruned model...")
        test_metrics = pruned_model.val(data=data_yaml, verbose=False)
        
        print(f"📈 Pruned model performance:")
        print(f"   mAP@0.5:0.95: {test_metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"   mAP@0.5: {test_metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   Precision: {test_metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
        print(f"   Recall: {test_metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
        
        # Print pruning details if available
        if hasattr(pruned_model, 'pruned_layers_details'):
            print(f"\n📋 Pruning Details:")
            for i, detail in enumerate(pruned_model.pruned_layers_details):
                print(f"   Layer {i+1}:")
                print(f"     Block: {detail.get('block_idx', 'N/A')}")
                print(f"     Original channels: {detail.get('original_channels', 'N/A')}")
                print(f"     Remaining channels: {detail.get('remaining_channels', 'N/A')}")
                print(f"     Status: {detail.get('status', 'N/A')}")
                if detail.get('status') == 'failed':
                    print(f"     Error: {detail.get('error', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sequential pruning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

