#!/usr/bin/env python3
"""
Test script to demonstrate guaranteed same-layer comparison between gamma and activation pruning.
This ensures you're testing the EXACT SAME layers with both methods.
"""

import sys
import os
sys.path.append('/Users/advahelman/Code/pruning/adva_yolo_pruning')

from pruning_yolo_v8 import run_comparison_experiment, get_layer_selection_info
from pruning_experiments import load_samples
import yaml

def test_comparison():
    """Test the comparison experiment with guaranteed same layers."""
    
    print("🔬 Testing Guaranteed Same-Layer Comparison")
    print("=" * 60)
    
    # Paths
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML not found: {data_yaml}")
        return
    
    # Load dataset info
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Get class names
    classes = data_config.get('names', [])
    print(f"📋 Dataset classes: {classes}")
    
    # Load samples for activation pruning
    print(f"📊 Loading samples for activation extraction...")
    train_data, valid_data = load_samples(data_yaml, max_samples=100)
    
    if not train_data or not valid_data:
        print(f"❌ Failed to load training/validation data")
        return
    
    print(f"✅ Loaded {len(train_data)} training samples, {len(valid_data)} validation samples")
    
    # Test layer selection info first
    print(f"\n🔍 Testing layer selection info...")
    
    # Get gamma layer selection
    gamma_layers = get_layer_selection_info(model_path, layers_to_prune=3, method="gamma")
    print(f"✅ Gamma pruning selected {len(gamma_layers)} layers")
    
    # Get activation layer selection
    activation_layers = get_layer_selection_info(model_path, layers_to_prune=3, method="activation")
    print(f"✅ Activation pruning selected {len(activation_layers)} layers")
    
    # Show the difference
    print(f"\n📊 Layer Selection Comparison:")
    print(f"  Gamma layers: {[(l['block_idx'], l['original_model_idx']) for l in gamma_layers]}")
    print(f"  Activation layers: {[(l['block_idx'], l['original_model_idx']) for l in activation_layers]}")
    
    # Run comparison experiment
    print(f"\n🧪 Running comparison experiment...")
    try:
        results = run_comparison_experiment(
            model_path=model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=3,
            data_yaml=data_yaml
        )
        
        print(f"\n✅ Comparison experiment completed successfully!")
        print(f"📊 Winner: {results['comparison_summary']['winner']}")
        print(f"📈 Improvement: {results['comparison_summary']['improvement_percent']:.2f}%")
        
    except Exception as e:
        print(f"❌ Comparison experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_comparison()
