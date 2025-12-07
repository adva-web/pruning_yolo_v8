#!/usr/bin/env python3
"""
Test script for gamma-based pruning with all blocks (skip block 0)
"""
import yaml
from pruning_yolo_v8 import apply_50_percent_gamma_pruning_blocks_3_4

def main():
    print("=" * 80)
    print("Testing Gamma Pruning with All Blocks (Skip Block 0)")
    print("=" * 80)
    
    # Configuration
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    layers_to_prune = 6  # Try to prune 6 layers
    
    # Run gamma pruning
    print(f"\n🚀 Running gamma pruning with {layers_to_prune} layers")
    print(f"   Model: {model_path}")
    print(f"   Data: {data_yaml}")
    print(f"   Will search ALL blocks (skip block 0) for layers with BatchNorm")
    
    try:
        pruned_model = apply_50_percent_gamma_pruning_blocks_3_4(
            model_path=model_path,
            data_yaml=data_yaml,
            layers_to_prune=layers_to_prune
        )
        
        print("\n✅ Gamma pruning completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Gamma pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

