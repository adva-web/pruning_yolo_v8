#!/usr/bin/env python3
"""
Test script for Strategy 3: Reverse Hybrid Pruning on C2f blocks.
Tests reverse hybrid pruning on Block 2 (C2f block).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from pruning_c2f_reverse_hybrid import prune_c2f_block_reverse_hybrid
from pruning_yolo_v8_sequential_fix import (
    load_training_data,
    load_validation_data
)
from c2f_utils import is_c2f_block
import yaml


def test_reverse_hybrid_pruning():
    """Test reverse hybrid pruning on C2f block (Block 2)."""
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    block_idx = 2  # C2f block
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False
    
    # Load data
    print("📥 Loading training data...")
    train_data = load_training_data(data_yaml, max_samples=50)
    if len(train_data) == 0:
        print("❌ No training data loaded.")
        return False
    
    print("📥 Loading validation data...")
    try:
        valid_data = load_validation_data(data_yaml, max_samples=30)
    except Exception:
        valid_data = train_data[:20]
    
    # Load classes
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        classes = list(range(len(data_config['names'])))
    
    # Load model
    print("\n🚀 Loading model...")
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Verify block is C2f
    if block_idx >= len(detection_model):
        print(f"❌ Block index {block_idx} out of range")
        return False
    
    block = detection_model[block_idx]
    if not is_c2f_block(block):
        print(f"❌ Block {block_idx} is not a C2f block")
        return False
    
    print(f"\n✅ C2f Block {block_idx} found")
    
    # Test reverse hybrid pruning
    print(f"\n{'='*70}")
    print("TESTING REVERSE HYBRID PRUNING (Strategy 3)")
    print(f"{'='*70}")
    print("Strategy:")
    print("  - Conv 0 (before concat): GAMMA pruning")
    print("  - Conv 1+ (after concat): ACTIVATION pruning")
    print(f"{'='*70}\n")
    
    try:
        # Prune using reverse hybrid strategy
        pruned_model = prune_c2f_block_reverse_hybrid(
            model_path=model_path,
            block_idx=block_idx,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            data_yaml=data_yaml,
            gamma_pruning_ratio=0.5,
            fine_tune_epochs=3  # Use 3 epochs for testing
        )
        
        if pruned_model is None:
            print("❌ Pruning returned None")
            return False
        
        print(f"\n{'='*70}")
        print("✅ Reverse hybrid pruning test completed successfully!")
        print(f"{'='*70}")
        
        # Evaluate model if desired
        print("\n📊 Evaluating pruned model...")
        try:
            metrics = pruned_model.val(data=data_yaml, verbose=False)
            map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0)
            map50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
            precision = metrics.results_dict.get('metrics/precision(B)', 0)
            recall = metrics.results_dict.get('metrics/recall(B)', 0)
            
            print(f"\n📈 Model Performance:")
            print(f"   mAP@0.5:0.95: {map50_95:.4f}")
            print(f"   mAP@0.5: {map50:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f}")
        except Exception as e:
            print(f"⚠️  Evaluation failed: {e}")
        
        return True
    
    except Exception as e:
        print(f"❌ Pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_reverse_hybrid_pruning()
    sys.exit(0 if success else 1)

