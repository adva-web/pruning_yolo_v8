#!/usr/bin/env python3
"""
Test script for hybrid C2f block pruning.
Tests the hybrid approach on Block 2 (first C2f block):
- Activation pruning on Conv 0 (before concat)
- Gamma pruning on Conv 1+ (after concat)
"""

import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
from pruning_yolo_v8_sequential_fix import (
    load_training_data,
    load_validation_data,
    count_active_channels
)
from yolov8_utils import get_all_conv2d_layers
from c2f_hybrid_utils import print_c2f_structure, get_c2f_conv_categories
from c2f_utils import is_c2f_block
from pruning_c2f_hybrid import prune_c2f_block_hybrid


def main():
    # Configuration
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    block_idx = 2  # Block 2 is the first C2f block
    gamma_pruning_ratio = 0.5  # Keep 50% of channels for gamma pruning
    fine_tune_epochs = 5  # Fine-tune for 5 epochs after each pruning step
    final_fine_tune_epochs = 20  # Final fine-tuning
    
    # Validate files
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False
    
    # Load model
    print("🚀 Loading model...")
    model = YOLO(model_path)
    
    # ========================================================================
    # MEASURE ORIGINAL MODEL INFERENCE TIME
    # ========================================================================
    print(f"\n{'='*70}")
    print("MEASURING ORIGINAL MODEL INFERENCE TIME")
    print(f"{'='*70}")
    
    print("   Running validation on original model...")
    try:
        original_metrics = model.val(data=data_yaml, verbose=False)
        # Extract inference time from metrics
        original_inference_time = None
        if hasattr(original_metrics, 'speed') and original_metrics.speed is not None:
            if isinstance(original_metrics.speed, dict):
                original_inference_time = original_metrics.speed.get('inference', None)
            elif isinstance(original_metrics.speed, (int, float)):
                original_inference_time = original_metrics.speed
        
        if original_inference_time is not None:
            print(f"✅ Original model inference time: {original_inference_time:.2f} ms/image")
        else:
            print(f"⚠️  Could not extract inference time from metrics")
            original_inference_time = None
    except Exception as e:
        print(f"⚠️  Error measuring original model inference time: {e}")
        original_inference_time = None
    
    torch_model = model.model
    detection_model = torch_model.model
    
    # Verify block is C2f
    if block_idx >= len(detection_model):
        print(f"❌ Block index {block_idx} out of range (model has {len(detection_model)} blocks)")
        return False
    
    block = detection_model[block_idx]
    if not is_c2f_block(block):
        print(f"❌ Block {block_idx} is not a C2f block (type: {type(block).__name__})")
        return False
    
    print(f"✅ Block {block_idx} is a C2f block")
    
    # Analyze structure
    print_c2f_structure(block, block_idx)
    categories = get_c2f_conv_categories(block)
    
    print(f"\n📋 Pruning Plan:")
    if len(categories['before_concat']) > 0:
        print(f"  ✅ Will prune Conv 0 (before concat) with ACTIVATION pruning")
    else:
        print(f"  ⚠️  No convs before concat found")
    
    if len(categories['after_concat']) > 0:
        print(f"  ✅ Will prune {len(categories['after_concat'])} conv(s) after concat with GAMMA pruning")
        for conv_info in categories['after_concat']:
            print(f"     - Conv {conv_info['idx']}: {conv_info['out_channels']} channels")
    else:
        print(f"  ⚠️  No convs after concat found")
    
    # Load data
    print(f"\n📥 Loading training data...")
    train_data = load_training_data(data_yaml, max_samples=None)
    if len(train_data) == 0:
        print("❌ No training data loaded.")
        return False
    
    print(f"✅ Loaded {len(train_data)} training samples")
    
    print(f"📥 Loading validation data...")
    try:
        valid_data = load_validation_data(data_yaml, max_samples=None)
        print(f"✅ Loaded {len(valid_data)} validation samples")
    except Exception as e:
        print(f"⚠️  Could not load validation data: {e}, using empty list")
        valid_data = []
    
    # Load classes
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        classes = list(range(len(data_config['names'])))
    
    print(f"✅ Loaded {len(classes)} classes")
    
    # Apply hybrid pruning
    print(f"\n{'='*70}")
    print(f"STARTING HYBRID C2F PRUNING")
    print(f"{'='*70}")
    
    pruned_model = prune_c2f_block_hybrid(
        model_path=model_path,
        block_idx=block_idx,
        train_data=train_data,
        valid_data=valid_data,
        classes=classes,
        data_yaml=data_yaml,
        gamma_pruning_ratio=gamma_pruning_ratio,
        prune_conv0=True,
        prune_conv1_plus=True,
        fine_tune_epochs=fine_tune_epochs
    )
    
    # Final fine-tuning
    if final_fine_tune_epochs > 0:
        print(f"\n{'='*70}")
        print(f"FINAL FINE-TUNING ({final_fine_tune_epochs} EPOCHS)")
        print(f"{'='*70}")
        try:
            pruned_model.train(data=data_yaml, epochs=final_fine_tune_epochs, verbose=True)
            print(f"✅ Final fine-tuning completed")
        except Exception as e:
            print(f"⚠️  Final fine-tuning failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Final evaluation
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION")
    print(f"{'='*70}")
    try:
        metrics = pruned_model.val(data=data_yaml, verbose=False)
        map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0)
        map50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
        precision = metrics.results_dict.get('metrics/precision(B)', 0)
        recall = metrics.results_dict.get('metrics/recall(B)', 0)
        
        print(f"\n📊 Performance Metrics:")
        print(f"mAP@0.5:0.95: {map50_95:.4f}")
        print(f"mAP@0.5: {map50:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        
        # Extract inference time from metrics
        pruned_inference_time = None
        if hasattr(metrics, 'speed') and metrics.speed is not None:
            if isinstance(metrics.speed, dict):
                pruned_inference_time = metrics.speed.get('inference', None)
            elif isinstance(metrics.speed, (int, float)):
                pruned_inference_time = metrics.speed
        
        # Inference time comparison
        print(f"\n⏱️  Inference Time Comparison:")
        print(f"{'Model':<25} {'Inference Time (ms/img)':<25} {'Speedup':<15}")
        print("-" * 65)
        
        if original_inference_time is not None:
            print(f"{'Original Model':<25} {original_inference_time:<25.2f} {'1.00x (baseline)':<15}")
        
        if pruned_inference_time is not None:
            if original_inference_time is not None and original_inference_time > 0:
                speedup = original_inference_time / pruned_inference_time
                print(f"{'Hybrid Pruned Model':<25} {pruned_inference_time:<25.2f} {speedup:.2f}x")
            else:
                print(f"{'Hybrid Pruned Model':<25} {pruned_inference_time:<25.2f} {'N/A':<15}")
        else:
            print(f"{'Hybrid Pruned Model':<25} {'N/A':<25} {'N/A':<15}")
            
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n✅ Hybrid C2f pruning test completed!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

