#!/usr/bin/env python3
"""
C2f Hybrid Activation + Gamma Pruning Experiment
- Conv0: Gamma pruning (BN gamma-based)
- After-concat convs: Activation pruning (hook-based + clustering)

Includes:
- Warm-up + averaged (3x) inference time measurement before/after
- mAP metrics (mAP@0.5, mAP@0.5:0.95, precision, recall) before/after
- Intermediate fine-tuning: 5 epochs after each layer
- Final fine-tuning: 20 epochs
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruning_yolo_v8_sequential_fix import (
    load_training_data,
    load_validation_data,
)
from yolov8_utils import (
    get_all_conv2d_layers,
)
from c2f_utils import is_c2f_block
from c2f_hybrid_utils import analyze_c2f_block_structure, get_c2f_conv_categories, print_c2f_structure
from c2f_hybrid_conv_pruning import (
    prune_c2f_conv0_gamma,
    prune_c2f_after_concat_activation
)


def measure_inference_time(model, data_yaml, num_runs=3):
    """Measure inference time for a model via model.val()."""
    inference_times = []
    for _ in range(num_runs):
        try:
            metrics = model.val(data=data_yaml, verbose=False)
            if hasattr(metrics, 'speed') and metrics.speed is not None:
                if isinstance(metrics.speed, dict):
                    ms = metrics.speed.get('inference', None)
                else:
                    ms = metrics.speed
                if ms is not None:
                    inference_times.append(ms)
        except Exception:
            continue
    if inference_times:
        return sum(inference_times) / len(inference_times)
    return None


def main():
    """Main experiment function."""
    print("=" * 80)
    print("C2F HYBRID ACTIVATION + GAMMA PRUNING EXPERIMENT")
    print("=" * 80)
    print("Strategy:")
    print("  - Conv0: Gamma pruning (BN gamma-based)")
    print("  - After-concat convs: Activation pruning (hook-based + clustering)")
    print("=" * 80)
    
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    
    # Configuration: which C2f blocks to prune
    target_c2f_blocks = [2]  # Can be extended to [2, 6, 8, ...]
    
    # Load model
    print("\n📥 Loading model...")
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Verify C2f blocks
    print("\n🔍 Verifying C2f blocks:")
    valid_blocks = []
    for block_idx in target_c2f_blocks:
        if block_idx < len(detection_model):
            block = detection_model[block_idx]
            if is_c2f_block(block):
                structure = analyze_c2f_block_structure(block)
                if structure:
                    valid_blocks.append(block_idx)
                    print(f"   ✅ Block {block_idx}: C2f block with {structure['conv_count']} Conv layers")
                    print_c2f_structure(block, block_idx)
                else:
                    print(f"   ⚠️  Block {block_idx}: C2f block but structure analysis failed")
            else:
                print(f"   ⚠️  Block {block_idx}: Not a C2f block (type: {type(block).__name__})")
        else:
            print(f"   ⚠️  Block {block_idx}: Out of range")
    
    if not valid_blocks:
        print("❌ No valid C2f blocks found")
        return
    
    # Measure original model metrics (inference + mAP)
    print("\n" + "=" * 80)
    print("MEASURING ORIGINAL MODEL METRICS")
    print("=" * 80)
    original_model = YOLO(model_path)
    original_inference_time = None
    original_metrics_dict = None
    try:
        print("   Warm-up run...")
        _ = original_model.val(data=data_yaml, verbose=False)
        print("   Averaging over 3 runs...")
        original_inference_time = measure_inference_time(original_model, data_yaml, num_runs=3)
        if original_inference_time:
            print(f"   ✅ Original inference time: {original_inference_time:.2f} ms/image")
        
        print("   Getting mAP metrics...")
        res = original_model.val(data=data_yaml, verbose=False)
        original_metrics_dict = {
            'mAP50-95': res.results_dict.get('metrics/mAP50-95(B)', 0),
            'mAP50': res.results_dict.get('metrics/mAP50(B)', 0),
            'precision': res.results_dict.get('metrics/precision(B)', 0),
            'recall': res.results_dict.get('metrics/recall(B)', 0)
        }
        print(f"   ✅ Original mAP@0.5: {original_metrics_dict['mAP50']:.4f}")
        print(f"   ✅ Original mAP@0.5:0.95: {original_metrics_dict['mAP50-95']:.4f}")
    except Exception as e:
        print(f"   ⚠️  Error measuring original metrics: {e}")
    
    # Load data - ALL available data for activation extraction
    print("\n📥 Loading ALL training data for activation extraction...")
    train_data = load_training_data(data_yaml, max_samples=None)
    if len(train_data) == 0:
        print("❌ No training data loaded.")
        return
    
    print(f"   ✅ Loaded {len(train_data)} training samples")
    
    print("📥 Loading ALL validation data for activation extraction...")
    try:
        valid_data = load_validation_data(data_yaml, max_samples=None)
        all_activation_data = train_data + valid_data
        print(f"   ✅ Total samples for activation extraction: {len(all_activation_data)}")
    except Exception as e:
        print(f"   ⚠️  Could not load validation data: {e}, using training data only")
        all_activation_data = train_data
    
    # Load classes
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        classes = list(range(len(data_config['names'])))
    
    # Store pruning results
    all_pruning_details = []
    
    print("\n" + "=" * 80)
    print("PRUNING PROCESS")
    print("=" * 80)
    
    # Prune each C2f block
    for block_idx in valid_blocks:
        block = detection_model[block_idx]
        structure = analyze_c2f_block_structure(block)
        categories = get_c2f_conv_categories(block)
        
        print(f"\n{'='*80}")
        print(f"PRUNING C2F BLOCK {block_idx}")
        print(f"{'='*80}")
        
        # Step 1: Prune Conv0 with gamma
        if categories['before_concat']:
            conv0_info = categories['before_concat'][0]  # First conv (Conv0)
            conv0_idx = conv0_info['idx']
            
            print(f"\n--- Step 1: Pruning Conv0 (Gamma) ---")
            model, details = prune_c2f_conv0_gamma(
                model=model,
                block_idx=block_idx,
                conv0_idx=conv0_idx,
                pruning_ratio=0.5
            )
            if details:
                all_pruning_details.append(details)
            
            # Fine-tune after Conv0
            print(f"\n🔄 Fine-tuning 5 epochs after Conv0...")
            try:
                model.train(data=data_yaml, epochs=5, verbose=True)
            except Exception as e:
                print(f"   ⚠️  Fine-tuning failed: {e}")
        
        # Step 2: Prune after-concat convs with activation
        if categories['after_concat']:
            for conv_info in categories['after_concat']:
                conv_idx = conv_info['idx']
                
                print(f"\n--- Step 2: Pruning Conv {conv_idx} After Concat (Activation) ---")
                model, details = prune_c2f_after_concat_activation(
                    model=model,
                    block_idx=block_idx,
                    conv_idx=conv_idx,
                    train_data=all_activation_data,
                    classes=classes,
                    data_yaml=data_yaml,
                    max_patches=100000
                )
                if details:
                    all_pruning_details.append(details)
                
                # Fine-tune after each after-concat conv
                print(f"\n🔄 Fine-tuning 5 epochs after Conv {conv_idx}...")
                try:
                    model.train(data=data_yaml, epochs=5, verbose=True)
                except Exception as e:
                    print(f"   ⚠️  Fine-tuning failed: {e}")
    
    # Final fine-tuning
    print("\n" + "=" * 80)
    print("FINAL FINE-TUNING (20 EPOCHS)")
    print("=" * 80)
    try:
        model.train(data=data_yaml, epochs=20, verbose=True)
        print("✅ Final fine-tuning completed")
    except Exception as e:
        print(f"⚠️  Final fine-tuning failed: {e}")
    
    # Measure pruned model metrics (inference + mAP)
    print("\n" + "=" * 80)
    print("MEASURING PRUNED MODEL METRICS")
    print("=" * 80)
    pruned_inference_time = None
    pruned_metrics_dict = None
    try:
        print("   Warm-up run...")
        _ = model.val(data=data_yaml, verbose=False)
        print("   Averaging over 3 runs...")
        pruned_inference_time = measure_inference_time(model, data_yaml, num_runs=3)
        if pruned_inference_time:
            print(f"   ✅ Pruned inference time: {pruned_inference_time:.2f} ms/image")
        
        print("   Getting mAP metrics...")
        res2 = model.val(data=data_yaml, verbose=False)
        pruned_metrics_dict = {
            'mAP50-95': res2.results_dict.get('metrics/mAP50-95(B)', 0),
            'mAP50': res2.results_dict.get('metrics/mAP50(B)', 0),
            'precision': res2.results_dict.get('metrics/precision(B)', 0),
            'recall': res2.results_dict.get('metrics/recall(B)', 0)
        }
        print(f"   ✅ Pruned mAP@0.5: {pruned_metrics_dict['mAP50']:.4f}")
        print(f"   ✅ Pruned mAP@0.5:0.95: {pruned_metrics_dict['mAP50-95']:.4f}")
    except Exception as e:
        print(f"   ⚠️  Error measuring pruned metrics: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Pruning stats
    print("\n📊 Pruning Results:")
    for d in all_pruning_details:
        method = d['method'].upper()
        print(f"  Block {d['block_idx']} Conv {d['conv_in_block_idx']} ({method}):")
        print(f"    {d['original_channels']} -> {d['remaining_channels']} channels (removed {d['pruned_channels']}, {d['pruning_ratio']*100:.1f}%)")
    
    # Inference time
    print("\n⏱️  Inference Time:")
    oi = f"{original_inference_time:.2f} ms/image" if original_inference_time else "N/A"
    pi = f"{pruned_inference_time:.2f} ms/image" if pruned_inference_time else "N/A"
    print(f"  Original: {oi}")
    print(f"  Pruned:   {pi}")
    if original_inference_time and pruned_inference_time and pruned_inference_time > 0:
        print(f"  Speedup:  {original_inference_time / pruned_inference_time:.2f}x")
    
    # mAP metrics
    print("\n📈 mAP Metrics:")
    if original_metrics_dict and pruned_metrics_dict:
        print(f"  Original mAP@0.5:      {original_metrics_dict['mAP50']:.4f}")
        print(f"  Original mAP@0.5:0.95: {original_metrics_dict['mAP50-95']:.4f}")
        print(f"  Original Precision:     {original_metrics_dict['precision']:.4f}")
        print(f"  Original Recall:        {original_metrics_dict['recall']:.4f}")
        print(f"  Pruned mAP@0.5:        {pruned_metrics_dict['mAP50']:.4f} ({pruned_metrics_dict['mAP50'] - original_metrics_dict['mAP50']:+.4f})")
        print(f"  Pruned mAP@0.5:0.95:   {pruned_metrics_dict['mAP50-95']:.4f} ({pruned_metrics_dict['mAP50-95'] - original_metrics_dict['mAP50-95']:+.4f})")
        print(f"  Pruned Precision:       {pruned_metrics_dict['precision']:.4f} ({pruned_metrics_dict['precision'] - original_metrics_dict['precision']:+.4f})")
        print(f"  Pruned Recall:          {pruned_metrics_dict['recall']:.4f} ({pruned_metrics_dict['recall'] - original_metrics_dict['recall']:+.4f})")
    else:
        print("  ⚠️  Could not retrieve full metrics")
    
    # Save
    out_path = "runs/detect/c2f_hybrid_activation_gamma.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    print(f"\n💾 Saved pruned model to: {out_path}")


if __name__ == "__main__":
    main()


