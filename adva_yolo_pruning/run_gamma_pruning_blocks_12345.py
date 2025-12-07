#!/usr/bin/env python3
"""
Pure Gamma Pruning Experiment: Blocks 1, 2, 3, 4, 5
- Blocks 1, 2, 3, 4, 5: Gamma pruning (BN gamma-based channel selection)

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


def find_following_bn(block: nn.Module, conv_in_block_idx: int):
    """Find BatchNorm following the specified Conv2d by matching channel count."""
    all_convs = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(all_convs):
        return None
    target_conv = all_convs[conv_in_block_idx]
    target_out_channels = target_conv.weight.shape[0]
    for module in block.modules():
        if isinstance(module, nn.BatchNorm2d) and module.num_features == target_out_channels:
            return module
    return None


def gamma_prune_block(model, block_idx, conv_in_block_idx, pruning_ratio=0.5):
    """Gamma prune a Conv in any block (works for C2f or regular)."""
    torch_model = model.model
    detection_model = torch_model.model
    if block_idx >= len(detection_model):
        print(f"⚠️  Block {block_idx} out of range")
        return model, None
    block = detection_model[block_idx]
    convs = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(convs):
        print(f"⚠️  Conv {conv_in_block_idx} out of range for block {block_idx}")
        return model, None

    conv = convs[conv_in_block_idx]
    bn = find_following_bn(block, conv_in_block_idx)
    if bn is None:
        print(f"⚠️  No BatchNorm found for Block {block_idx}, Conv {conv_in_block_idx}")
        return model, None

    num_channels = conv.weight.shape[0]
    keep = int(num_channels * (1 - pruning_ratio))
    remove = num_channels - keep

    gamma_values = bn.weight.data.abs().cpu().numpy()
    order = np.argsort(gamma_values)  # ascending
    indices_to_remove = order[:remove]

    print(f"📊 Gamma pruning: Block {block_idx}, Conv {conv_in_block_idx}, {num_channels} -> {keep} (remove {remove})")

    with torch.no_grad():
        conv.weight[indices_to_remove] = 0
        if conv.bias is not None:
            conv.bias[indices_to_remove] = 0
        bn.weight[indices_to_remove] = 0
        bn.bias[indices_to_remove] = 0
        bn.running_mean[indices_to_remove] = 0
        bn.running_var[indices_to_remove] = 1

    details = {
        'block_idx': block_idx,
        'conv_in_block_idx': conv_in_block_idx,
        'original_channels': num_channels,
        'remaining_channels': keep,
        'pruned_channels': remove,
        'pruning_ratio': remove / num_channels,
        'method': 'gamma'
    }
    return model, details


def main():
    print("=" * 80)
    print("PURE GAMMA PRUNING EXPERIMENT: BLOCKS 1, 2, 3, 4, 5")
    print("=" * 80)

    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"

    # Load model
    print("\n📥 Loading model...")
    model = YOLO(model_path)
    detection_model = model.model.model

    # Measure original metrics (inference + mAP)
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
        print("   Getting mAP metrics...")
        res = original_model.val(data=data_yaml, verbose=False)
        original_metrics_dict = {
            'mAP50-95': res.results_dict.get('metrics/mAP50-95(B)', 0),
            'mAP50': res.results_dict.get('metrics/mAP50(B)', 0),
            'precision': res.results_dict.get('metrics/precision(B)', 0),
            'recall': res.results_dict.get('metrics/recall(B)', 0)
        }
    except Exception as e:
        print(f"   ⚠️  Error measuring original metrics: {e}")

    # Pruning plan: blocks 1,2,3,4,5 (conv 0)
    pruning_config = [
        {'block_idx': 1, 'conv_idx': 0},
        {'block_idx': 2, 'conv_idx': 0},
        {'block_idx': 3, 'conv_idx': 0},
        {'block_idx': 4, 'conv_idx': 0},
        {'block_idx': 5, 'conv_idx': 0},
    ]

    all_pruning_details = []

    print("\n" + "=" * 80)
    print("PRUNING PROCESS (GAMMA)")
    print("=" * 80)

    for i, cfg in enumerate(pruning_config):
        b = cfg['block_idx']
        c = cfg['conv_idx']
        print(f"\n{'='*80}\nPRUNING LAYER {i+1}/{len(pruning_config)}: Block {b}, Conv {c} (GAMMA)\n{'='*80}")
        model, details = gamma_prune_block(model, b, c, pruning_ratio=0.5)
        if details:
            all_pruning_details.append(details)
        # Fine-tune after each layer
        print(f"\n🔄 Fine-tuning 5 epochs...")
        try:
            model.train(data=data_yaml, epochs=5, verbose=True)
        except Exception as e:
            print(f"   ⚠️  Fine-tuning failed: {e}")

    # Final fine-tuning
    print("\nFINAL FINE-TUNING (20 EPOCHS)")
    try:
        model.train(data=data_yaml, epochs=20, verbose=True)
    except Exception as e:
        print(f"   ⚠️  Final fine-tuning failed: {e}")

    # Measure pruned metrics (inference + mAP)
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
        print("   Getting mAP metrics...")
        res2 = model.val(data=data_yaml, verbose=False)
        pruned_metrics_dict = {
            'mAP50-95': res2.results_dict.get('metrics/mAP50-95(B)', 0),
            'mAP50': res2.results_dict.get('metrics/mAP50(B)', 0),
            'precision': res2.results_dict.get('metrics/precision(B)', 0),
            'recall': res2.results_dict.get('metrics/recall(B)', 0)
        }
    except Exception as e:
        print(f"   ⚠️  Error measuring pruned metrics: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Pruning stats
    print("\n📊 Pruning Results:")
    for d in all_pruning_details:
        print(f"  Block {d['block_idx']} Conv {d['conv_in_block_idx']} -> removed {d['pruned_channels']} / {d['original_channels']} ({d['pruning_ratio']*100:.1f}%)")

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
    out_path = "runs/detect/gamma_pruning_blocks_12345.pt"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    print(f"\n💾 Saved pruned model to: {out_path}")


if __name__ == "__main__":
    main()
