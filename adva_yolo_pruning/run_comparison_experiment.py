#!/usr/bin/env python3
"""
Apples-to-apples comparison between activation-based and gamma-based pruning.
 - Prunes blocks 1, 3, 5 in order
 - Activation method uses its clustering+MSS to choose which channels to keep
 - Gamma method prunes the EXACT SAME NUMBER of channels per layer
 - Final fine-tuning at the end for 20 epochs (not after each layer)
"""

import os
import sys
import yaml
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import torch
import torch.nn as nn

from pruning_yolo_v8_sequential_fix import (
    load_training_data,
    load_validation_data,
    count_active_channels
)
from yolov8_utils import get_all_conv2d_layers
from pruning_yolo_v8 import (
    prune_conv2d_in_block_with_activations,
    YoloLayerPruner,
    select_optimal_components
)


def select_layers_blocks_1_3_5(detection_model, total_layers=3):
    """Select Conv 0 from blocks 1, 3, 5 in order"""
    target_blocks = [1, 3, 5]
    layers = []
    for b in target_blocks:
        if b < len(detection_model):
            block = detection_model[b]
            convs = get_all_conv2d_layers(block)
            if len(convs) >= 1:
                conv0 = convs[0]
                layers.append({
                    'block_idx': b,
                    'conv_in_block_idx': 0,
                    'name': f"Block {b}, Conv 0",
                    'num_channels': conv0.weight.shape[0]
                })
        if len(layers) >= total_layers:
            break
    return layers[:total_layers]


def find_following_bn(block: nn.Module, conv_in_block_idx: int):
    """Find BN after specified Conv in block"""
    hit = -1
    for m in block.children():
        if isinstance(m, nn.Conv2d):
            hit += 1
            if hit == conv_in_block_idx:
                for n in block.children():
                    if isinstance(n, nn.BatchNorm2d):
                        return n
                return None
    return None


def main():
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    epochs_final = 20
    total_layers_to_prune = 3

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False

    # Load data - use ALL available data for better activation statistics
    print("📥 Loading ALL training data for activation extraction...")
    train_data = load_training_data(data_yaml, max_samples=None)  # None = load all available
    if len(train_data) == 0:
        print("❌ No training data loaded.")
        return False
    
    print("📥 Loading ALL validation data for activation extraction...")
    try:
        valid_data = load_validation_data(data_yaml, max_samples=None)  # None = load all available
        # Combine train + validation for maximum activation coverage
        print(f"   Combining {len(train_data)} training + {len(valid_data)} validation samples")
        all_activation_data = train_data + valid_data
        print(f"   ✅ Total samples for activation extraction: {len(all_activation_data)}")
    except Exception as e:
        print(f"   ⚠️  Could not load validation data: {e}, using training data only")
        all_activation_data = train_data
        valid_data = []
    
    # Use combined data for activation extraction
    activation_data = all_activation_data

    # ========================================================================
    # MEASURE ORIGINAL MODEL INFERENCE TIME
    # ========================================================================
    print(f"\n{'='*70}")
    print("MEASURING ORIGINAL MODEL INFERENCE TIME")
    print(f"{'='*70}")
    
    original_model = YOLO(model_path)
    print("   Running validation on original model...")
    try:
        original_metrics = original_model.val(data=data_yaml, verbose=False)
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

    # Create two models: one for activation pruning, one for gamma
    print("\n🚀 Loading models for comparison...")
    model_activation = YOLO(model_path)
    model_gamma = YOLO(model_path)

    torch_model_act = model_activation.model
    torch_model_gam = model_gamma.model
    detection_model_act = torch_model_act.model
    detection_model_gam = torch_model_gam.model

    targets = select_layers_blocks_1_3_5(detection_model_act, total_layers=total_layers_to_prune)
    if len(targets) == 0:
        print("❌ No eligible layers found.")
        return False

    print("\n🎯 Target layers (in order):")
    for i, t in enumerate(targets):
        print(f"  {i+1}. {t['name']} ({t['num_channels']} channels)")

    classes = list(range(len(yaml.safe_load(open(data_yaml))['names'])))

    # ============================================================
    # METHOD 1: ACTIVATION-BASED PRUNING (determine how many channels to keep)
    # ============================================================
    print(f"\n{'='*70}")
    print("METHOD 1: ACTIVATION-BASED PRUNING")
    print(f"{'='*70}")

    channels_pruned_per_layer = []

    for i, t in enumerate(targets):
        print(f"\n--- Layer {i+1}/{len(targets)}: {t['name']} ---")
        temp_path = f"temp_act_iter_{i+1}.pt"
        model_activation.save(temp_path)

        try:
            updated_model = prune_conv2d_in_block_with_activations(
                model_path=temp_path,
                train_data=activation_data,  # Use all available data (train+val combined)
                valid_data=valid_data,
                classes=classes,
                block_idx=t['block_idx'],
                conv_in_block_idx=t['conv_in_block_idx'],
                log_file=f"activation_iter_{i+1}.txt",
                data_yaml=data_yaml
            )
        except Exception as e:
            print(f"❌ Pruning failed: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            # Still record 0 pruned for matching
            block = detection_model_act[t['block_idx']]
            conv = get_all_conv2d_layers(block)[t['conv_in_block_idx']]
            original_ch = conv.weight.shape[0]
            channels_pruned_per_layer.append((original_ch, original_ch))
            continue

        if updated_model is None:
            print("❌ Pruning returned None")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            block = detection_model_act[t['block_idx']]
            conv = get_all_conv2d_layers(block)[t['conv_in_block_idx']]
            original_ch = conv.weight.shape[0]
            channels_pruned_per_layer.append((original_ch, original_ch))
            continue

        # Copy pruned weights to activation model
        block_act = detection_model_act[t['block_idx']]
        conv_act = get_all_conv2d_layers(block_act)[t['conv_in_block_idx']]
        block_upd = updated_model.model.model[t['block_idx']]
        conv_upd = get_all_conv2d_layers(block_upd)[t['conv_in_block_idx']]

        with torch.no_grad():
            conv_act.weight.copy_(conv_upd.weight)
            if conv_act.bias is not None and conv_upd.bias is not None:
                conv_act.bias.copy_(conv_upd.bias)

        remaining = count_active_channels(conv_act)
        original_ch = conv_act.weight.shape[0]
        channels_pruned_per_layer.append((original_ch, remaining))
        print(f"✅ Activation pruning: {original_ch} → {remaining} channels")

        if os.path.exists(temp_path):
            os.remove(temp_path)

    # ============================================================
    # METHOD 2: GAMMA-BASED PRUNING (use same channel counts)
    # ============================================================
    print(f"\n{'='*70}")
    print("METHOD 2: GAMMA-BASED PRUNING (matching channel counts)")
    print(f"{'='*70}")

    for i, t in enumerate(targets):
        original_ch, target_ch = channels_pruned_per_layer[i]
        print(f"\n--- Layer {i+1}/{len(targets)}: {t['name']} ---")
        print(f"   Matching: prune {original_ch - target_ch} channels (→ {target_ch} remaining)")

        block = detection_model_gam[t['block_idx']]
        conv = get_all_conv2d_layers(block)[t['conv_in_block_idx']]
        bn = find_following_bn(block, t['conv_in_block_idx'])

        out_ch = conv.weight.shape[0]
        k_keep = target_ch

        # Compute scores (BN gamma or weight magnitude)
        if bn is not None:
            scores = bn.weight.detach().abs()
        else:
            scores = conv.weight.detach().abs().mean(dim=(1, 2, 3))

        order = torch.argsort(scores, descending=True)
        keep_idx = set(order[:k_keep].tolist())
        remove_idx = [j for j in range(out_ch) if j not in keep_idx]

        if len(remove_idx) > 0:
            with torch.no_grad():
                conv.weight[remove_idx] = 0
                if conv.bias is not None:
                    conv.bias[remove_idx] = 0
                if bn is not None:
                    bn.weight[remove_idx] = 0
                    bn.bias[remove_idx] = 0

        remaining = count_active_channels(conv)
        print(f"✅ Gamma pruning: {out_ch} → {remaining} channels")

    # ============================================================
    # FINAL FINE-TUNING (20 epochs each)
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL FINE-TUNING (20 epochs each)")
    print(f"{'='*70}")

    print("\n--- Fine-tuning Activation Model ---")
    try:
        model_activation.train(data=data_yaml, epochs=epochs_final, verbose=True)
        print("✅ Activation model fine-tuned")
    except Exception as e:
        print(f"⚠️  Fine-tuning failed: {e}")

    print("\n--- Fine-tuning Gamma Model ---")
    try:
        model_gamma.train(data=data_yaml, epochs=epochs_final, verbose=True)
        print("✅ Gamma model fine-tuned")
    except Exception as e:
        print(f"⚠️  Fine-tuning failed: {e}")

    # ============================================================
    # EVALUATION
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")

    try:
        print("\n--- Evaluating Activation Model ---")
        metrics_act = model_activation.val(data=data_yaml, verbose=False)
        
        # Extract inference time for activation model
        activation_inference_time = None
        if hasattr(metrics_act, 'speed') and metrics_act.speed is not None:
            if isinstance(metrics_act.speed, dict):
                activation_inference_time = metrics_act.speed.get('inference', None)
            elif isinstance(metrics_act.speed, (int, float)):
                activation_inference_time = metrics_act.speed
        
        print("\n--- Evaluating Gamma Model ---")
        metrics_gam = model_gamma.val(data=data_yaml, verbose=False)
        
        # Extract inference time for gamma model
        gamma_inference_time = None
        if hasattr(metrics_gam, 'speed') and metrics_gam.speed is not None:
            if isinstance(metrics_gam.speed, dict):
                gamma_inference_time = metrics_gam.speed.get('inference', None)
            elif isinstance(metrics_gam.speed, (int, float)):
                gamma_inference_time = metrics_gam.speed

        print("\n📊 Performance Comparison:")
        print(f"{'Metric':<25} {'Activation':<15} {'Gamma':<15} {'Diff':<15}")
        print("-"*70)

        for metric_key in ['metrics/mAP50-95(B)', 'metrics/mAP50(B)', 'metrics/precision(B)', 'metrics/recall(B)']:
            act_val = metrics_act.results_dict.get(metric_key, 0)
            gam_val = metrics_gam.results_dict.get(metric_key, 0)
            diff = act_val - gam_val
            print(f"{metric_key:<25} {act_val:<15.4f} {gam_val:<15.4f} {diff:<15.4f}")
        
        # Inference time comparison
        print("\n⏱️  Inference Time Comparison:")
        print(f"{'Model':<25} {'Inference Time (ms/img)':<25} {'Speedup':<15}")
        print("-" * 65)
        
        if original_inference_time is not None:
            print(f"{'Original Model':<25} {original_inference_time:<25.2f} {'1.00x (baseline)':<15}")
        
        if activation_inference_time is not None:
            if original_inference_time is not None and original_inference_time > 0:
                speedup = original_inference_time / activation_inference_time
                print(f"{'Activation Pruning':<25} {activation_inference_time:<25.2f} {speedup:.2f}x")
            else:
                print(f"{'Activation Pruning':<25} {activation_inference_time:<25.2f} {'N/A':<15}")
        else:
            print(f"{'Activation Pruning':<25} {'N/A':<25} {'N/A':<15}")
        
        if gamma_inference_time is not None:
            if original_inference_time is not None and original_inference_time > 0:
                speedup = original_inference_time / gamma_inference_time
                print(f"{'Gamma Pruning':<25} {gamma_inference_time:<25.2f} {speedup:.2f}x")
            else:
                print(f"{'Gamma Pruning':<25} {gamma_inference_time:<25.2f} {'N/A':<15}")
        else:
            print(f"{'Gamma Pruning':<25} {'N/A':<25} {'N/A':<15}")

    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n✅ Comparison experiment completed!")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
