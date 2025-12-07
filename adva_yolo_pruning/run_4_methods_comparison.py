#!/usr/bin/env python3
"""
Comprehensive comparison of 4 pruning methods:
1. Activation with max weight (original method)
2. Activation with k-medoid (geometric center selection)
3. Activation with max gamma (BN gamma-based selection)
4. Pure gamma pruning (BN gamma magnitude only)

Prunes Conv 0 from blocks: 1, 3, 5 (3 layers)
Final fine-tuning: 20 epochs for each method
Includes inference time measurement for original model and all methods

NOTE: For experiments with additional blocks, see:
- run_4_methods_comparison_blocks_13579.py (blocks 1, 3, 5, 7, 9)
- run_4_methods_comparison_blocks_1357_16.py (blocks 1, 3, 5, 7, 16)
- run_4_methods_comparison_blocks_1357_19.py (blocks 1, 3, 5, 7, 19)
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
    count_active_channels
)
from yolov8_utils import (
    get_all_conv2d_layers,
    build_mini_net,
    get_raw_objects_debug_v8,
    aggregate_activations_from_matches
)
from yolo_layer_pruner import YoloLayerPruner
from clustering_variants import (
    select_optimal_components_medoid,
    select_optimal_components_max_gamma
)
from pruning_yolo_v8 import (
    prune_conv2d_in_block_with_activations,
    select_optimal_components
)
from c2f_utils import is_c2f_block
import time


def select_layers_blocks_1_3_5(detection_model):
    """Select Conv 0 from blocks 1, 3, 5 (3 layers)."""
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
    return layers


def find_following_bn(block: nn.Module, conv_in_block_idx: int):
    """Find the BatchNorm layer following the specified Conv2d in a block."""
    hit = -1
    found_conv = False
    for m in block.children():
        if isinstance(m, nn.Conv2d):
            hit += 1
            if hit == conv_in_block_idx:
                found_conv = True
        elif isinstance(m, nn.BatchNorm2d) and found_conv:
            return m
    return None


def apply_pruned_weights(main_model, pruned_model, block_idx, conv_in_block_idx):
    """Copy pruned conv (and following BN if exists) from pruned_model into main_model."""
    tm_main = main_model.model
    dm_main = tm_main.model
    tm_pruned = pruned_model.model
    dm_pruned = tm_pruned.model

    block_main = dm_main[block_idx]
    block_pruned = dm_pruned[block_idx]

    convs_main = get_all_conv2d_layers(block_main)
    convs_pruned = get_all_conv2d_layers(block_pruned)

    target_conv_main = convs_main[conv_in_block_idx]
    target_conv_pruned = convs_pruned[conv_in_block_idx]

    with torch.no_grad():
        target_conv_main.weight.copy_(target_conv_pruned.weight)
        if target_conv_main.bias is not None and target_conv_pruned.bias is not None:
            target_conv_main.bias.copy_(target_conv_pruned.bias)

    # Copy the next BatchNorm if present
    def find_next_bn(block):
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

    bn_main = find_next_bn(block_main)
    bn_pruned = find_next_bn(block_pruned)
    if bn_main is not None and bn_pruned is not None:
        with torch.no_grad():
            bn_main.weight.copy_(bn_pruned.weight)
            bn_main.bias.copy_(bn_pruned.bias)


# ============================================================================
# METHOD 1: ACTIVATION WITH MAX WEIGHT (ORIGINAL)
# ============================================================================
def prune_method1_activation_max_weight(model_path, train_data, valid_data, classes,
                                        block_idx, conv_in_block_idx, data_yaml):
    """Method 1: Activation-based pruning with max weight selection (original)."""
    return prune_conv2d_in_block_with_activations(
        model_path=model_path,
        train_data=train_data,
        valid_data=valid_data,
        classes=classes,
        block_idx=block_idx,
        conv_in_block_idx=conv_in_block_idx,
        log_file=None,
        data_yaml=data_yaml
    )


# ============================================================================
# METHOD 2: ACTIVATION WITH MEDOID
# ============================================================================
def prune_method2_activation_medoid(model_path, train_data, valid_data, classes,
                                    block_idx, conv_in_block_idx, data_yaml):
    """Method 2: Activation-based pruning with medoid selection."""
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    block = detection_model[block_idx]
    conv_layers_in_block = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(conv_layers_in_block):
        print(f"⚠️  conv_in_block_idx {conv_in_block_idx} out of range")
        return None
    
    target_conv_layer = conv_layers_in_block[conv_in_block_idx]
    
    # Build sliced_block for activation extraction
    blocks_up_to = list(detection_model[:block_idx])
    submodules = []
    conv_count = 0
    for sublayer in block.children():
        submodules.append(sublayer)
        if isinstance(sublayer, nn.Conv2d):
            if conv_count == conv_in_block_idx:
                break
            conv_count += 1
    partial_block = nn.Sequential(*submodules)
    sliced_block = nn.Sequential(*(blocks_up_to + [partial_block]))
    
    # Extract activations
    mini_net = build_mini_net(sliced_block, target_conv_layer)

    try:
        train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
        train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
    except Exception as e:
        print(f"⚠️  Activation extraction failed (medoid): {e}")
        train_activations = None
    
    # Target pruning ratio
    num_channels = target_conv_layer.weight.shape[0]
    target_channels = max(num_channels // 2, num_channels // 4)

    if train_activations and not all(len(v) == 0 for v in train_activations.values()):
        # Create layer space and select optimal components (MEDOID-BASED)
        graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
        layer_weights = target_conv_layer.weight.data.detach().cpu().numpy()
        optimal_components = select_optimal_components_medoid(
            graph_space, layer_weights, num_channels, target_channels
        )
    else:
        print("⚠️  Falling back to weight-magnitude selection (medoid unavailable)")
        weights = target_conv_layer.weight.data.detach().abs().view(num_channels, -1).mean(dim=1).cpu().numpy()
        order = np.argsort(weights)[::-1]
        optimal_components = order[:target_channels].tolist()
 
    # Apply pruning
    with torch.no_grad():
        all_indices = list(range(num_channels))
        indices_to_remove = [i for i in all_indices if i not in optimal_components]
        
        target_conv_layer.weight[indices_to_remove] = 0
        if target_conv_layer.bias is not None:
            target_conv_layer.bias[indices_to_remove] = 0
        
        # Also zero corresponding BN parameters if present
        hit = -1
        for m in block.children():
            if isinstance(m, nn.Conv2d):
                hit += 1
                if hit == conv_in_block_idx:
                    found_conv = True
                    for n in block.children():
                        if found_conv and isinstance(n, nn.BatchNorm2d):
                            n.weight[indices_to_remove] = 0
                            n.bias[indices_to_remove] = 0
                            break
    
    return model


# ============================================================================
# METHOD 3: ACTIVATION WITH MAX GAMMA
# ============================================================================
def prune_method3_activation_max_gamma(model_path, train_data, valid_data, classes,
                                       block_idx, conv_in_block_idx, data_yaml):
    """Method 3: Activation-based pruning with max gamma selection."""
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    block = detection_model[block_idx]
    conv_layers_in_block = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(conv_layers_in_block):
        print(f"⚠️  conv_in_block_idx {conv_in_block_idx} out of range")
        return None
    
    target_conv_layer = conv_layers_in_block[conv_in_block_idx]
    
    # Find corresponding BN layer
    bn_layer = find_following_bn(block, conv_in_block_idx)
    if bn_layer is None:
        gamma_values = target_conv_layer.weight.data.detach().abs().mean(dim=(1, 2, 3)).cpu().numpy()
    else:
        gamma_values = bn_layer.weight.data.detach().abs().cpu().numpy()
    
    # Build sliced_block for activation extraction
    blocks_up_to = list(detection_model[:block_idx])
    submodules = []
    conv_count = 0
    for sublayer in block.children():
        submodules.append(sublayer)
        if isinstance(sublayer, nn.Conv2d):
            if conv_count == conv_in_block_idx:
                break
            conv_count += 1
    partial_block = nn.Sequential(*submodules)
    sliced_block = nn.Sequential(*(blocks_up_to + [partial_block]))
    
    # Extract activations
    mini_net = build_mini_net(sliced_block, target_conv_layer)

    try:
        train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
        train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
    except Exception as e:
        print(f"⚠️  Activation extraction failed (max gamma): {e}")
        train_activations = None

    # Target pruning ratio
    num_channels = target_conv_layer.weight.shape[0]
    target_channels = max(num_channels // 2, num_channels // 4)

    if train_activations and not all(len(v) == 0 for v in train_activations.values()):
        # Create layer space and select optimal components (MAX GAMMA-BASED)
        graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
        optimal_components = select_optimal_components_max_gamma(
            graph_space, gamma_values, num_channels, target_channels
        )
    else:
        print("⚠️  Falling back to pure gamma ranking (activation unavailable)")
        order = np.argsort(gamma_values)[::-1]
        optimal_components = order[:target_channels].tolist()
 
    # Apply pruning
    with torch.no_grad():
        all_indices = list(range(num_channels))
        indices_to_remove = [i for i in all_indices if i not in optimal_components]
        
        target_conv_layer.weight[indices_to_remove] = 0
        if target_conv_layer.bias is not None:
            target_conv_layer.bias[indices_to_remove] = 0
        
        if bn_layer is not None:
            bn_layer.weight[indices_to_remove] = 0
            bn_layer.bias[indices_to_remove] = 0
    
    return model


# ============================================================================
# METHOD 4: PURE GAMMA PRUNING
# ============================================================================
def prune_method4_pure_gamma(model, block_idx, conv_in_block_idx, target_channels):
    """Method 4: Pure gamma-based pruning (no activation analysis)."""
    torch_model = model.model
    detection_model = torch_model.model
    
    block = detection_model[block_idx]
    conv = get_all_conv2d_layers(block)[conv_in_block_idx]
    bn = find_following_bn(block, conv_in_block_idx)
    
    out_ch = conv.weight.shape[0]
    k_keep = target_channels
    
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
    
    return model


def measure_inference_time(model, data_yaml, num_runs=3):
    """
    Measure inference time for a model.
    
    Args:
        model: YOLO model
        data_yaml: Path to data YAML file
        num_runs: Number of validation runs to average (default: 3)
    
    Returns:
        Average inference time in ms per image, or None if measurement fails
    """
    inference_times = []
    
    for i in range(num_runs):
        try:
            metrics = model.val(data=data_yaml, verbose=False)
            # Get inference time from speed dict
            if hasattr(metrics, 'speed') and metrics.speed is not None:
                if isinstance(metrics.speed, dict):
                    inference_ms = metrics.speed.get('inference', None)
                    if inference_ms is not None:
                        inference_times.append(inference_ms)
                elif isinstance(metrics.speed, (int, float)):
                    inference_times.append(metrics.speed)
        except Exception as e:
            print(f"⚠️  Inference time measurement run {i+1} failed: {e}")
            continue
    
    if len(inference_times) > 0:
        avg_inference_time = sum(inference_times) / len(inference_times)
        return avg_inference_time
    else:
        return None


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def main():
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    epochs_final = 20
    epochs_per_finetune = 5
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False

    # Load data - use ALL available data
    print("📥 Loading ALL training data for activation extraction...")
    train_data = load_training_data(data_yaml, max_samples=None)
    if len(train_data) == 0:
        print("❌ No training data loaded.")
        return False
    
    print("📥 Loading ALL validation data for activation extraction...")
    try:
        valid_data = load_validation_data(data_yaml, max_samples=None)
        all_activation_data = train_data + valid_data
        print(f"   ✅ Total samples for activation extraction: {len(all_activation_data)}")
    except Exception as e:
        print(f"   ⚠️  Could not load validation data: {e}, using training data only")
        all_activation_data = train_data
        valid_data = []
    
    activation_data = all_activation_data

    # Load classes
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        classes = list(range(len(data_config['names'])))

    # Create 4 models (one for each method)
    print("\n🚀 Loading 4 models (one per method)...")
    models = {
        'method1_activation_max_weight': YOLO(model_path),
        'method2_activation_medoid': YOLO(model_path),
        'method3_activation_max_gamma': YOLO(model_path),
        'method4_pure_gamma': YOLO(model_path)
    }
    
    # Get detection models
    detection_models = {}
    for method_name, model in models.items():
        detection_models[method_name] = model.model.model

    # ========================================================================
    # MEASURE ORIGINAL MODEL INFERENCE TIME
    # ========================================================================
    print(f"\n{'='*70}")
    print("MEASURING ORIGINAL MODEL INFERENCE TIME")
    print(f"{'='*70}")
    
    original_model = YOLO(model_path)
    original_inference_time = None
    try:
        print("   Warm-up run (first validation)...")
        _ = original_model.val(data=data_yaml, verbose=False)  # Warm-up run
        
        print("   Running multiple validation runs for accurate measurement...")
        original_inference_time = measure_inference_time(original_model, data_yaml, num_runs=3)
        
        if original_inference_time is not None:
            print(f"✅ Original model inference time (averaged over 3 runs): {original_inference_time:.2f} ms/image")
        else:
            # Fallback: try single run
            print("   ⚠️  Fallback to single run...")
            original_metrics = original_model.val(data=data_yaml, verbose=False)
            if hasattr(original_metrics, 'speed') and original_metrics.speed is not None:
                if isinstance(original_metrics.speed, dict):
                    original_inference_time = original_metrics.speed.get('inference', None)
                elif isinstance(original_metrics.speed, (int, float)):
                    original_inference_time = original_metrics.speed
            
            if original_inference_time is not None:
                print(f"✅ Original model inference time (single run): {original_inference_time:.2f} ms/image")
            else:
                print(f"⚠️  Could not measure original model inference time")
                original_inference_time = None
    except Exception as e:
        print(f"⚠️  Error measuring original model inference time: {e}")
        original_inference_time = None

    # Select target layers (blocks 1, 3, 5)
    targets = select_layers_blocks_1_3_5(detection_models['method1_activation_max_weight'])
    if len(targets) == 0:
        print("❌ No eligible layers found.")
        return False

    print("\n" + "="*70)
    print("4-METHOD PRUNING COMPARISON EXPERIMENT (BLOCKS 1, 3, 5)")
    print("="*70)
    print("\n🎯 Target layers (in order):")
    for i, t in enumerate(targets):
        print(f"  {i+1}. {t['name']} ({t['num_channels']} channels)")

    # Store pruning results for each method
    method_results = {
        'method1_activation_max_weight': [],
        'method2_activation_medoid': [],
        'method3_activation_max_gamma': [],
        'method4_pure_gamma': []
    }

    # ========================================================================
    # PRUNE EACH METHOD
    # ========================================================================
    method_names = {
        'method1_activation_max_weight': 'Activation + Max Weight',
        'method2_activation_medoid': 'Activation + Medoid',
        'method3_activation_max_gamma': 'Activation + Max Gamma',
        'method4_pure_gamma': 'Pure Gamma'
    }

    for method_name in method_names.keys():
        print(f"\n{'='*70}")
        print(f"METHOD: {method_names[method_name]}")
        print(f"{'='*70}")
        
        model = models[method_name]
        pruned_details = []

        for i, t in enumerate(targets):
            print(f"\n--- Layer {i+1}/{len(targets)}: {t['name']} ---")

            block = detection_models[method_name][t['block_idx']]
            if is_c2f_block(block):
                print(f"⚠️  Skipping {t['name']} (C2f block)")
                pruned_details.append({
                    'name': t['name'],
                    'status': 'skipped',
                    'original': t['num_channels'],
                    'remaining': t['num_channels'],
                    'removed': 0
                })
                continue

            temp_path = f"temp_{method_name}_iter_{i+1}.pt"
            model.save(temp_path)

            try:
                if method_name == 'method1_activation_max_weight':
                    updated_model = prune_method1_activation_max_weight(
                        temp_path, activation_data, valid_data, classes,
                        t['block_idx'], t['conv_in_block_idx'], data_yaml
                    )
                elif method_name == 'method2_activation_medoid':
                    updated_model = prune_method2_activation_medoid(
                        temp_path, activation_data, valid_data, classes,
                        t['block_idx'], t['conv_in_block_idx'], data_yaml
                    )
                elif method_name == 'method3_activation_max_gamma':
                    updated_model = prune_method3_activation_max_gamma(
                        temp_path, activation_data, valid_data, classes,
                        t['block_idx'], t['conv_in_block_idx'], data_yaml
                    )
                elif method_name == 'method4_pure_gamma':
                    reference = method_results.get('method1_activation_max_weight', [])
                    if i >= len(reference):
                        print("⚠️  No reference pruning info from Activation + Max Weight; skipping")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        pruned_details.append({
                            'name': t['name'],
                            'status': 'skipped',
                            'original': t['num_channels'],
                            'remaining': t['num_channels'],
                            'removed': 0
                        })
                        continue

                    ref_entry = reference[i]
                    if ref_entry['status'] != 'success':
                        print(f"⚠️  Reference method skipped/failed ({ref_entry['status']}); skipping to match")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        pruned_details.append({
                            'name': t['name'],
                            'status': 'skipped',
                            'original': t['num_channels'],
                            'remaining': t['num_channels'],
                            'removed': 0
                        })
                        continue

                    num_channels = t['num_channels']
                    target_channels = ref_entry.get('remaining', num_channels)
                    target_channels = max(min(target_channels, num_channels), 0)

                    # Load fresh model state for consistency
                    temp_model = YOLO(temp_path)
                    updated_model = prune_method4_pure_gamma(
                        temp_model, t['block_idx'], t['conv_in_block_idx'], target_channels
                    )
                    # Copy pruned weights back to main model
                    apply_pruned_weights(model, updated_model, t['block_idx'], t['conv_in_block_idx'])
                    remaining = count_active_channels(get_all_conv2d_layers(
                        detection_models[method_name][t['block_idx']])[t['conv_in_block_idx']])
                    pruned_details.append({
                        'name': t['name'],
                        'status': 'success',
                        'original': num_channels,
                        'remaining': remaining,
                        'removed': num_channels - remaining
                    })
                    print(f"✅ Pruned {t['name']}: {num_channels} → {remaining} (matched Activation + Max Weight targets)")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    continue
                else:
                    raise ValueError(f"Unknown method: {method_name}")

                if updated_model is None:
                    print(f"❌ Pruning returned None")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    pruned_details.append({
                        'name': t['name'],
                        'status': 'failed',
                        'original': t['num_channels'],
                        'remaining': t['num_channels'],
                        'removed': 0
                    })
                    continue

                # Get remaining channels and copy weights
                updated_detection_model = updated_model.model.model
                updated_block = updated_detection_model[t['block_idx']]
                updated_conv = get_all_conv2d_layers(updated_block)[t['conv_in_block_idx']]
                remaining = count_active_channels(updated_conv)

                apply_pruned_weights(model, updated_model, t['block_idx'], t['conv_in_block_idx'])

                print(f"✅ Pruned {t['name']}: {t['num_channels']} → {remaining}")
                pruned_details.append({
                    'name': t['name'],
                    'status': 'success',
                    'original': t['num_channels'],
                    'remaining': remaining,
                    'removed': t['num_channels'] - remaining
                })

            except Exception as e:
                print(f"❌ Pruning failed: {e}")
                import traceback
                traceback.print_exc()
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                pruned_details.append({
                    'name': t['name'],
                    'status': 'failed',
                    'original': t['num_channels'],
                    'remaining': t['num_channels'],
                    'removed': 0
                })
                continue

            if os.path.exists(temp_path):
                os.remove(temp_path)

            # Fine-tune after each layer
            try:
                print(f"🔄 Fine-tuning for {epochs_per_finetune} epochs...")
                model.train(data=data_yaml, epochs=epochs_per_finetune, verbose=True)
                print("✅ Fine-tuning done")
            except Exception as e:
                print(f"⚠️  Fine-tuning failed: {e}")

        method_results[method_name] = pruned_details

    # ========================================================================
    # FINAL FINE-TUNING (20 epochs each)
    # ========================================================================
    print(f"\n{'='*70}")
    print("FINAL FINE-TUNING (20 EPOCHS EACH)")
    print(f"{'='*70}")

    for method_name, model in models.items():
        print(f"\n--- Fine-tuning {method_names[method_name]} ---")
        try:
            model.train(data=data_yaml, epochs=epochs_final, verbose=True)
            print(f"✅ {method_names[method_name]} fine-tuned")
        except Exception as e:
            print(f"⚠️  Fine-tuning failed: {e}")

    # ========================================================================
    # EVALUATION
    # ========================================================================
    print(f"\n{'='*70}")
    print("FINAL EVALUATION")
    print(f"{'='*70}")

    metrics_results = {}
    inference_times = {}
    
    for method_name, model in models.items():
        print(f"\n--- Evaluating {method_names[method_name]} ---")
        try:
            # Warm-up run
            print(f"   Warm-up run...")
            _ = model.val(data=data_yaml, verbose=False)
            
            # Run multiple validations for accurate inference time
            print(f"   Running multiple validation runs for accurate measurement...")
            inference_time = measure_inference_time(model, data_yaml, num_runs=3)
            
            if inference_time is not None:
                inference_times[method_name] = inference_time
                print(f"   ✅ Inference time (averaged over 3 runs): {inference_time:.2f} ms/image")
            else:
                # Fallback: try single run
                print(f"   ⚠️  Fallback to single run...")
                metrics = model.val(data=data_yaml, verbose=False)
                if hasattr(metrics, 'speed') and metrics.speed is not None:
                    if isinstance(metrics.speed, dict):
                        inference_time = metrics.speed.get('inference', None)
                    elif isinstance(metrics.speed, (int, float)):
                        inference_time = metrics.speed
                
                if inference_time is not None:
                    inference_times[method_name] = inference_time
                    print(f"   ✅ Inference time (single run): {inference_time:.2f} ms/image")
                else:
                    inference_times[method_name] = None
                    print(f"   ⚠️  Could not extract inference time from metrics")
            
            # Get metrics from last validation run
            if inference_time is not None:
                metrics = model.val(data=data_yaml, verbose=False)
                metrics_results[method_name] = {
                    'mAP50-95': metrics.results_dict.get('metrics/mAP50-95(B)', 0),
                    'mAP50': metrics.results_dict.get('metrics/mAP50(B)', 0),
                    'precision': metrics.results_dict.get('metrics/precision(B)', 0),
                    'recall': metrics.results_dict.get('metrics/recall(B)', 0)
                }
        except Exception as e:
            print(f"⚠️  Evaluation failed for {method_names[method_name]}: {e}")
            metrics_results[method_name] = None
            inference_times[method_name] = None

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print("COMPREHENSIVE SUMMARY")
    print(f"{'='*70}")

    # Pruning statistics
    print("\n📊 Pruning Statistics:")
    print(f"{'Method':<35} {'Success':<10} {'Total Removed':<15} {'Pruning Ratio':<15}")
    print("-" * 75)
    
    for method_name in method_names.keys():
        details = method_results[method_name]
        ok = sum(1 for d in details if d['status'] == 'success')
        total_original = sum(d['original'] for d in details)
        total_removed = sum(d.get('removed', 0) for d in details)
        pruning_ratio = (total_removed / total_original * 100) if total_original > 0 else 0
        print(f"{method_names[method_name]:<35} {ok}/{len(details):<10} {total_removed:<15} {pruning_ratio:.1f}%")

    # Performance comparison
    print("\n📈 Performance Comparison:")
    print(f"{'Method':<35} {'mAP@0.5:0.95':<15} {'mAP@0.5':<15} {'Precision':<15} {'Recall':<15}")
    print("-" * 95)
    
    for method_name in method_names.keys():
        if metrics_results[method_name] is not None:
            m = metrics_results[method_name]
            print(f"{method_names[method_name]:<35} {m['mAP50-95']:<15.4f} {m['mAP50']:<15.4f} "
                  f"{m['precision']:<15.4f} {m['recall']:<15.4f}")
        else:
            print(f"{method_names[method_name]:<35} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}")
    
    # Inference time comparison
    print("\n⏱️  Inference Time Comparison:")
    print(f"{'Method':<35} {'Inference Time (ms/img)':<25} {'Speedup':<15}")
    print("-" * 75)
    
    if original_inference_time is not None:
        print(f"{'Original Model':<35} {original_inference_time:<25.2f} {'1.00x (baseline)':<15}")
    
    for method_name in method_names.keys():
        if inference_times.get(method_name) is not None:
            inf_time = inference_times[method_name]
            if original_inference_time is not None and original_inference_time > 0:
                speedup = original_inference_time / inf_time
                print(f"{method_names[method_name]:<35} {inf_time:<25.2f} {speedup:.2f}x")
            else:
                print(f"{method_names[method_name]:<35} {inf_time:<25.2f} {'N/A':<15}")
        else:
            print(f"{method_names[method_name]:<35} {'N/A':<25} {'N/A':<15}")

    # Detailed layer information
    print("\n📋 Detailed Layer Information:")
    for method_name in method_names.keys():
        print(f"\n{method_names[method_name]}:")
        for d in method_results[method_name]:
            status_icon = "✅" if d['status'] == 'success' else "❌"
            removed = d.get('removed', 0)
            pruning_ratio = (removed / d['original'] * 100) if d['original'] > 0 else 0
            print(f"   {status_icon} {d['name']}: {d['original']} → {d.get('remaining', d['original'])} "
                  f"(removed {removed}, {pruning_ratio:.1f}%)")

    print(f"\n✅ 4-method comparison experiment completed!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

