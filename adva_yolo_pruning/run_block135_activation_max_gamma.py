#!/usr/bin/env python3
"""
Experiment 2: Activation-based pruning with MAX GAMMA selection
Prunes Conv 0 from blocks 1, 3, 5 using activation algorithm but selects channels
with MAX BN GAMMA from each cluster (instead of max weight).

This allows comparing:
- Max gamma selection (BN importance)
- vs. Max weight selection (current default)
- vs. Medoid selection (geometric center)
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
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
from clustering_variants import select_optimal_components_max_gamma


def select_layers_blocks_1_3_5(detection_model, total_layers=3):
    """Select Conv 0 from blocks 1, 3, 5."""
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


def prune_conv2d_with_activations_max_gamma(model_path, train_data, valid_data, classes,
                                            # train_data now contains all available data (train+val combined) 
                                            block_idx, conv_in_block_idx, data_yaml):
    """
    Prune Conv2d using activation algorithm with MAX GAMMA selection (not max weight).
    Same as original but uses select_optimal_components_max_gamma.
    """
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    # Get target layer
    block = detection_model[block_idx]
    conv_layers_in_block = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(conv_layers_in_block):
        print(f"⚠️  conv_in_block_idx {conv_in_block_idx} out of range")
        return None
    
    target_conv_layer = conv_layers_in_block[conv_in_block_idx]
    
    # Find corresponding BN layer
    bn_layer = find_following_bn(block, conv_in_block_idx)
    if bn_layer is None:
        print(f"⚠️  No BN layer found for {block_idx}, Conv {conv_in_block_idx}, falling back to weight magnitude")
        # Fallback: use weight L1 norm as gamma proxy
        gamma_values = target_conv_layer.weight.data.detach().abs().mean(dim=(1, 2, 3)).cpu().numpy()
    else:
        # Extract BN gamma values (absolute)
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
    train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
    train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
    
    if not train_activations or all(len(v) == 0 for v in train_activations.values()):
        print("⚠️  No activations found, skipping pruning.")
        return None
    
    # Create layer space and select optimal components (MAX GAMMA-BASED)
    print("   🔍 Using MAX GAMMA selection (BN gamma, not weight)")
    graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
    
    # Target pruning ratio
    num_channels = target_conv_layer.weight.shape[0]
    target_channels = max(num_channels // 2, num_channels // 4)
    
    # Use MAX GAMMA selection (selects channels with highest BN gamma per cluster)
    optimal_components = select_optimal_components_max_gamma(
        graph_space, gamma_values, num_channels, target_channels
    )
    
    channels_to_keep = len(optimal_components)
    channels_to_remove = num_channels - channels_to_keep
    
    print(f"   📊 Activation analysis (MAX GAMMA selection):")
    print(f"      - Total channels: {num_channels}")
    print(f"      - Channels to keep: {channels_to_keep}")
    print(f"      - Channels to remove: {channels_to_remove}")
    print(f"      - Pruning ratio: {(channels_to_remove/num_channels*100):.1f}%")
    
    # Apply pruning
    with torch.no_grad():
        all_indices = list(range(num_channels))
        indices_to_remove = [i for i in all_indices if i not in optimal_components]
        
        target_conv_layer.weight[indices_to_remove] = 0
        if target_conv_layer.bias is not None:
            target_conv_layer.bias[indices_to_remove] = 0
        
        # Also zero corresponding BN parameters if present
        if bn_layer is not None:
            bn_layer.weight[indices_to_remove] = 0
            bn_layer.bias[indices_to_remove] = 0
    
    print(f"   ✅ Activation-based pruning (MAX GAMMA) applied successfully!")
    return model


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
        for i, m in enumerate(block.children()):
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


def main():
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    epochs_per_finetune = 5
    total_layers_to_prune = 3

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False

    # Load data - use ALL available data for better activation statistics
    # Using max_samples=None loads all available images
    print("📥 Loading ALL training data for activation extraction...")
    train_data = load_training_data(data_yaml, max_samples=None)  # None = load all
    if len(train_data) == 0:
        print("❌ No training data loaded.")
        return False
    
    print("📥 Loading ALL validation data for activation extraction...")
    try:
        valid_data = load_validation_data(data_yaml, max_samples=None)  # None = load all
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

    # Load classes
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        classes = list(range(len(data_config['names'])))

    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model

    # Select targets
    targets = select_layers_blocks_1_3_5(detection_model, total_layers=total_layers_to_prune)
    if len(targets) == 0:
        print("❌ No eligible layers found in blocks 1, 3, 5.")
        return False

    print("\n" + "="*70)
    print("EXPERIMENT 2: ACTIVATION PRUNING WITH MAX GAMMA SELECTION")
    print("="*70)
    print("\n🎯 Target layers (in order):")
    for i, t in enumerate(targets):
        print(f"  {i+1}. {t['name']} ({t['num_channels']} channels)")

    pruned_details = []

    # Iterate and prune with max gamma selection
    for i, t in enumerate(targets):
        print(f"\n{'='*70}\nITERATION {i+1}/{len(targets)}\n{'='*70}")
        print(f"   🔧 Pruning {t['name']} using MAX GAMMA selection")
        
        # Save model state and prune independently
        temp_path = f"temp_gamma_iter_{i+1}.pt"
        model.save(temp_path)
        
        try:
            updated_model = prune_conv2d_with_activations_max_gamma(
                model_path=temp_path,
                train_data=activation_data,  # Use all available data
                valid_data=valid_data,
                classes=classes,
                block_idx=t['block_idx'],
                conv_in_block_idx=t['conv_in_block_idx'],
                data_yaml=data_yaml
            )
        except Exception as e:
            print(f"❌ Pruning failed for {t['name']}: {e}")
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
        
        if updated_model is None:
            print(f"❌ Pruning returned None for {t['name']}")
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
        
        # Get actual remaining channels BEFORE copying (from updated_model)
        updated_detection_model = updated_model.model.model
        updated_block = updated_detection_model[t['block_idx']]
        updated_conv = get_all_conv2d_layers(updated_block)[t['conv_in_block_idx']]
        remaining = count_active_channels(updated_conv)
        
        # Copy pruned weights back to main model
        apply_pruned_weights(model, updated_model, t['block_idx'], t['conv_in_block_idx'])
        
        print(f"✅ Pruned {t['name']}: {t['num_channels']} → {remaining} (removed {t['num_channels'] - remaining})")
        pruned_details.append({
            'name': t['name'], 
            'status': 'success', 
            'original': t['num_channels'], 
            'remaining': remaining,
            'removed': t['num_channels'] - remaining
        })
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Fine-tune
        try:
            print(f"🔄 Fine-tuning for {epochs_per_finetune} epochs...")
            model.train(data=data_yaml, epochs=epochs_per_finetune, verbose=True)
            print("✅ Fine-tuning done")
        except Exception as e:
            print(f"⚠️  Fine-tuning failed: {e}")

    # Final fine-tuning (20 epochs)
    print("\n" + "="*70)
    print("FINAL FINE-TUNING (20 EPOCHS)")
    print("="*70)
    try:
        print("🔄 Starting final fine-tuning for 20 epochs...")
        model.train(data=data_yaml, epochs=20, verbose=True)
        print("✅ Final fine-tuning completed!")
    except Exception as e:
        print(f"⚠️  Final fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    try:
        metrics = model.val(data=data_yaml, verbose=False)
        map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0)
        map50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
        precision = metrics.results_dict.get('metrics/precision(B)', 0)
        recall = metrics.results_dict.get('metrics/recall(B)', 0)
        print(f"mAP@0.5:0.95: {map50_95:.4f}")
        print(f"mAP@0.5: {map50:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    except Exception as e:
        print(f"⚠️  Evaluation failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY (ACTIVATION + MAX GAMMA SELECTION)")
    print("="*70)
    ok = sum(1 for d in pruned_details if d['status'] == 'success')
    total_original = sum(d['original'] for d in pruned_details)
    total_remaining = sum(d.get('remaining', d['original']) for d in pruned_details)
    total_removed = sum(d.get('removed', 0) for d in pruned_details)
    overall_pruning_ratio = (total_removed / total_original * 100) if total_original > 0 else 0
    
    print(f"Successfully pruned: {ok}/{len(pruned_details)} layers")
    print(f"\n📊 Pruning Statistics:")
    print(f"   Total original channels: {total_original}")
    print(f"   Total remaining channels: {total_remaining}")
    print(f"   Total removed channels: {total_removed}")
    print(f"   Overall pruning ratio: {overall_pruning_ratio:.1f}%")
    print(f"\n📋 Layer Details:")
    for d in pruned_details:
        status_icon = "✅" if d['status'] == 'success' else "❌"
        removed = d.get('removed', 0)
        pruning_ratio = (removed / d['original'] * 100) if d['original'] > 0 else 0
        print(f"   {status_icon} {d['name']}: {d['original']} → {d.get('remaining', d['original'])} channels "
              f"(removed {removed}, {pruning_ratio:.1f}%)")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
