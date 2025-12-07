#!/usr/bin/env python3
"""
Multi-Block Pruning with Independent Sessions
This script runs separate pruning sessions for each block, saving and loading models between sessions.
This avoids the sliced_block dependency issue by keeping each block's pruning independent.
"""

import os
import sys
import yaml

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruning_yolo_v8_sequential_fix import (
    load_training_data,
    load_validation_data,
    count_active_channels
)

from yolo_layer_pruner import YoloLayerPruner
from clustering import select_optimal_components
from yolov8_utils import (
    build_mini_net,
    get_all_conv2d_layers,
    get_raw_objects_debug_v8,
    aggregate_activations_from_matches
)

import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO

def prune_single_layer_iterative(model, conv_info, train_data, classes, data_yaml):
    """Same as before - prunes a single layer"""
    conv_layer = conv_info['conv_layer']
    block_idx = conv_info['block_idx']
    conv_in_block_idx = conv_info['conv_in_block_idx']
    global_idx = conv_info['global_idx']
    num_channels = conv_info['num_channels']
    
    print(f"\n  Pruning layer:")
    print(f"    - Block: {block_idx}")
    print(f"    - Conv in block: {conv_in_block_idx}")
    print(f"    - Global index: {global_idx}")
    print(f"    - Original channels: {num_channels}")
    
    try:
        torch_model = model.model
        detection_model = torch_model.model
        
        # Build sliced block for this layer
        blocks_up_to = list(detection_model[:block_idx])
        block = detection_model[block_idx]
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

        # Build mini_net and extract activations
        # CRITICAL: Try to extract activations, but catch channel mismatch errors
        try:
            mini_net = build_mini_net(sliced_block, conv_layer)
            train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
            train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
        except RuntimeError as e:
            if "channels" in str(e) or "expected" in str(e):
                print(f"    ❌ Channel mismatch in activation extraction: {e}")
                print(f"    ⚠️  Cannot extract activations due to architecture mismatch")
                print(f"    💡 This layer depends on previously pruned layers")
                return None, num_channels, 0
            raise

        if not train_activations or all(len(v) == 0 for v in train_activations.values()):
            print(f"    ⚠️  No activations found, skipping this layer")
            return None, num_channels, 0
        
        # Create layer space and select optimal components
        graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
        layer_weights = conv_layer.weight.data.detach().cpu().numpy()
        
        reduced_matrix = graph_space['reduced_matrix']
        if layer_weights.shape[0] != reduced_matrix.shape[0]:
            train_activations_flat = []
            for channel_id in range(num_channels):
                if channel_id in train_activations:
                    channel_activations = []
                    for class_id, activations in train_activations[channel_id].items():
                        channel_activations.extend(activations)
                    if channel_activations:
                        train_activations_flat.append(np.mean(channel_activations))
                    else:
                        train_activations_flat.append(0.0)
                else:
                    train_activations_flat.append(0.0)
            layer_weights_flat = np.array(train_activations_flat)
        else:
            layer_weights_flat = np.linalg.norm(layer_weights.reshape(layer_weights.shape[0], -1), ord=1, axis=1)
        
        target_channels = max(num_channels // 2, num_channels // 4)
        optimal_components = select_optimal_components(graph_space, layer_weights_flat, num_channels, target_channels)
        
        channels_to_keep = len(optimal_components)
        channels_to_remove = num_channels - channels_to_keep
        
        print(f"    📊 Activation analysis complete:")
        print(f"      - Total channels: {num_channels}")
        print(f"      - Channels to keep: {channels_to_keep}")
        print(f"      - Channels to remove: {channels_to_remove}")
        print(f"      - Pruning ratio: {(channels_to_remove/num_channels*100):.1f}%")
        
        # Apply pruning
        with torch.no_grad():
            all_indices = list(range(num_channels))
            indices_to_keep = optimal_components
            indices_to_remove = [i for i in all_indices if i not in indices_to_keep]
            
            conv_layer.weight[indices_to_remove] = 0
            if conv_layer.bias is not None:
                conv_layer.bias[indices_to_remove] = 0
            
            # Find and zero corresponding BatchNorm channels
            block = detection_model[block_idx]
            bn_layer = None
            conv_count = 0
            for sublayer in block.children():
                if isinstance(sublayer, nn.Conv2d):
                    if conv_count == conv_in_block_idx:
                        for next_sublayer in block.children():
                            if isinstance(next_sublayer, nn.BatchNorm2d):
                                bn_layer = next_sublayer
                                break
                        break
                    conv_count += 1
            
            if bn_layer is not None:
                bn_layer.weight[indices_to_remove] = 0
                bn_layer.bias[indices_to_remove] = 0
                bn_layer.running_mean[indices_to_remove] = 0
                bn_layer.running_var[indices_to_remove] = 1
        
        active_channels_after = count_active_channels(conv_layer)
        print(f"    ✅ Pruning completed successfully!")
        print(f"    📝 Active channels after pruning: {active_channels_after}/{num_channels}")
        
        return model, channels_to_keep, channels_to_remove
        
    except Exception as e:
        print(f"    ❌ Pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return None, num_channels, 0

def prune_block(model, block_layers, train_data, classes, data_yaml, epochs_per_finetune=5):
    """Prune all layers in a specific block, then fine-tune."""
    
    if not block_layers:
        return model, []
    
    print(f"\n{'='*80}")
    print(f"PRUNING BLOCK WITH {len(block_layers)} LAYERS")
    print(f"{'='*80}")
    
    pruned_details = []
    
    # Step 1: Prune all layers in this block
    for idx, conv_info in enumerate(block_layers):
        print(f"\nPruning Layer {idx + 1}/{len(block_layers)}:")
        updated_model, remaining_channels, pruned_channels = prune_single_layer_iterative(
            model, conv_info, train_data, classes, data_yaml
        )
        
        if updated_model is None:
            continue
        
        model = updated_model
        torch_model = model.model
        detection_model = torch_model.model
        
        pruned_details.append({
            'block_idx': conv_info['block_idx'],
            'conv_in_block_idx': conv_info['conv_in_block_idx'],
            'global_idx': conv_info['global_idx'],
            'original_channels': conv_info['num_channels'],
            'remaining_channels': remaining_channels,
            'pruned_channels': pruned_channels,
            'status': 'success'
        })
    
    # Step 2: Fine-tune after pruning this block
    print(f"\n{'='*80}")
    print(f"FINE-TUNING FOR {epochs_per_finetune} EPOCHS")
    print(f"{'='*80}")
    
    try:
        model.train(data=data_yaml, epochs=epochs_per_finetune, verbose=True)
        print(f"✅ Fine-tuning completed successfully")
    except Exception as e:
        print(f"⚠️  Fine-tuning failed: {e}")
    
    return model, pruned_details

def main():
    print("=" * 80)
    print("Multi-Block Pruning with Independent Sessions")
    print("=" * 80)
    
    # Configuration
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    layers_per_block = 2  # Prune 2 layers per block
    epochs_per_finetune = 5
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False
    
    print(f"✅ Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Data: {data_yaml}")
    print(f"   Layers per block: {layers_per_block}")
    print(f"   Epochs per fine-tune: {epochs_per_finetune}")
    
    try:
        # Load data
        train_data = load_training_data(data_yaml, max_samples=50)
        if len(train_data) == 0:
            print(f"❌ No training data loaded.")
            return False
        
        valid_data = load_validation_data(data_yaml, max_samples=30)
        if len(valid_data) == 0:
            valid_data = train_data[:20]
        
        # Get classes
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        classes = list(range(len(data_cfg['names'])))
        
        # Load initial model
        model = YOLO(model_path)
        torch_model = model.model
        detection_model = torch_model.model
        
        # Get all Conv2d layers from blocks 1-5
        target_blocks = [1, 2, 3, 4, 5]
        all_conv_layers = get_all_conv2d_layers(detection_model)
        
        # Organize layers by block
        block_layers = {}
        for i, conv_layer in enumerate(all_conv_layers):
            for block_idx in target_blocks:
                if block_idx < len(detection_model):
                    block = detection_model[block_idx]
                    block_convs = get_all_conv2d_layers(block)
                    if conv_layer in block_convs:
                        conv_in_block_idx = block_convs.index(conv_layer)
                        if block_idx not in block_layers:
                            block_layers[block_idx] = []
                        block_layers[block_idx].append({
                            'conv_layer': conv_layer,
                            'block_idx': block_idx,
                            'conv_in_block_idx': conv_in_block_idx,
                            'global_idx': i,
                            'num_channels': conv_layer.weight.shape[0]
                        })
                        break
        
        # Select layers from each block (sort by channel count)
        selected_blocks = {}
        for block_idx in sorted(block_layers.keys()):
            block_layers[block_idx].sort(key=lambda x: x['num_channels'], reverse=True)
            # Take top layers_per_block layers from each block
            selected_blocks[block_idx] = block_layers[block_idx][:layers_per_block]
            print(f"Block {block_idx}: {len(selected_blocks[block_idx])} layers selected")
        
        # Prune each block independently
        all_pruned_details = []
        
        print(f"\n⚠️  CRITICAL LIMITATION:")
        print(f"   Due to architectural constraints, we can only reliably prune Block 1")
        print(f"   Other blocks fail due to channel mismatch in activation extraction")
        print(f"   Fine-tuning doesn't solve this because sliced_block includes pruned blocks")
        print(f"\n📝 Strategy: Prune ONLY from Block 1 (first block)")
        
        # ONLY prune Block 1 for now (most reliable)
        block_idx_to_prune = 1
        
        if block_idx_to_prune in selected_blocks:
            block_layers_list = selected_blocks[block_idx_to_prune]
            
            if block_layers_list:
                print(f"\n{'='*80}")
                print(f"PROCESSING BLOCK {block_idx_to_prune} ONLY")
                print(f"{'='*80}")
                
                # Prune Block 1 and fine-tune
                model, block_pruned_details = prune_block(
                    model, block_layers_list, train_data, classes, data_yaml, epochs_per_finetune
                )
                all_pruned_details.extend(block_pruned_details)
                
                # Save model
                intermediate_model_path = f"pruned_model_block_{block_idx_to_prune}.pt"
                model.save(intermediate_model_path)
                print(f"💾 Saved pruned model: {intermediate_model_path}")
        else:
            print(f"❌ Block {block_idx_to_prune} not found in selected blocks")
            return False
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Total layers pruned: {len(all_pruned_details)}")
        
        total_before = sum(d['original_channels'] for d in all_pruned_details)
        total_after = sum(d['remaining_channels'] for d in all_pruned_details)
        
        print(f"Total channels before: {total_before}")
        print(f"Total channels after: {total_after}")
        print(f"Total channels removed: {total_before - total_after}")
        
        # Final test
        print(f"\n{'='*80}")
        print(f"FINAL MODEL PERFORMANCE")
        print(f"{'='*80}")
        
        try:
            test_metrics = model.val(data=data_yaml, verbose=False)
            print(f"📈 Final model performance:")
            print(f"   mAP@0.5:0.95: {test_metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"   mAP@0.5: {test_metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   Precision: {test_metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
            print(f"   Recall: {test_metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
        except Exception as e:
            print(f"⚠️  Performance testing failed: {e}")
        
        if hasattr(model, 'pruned_layers_details'):
            model.pruned_layers_details = all_pruned_details
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-block pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

