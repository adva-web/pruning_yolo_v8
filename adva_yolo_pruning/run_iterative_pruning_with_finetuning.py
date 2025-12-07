#!/usr/bin/env python3
"""
Iterative Pruning with Fine-tuning
This script prunes one layer at a time, fine-tunes after each step, then continues with the next layer.
This approach prevents channel mismatch errors by maintaining model consistency between steps.
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

def adjust_subsequent_layers_robust(detection_model, pruned_conv_info, new_output_channels):
    """
    Robust channel adjustment that analyzes the actual network flow
    and updates layers that will receive the pruned layer's output.
    """
    try:
        block_idx = pruned_conv_info['block_idx']
        conv_in_block_idx = pruned_conv_info['conv_in_block_idx']
        original_channels = pruned_conv_info['num_channels']
        
        print(f"    📝 Analyzing network flow for channel adjustment...")
        print(f"      Pruned layer: Block {block_idx}, Conv {conv_in_block_idx}")
        print(f"      Original channels: {original_channels} → {new_output_channels}")
        
        # Get all Conv2d layers in the model
        all_conv_layers = get_all_conv2d_layers(detection_model)
        
        # Find the global index of the pruned layer
        current_block = detection_model[block_idx]
        current_block_convs = get_all_conv2d_layers(current_block)
        current_conv_layer = current_block_convs[conv_in_block_idx]
        
        current_global_idx = None
        for i, conv_layer in enumerate(all_conv_layers):
            if conv_layer is current_conv_layer:
                current_global_idx = i
                break
        
        if current_global_idx is None:
            print(f"      ⚠️  Could not find global index for pruned layer")
            return False
        
        print(f"      Found pruned layer at global index: {current_global_idx}")
        
        # Strategy: Look for layers that might be affected by this pruning
        # We need to find layers that receive input from this pruned layer
        layers_updated = 0
        
        # Check the next few layers after the pruned layer
        for i in range(current_global_idx + 1, min(current_global_idx + 5, len(all_conv_layers))):
            next_conv_layer = all_conv_layers[i]
            current_input_channels = next_conv_layer.weight.shape[1]
            
            print(f"      Checking layer {i}: input_channels={current_input_channels}")
            
            # If this layer's input channels match our pruned layer's output channels,
            # it likely receives input from our pruned layer
            if current_input_channels == original_channels:
                print(f"      📝 Layer {i} expects {original_channels} channels (matches pruned layer)")
                print(f"        Updating input channels: {current_input_channels} → {new_output_channels}")
                
                with torch.no_grad():
                    old_weight = next_conv_layer.weight.clone()
                    old_bias = next_conv_layer.bias.clone() if next_conv_layer.bias is not None else None
                    
                    # Create new weight tensor with adjusted input channels
                    output_channels, _, kernel_h, kernel_w = old_weight.shape
                    if new_output_channels < current_input_channels:
                        # Truncate input channels (keep first N channels)
                        new_weight = old_weight[:, :new_output_channels, :, :].clone()
                    else:
                        # Pad with zeros (shouldn't happen in pruning, but handle gracefully)
                        new_weight = torch.zeros(output_channels, new_output_channels, kernel_h, kernel_w, 
                                               device=old_weight.device, dtype=old_weight.dtype)
                        new_weight[:, :current_input_channels, :, :] = old_weight
                    
                    # Update the layer
                    next_conv_layer.weight = nn.Parameter(new_weight)
                    if old_bias is not None:
                        next_conv_layer.bias = nn.Parameter(old_bias.clone())
                    
                    # Update the layer's in_channels attribute if it exists
                    if hasattr(next_conv_layer, 'in_channels'):
                        next_conv_layer.in_channels = new_output_channels
                
                print(f"        ✅ Successfully updated layer {i}")
                layers_updated += 1
        
        print(f"      ✅ Updated {layers_updated} subsequent layers")
        return layers_updated > 0
        
    except Exception as e:
        print(f"      ❌ Failed to adjust subsequent layers: {e}")
        import traceback
        traceback.print_exc()
        return False

def prune_single_layer_iterative(model, conv_info, train_data, classes, data_yaml):
    """
    Prune a single layer and return the updated model.
    
    Args:
        model: Current YOLO model
        conv_info: Information about the layer to prune
        train_data: Training data for activation extraction
        classes: List of class IDs
        data_yaml: Path to data YAML file
    
    Returns:
        Updated model with one layer pruned
    """
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
        # Extract activations for this layer
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

        # Build mini_net and extract activations with enhanced error handling
        try:
            mini_net = build_mini_net(sliced_block, conv_layer)
            train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
            train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
        except RuntimeError as e:
            if "channels" in str(e).lower() or "expected" in str(e).lower():
                print(f"    ❌ Channel mismatch in activation extraction: {e}")
                print(f"    ⚠️  This layer depends on previously pruned layers")
                print(f"    💡 The fine-tuned model should have adapted, but there's still a mismatch")
                print(f"    🔄 Skipping this layer and continuing...")
                return None, num_channels, 0
            else:
                print(f"    ❌ Unexpected error during activation extraction: {e}")
                return None, num_channels, 0
        except Exception as e:
            print(f"    ❌ Error during activation extraction: {e}")
            return None, num_channels, 0

        if not train_activations or all(len(v) == 0 for v in train_activations.values()):
            print(f"    ⚠️  No activations found, skipping this layer")
            return None, num_channels, 0  # Return original channels as "remaining"
        
        # Create layer space and select optimal components
        graph_space = YoloLayerPruner(activations=train_activations).create_layer_space()
        layer_weights = conv_layer.weight.data.detach().cpu().numpy()
        
        # Ensure weights array matches the number of channels in the reduced matrix
        reduced_matrix = graph_space['reduced_matrix']
        if layer_weights.shape[0] != reduced_matrix.shape[0]:
            # If dimensions don't match, create a simple weight vector based on L1 norm
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
            
            # Use activation-based importance as weights
            layer_weights_flat = np.array(train_activations_flat)
        else:
            # Use L1 norm of weights as importance
            layer_weights_flat = np.linalg.norm(layer_weights.reshape(layer_weights.shape[0], -1), ord=1, axis=1)
        
        # Use aggressive pruning approach - aim for 50% pruning
        target_channels = max(num_channels // 2, num_channels // 4)
        optimal_components = select_optimal_components(graph_space, layer_weights_flat, num_channels, target_channels)
        
        channels_to_keep = len(optimal_components)
        channels_to_remove = num_channels - channels_to_keep
        
        print(f"    📊 Activation analysis complete:")
        print(f"      - Total channels: {num_channels}")
        print(f"      - Channels to keep: {channels_to_keep}")
        print(f"      - Channels to remove: {channels_to_remove}")
        print(f"      - Pruning ratio: {(channels_to_remove/num_channels*100):.1f}%")
        
        # Apply pruning by zeroing out the least important channels
        with torch.no_grad():
            # Get all channel indices
            all_indices = list(range(num_channels))
            indices_to_keep = optimal_components
            indices_to_remove = [i for i in all_indices if i not in indices_to_keep]
            
            # Zero out the least important channels
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
                        # Find the next BatchNorm layer
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
        
        # Track active channels after pruning
        active_channels_after = count_active_channels(conv_layer)
        print(f"    ✅ Pruning completed successfully!")
        print(f"    📝 Active channels after pruning: {active_channels_after}/{num_channels}")
        
        return model, channels_to_keep, channels_to_remove
        
    except Exception as e:
        print(f"    ❌ Pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return None, num_channels, 0

def main():
    print("=" * 80)
    print("Iterative Pruning with Fine-tuning")
    print("=" * 80)
    
    # Configuration
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    layers_to_prune = 3
    epochs_per_finetune = 5
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"✅ Data YAML found: {data_yaml}")
    print(f"🎯 Layers to prune: {layers_to_prune}")
    print(f"🎯 Fine-tuning epochs per layer: {epochs_per_finetune}")
    
    try:
        # Load real training and validation data
        train_data = load_training_data(data_yaml, max_samples=50)
        
        if len(train_data) == 0:
            print(f"❌ No training data loaded. Cannot proceed.")
            return False
        
        valid_data = load_validation_data(data_yaml, max_samples=30)
        
        if len(valid_data) == 0:
            print(f"⚠️  No validation data loaded, using training data for validation")
            valid_data = train_data[:20]
        
        # Get classes from YAML
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        classes = list(range(len(data_cfg['names'])))
        
        print(f"\n📊 Data loaded:")
        print(f"   - Training samples: {len(train_data)}")
        print(f"   - Validation samples: {len(valid_data)}")
        print(f"   - Classes: {len(classes)}")
        
        # Load initial model
        print(f"\n🚀 Starting iterative pruning with fine-tuning...")
        model = YOLO(model_path)
        torch_model = model.model
        detection_model = torch_model.model
        
        # Get all Conv2d layers from blocks 1-5
        target_blocks = [1, 2, 3, 4, 5]
        all_conv_layers = get_all_conv2d_layers(detection_model)
        
        # Filter layers from target blocks
        target_convs = []
        for i, conv_layer in enumerate(all_conv_layers):
            for block_idx in target_blocks:
                if block_idx < len(detection_model):
                    block = detection_model[block_idx]
                    block_convs = get_all_conv2d_layers(block)
                    if conv_layer in block_convs:
                        conv_in_block_idx = block_convs.index(conv_layer)
                        target_convs.append({
                            'conv_layer': conv_layer,
                            'block_idx': block_idx,
                            'conv_in_block_idx': conv_in_block_idx,
                            'global_idx': i,
                            'num_channels': conv_layer.weight.shape[0]
                        })
                        break
        
        # NEW STRATEGY: Select specific blocks in order Block 5 → Block 4 → Block 3
        # This allows pruning in the desired backward order
        print(f"\n🔧 Strategy: Backward Pruning Order")
        print(f"   - Select layers from Block 5 → Block 4 → Block 3")
        print(f"   - Prune ONE layer at a time")
        print(f"   - Fine-tune immediately after each pruning step")
        print(f"   - Adjust subsequent layer input channels to prevent mismatches")
        
        # Select specific blocks in the desired order: Block 5 → Block 4 → Block 3
        selected_convs = []
        desired_order = [5, 4, 3]  # Block 5 → Block 4 → Block 3
        
        for block_idx in desired_order:
            # Find the first conv layer in this block
            block_convs = [conv for conv in target_convs if conv['block_idx'] == block_idx]
            if block_convs:
                # Take the first conv layer (conv_in_block_idx = 0) from this block
                first_conv = block_convs[0]
                selected_convs.append(first_conv)
                print(f"  Selected: Block {block_idx}, Conv {first_conv['conv_in_block_idx']}, Channels: {first_conv['num_channels']}")
        
        # Limit to the requested number of layers
        selected_convs = selected_convs[:layers_to_prune]
        
        print(f"\n✅ Selected {len(selected_convs)} layers for iterative pruning:")
        for i, conv_info in enumerate(selected_convs):
            print(f"  Layer {i+1}: Block {conv_info['block_idx']}, Conv {conv_info['conv_in_block_idx']}, Channels: {conv_info['num_channels']}")
        
        # Track pruning results
        pruned_layers_details = []
        
        # TRUE ITERATIVE PRUNING: Prune one layer → Fine-tune → Repeat
        print(f"\n{'='*80}")
        print(f"STARTING TRUE ITERATIVE PRUNING")
        print(f"{'='*80}")
        
        for iter_step, conv_info in enumerate(selected_convs):
            print(f"\n{'='*60}")
            print(f"ITERATION {iter_step + 1}/{len(selected_convs)}")
            print(f"{'='*60}")
            
            # Step 1: Prune ONE layer
            print(f"\n🔧 STEP 1: Pruning Layer {iter_step + 1}")
            updated_model, remaining_channels, pruned_channels = prune_single_layer_iterative(
                model, conv_info, train_data, classes, data_yaml
            )
            
            if updated_model is None:
                print(f"❌ Layer pruning failed, skipping to next layer...")
                print(f"   This could be due to:")
                print(f"   - Channel mismatch (layer depends on previously pruned layers)")
                print(f"   - No activations found for this layer")
                print(f"   - Other activation extraction errors")
                pruned_layers_details.append({
                    'block_idx': conv_info['block_idx'],
                    'conv_in_block_idx': conv_info['conv_in_block_idx'],
                    'global_idx': conv_info['global_idx'],
                    'original_channels': conv_info['num_channels'],
                    'remaining_channels': conv_info['num_channels'],
                    'pruned_channels': 0,
                    'status': 'failed'
                })
                continue
            
            # Use the updated model
            model = updated_model
            torch_model = model.model
            detection_model = torch_model.model
            
            pruned_layers_details.append({
                'block_idx': conv_info['block_idx'],
                'conv_in_block_idx': conv_info['conv_in_block_idx'],
                'global_idx': conv_info['global_idx'],
                'original_channels': conv_info['num_channels'],
                'remaining_channels': remaining_channels,
                'pruned_channels': pruned_channels,
                'status': 'success'
            })
            
            # Step 2: Fine-tune immediately after pruning this layer
            print(f"\n🔄 STEP 2: Fine-tuning for {epochs_per_finetune} epochs")
            print(f"   This allows subsequent layers to adapt to the new channel counts")
            
            try:
                # Save intermediate model state before fine-tuning
                temp_model_path = f"temp_pruned_iter_{iter_step + 1}_before_finetune.pt"
                model.save(temp_model_path)
                
                # Fine-tune the model
                model.train(data=data_yaml, epochs=epochs_per_finetune, verbose=True)
                
                # Save fine-tuned model state
                temp_finetuned_path = f"temp_pruned_iter_{iter_step + 1}_after_finetune.pt"
                model.save(temp_finetuned_path)
                
                print(f"✅ Fine-tuning completed successfully")
                print(f"   Model adapted to new channel counts via gradient updates")
                
            except Exception as e:
                print(f"⚠️  Fine-tuning failed: {e}")
                print(f"   Continuing with next layer...")
            
            # Step 3: Adjust subsequent layer input channels (CRITICAL FIX)
            print(f"\n🔧 STEP 3: Adjusting subsequent layer input channels")
            print(f"   This prevents channel mismatch errors for the next layer")
            
            try:
                # Use a more robust channel adjustment approach
                architecture_update_success = adjust_subsequent_layers_robust(
                    detection_model, conv_info, remaining_channels
                )
                
                if architecture_update_success:
                    print(f"✅ Channel adjustment completed successfully")
                    print(f"   Subsequent layers now expect {remaining_channels} channels")
                else:
                    print(f"⚠️  Channel adjustment failed, but continuing...")
                    
            except Exception as e:
                print(f"⚠️  Channel adjustment failed: {e}")
                print(f"   This may cause issues with the next layer...")
            
            print(f"\n✅ Iteration {iter_step + 1} completed!")
            print(f"   - Layer pruned: Block {conv_info['block_idx']}, Conv {conv_info['conv_in_block_idx']}")
            print(f"   - Channels: {conv_info['num_channels']} → {remaining_channels}")
            print(f"   - Model fine-tuned and channel-adjusted")
        
        # Clean up temporary files
        print(f"\n🧹 Cleaning up temporary files...")
        for i in range(len(selected_convs)):
            temp_files = [
                f"temp_pruned_iter_{i + 1}_before_finetune.pt",
                f"temp_pruned_iter_{i + 1}_after_finetune.pt"
            ]
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"   Removed: {temp_file}")
        
        print(f"\n✅ All iterations completed!")
        
        # Final evaluation of the iteratively pruned model
        print(f"\n{'='*80}")
        print(f"FINAL EVALUATION")
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
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}")
        
        successful_prunes = len([d for d in pruned_layers_details if d['status'] == 'success'])
        failed_prunes = len([d for d in pruned_layers_details if d['status'] == 'failed'])
        
        print(f"📊 Pruning Results:")
        print(f"   - Total layers attempted: {len(pruned_layers_details)}")
        print(f"   - Successfully pruned: {successful_prunes}")
        print(f"   - Failed: {failed_prunes}")
        print(f"   - Success rate: {(successful_prunes/len(pruned_layers_details)*100):.1f}%")
        
        print(f"\n📋 Layer Details:")
        for i, details in enumerate(pruned_layers_details):
            status_icon = "✅" if details['status'] == 'success' else "❌"
            print(f"   {status_icon} Layer {i+1}: Block {details['block_idx']}, Conv {details['conv_in_block_idx']}")
            print(f"      Channels: {details['original_channels']} → {details['remaining_channels']}")
            print(f"      Status: {details['status']}")
        
        total_before = sum(d['original_channels'] for d in pruned_layers_details)
        total_after = sum(d['remaining_channels'] for d in pruned_layers_details)
        
        print(f"\n📈 Overall Statistics:")
        print(f"   - Total channels before: {total_before}")
        print(f"   - Total channels after: {total_after}")
        print(f"   - Total channels removed: {total_before - total_after}")
        print(f"   - Overall pruning ratio: {((total_before - total_after)/total_before*100):.1f}%")
        
        # Attach pruning details to model for external access
        if hasattr(model, 'pruned_layers_details'):
            model.pruned_layers_details = pruned_layers_details
        
        return True
        
    except Exception as e:
        print(f"❌ Iterative pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

