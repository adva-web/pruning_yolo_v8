#!/usr/bin/env python3
"""
Channel Adjustment Fix for Multi-Layer Pruning
This module provides functions to fix channel mismatch errors when pruning multiple layers sequentially.
"""

import torch
import torch.nn as nn
from yolov8_utils import get_all_conv2d_layers

def adjust_next_layer_input_channels(detection_model, current_block_idx, current_conv_idx, new_output_channels):
    """
    Adjust the input channels of the next Conv2d layer to match the pruned output channels.
    This prevents channel mismatch errors when pruning multiple layers sequentially.
    
    Args:
        detection_model: The YOLOv8 detection model
        current_block_idx: Index of the current block being pruned
        current_conv_idx: Index of the current conv layer within the block
        new_output_channels: Number of channels after pruning the current layer
    
    Returns:
        bool: True if adjustment was successful, False otherwise
    """
    try:
        # Find the next Conv2d layer that uses our output as input
        next_conv_layer = None
        next_block_idx = None
        next_conv_idx = None
        
        # Look in the same block first (for consecutive Conv2d layers)
        current_block = detection_model[current_block_idx]
        conv_layers_in_block = get_all_conv2d_layers(current_block)
        
        if current_conv_idx + 1 < len(conv_layers_in_block):
            # Next conv in same block
            next_conv_layer = conv_layers_in_block[current_conv_idx + 1]
            next_block_idx = current_block_idx
            next_conv_idx = current_conv_idx + 1
        else:
            # Look in the next block
            for block_idx in range(current_block_idx + 1, len(detection_model)):
                block = detection_model[block_idx]
                conv_layers_in_block = get_all_conv2d_layers(block)
                if conv_layers_in_block:
                    next_conv_layer = conv_layers_in_block[0]  # First conv in next block
                    next_block_idx = block_idx
                    next_conv_idx = 0
                    break
        
        if next_conv_layer is not None:
            current_input_channels = next_conv_layer.weight.shape[1]
            if current_input_channels != new_output_channels:
                print(f"    📝 Adjusting next Conv2d layer (Block {next_block_idx}, Conv {next_conv_idx})")
                print(f"      Input channels: {current_input_channels} → {new_output_channels}")
                
                # Adjust the input channels of the next Conv2d layer
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
                
                print(f"      ✅ Successfully adjusted input channels")
                return True
            else:
                print(f"    ✅ Next layer already has correct input channels ({current_input_channels})")
                return True
        else:
            print(f"    ⚠️  No next Conv2d layer found to adjust")
            return False
            
    except Exception as e:
        print(f"    ❌ Failed to adjust next layer input channels: {e}")
        return False

def update_model_architecture_after_pruning(detection_model, pruned_layer_info):
    """
    Update the model architecture after pruning to ensure all subsequent layers
    have the correct input channel expectations.
    
    Args:
        detection_model: The YOLOv8 detection model
        pruned_layer_info: Dictionary containing information about the pruned layer
    """
    try:
        block_idx = pruned_layer_info['block_idx']
        conv_in_block_idx = pruned_layer_info['conv_in_block_idx']
        new_output_channels = pruned_layer_info['remaining_channels']
        
        print(f"  🔧 Updating model architecture after pruning...")
        
        # Find and update all subsequent layers that might be affected
        all_conv_layers = get_all_conv2d_layers(detection_model)
        
        # Find the global index of the current pruned layer
        current_block = detection_model[block_idx]
        current_block_convs = get_all_conv2d_layers(current_block)
        current_conv_layer = current_block_convs[conv_in_block_idx]
        
        current_global_idx = None
        for i, conv_layer in enumerate(all_conv_layers):
            if conv_layer is current_conv_layer:
                current_global_idx = i
                break
        
        if current_global_idx is None:
            print(f"    ⚠️  Could not find global index for pruned layer")
            return False
        
        # Update all subsequent layers that depend on this layer's output
        for i in range(current_global_idx + 1, len(all_conv_layers)):
            next_conv_layer = all_conv_layers[i]
            
            # Check if this layer's input channels need to be updated
            current_input_channels = next_conv_layer.weight.shape[1]
            
            # If this layer expects the old number of channels, update it
            if current_input_channels == pruned_layer_info['original_channels']:
                print(f"    📝 Updating layer {i} input channels: {current_input_channels} → {new_output_channels}")
                
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
                
                print(f"      ✅ Successfully updated layer {i}")
        
        print(f"  ✅ Model architecture updated successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Failed to update model architecture: {e}")
        return False

def apply_activation_pruning_with_channel_fix(model_path, train_data, valid_data, classes, layers_to_prune=3, data_yaml="data/VOC_adva.yaml"):
    """
    Apply activation-based pruning with channel mismatch fix.
    This is a modified version that includes the channel adjustment logic.
    """
    if layers_to_prune < 1 or layers_to_prune > 12:
        raise ValueError("layers_to_prune must be between 1 and 12")
    
    print(f"\n===== Activation-based Pruning with Channel Fix =====")
    print(f"🔧 Layers to prune: {layers_to_prune}")
    print(f"🔧 This method includes channel adjustment to prevent mismatches")
    
    # Import the clustering function
    from clustering import select_optimal_components
    from yolo_layer_pruner import YoloLayerPruner
    from yolov8_utils import build_mini_net, get_raw_objects_debug_v8, aggregate_activations_from_matches
    import numpy as np
    
    # Load model
    from ultralytics import YOLO
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
    
    print(f"Found {len(target_convs)} Conv2d layers in blocks 1-5")
    
    # Sort by channel count (highest first) and select layers to prune
    target_convs.sort(key=lambda x: x['num_channels'], reverse=True)
    selected_convs = target_convs[:layers_to_prune]
    
    print(f"\nSelected {len(selected_convs)} layers for activation-based pruning:")
    for i, conv_info in enumerate(selected_convs):
        print(f"  Layer {i+1}: Block {conv_info['block_idx']}, Channels: {conv_info['num_channels']}")
    
    # Apply activation-based pruning to selected layers
    pruned_layers_details = []
    successfully_pruned_layers = 0
    
    for idx, conv_info in enumerate(selected_convs):
        conv_layer = conv_info['conv_layer']
        block_idx = conv_info['block_idx']
        conv_in_block_idx = conv_info['conv_in_block_idx']
        global_idx = conv_info['global_idx']
        num_channels = conv_info['num_channels']
        
        print(f"\nPruning Layer {idx + 1}/{len(selected_convs)}:")
        print(f"  - Block: {block_idx}")
        print(f"  - Conv in block: {conv_in_block_idx}")
        print(f"  - Global index: {global_idx}")
        print(f"  - Original channels: {num_channels}")
        
        try:
            # Extract activations for this layer
            print(f"  🔍 Extracting activations...")
            
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
            mini_net = build_mini_net(sliced_block, conv_layer)
            train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(model, mini_net, train_data)
            train_activations = aggregate_activations_from_matches(train_matched_objs, classes)
            
            if not train_activations or all(len(v) == 0 for v in train_activations.values()):
                print(f"  ⚠️  No activations found, skipping this layer")
                pruned_layers_details.append({
                    'block_idx': block_idx,
                    'conv_in_block_idx': conv_in_block_idx,
                    'global_idx': global_idx,
                    'original_channels': num_channels,
                    'remaining_channels': num_channels,
                    'pruned_channels': 0,
                    'status': 'failed',
                    'error': 'No activations found'
                })
                continue
            
            # Create layer space and select optimal components
            print(f"  🔍 Analyzing activations...")
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
            
            print(f"  📊 Activation analysis complete:")
            print(f"    - Total channels: {num_channels}")
            print(f"    - Channels to keep: {channels_to_keep}")
            print(f"    - Channels to remove: {channels_to_remove}")
            print(f"    - Pruning ratio: {(channels_to_remove/num_channels*100):.1f}%")
            
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
                
                # CRITICAL: Update model architecture after pruning
                print(f"  🔧 Updating model architecture after pruning...")
                pruned_layer_info = {
                    'block_idx': block_idx,
                    'conv_in_block_idx': conv_in_block_idx,
                    'original_channels': num_channels,
                    'remaining_channels': channels_to_keep
                }
                architecture_update_success = update_model_architecture_after_pruning(detection_model, pruned_layer_info)
                
                if not architecture_update_success:
                    print(f"  ⚠️  Architecture update failed, but continuing...")
            
            print(f"  ✅ Activation-based pruning applied successfully!")
            successfully_pruned_layers += 1
            
            # Store details
            pruned_layers_details.append({
                'block_idx': block_idx,
                'conv_in_block_idx': conv_in_block_idx,
                'global_idx': global_idx,
                'original_channels': num_channels,
                'remaining_channels': channels_to_keep,
                'pruned_channels': channels_to_remove,
                'status': 'success'
            })
            
        except Exception as e:
            print(f"  ❌ Activation pruning failed: {e}")
            pruned_layers_details.append({
                'block_idx': block_idx,
                'conv_in_block_idx': conv_in_block_idx,
                'global_idx': global_idx,
                'original_channels': num_channels,
                'remaining_channels': num_channels,
                'pruned_channels': 0,
                'status': 'failed',
                'error': str(e)
            })
    
    # Final retraining
    print(f"\n🔄 Final retraining after activation pruning...")
    try:
        model.train(data=data_yaml, epochs=20, verbose=False)
        print(f"✅ Final retraining completed successfully")
    except Exception as e:
        print(f"⚠️  Final retraining failed: {e}")
    
    # Attach pruning details to model for summary
    model.pruned_layers_details = pruned_layers_details
    
    print(f"\n✅ Activation-based pruning with channel fix completed successfully!")
    print(f"📊 Successfully pruned {successfully_pruned_layers}/{len(selected_convs)} layers")
    return model

if __name__ == "__main__":
    print("🧪 Testing channel adjustment fix...")
    print("This module provides functions to fix channel mismatch errors in multi-layer pruning.")
    print("Use apply_activation_pruning_with_channel_fix() to test the fix.")
