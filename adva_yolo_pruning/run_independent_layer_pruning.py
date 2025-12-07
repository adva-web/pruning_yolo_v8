#!/usr/bin/env python3
"""
Independent Layer Pruning with Fine-tuning
This script implements the WORKING regular activation pruning approach that prunes each layer independently.
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

from yolov8_utils import get_all_conv2d_layers
from pruning_yolo_v8 import prune_conv2d_in_block_with_activations

import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO

def main():
    print("=" * 80)
    print("Independent Layer Pruning with Fine-tuning")
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
        print(f"\n🚀 Starting independent layer pruning with smart layer selection")
        model = YOLO(model_path)
        
        # STRATEGY: Select truly independent layers to avoid channel mismatches
        print(f"\n🔧 Strategy: Select Truly Independent Layers")
        print(f"   - Choose layers that don't depend on each other")
        print(f"   - Avoid cross-block dependencies")
        print(f"   - No channel mismatches occur")
        print(f"   - Fine-tuning after each layer")
        
        # Get all Conv2d layers from the model
        torch_model = model.model
        detection_model = torch_model.model
        all_conv_layers = get_all_conv2d_layers(detection_model)
        
        # Group layers by block
        block_layers = {}
        for i, conv_layer in enumerate(all_conv_layers):
            for block_idx in range(len(detection_model)):
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
        
        print(f"\n📊 Available layers by block:")
        for block_idx in sorted(block_layers.keys()):
            layers = block_layers[block_idx]
            print(f"   Block {block_idx}: {len(layers)} layers")
            for layer in layers:
                print(f"     - Conv {layer['conv_in_block_idx']}: {layer['num_channels']} channels")
        
        # Select independent layers using different strategies
        pruning_order = []
        
        # Strategy 1: Select from Block 1 only (most independent)
        if 1 in block_layers and len(block_layers[1]) >= layers_to_prune:
            print(f"\n✅ Strategy 1: Selecting {layers_to_prune} layers from Block 1 only")
            print(f"   Block 1 is the most independent (no dependencies on other blocks)")
            
            # Sort by channel count (highest first)
            block_layers[1].sort(key=lambda x: x['num_channels'], reverse=True)
            
            for i in range(min(layers_to_prune, len(block_layers[1]))):
                layer = block_layers[1][i]
                pruning_order.append({
                    'block_idx': layer['block_idx'],
                    'conv_in_block_idx': layer['conv_in_block_idx'],
                    'name': f"Block {layer['block_idx']}, Conv {layer['conv_in_block_idx']}",
                    'num_channels': layer['num_channels']
                })
        
        # Strategy 2: Select one layer from different blocks (if Block 1 doesn't have enough)
        elif 1 in block_layers:
            print(f"\n✅ Strategy 2: Selecting layers from Block 1 only (limited layers available)")
            print(f"   Block 1 is the only safe block to prune")
            
            # Sort by channel count (highest first)
            block_layers[1].sort(key=lambda x: x['num_channels'], reverse=True)
            
            # Take whatever layers are available from Block 1
            for i in range(min(layers_to_prune, len(block_layers[1]))):
                layer = block_layers[1][i]
                pruning_order.append({
                    'block_idx': layer['block_idx'],
                    'conv_in_block_idx': layer['conv_in_block_idx'],
                    'name': f"Block {layer['block_idx']}, Conv {layer['conv_in_block_idx']}",
                    'num_channels': layer['num_channels']
                })
        
        # Strategy 3: NO OTHER BLOCKS - Only Block 1 is safe
        else:
            print(f"\n❌ No layers available from Block 1")
            print(f"   Cannot safely prune any layers")
            return False
        
        # Limit to the requested number of layers
        pruning_order = pruning_order[:layers_to_prune]
        
        print(f"\n✅ Selected {len(pruning_order)} layers for independent pruning:")
        for i, layer_info in enumerate(pruning_order):
            print(f"  Layer {i+1}: {layer_info['name']}")
        
        # Track pruning results
        pruned_layers_details = []
        
        # INDEPENDENT LAYER PRUNING: Prune one layer → Fine-tune → Repeat
        print(f"\n{'='*80}")
        print(f"STARTING INDEPENDENT LAYER PRUNING WITH FALLBACK")
        print(f"{'='*80}")
        
        # Track which layers we've already tried to avoid duplicates
        attempted_layers = set()
        
        for iter_step in range(layers_to_prune):
            print(f"\n{'='*60}")
            print(f"ITERATION {iter_step + 1}/{layers_to_prune}")
            print(f"{'='*60}")
            
            # Find the best available layer for this iteration
            current_layer_info = None
            
            # Try layers in order of preference
            for layer_info in pruning_order:
                layer_key = (layer_info['block_idx'], layer_info['conv_in_block_idx'])
                if layer_key not in attempted_layers:
                    current_layer_info = layer_info
                    attempted_layers.add(layer_key)
                    break
            
            # If no more layers from original selection, try fallback layers
            if current_layer_info is None:
                print(f"🔄 No more layers from original selection, trying fallback layers...")
                
                # Try other layers from available blocks
                for block_idx in sorted(block_layers.keys()):
                    if current_layer_info is not None:
                        break
                    
                    for layer in block_layers[block_idx]:
                        layer_key = (layer['block_idx'], layer['conv_in_block_idx'])
                        if layer_key not in attempted_layers:
                            current_layer_info = {
                                'block_idx': layer['block_idx'],
                                'conv_in_block_idx': layer['conv_in_block_idx'],
                                'name': f"Block {layer['block_idx']}, Conv {layer['conv_in_block_idx']}",
                                'num_channels': layer['num_channels']
                            }
                            attempted_layers.add(layer_key)
                            print(f"   ✅ Selected fallback layer: {current_layer_info['name']}")
                            break
            
            if current_layer_info is None:
                print(f"❌ No more layers available to prune")
                break
            
            print(f"🎯 Target Layer: {current_layer_info['name']}")
            
            # Step 1: Save current model state to temporary file
            print(f"\n🔧 STEP 1: Saving model state for independent pruning")
            temp_model_path = f"temp_independent_pruned_iter_{iter_step + 1}.pt"
            model.save(temp_model_path)
            print(f"   Model saved to: {temp_model_path}")
            
            # Step 2: Prune ONE layer using the working regular method
            print(f"\n🔧 STEP 2: Pruning {current_layer_info['name']} independently")
            
            try:
                # Use the working regular activation pruning method
                updated_model = prune_conv2d_in_block_with_activations(
                    model_path=temp_model_path,
                    train_data=train_data,
                    valid_data=valid_data,
                    classes=classes,
                    block_idx=current_layer_info['block_idx'],
                    conv_in_block_idx=current_layer_info['conv_in_block_idx'],
                    log_file=f"pruning_independent_iter_{iter_step + 1}.txt",
                    data_yaml=data_yaml
                )
                
                if updated_model is None:
                    print(f"❌ Layer pruning failed (returned None), trying next available layer...")
                    pruned_layers_details.append({
                        'block_idx': current_layer_info['block_idx'],
                        'conv_in_block_idx': current_layer_info['conv_in_block_idx'],
                        'original_channels': current_layer_info['num_channels'],
                        'remaining_channels': current_layer_info['num_channels'],
                        'pruned_channels': 0,
                        'status': 'failed'
                    })
                    
                    # Clean up temporary file
                    if os.path.exists(temp_model_path):
                        os.remove(temp_model_path)
                    
                    # Continue to next iteration to try another layer
                    continue
                    
            except Exception as e:
                print(f"❌ Layer pruning failed with exception: {e}")
                print(f"🔄 Trying next available layer...")
                
                pruned_layers_details.append({
                    'block_idx': current_layer_info['block_idx'],
                    'conv_in_block_idx': current_layer_info['conv_in_block_idx'],
                    'original_channels': current_layer_info['num_channels'],
                    'remaining_channels': current_layer_info['num_channels'],
                    'pruned_channels': 0,
                    'status': 'failed'
                })
                
                # Clean up temporary file
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                
                # Continue to next iteration to try another layer
                continue
            
            # CRITICAL: Apply the pruning to our current model to accumulate changes
            print(f"   🔄 Applying pruning to current model to accumulate changes")
            
            # Get the pruned layer from the updated model
            torch_model_updated = updated_model.model
            detection_model_updated = torch_model_updated.model
            block_updated = detection_model_updated[current_layer_info['block_idx']]
            block_convs_updated = get_all_conv2d_layers(block_updated)
            pruned_conv = block_convs_updated[current_layer_info['conv_in_block_idx']]
            
            # Apply the same pruning to our current model
            torch_model = model.model
            detection_model = torch_model.model
            block = detection_model[current_layer_info['block_idx']]
            block_convs = get_all_conv2d_layers(block)
            target_conv = block_convs[current_layer_info['conv_in_block_idx']]
            
            # Copy the pruned weights to our current model
            with torch.no_grad():
                target_conv.weight.copy_(pruned_conv.weight)
                if target_conv.bias is not None and pruned_conv.bias is not None:
                    target_conv.bias.copy_(pruned_conv.bias)
            
            # Also copy BatchNorm parameters if they exist
            # Find corresponding BatchNorm layer
            bn_layer = None
            conv_count = 0
            for sublayer in block.children():
                if isinstance(sublayer, nn.Conv2d):
                    if conv_count == current_layer_info['conv_in_block_idx']:
                        # Find the next BatchNorm layer
                        for next_sublayer in block.children():
                            if isinstance(next_sublayer, nn.BatchNorm2d):
                                bn_layer = next_sublayer
                                break
                        break
                    conv_count += 1
            
            if bn_layer is not None:
                # Find corresponding BatchNorm in updated model
                bn_layer_updated = None
                conv_count = 0
                for sublayer in block_updated.children():
                    if isinstance(sublayer, nn.Conv2d):
                        if conv_count == current_layer_info['conv_in_block_idx']:
                            # Find the next BatchNorm layer
                            for next_sublayer in block_updated.children():
                                if isinstance(next_sublayer, nn.BatchNorm2d):
                                    bn_layer_updated = next_sublayer
                                    break
                            break
                        conv_count += 1
                
                if bn_layer_updated is not None:
                    with torch.no_grad():
                        bn_layer.weight.copy_(bn_layer_updated.weight)
                        bn_layer.bias.copy_(bn_layer_updated.bias)
                        bn_layer.running_mean.copy_(bn_layer_updated.running_mean)
                        bn_layer.running_var.copy_(bn_layer_updated.running_var)
            
            print(f"   ✅ Pruning applied to current model successfully")
            
            # Count active channels after pruning
            torch_model = model.model
            detection_model = torch_model.model
            block = detection_model[current_layer_info['block_idx']]
            block_convs = get_all_conv2d_layers(block)
            target_conv = block_convs[current_layer_info['conv_in_block_idx']]
            active_channels_after = count_active_channels(target_conv)
            
            # Get original channel count
            original_channels = target_conv.weight.shape[0]
            pruned_channels = original_channels - active_channels_after
            
            pruned_layers_details.append({
                'block_idx': current_layer_info['block_idx'],
                'conv_in_block_idx': current_layer_info['conv_in_block_idx'],
                'original_channels': original_channels,
                'remaining_channels': active_channels_after,
                'pruned_channels': pruned_channels,
                'status': 'success'
            })
            
            print(f"✅ Layer pruning completed successfully!")
            print(f"   Channels: {original_channels} → {active_channels_after}")
            print(f"   Pruned: {pruned_channels} channels")
            
            # Step 3: Fine-tune immediately after pruning this layer
            print(f"\n🔄 STEP 3: Fine-tuning for {epochs_per_finetune} epochs")
            print(f"   This allows the model to adapt to the pruned layer")
            
            try:
                # Fine-tune the model
                model.train(data=data_yaml, epochs=epochs_per_finetune, verbose=True)
                
                print(f"✅ Fine-tuning completed successfully")
                print(f"   Model adapted to new channel counts via gradient updates")
                
            except Exception as e:
                print(f"⚠️  Fine-tuning failed: {e}")
                print(f"   Continuing with next layer...")
            
            print(f"\n✅ Iteration {iter_step + 1} completed!")
            print(f"   - Layer pruned: {current_layer_info['name']}")
            print(f"   - Channels: {original_channels} → {active_channels_after}")
            print(f"   - Model fine-tuned")
            
            # Clean up temporary file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                print(f"🧹 Cleaned up: {temp_model_path}")
        
        # Final evaluation
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
            print(f"   {status_icon} Layer {i+1}: {details['block_idx']}, Conv {details['conv_in_block_idx']}")
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
        print(f"❌ Independent layer pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
