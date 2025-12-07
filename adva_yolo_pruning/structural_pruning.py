#!/usr/bin/env python3
"""
Structural Pruning Implementation for YOLOv8
This module implements true structural pruning that modifies the model architecture
instead of just zeroing weights, enabling proper multi-layer pruning.
"""

import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
import copy
from typing import List, Dict, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class StructuralPruner:
    """
    Implements structural pruning for YOLOv8 models using activation-based analysis.
    This class can actually remove channels from the model architecture,
    enabling proper multi-layer pruning without channel mismatches.
    """
    
    def __init__(self, model_path: str, train_data, valid_data, classes):
        """Initialize the structural pruner with a YOLOv8 model and training data."""
        self.model = YOLO(model_path)
        self.torch_model = self.model.model
        self.detection_model = self.torch_model.model
        self.original_model = copy.deepcopy(self.model)
        self.train_data = train_data
        self.valid_data = valid_data
        self.classes = classes
        
    def get_all_conv2d_layers(self, model: nn.Module) -> List[nn.Conv2d]:
        """Get all Conv2d layers in the model."""
        return [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    
    def get_conv_bn_pairs(self, block: nn.Module) -> List[Dict[str, Any]]:
        """Get Conv2d + BatchNorm pairs from a block."""
        pairs = []
        conv_layers = self.get_all_conv2d_layers(block)
        
        for i, conv in enumerate(conv_layers):
            # Find corresponding BatchNorm layer
            bn_layer = None
            for name, module in block.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    # Check if this BN follows the conv layer
                    if hasattr(module, 'weight') and module.weight is not None:
                        if module.weight.shape[0] == conv.weight.shape[0]:
                            bn_layer = module
                            break
            
            if bn_layer is not None:
                pairs.append({
                    'conv': conv,
                    'bn': bn_layer,
                    'index': i
                })
        
        return pairs
    
    def extract_bn_gamma(self, bn_layer: nn.BatchNorm2d) -> np.ndarray:
        """Extract gamma values from BatchNorm layer."""
        return bn_layer.weight.data.detach().cpu().numpy()
    
    def prune_conv2d_structurally(self, conv_layer: nn.Conv2d, indices_to_keep: List[int]) -> nn.Conv2d:
        """
        Structurally prune a Conv2d layer by creating a new layer with fewer channels.
        This actually removes channels from the architecture, not just zeroing them.
        """
        # Get original layer parameters
        original_weight = conv_layer.weight.data
        original_bias = conv_layer.bias.data if conv_layer.bias is not None else None
        
        # Create new Conv2d layer with reduced output channels
        new_conv = nn.Conv2d(
            in_channels=conv_layer.in_channels,
            out_channels=len(indices_to_keep),
            kernel_size=conv_layer.kernel_size,
            stride=conv_layer.stride,
            padding=conv_layer.padding,
            dilation=conv_layer.dilation,
            groups=conv_layer.groups,
            bias=conv_layer.bias is not None,
            padding_mode=conv_layer.padding_mode
        )
        
        # Copy weights for the channels we want to keep
        new_conv.weight.data = original_weight[indices_to_keep, :, :, :]
        if original_bias is not None:
            new_conv.bias.data = original_bias[indices_to_keep]
        
        return new_conv
    
    def prune_bn_structurally(self, bn_layer: nn.BatchNorm2d, indices_to_keep: List[int]) -> nn.BatchNorm2d:
        """
        Structurally prune a BatchNorm layer by creating a new layer with fewer channels.
        """
        # Get original layer parameters
        original_weight = bn_layer.weight.data
        original_bias = bn_layer.bias.data
        original_running_mean = bn_layer.running_mean
        original_running_var = bn_layer.running_var
        original_num_batches_tracked = bn_layer.num_batches_tracked
        
        # Create new BatchNorm layer with reduced channels
        new_bn = nn.BatchNorm2d(
            num_features=len(indices_to_keep),
            eps=bn_layer.eps,
            momentum=bn_layer.momentum,
            affine=bn_layer.affine,
            track_running_stats=bn_layer.track_running_stats
        )
        
        # Copy parameters for the channels we want to keep
        new_bn.weight.data = original_weight[indices_to_keep]
        new_bn.bias.data = original_bias[indices_to_keep]
        new_bn.running_mean = original_running_mean[indices_to_keep]
        new_bn.running_var = original_running_var[indices_to_keep]
        new_bn.num_batches_tracked = original_num_batches_tracked
        
        return new_bn
    
    def update_next_layer_input_channels(self, current_conv: nn.Conv2d, next_conv: nn.Conv2d, 
                                       indices_to_keep: List[int]) -> nn.Conv2d:
        """
        Update the next layer's input channels to match the pruned current layer's output channels.
        This is crucial for structural pruning to work properly.
        """
        # Get original layer parameters
        original_weight = next_conv.weight.data
        original_bias = next_conv.bias.data if next_conv.bias is not None else None
        
        # Create new Conv2d layer with reduced input channels
        new_conv = nn.Conv2d(
            in_channels=len(indices_to_keep),
            out_channels=next_conv.out_channels,
            kernel_size=next_conv.kernel_size,
            stride=next_conv.stride,
            padding=next_conv.padding,
            dilation=next_conv.dilation,
            groups=next_conv.groups,
            bias=next_conv.bias is not None,
            padding_mode=next_conv.padding_mode
        )
        
        # Copy weights for the input channels we want to keep
        new_conv.weight.data = original_weight[:, indices_to_keep, :, :]
        if original_bias is not None:
            new_conv.bias.data = original_bias
        
        return new_conv
    
    def replace_layer_in_model(self, old_layer: nn.Module, new_layer: nn.Module, model: nn.Module) -> None:
        """
        Replace a layer in the model with a new layer.
        This handles the complex task of updating the model structure.
        """
        for name, module in model.named_modules():
            if module is old_layer:
                # Find the parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = model
                    for part in parent_name.split('.'):
                        parent_module = getattr(parent_module, part)
                    setattr(parent_module, child_name, new_layer)
                else:
                    # Root level replacement
                    setattr(model, child_name, new_layer)
                break
    
    def prune_layer_structurally(self, conv_layer: nn.Conv2d, bn_layer: nn.BatchNorm2d, 
                               indices_to_keep: List[int], next_conv_layer: nn.Conv2d = None) -> Dict[str, Any]:
        """
        Structurally prune a Conv2d + BatchNorm pair and update the next layer if needed.
        """
        results = {}
        
        # Prune the Conv2d layer
        new_conv = self.prune_conv2d_structurally(conv_layer, indices_to_keep)
        self.replace_layer_in_model(conv_layer, new_conv, self.detection_model)
        results['pruned_conv'] = new_conv
        
        # Prune the BatchNorm layer
        if bn_layer is not None:
            new_bn = self.prune_bn_structurally(bn_layer, indices_to_keep)
            self.replace_layer_in_model(bn_layer, new_bn, self.detection_model)
            results['pruned_bn'] = new_bn
        
        # Update the next layer's input channels if it exists
        if next_conv_layer is not None:
            new_next_conv = self.update_next_layer_input_channels(conv_layer, next_conv_layer, indices_to_keep)
            self.replace_layer_in_model(next_conv_layer, new_next_conv, self.detection_model)
            results['updated_next_conv'] = new_next_conv
        
        return results
    
    def structural_activation_pruning(self, layers_to_prune: int = 3, target_blocks: List[int] = None) -> Dict[str, Any]:
        """
        Perform structural activation-based pruning on multiple layers.
        """
        if target_blocks is None:
            target_blocks = [1, 2, 3, 4, 5]  # Start from block 1 as requested
        
        print(f"\n===== Structural Activation-based Pruning of {layers_to_prune} layers =====")
        print(f"Target blocks: {target_blocks}")
        
        # Import required modules for activation analysis
        from yolov8_utils import get_all_conv2d_layers, get_raw_objects_debug_v8, aggregate_activations_from_matches, build_mini_net
        from yolo_layer_pruner import YoloLayerPruner
        from clustering import select_optimal_components
        
        # Collect all Conv2d layers from target blocks
        all_conv_layers = get_all_conv2d_layers(self.detection_model)
        available_convs = []
        
        for block_idx in target_blocks:
            if block_idx < len(self.detection_model):
                block = self.detection_model[block_idx]
                conv_layers_in_block = get_all_conv2d_layers(block)
                
                for conv_in_block_idx, conv_layer in enumerate(conv_layers_in_block):
                    num_channels = conv_layer.weight.shape[0]
                    
                    # Skip layers with too few channels (need at least 8 channels for meaningful pruning)
                    if num_channels < 8:
                        continue
                    
                    # Find global index
                    global_idx = all_conv_layers.index(conv_layer)
                    
                    available_convs.append({
                        'block_idx': block_idx,
                        'conv_in_block_idx': conv_in_block_idx,
                        'global_idx': global_idx,
                        'conv_layer': conv_layer,
                        'num_channels': num_channels
                    })
        
        print(f"Found {len(available_convs)} suitable Conv2d layers in target blocks")
        
        # Select layers with most channels for activation-based pruning
        available_convs.sort(key=lambda x: x['num_channels'], reverse=True)
        selected_convs = available_convs[:layers_to_prune]
        
        print(f"\nSelected {len(selected_convs)} layers for structural activation-based pruning:")
        for i, conv_info in enumerate(selected_convs):
            print(f"  Layer {i+1}: Block {conv_info['block_idx']}, Channels: {conv_info['num_channels']}")
        
        # Perform structural pruning
        pruned_layers_details = []
        
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
                
                # Get the target block and build sliced block
                target_block = self.detection_model[block_idx]
                target_conv_layer = conv_layer
                
                # Build sliced block: all blocks before, plus partial block up to target Conv2d
                blocks_up_to = list(self.detection_model[:block_idx])
                submodules = []
                conv_count = 0
                for sublayer in target_block.children():
                    submodules.append(sublayer)
                    if isinstance(sublayer, nn.Conv2d):
                        if conv_count == conv_in_block_idx:
                            break
                        conv_count += 1
                partial_block = nn.Sequential(*submodules)
                sliced_block = nn.Sequential(*(blocks_up_to + [partial_block]))
                
                # CRITICAL FIX: Ensure sliced_block is on the same device as the model
                device = next(self.detection_model[0].parameters()).device
                sliced_block = sliced_block.to(device)
                
                # Build mini_net for this layer first
                mini_net = build_mini_net(sliced_block, target_conv_layer)
                
                train_matched_objs, train_unmatched_objs = get_raw_objects_debug_v8(
                    self.model, mini_net, self.train_data
                )
                train_activations = aggregate_activations_from_matches(train_matched_objs, self.classes)
                
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
                
                # Use aggressive pruning approach - aim for 50% pruning
                target_channels = max(len(train_activations) // 2, len(train_activations) // 4)
                optimal_components = select_optimal_components(graph_space, layer_weights, len(train_activations), target_channels)
                
                channels_to_keep = len(optimal_components)
                channels_to_remove = num_channels - channels_to_keep
                
                print(f"  📊 Activation analysis complete:")
                print(f"    - Total channels: {num_channels}")
                print(f"    - Channels to keep: {channels_to_keep}")
                print(f"    - Channels to remove: {channels_to_remove}")
                print(f"    - Pruning ratio: {(channels_to_remove/num_channels*100):.1f}%")
                
                # Find the next Conv2d layer (if it exists)
                next_conv_layer = None
                if global_idx + 1 < len(all_conv_layers):
                    next_conv_layer = all_conv_layers[global_idx + 1]
                
                # Perform structural pruning
                print(f"  🔧 Applying structural pruning...")
                results = self.prune_layer_structurally(conv_layer, None, optimal_components, next_conv_layer)
                
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
                
                print(f"  ✅ Structural activation pruning successful: {num_channels} → {channels_to_keep} channels")
                
            except Exception as e:
                print(f"  ❌ Structural activation pruning failed: {e}")
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
        
        # Update the model's pruned_layers_details
        self.model.pruned_layers_details = pruned_layers_details
        
        return {
            'model': self.model,
            'pruned_layers_details': pruned_layers_details,
            'success_count': sum(1 for detail in pruned_layers_details if detail['status'] == 'success')
        }


def apply_structural_activation_pruning(model_path: str, train_data, valid_data, classes, data_yaml: str, layers_to_prune: int = 3) -> YOLO:
    """
    Apply structural activation-based pruning to a YOLOv8 model.
    This function performs true structural pruning that modifies the model architecture.
    """
    print(f"🔧 Starting Structural Activation-based Pruning")
    print(f"Model: {model_path}")
    print(f"Layers to prune: {layers_to_prune}")
    
    # Initialize structural pruner with training data
    pruner = StructuralPruner(model_path, train_data, valid_data, classes)
    
    # Perform structural pruning
    results = pruner.structural_activation_pruning(layers_to_prune)
    
    # Retrain the structurally pruned model
    print(f"\n🔄 Retraining structurally pruned model...")
    try:
        results['model'].train(data=data_yaml, epochs=20, verbose=False)
        print(f"✅ Retraining completed successfully")
    except Exception as e:
        print(f"⚠️  Retraining failed: {e}")
    
    # Final evaluation
    print(f"\n📊 Final evaluation of structurally pruned model...")
    try:
        final_metrics = results['model'].val(data=data_yaml, verbose=False)
        print(f"✅ Final evaluation completed")
        print(f"📈 Final metrics: {final_metrics.results_dict}")
    except Exception as e:
        print(f"⚠️  Final evaluation failed: {e}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"📊 STRUCTURAL ACTIVATION PRUNING SUMMARY")
    print(f"{'='*80}")
    print(f"Layers attempted: {len(results['pruned_layers_details'])}")
    print(f"Successfully pruned: {results['success_count']}")
    print(f"Failed attempts: {len(results['pruned_layers_details']) - results['success_count']}")
    print(f"Success rate: {results['success_count'] / len(results['pruned_layers_details']) * 100:.1f}%")
    
    print(f"\nDetailed Results:")
    print(f"{'Layer':<8} {'Block':<6} {'Conv#':<7} {'Original':<10} {'Remaining':<10} {'Pruned':<8} {'Status':<10}")
    print(f"{'-'*70}")
    
    for i, detail in enumerate(results['pruned_layers_details']):
        conv_num = detail.get('conv_in_block_idx', 'N/A')
        print(f"{i+1:<8} {detail['block_idx']:<6} {conv_num:<7} {detail['original_channels']:<10} "
              f"{detail['remaining_channels']:<10} {detail['pruned_channels']:<8} {detail['status']:<10}")
    
    print(f"{'='*80}")
    
    return results['model']


if __name__ == "__main__":
    # Test the structural pruning
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    
    # Load sample data for activation analysis
    from yolov8_utils import load_samples
    train_data, valid_data, classes = load_samples(data_yaml, max_samples=100)
    
    pruned_model = apply_structural_activation_pruning(
        model_path=model_path,
        train_data=train_data,
        valid_data=valid_data,
        classes=classes,
        data_yaml=data_yaml,
        layers_to_prune=3
    )
    print(f"✅ Structural activation pruning completed successfully!")
