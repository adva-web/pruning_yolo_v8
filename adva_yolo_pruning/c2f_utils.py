#!/usr/bin/env python3
"""
C2f Utilities for Activation-Based Pruning
Provides functions to analyze C2f block structure and build C2f-aware mini-nets
for activation extraction in C2f blocks.
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Optional
import numpy as np


def is_c2f_block(block: nn.Module) -> bool:
    """Detect if a block is a C2f module by inspecting its class name."""
    return block.__class__.__name__.lower() == 'c2f'


def get_c2f_structure_info(c2f_block: nn.Module) -> dict:
    """
    Analyze C2f block structure to extract key information.
    
    Returns dict with:
    - input_channels: input channels to C2f
    - output_channels: output channels from C2f
    - has_concat: whether C2f has internal concatenation
    - conv_count: number of Conv layers in C2f
    """
    # Get all Conv layers in the C2f block
    all_convs = []
    for module in c2f_block.modules():
        if isinstance(module, nn.Conv2d):
            all_convs.append(module)
    
    if len(all_convs) == 0:
        return None
    
    first_conv = all_convs[0]
    last_conv = all_convs[-1]
    
    return {
        'input_channels': first_conv.weight.shape[1],
        'output_channels': last_conv.weight.shape[0],
        'conv_count': len(all_convs),
        'first_conv': first_conv,
        'last_conv': last_conv
    }


def is_conv_after_c2f_concat(c2f_block: nn.Module, conv_idx: int, get_all_conv2d_layers) -> Tuple[bool, Optional[int]]:
    """
    Determine if a Conv layer is after the internal concatenation in C2f.
    
    Args:
        c2f_block: The C2f module
        conv_idx: Index of the Conv layer within the block
        get_all_conv2d_layers: Function to get all Conv layers from a module
    
    Returns:
        (is_after_concat, concat_channel_width)
        - is_after_concat: True if Conv is after concat, False otherwise
        - concat_channel_width: Expected input channels for this Conv (from concat)
    """
    all_convs = get_all_conv2d_layers(c2f_block)
    
    if conv_idx >= len(all_convs):
        return False, None
    
    target_conv = all_convs[conv_idx]
    
    # If this Conv expects more channels than the input to C2f, it's likely after concat
    # We can also look at the channel progression through the block
    structure_info = get_c2f_structure_info(c2f_block)
    if structure_info is None:
        return False, None
    
    input_ch = structure_info['input_channels']
    target_in_ch = target_conv.weight.shape[1]
    
    # Primary heuristic: channel count comparison
    # If Conv expects more channels than input to C2f, it's after concat
    if target_in_ch > input_ch:
        return True, target_in_ch
    
    # If channels match or less, it's likely before concat
    # (projection or initial processing layer)
    return False, None


class C2fSplitWrapper(nn.Module):
    """
    Wrapper that replicates C2f's split behavior without full C2f forward.
    Splits input tensor into two branches for mini-net construction.
    """
    def __init__(self, c2f_block: nn.Module):
        super().__init__()
        self.c2f_block = c2f_block
        # Store reference to original C2f for forward pass
        self.cached_input_shape = None
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Run C2f forward and extract intermediate branches for mini-net.
        This is a workaround: we actually run the full C2f forward,
        then split the output if needed.
        
        Note: This doesn't truly replicate the split, but for mini-net purposes
        it should work if we're only interested in activations at a specific Conv.
        """
        # Actually run the full C2f forward
        with torch.no_grad():
            output = self.c2f_block(x)
        
        # Return as single tensor for now
        # In a true split, we'd return [branch1, branch2]
        # But for mini-net activation extraction, running full forward is acceptable
        return output


def build_c2f_aware_mini_net(
    detection_model: nn.ModuleList,
    block_idx: int,
    conv_in_block_idx: int,
    target_conv: nn.Conv2d,
    get_all_conv2d_layers,
    build_mini_net_standard
) -> nn.Sequential:
    """
    Build a mini-net for activation extraction in a C2f block.
    
    For Convs before concat: uses standard sliced_block construction.
    For Convs after concat: builds a mini-net that includes blocks before C2f
    and then passes through the full C2f to reach the target Conv.
    
    Args:
        detection_model: The detection model containing all blocks
        block_idx: Index of the C2f block
        conv_in_block_idx: Index of Conv within the C2f block
        target_conv: The Conv layer we want to extract activations from
        get_all_conv2d_layers: Function to get all Conv layers
        build_mini_net_standard: Function to build standard mini-net
    
    Returns:
        nn.Sequential mini-net for activation extraction
    """
    c2f_block = detection_model[block_idx]
    
    # Check if this Conv is after concat
    is_after_concat, concat_width = is_conv_after_c2f_concat(
        c2f_block, conv_in_block_idx, get_all_conv2d_layers
    )
    
    if not is_after_concat:
        # Conv is before concat - need special handling for C2f
        # Since block.children() doesn't work correctly for C2f blocks,
        # we use a wrapper that runs the C2f forward and extracts Conv 0's output via hook
        print(f"   ℹ️  Conv is before concat in C2f, using hook-based extraction")
        
        # For Conv 0 (first conv before concat), we need to extract its output
        # directly from the C2f block using a forward hook
        class C2fConv0Extractor(nn.Module):
            """Wrapper to extract Conv 0 output from C2f block."""
            def __init__(self, c2f_block, target_conv):
                super().__init__()
                self.c2f_block = c2f_block
                self.target_conv = target_conv
                self.conv0_output = None
                
                # Register forward hook on target_conv to capture its output
                def hook_fn(module, input, output):
                    self.conv0_output = output
                
                self.hook_handle = target_conv.register_forward_hook(hook_fn)
            
            def forward(self, x):
                # Clear previous output
                self.conv0_output = None
                # Run C2f forward - hook will capture Conv 0 output
                _ = self.c2f_block(x)
                # Return the captured Conv 0 output (hook should have fired during forward)
                if self.conv0_output is None:
                    raise RuntimeError("Hook did not capture Conv 0 output")
                return self.conv0_output
            
            def __del__(self):
                if hasattr(self, 'hook_handle'):
                    self.hook_handle.remove()
        
        blocks_up_to = list(detection_model[:block_idx])
        conv0_extractor = C2fConv0Extractor(c2f_block, target_conv)
        sliced_block = nn.Sequential(*(blocks_up_to + [conv0_extractor]))
        
        # Build mini-net: the mini-net should just pass through the Conv 0 output
        # since we're extracting from Conv 0 itself
        return build_mini_net_standard(sliced_block, target_conv)
    
    # Conv is after concat - need special handling
    print(f"   ℹ️  Conv is after concat in C2f (expects {concat_width} channels)")
    print(f"   🔧 Building C2f-aware mini-net with full C2f forward...")
    
    # Strategy: Build mini-net that includes blocks before C2f,
    # then runs full C2f forward to get concat output,
    # then extracts activations up to the target Conv
    blocks_up_to = list(detection_model[:block_idx])
    
    # For now, use a simplified approach: include full C2f block
    # and the target Conv within it
    # This means we build a mini-net as: [blocks_before, partial_c2f]
    # where partial_c2f goes from input to target Conv
    
    # Get all submodules in C2f up to target Conv
    submodules = []
    conv_count = 0
    for sublayer in c2f_block.children():
        submodules.append(sublayer)
        if isinstance(sublayer, nn.Conv2d):
            if conv_count == conv_in_block_idx:
                break
            conv_count += 1
    
    partial_c2f = nn.Sequential(*submodules)
    sliced_block = nn.Sequential(*(blocks_up_to + [partial_c2f]))
    
    # Build standard mini-net from this sliced_block
    return build_mini_net_standard(sliced_block, target_conv)

