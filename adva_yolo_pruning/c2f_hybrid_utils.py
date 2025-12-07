#!/usr/bin/env python3
"""
C2f Hybrid Pruning Utilities
Provides functions to analyze C2f block structure and categorize conv layers
for hybrid pruning (activation on Conv 0, gamma on Conv 1+).
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from yolov8_utils import get_all_conv2d_layers
from c2f_utils import is_c2f_block, get_c2f_structure_info, is_conv_after_c2f_concat


def analyze_c2f_block_structure(c2f_block: nn.Module) -> Dict:
    """
    Analyze C2f block structure to extract detailed conv layer information.
    
    Returns dict with:
    - input_channels: input channels to C2f
    - output_channels: output channels from C2f
    - conv_count: number of Conv layers in C2f
    - convs: list of conv layer info (idx, layer, in_ch, out_ch, is_after_concat)
    """
    if not is_c2f_block(c2f_block):
        return None
    
    structure_info = get_c2f_structure_info(c2f_block)
    if structure_info is None:
        return None
    
    all_convs = get_all_conv2d_layers(c2f_block)
    input_channels = structure_info['input_channels']
    
    convs_info = []
    for idx, conv in enumerate(all_convs):
        in_ch = conv.weight.shape[1]
        out_ch = conv.weight.shape[0]
        is_after_concat, concat_width = is_conv_after_c2f_concat(
            c2f_block, idx, get_all_conv2d_layers
        )
        
        convs_info.append({
            'idx': idx,
            'conv': conv,
            'in_channels': in_ch,
            'out_channels': out_ch,
            'is_after_concat': is_after_concat,
            'concat_width': concat_width if is_after_concat else None
        })
    
    return {
        'input_channels': input_channels,
        'output_channels': structure_info['output_channels'],
        'conv_count': len(all_convs),
        'convs': convs_info
    }


def get_c2f_conv_categories(c2f_block: nn.Module) -> Dict[str, List[Dict]]:
    """
    Categorize conv layers in C2f block for hybrid pruning.
    
    Returns dict with:
    - 'before_concat': List of convs before concatenation (use activation pruning)
    - 'after_concat': List of convs after concatenation (use gamma pruning)
    
    Each conv dict contains: idx, conv, in_ch, out_ch
    """
    structure = analyze_c2f_block_structure(c2f_block)
    if structure is None:
        return {'before_concat': [], 'after_concat': []}
    
    before_concat = []
    after_concat = []
    
    for conv_info in structure['convs']:
        if conv_info['is_after_concat']:
            after_concat.append(conv_info)
        else:
            before_concat.append(conv_info)
    
    return {
        'before_concat': before_concat,
        'after_concat': after_concat
    }


def print_c2f_structure(c2f_block: nn.Module, block_idx: int = None):
    """
    Print detailed C2f block structure analysis.
    """
    structure = analyze_c2f_block_structure(c2f_block)
    if structure is None:
        print("❌ Not a C2f block or structure analysis failed")
        return
    
    block_name = f"Block {block_idx}" if block_idx is not None else "C2f block"
    print(f"\n{'='*70}")
    print(f"C2F BLOCK STRUCTURE: {block_name}")
    print(f"{'='*70}")
    print(f"Input channels: {structure['input_channels']}")
    print(f"Output channels: {structure['output_channels']}")
    print(f"Total Conv layers: {structure['conv_count']}")
    
    categories = get_c2f_conv_categories(c2f_block)
    
    print(f"\n📊 Conv Layers:")
    for conv_info in structure['convs']:
        idx = conv_info['idx']
        in_ch = conv_info['in_channels']
        out_ch = conv_info['out_channels']
        is_after = conv_info['is_after_concat']
        category = "AFTER concat" if is_after else "BEFORE concat"
        pruning_method = "GAMMA" if is_after else "ACTIVATION"
        
        print(f"  Conv {idx}: {in_ch} → {out_ch} channels [{category}] → {pruning_method} pruning")
    
    print(f"\n📋 Pruning Strategy:")
    print(f"  Before concat (Activation pruning): {len(categories['before_concat'])} convs")
    for conv_info in categories['before_concat']:
        print(f"    - Conv {conv_info['idx']}: {conv_info['in_channels']} → {conv_info['out_channels']} channels")
    
    print(f"  After concat (Gamma pruning): {len(categories['after_concat'])} convs")
    for conv_info in categories['after_concat']:
        print(f"    - Conv {conv_info['idx']}: {conv_info['in_channels']} → {conv_info['out_channels']} channels")
    
    print(f"{'='*70}\n")


def find_following_bn(block: nn.Module, conv_in_block_idx: int) -> Optional[nn.BatchNorm2d]:
    """
    Find the BatchNorm layer following a specific Conv2d in a block.
    
    In C2f blocks, the structure is complex, so we use a heuristic:
    - Find the target conv by index
    - Look for BN layers with matching output channel count
    - Return the first matching BN (typically there's one per conv)
    
    Args:
        block: The block containing the conv
        conv_in_block_idx: Index of the Conv layer within the block
    
    Returns:
        BatchNorm2d layer if found, None otherwise
    """
    all_convs = get_all_conv2d_layers(block)
    if conv_in_block_idx >= len(all_convs):
        return None
    
    target_conv = all_convs[conv_in_block_idx]
    target_out_channels = target_conv.weight.shape[0]
    
    # In C2f blocks, we need to find BN by matching channel count
    # Since C2f has nested modules, we search all modules
    # and match by num_features == conv output channels
    matching_bns = []
    for module in block.modules():
        if isinstance(module, nn.BatchNorm2d):
            if module.num_features == target_out_channels:
                matching_bns.append(module)
    
    # If we find exactly one matching BN, return it
    # If multiple, return the first one (heuristic)
    # This works for most cases since each conv typically has one BN
    if len(matching_bns) > 0:
        return matching_bns[0]
    
    return None

