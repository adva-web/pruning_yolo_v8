#!/usr/bin/env python3
"""
Script to identify which blocks in YOLOv8 are regular Conv blocks (not C2f/Concat).
This helps identify which blocks can be safely pruned using the activation algorithm.
"""

import torch
import torch.nn as nn
from ultralytics import YOLO
from yolov8_utils import get_all_conv2d_layers
from c2f_utils import is_c2f_block

def identify_conv_blocks(model_path="data/best.pt"):
    """Identify all blocks and categorize them as Conv, C2f, or Concat."""
    
    print("=" * 80)
    print("YOLOV8 BLOCK IDENTIFICATION")
    print("=" * 80)
    
    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model
    
    print(f"\n📊 Total blocks in model: {len(detection_model)}\n")
    
    conv_blocks = []
    c2f_blocks = []
    concat_blocks = []
    other_blocks = []
    
    for block_idx in range(len(detection_model)):
        block = detection_model[block_idx]
        block_type = type(block).__name__
        block_convs = get_all_conv2d_layers(block)
        
        # Check if it's a C2f block
        is_c2f = False
        try:
            is_c2f = is_c2f_block(block)
        except Exception:
            pass
        
        # Check if it's a Concat block
        is_concat = 'Concat' in block_type
        
        # Categorize
        if is_c2f:
            c2f_blocks.append(block_idx)
            category = "C2f"
        elif is_concat:
            concat_blocks.append(block_idx)
            category = "Concat"
        elif len(block_convs) > 0 and not is_concat:
            # Regular Conv block (has Conv layers but not C2f or Concat)
            conv_blocks.append(block_idx)
            category = "Conv"
        else:
            other_blocks.append(block_idx)
            category = "Other"
        
        # Print block info
        conv_info = ""
        if len(block_convs) > 0:
            conv0 = block_convs[0]
            out_ch = conv0.weight.shape[0]
            in_ch = conv0.weight.shape[1]
            conv_info = f" | Conv 0: {in_ch}→{out_ch} channels"
        
        print(f"Block {block_idx:2d}: {block_type:25s} [{category:7s}] | {len(block_convs)} Conv layers{conv_info}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✅ Regular Conv blocks (can be pruned with activation algorithm):")
    print(f"   Blocks: {conv_blocks}")
    print(f"   Total: {len(conv_blocks)} blocks")
    
    print(f"\n⚠️  C2f blocks (complex structure, skip for now):")
    print(f"   Blocks: {c2f_blocks}")
    print(f"   Total: {len(c2f_blocks)} blocks")
    
    print(f"\n⚠️  Concat blocks (feature fusion, skip for now):")
    print(f"   Blocks: {concat_blocks}")
    print(f"   Total: {len(concat_blocks)} blocks")
    
    if other_blocks:
        print(f"\n❓ Other blocks:")
        print(f"   Blocks: {other_blocks}")
        print(f"   Total: {len(other_blocks)} blocks")
    
    # Detailed info for Conv blocks
    print(f"\n" + "=" * 80)
    print("DETAILED CONV BLOCK INFORMATION")
    print("=" * 80)
    
    for block_idx in conv_blocks:
        block = detection_model[block_idx]
        block_convs = get_all_conv2d_layers(block)
        
        print(f"\n🔷 Block {block_idx} (Regular Conv):")
        for conv_idx, conv in enumerate(block_convs):
            weight_shape = conv.weight.shape
            in_channels = weight_shape[1]
            out_channels = weight_shape[0]
            kernel_size = weight_shape[2:]
            
            print(f"   Conv {conv_idx}: {in_channels} → {out_channels} channels, kernel={kernel_size}")
    
    return conv_blocks, c2f_blocks, concat_blocks, other_blocks


if __name__ == "__main__":
    conv_blocks, c2f_blocks, concat_blocks, other_blocks = identify_conv_blocks()
    
    print(f"\n" + "=" * 80)
    print("RECOMMENDATION FOR PRUNING")
    print("=" * 80)
    print(f"\n✅ Safe to prune (Regular Conv blocks): {conv_blocks}")
    print(f"   These blocks have simple Conv structure and can use activation-based pruning.")
    print(f"\n💡 Current selection in run_4_methods_comparison.py: [1, 3, 5, 7]")
    print(f"   Available additional blocks: {[b for b in conv_blocks if b not in [1, 3, 5, 7]]}")

