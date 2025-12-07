#!/usr/bin/env python3
"""
Comprehensive comparison of 4 pruning methods on COCO dataset:
1. Activation with max weight (original method)
2. Activation with k-medoid (geometric center selection)
3. Activation with max gamma (BN gamma-based selection)
4. Pure gamma pruning (BN gamma magnitude only)

Prunes Conv 0 from blocks: 1, 3, 5, 7 (4 layers)
Final fine-tuning: 20 epochs for each method
Includes inference time, FLOPs, and sparsity measurement for original model and all methods

COCO Dataset:
- Train images: data/coco/train2017.zip
- Val images: data/coco/val2017.zip
- Annotations: data/coco/annotations_trainval2017.zip
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
import zipfile
import json
import cv2
import glob
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import thop for FLOPs calculation
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("⚠️  thop not available. FLOPs calculation will be skipped. Install with: pip install thop")

from pruning_yolo_v8_sequential_fix import (
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


def has_concatenation(block: nn.Module) -> bool:
    """Check if block contains concatenation operations (C2f or Concat blocks)."""
    # Check if it's a C2f block
    if is_c2f_block(block):
        return True
    
    # Check if it's a Concat block by class name
    block_type = block.__class__.__name__.lower()
    if 'concat' in block_type or 'cat' in block_type:
        return True
    
    # Check for Concat modules within the block
    for module in block.modules():
        module_type = module.__class__.__name__.lower()
        if 'concat' in module_type or 'cat' in module_type:
            return True
    
    return False


def select_layers_blocks_1_3_5_7(detection_model):
    """Select Conv 0 from blocks 1, 3, 5, 7 (4 layers)."""
    target_blocks = [1, 3, 5, 7]
    layers = []
    print(f"🔍 Checking blocks: {target_blocks} (model has {len(detection_model)} blocks)")
    for b in target_blocks:
        if b < len(detection_model):
            block = detection_model[b]
            block_type = block.__class__.__name__
            convs = get_all_conv2d_layers(block)
            print(f"   Block {b}: type={block_type}, convs={len(convs)}")
            if len(convs) >= 1:
                conv0 = convs[0]
                layers.append({
                    'block_idx': b,
                    'conv_in_block_idx': 0,
                    'name': f"Block {b}, Conv 0",
                    'num_channels': conv0.weight.shape[0]
                })
                print(f"      ✅ Added Block {b}, Conv 0 ({conv0.weight.shape[0]} channels)")
            else:
                print(f"      ⚠️  Skipped Block {b} (no Conv layers found)")
        else:
            print(f"   Block {b}: ⚠️  Block index out of range (model has {len(detection_model)} blocks)")
    
    if len(layers) < len(target_blocks):
        print(f"⚠️  Warning: Only found {len(layers)} eligible layers out of {len(target_blocks)} target blocks")
    
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
# COCO DATASET HANDLING
# ============================================================================
def extract_zip_if_needed(zip_path, extract_to):
    """Extract zip file if it exists and target directory doesn't exist."""
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"   ✅ {extract_to} already exists, skipping extraction")
        # Check if images are in a subdirectory
        for item in os.listdir(extract_to):
            item_path = os.path.join(extract_to, item)
            if os.path.isdir(item_path) and item in ['train2017', 'val2017', 'images']:
                print(f"   📁 Found subdirectory: {item_path}")
        return True
    
    if not os.path.exists(zip_path):
        print(f"   ❌ Zip file not found: {zip_path}")
        return False
    
    print(f"   📦 Extracting {zip_path} to {extract_to}...")
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"   ✅ Extraction complete")
        
        # Check if images are in a nested subdirectory
        for item in os.listdir(extract_to):
            item_path = os.path.join(extract_to, item)
            if os.path.isdir(item_path):
                # Check if this subdirectory contains images
                subdir_files = os.listdir(item_path)
                if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in subdir_files):
                    print(f"   📁 Images are in subdirectory: {item_path}")
        return True
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        return False


def convert_coco_to_yolo_format(coco_json_path, images_dir, labels_dir, split='train'):
    """
    Convert COCO JSON annotations to YOLO format.
    
    Args:
        coco_json_path: Path to COCO JSON annotation file
        images_dir: Directory containing images
        labels_dir: Directory to save YOLO format labels
        split: 'train' or 'val'
    """
    print(f"   🔄 Converting COCO annotations to YOLO format for {split}...")
    
    if not os.path.exists(coco_json_path):
        print(f"   ❌ COCO JSON file not found: {coco_json_path}")
        return False
    
    if not os.path.exists(images_dir):
        print(f"   ❌ Images directory not found: {images_dir}")
        return False
    
    os.makedirs(labels_dir, exist_ok=True)
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    print(f"   📊 Found {len(coco_data['images'])} images in JSON")
    print(f"   📊 Found {len(coco_data['annotations'])} annotations")
    
    # Create mappings
    images_dict = {img['id']: img for img in coco_data['images']}
    categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Check what images actually exist in the directory
    existing_images = set()
    if os.path.exists(images_dir):
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                existing_images.add(filename)
    
    print(f"   📊 Found {len(existing_images)} image files in directory")
    
    # Convert each image's annotations
    converted_count = 0
    skipped_count = 0
    labels_created = 0
    images_with_annotations = set()  # Track which images got labels from JSON
    
    for image_id, image_info in images_dict.items():
        image_filename = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']
        
        # Check if image exists - try both exact match and case-insensitive
        image_path = os.path.join(images_dir, image_filename)
        if not os.path.exists(image_path):
            # Try case-insensitive match
            found = False
            for existing_file in existing_images:
                if existing_file.lower() == image_filename.lower():
                    image_filename = existing_file
                    image_path = os.path.join(images_dir, image_filename)
                    found = True
                    break
            
            if not found:
                skipped_count += 1
                if skipped_count <= 5:  # Show first 5 skipped files
                    print(f"   ⚠️  Image not found: {image_filename}")
                continue
        
        # Create label file
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        # Write label file
        label_lines = []
        if image_id in annotations_by_image:
            for ann in annotations_by_image[image_id]:
                # COCO bbox format: [x_min, y_min, width, height]
                x_min, y_min, bbox_width, bbox_height = ann['bbox']
                category_id = ann['category_id']
                
                # Convert to YOLO format: class_id x_center y_center width height (normalized)
                x_center = (x_min + bbox_width / 2) / image_width
                y_center = (y_min + bbox_height / 2) / image_height
                width = bbox_width / image_width
                height = bbox_height / image_height
                
                # COCO category_id is 1-indexed, YOLO uses 0-indexed
                class_id = category_id - 1
                
                # Clamp values to [0, 1] range
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Write label file (even if empty, YOLO needs the file to exist)
        with open(label_path, 'w') as f:
            f.writelines(label_lines)
        
        if len(label_lines) > 0:
            labels_created += 1
            images_with_annotations.add(image_filename)
        
        converted_count += 1
    
    # Create empty label files for images that exist but aren't in COCO JSON
    # This ensures all images have corresponding label files (even if empty)
    empty_labels_created = 0
    for image_filename in existing_images:
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        # Only create if it doesn't exist yet
        if not os.path.exists(label_path):
            with open(label_path, 'w') as f:
                pass  # Create empty file
            empty_labels_created += 1
    
    if skipped_count > 5:
        print(f"   ⚠️  ... and {skipped_count - 5} more images not found in JSON")
    
    print(f"   ✅ Converted {converted_count} images from JSON (skipped {skipped_count} not found)")
    print(f"   ✅ Created {labels_created} label files with annotations")
    if empty_labels_created > 0:
        print(f"   ✅ Created {empty_labels_created} empty label files for images without annotations")
    
    # Verify some label files exist
    if os.path.exists(labels_dir):
        sample_labels = [f for f in os.listdir(labels_dir) if f.endswith('.txt')][:5]
        if sample_labels:
            sample_path = os.path.join(labels_dir, sample_labels[0])
            if os.path.exists(sample_path):
                with open(sample_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        lines = content.split('\n')
                        print(f"   ✅ Sample label file '{sample_labels[0]}' has {len(lines)} annotations")
                        print(f"      First annotation: {lines[0] if lines else 'empty'}")
                    else:
                        print(f"   ⚠️  Sample label file '{sample_labels[0]}' is empty")
    
    return converted_count > 0


def setup_coco_dataset():
    """Extract COCO zips and convert annotations to YOLO format."""
    print(f"\n{'='*70}")
    print("SETTING UP COCO DATASET")
    print(f"{'='*70}")
    
    base_dir = "data/coco"
    train_zip = os.path.join(base_dir, "train2017.zip")
    val_zip = os.path.join(base_dir, "val2017.zip")
    annotations_zip = os.path.join(base_dir, "annotations_trainval2017.zip")
    
    # Extract images
    train_images_dir = os.path.join(base_dir, "train2017")
    val_images_dir = os.path.join(base_dir, "val2017")
    
    if not extract_zip_if_needed(train_zip, train_images_dir):
        return None, None, None
    
    # Check if images are in a nested subdirectory
    if os.path.exists(train_images_dir):
        for item in os.listdir(train_images_dir):
            item_path = os.path.join(train_images_dir, item)
            if os.path.isdir(item_path) and item in ['train2017', 'images']:
                # Images are in a subdirectory, update path
                train_images_dir = item_path
                print(f"   📁 Using nested train images directory: {train_images_dir}")
                break
    
    if not extract_zip_if_needed(val_zip, val_images_dir):
        return None, None, None
    
    # Check if images are in a nested subdirectory
    if os.path.exists(val_images_dir):
        for item in os.listdir(val_images_dir):
            item_path = os.path.join(val_images_dir, item)
            if os.path.isdir(item_path) and item in ['val2017', 'images']:
                # Images are in a subdirectory, update path
                val_images_dir = item_path
                print(f"   📁 Using nested val images directory: {val_images_dir}")
                break
    
    # Extract annotations
    annotations_dir = os.path.join(base_dir, "annotations")
    if not extract_zip_if_needed(annotations_zip, annotations_dir):
        return None, None, None
    
    # Try different possible paths for the JSON files
    # Based on user's file structure, annotations are directly in annotations/ folder
    possible_train_json_paths = [
        os.path.join(annotations_dir, "instances_train2017.json"),  # Most likely
        os.path.join(annotations_dir, "annotations", "instances_train2017.json"),
        os.path.join(base_dir, "annotations", "instances_train2017.json")
    ]
    
    possible_val_json_paths = [
        os.path.join(annotations_dir, "instances_val2017.json"),  # Most likely
        os.path.join(annotations_dir, "annotations", "instances_val2017.json"),
        os.path.join(base_dir, "annotations", "instances_val2017.json")
    ]
    
    train_json = None
    for path in possible_train_json_paths:
        if os.path.exists(path):
            train_json = path
            print(f"   ✅ Found train JSON: {train_json}")
            break
    
    val_json = None
    for path in possible_val_json_paths:
        if os.path.exists(path):
            val_json = path
            print(f"   ✅ Found val JSON: {val_json}")
            break
    
    if train_json is None:
        print(f"   ❌ Could not find train JSON file. Tried:")
        for path in possible_train_json_paths:
            print(f"      - {path}")
        return None, None, None
    
    if val_json is None:
        print(f"   ❌ Could not find val JSON file. Tried:")
        for path in possible_val_json_paths:
            print(f"      - {path}")
        return None, None, None
    
    # Determine actual image directories (handle nested structure)
    actual_train_img_dir = train_images_dir
    actual_val_img_dir = val_images_dir
    
    # Check for nested structure in train
    if os.path.exists(train_images_dir):
        for item in os.listdir(train_images_dir):
            item_path = os.path.join(train_images_dir, item)
            if os.path.isdir(item_path):
                files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(files) > 0:
                    actual_train_img_dir = item_path
                    print(f"   📁 Found nested train images in: {actual_train_img_dir}")
                    break
    
    # Check for nested structure in val
    if os.path.exists(val_images_dir):
        for item in os.listdir(val_images_dir):
            item_path = os.path.join(val_images_dir, item)
            if os.path.isdir(item_path):
                files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(files) > 0:
                    actual_val_img_dir = item_path
                    print(f"   📁 Found nested val images in: {actual_val_img_dir}")
                    break
    
    # YOLO format: labels must be in the SAME directory level as images
    # If images are in train2017/train2017/, labels must be in train2017/train2017/labels/
    # This is where YOLO will look for them
    train_labels_dir = os.path.join(actual_train_img_dir, "labels")
    val_labels_dir = os.path.join(actual_val_img_dir, "labels")
    
    # Also save to data/coco/labels/ for organization (backup location)
    train_labels_backup = os.path.join(base_dir, "labels", "train2017")
    val_labels_backup = os.path.join(base_dir, "labels", "val2017")
    
    # Ensure labels directories exist
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    os.makedirs(train_labels_backup, exist_ok=True)
    os.makedirs(val_labels_backup, exist_ok=True)
    
    print(f"   📝 Train labels will be saved to: {train_labels_dir} (YOLO location)")
    print(f"   📝 Val labels will be saved to: {val_labels_dir} (YOLO location)")
    
    # Convert annotations and save labels where YOLO expects them
    if not convert_coco_to_yolo_format(train_json, actual_train_img_dir, train_labels_dir, 'train'):
        return None, None, None
    
    if not convert_coco_to_yolo_format(val_json, actual_val_img_dir, val_labels_dir, 'val'):
        return None, None, None
    
    # Also copy to backup location for organization
    import shutil
    if os.path.exists(train_labels_dir) and os.path.exists(train_labels_backup):
        # Copy labels to backup location
        for label_file in os.listdir(train_labels_dir):
            if label_file.endswith('.txt'):
                src = os.path.join(train_labels_dir, label_file)
                dst = os.path.join(train_labels_backup, label_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        print(f"   ✅ Also saved train labels to backup: {train_labels_backup}")
    
    if os.path.exists(val_labels_dir) and os.path.exists(val_labels_backup):
        # Copy labels to backup location
        for label_file in os.listdir(val_labels_dir):
            if label_file.endswith('.txt'):
                src = os.path.join(val_labels_dir, label_file)
                dst = os.path.join(val_labels_backup, label_file)
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
        print(f"   ✅ Also saved val labels to backup: {val_labels_backup}")
    
    # Create COCO YAML file
    coco_yaml_path = os.path.join(base_dir, "coco.yaml")
    coco_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    ]
    
    # Determine actual image directories (handle nested structure)
    actual_train_dir = train_images_dir
    actual_val_dir = val_images_dir
    
    # Check for nested structure (e.g., train2017/train2017/)
    if os.path.exists(train_images_dir):
        for item in os.listdir(train_images_dir):
            item_path = os.path.join(train_images_dir, item)
            if os.path.isdir(item_path) and item != "labels":
                files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(files) > 0:
                    actual_train_dir = item_path
                    print(f"   📁 Using nested train images: {actual_train_dir}")
                    break
    
    if os.path.exists(val_images_dir):
        for item in os.listdir(val_images_dir):
            item_path = os.path.join(val_images_dir, item)
            if os.path.isdir(item_path) and item != "labels":
                files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(files) > 0:
                    actual_val_dir = item_path
                    print(f"   📁 Using nested val images: {actual_val_dir}")
                    break
    
    # Create YAML file for YOLO
    # YOLO format: labels must be in the same directory level as images
    # If images are in train2017/train2017/, labels must be in train2017/train2017/labels/
    # YAML should point to the actual image directory (train2017/train2017/)
    base_abs = os.path.abspath(base_dir)
    train_path_abs = os.path.abspath(actual_train_img_dir)  # data/coco/train2017/train2017/
    val_path_abs = os.path.abspath(actual_val_img_dir)      # data/coco/val2017/val2017/
    
    print(f"\n   📄 Final dataset structure:")
    print(f"      Train images: {train_path_abs}")
    print(f"      Val images:   {val_path_abs}")
    print(f"      Train labels: {os.path.abspath(train_labels_dir)}")
    print(f"      Val labels:   {os.path.abspath(val_labels_dir)}")
    
    # YOLO format: When YAML points to a directory, YOLO looks for:
    # - Images: in that directory (or subdirectories)
    # - Labels: in a 'labels' subdirectory at the SAME level
    # 
    # If YAML points to train2017/train2017/, YOLO expects:
    # - Images: train2017/train2017/*.jpg
    # - Labels: train2017/train2017/labels/*.txt
    #
    # Use ABSOLUTE paths directly - this is more reliable than using 'path' field
    # YOLO will look for labels in <train_path>/labels/ and <val_path>/labels/
    
    # CRITICAL: When using 'path' field, train/val must be RELATIVE to path
    # YOLO resolves: <path>/<train> and <path>/<val>
    # For labels, YOLO looks in: <path>/<train>/labels/ and <path>/<val>/labels/
    # 
    # Since we have nested structure (train2017/train2017/), we need to use relative paths
    train_relative = os.path.relpath(train_path_abs, base_abs)
    val_relative = os.path.relpath(val_path_abs, base_abs)
    
    coco_yaml_content = {
        'path': base_abs,  # Base directory
        'train': train_relative,  # Relative to path: train2017/train2017
        'val': val_relative,       # Relative to path: val2017/val2017
        'names': coco_names,
        'nc': len(coco_names)
    }
    
    print(f"   📋 Path resolution:")
    print(f"      Base path: {base_abs}")
    print(f"      Train relative: {train_relative} → {os.path.join(base_abs, train_relative)}")
    print(f"      Val relative: {val_relative} → {os.path.join(base_abs, val_relative)}")
    print(f"      Train labels: {os.path.join(base_abs, train_relative, 'labels')}")
    print(f"      Val labels: {os.path.join(base_abs, val_relative, 'labels')}")
    
    with open(coco_yaml_path, 'w') as f:
        yaml.dump(coco_yaml_content, f, default_flow_style=False)
    
    print(f"   ✅ Created COCO YAML file: {coco_yaml_path}")
    print(f"\n   📋 YAML file content:")
    print(f"   {'-'*60}")
    with open(coco_yaml_path, 'r') as f:
        yaml_content = f.read()
        # Print with indentation for readability
        for line in yaml_content.split('\n'):
            print(f"   {line}")
    print(f"   {'-'*60}")
    print(f"\n   📋 YAML structure summary:")
    print(f"      path: {base_abs}")
    print(f"      train: {coco_yaml_content['train']} (resolves to: {os.path.join(base_abs, train_relative)})")
    print(f"      val: {coco_yaml_content['val']} (resolves to: {os.path.join(base_abs, val_relative)})")
    print(f"      nc: {coco_yaml_content['nc']} classes")
    print(f"   💡 YOLO will look for labels in: {os.path.join(base_abs, train_relative, 'labels')}")
    print(f"   💡 YOLO will look for labels in: {os.path.join(base_abs, val_relative, 'labels')}")
    
    # Verify labels were created in YOLO location
    train_label_count = len([f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]) if os.path.exists(train_labels_dir) else 0
    val_label_count = len([f for f in os.listdir(val_labels_dir) if f.endswith('.txt')]) if os.path.exists(val_labels_dir) else 0
    print(f"   ✅ Verified: {train_label_count} train label files, {val_label_count} val label files")
    print(f"   ✅ Labels are in YOLO-expected location (same level as images)")
    
    # Clear YOLO cache files so it will re-scan and find labels
    # YOLO creates cache files in the image directory
    cache_files_to_remove = []
    
    # Check parent directories too (YOLO might create cache there)
    for img_dir in [train_path_abs, val_path_abs, train_images_dir, val_images_dir]:
        if img_dir:
            img_dir_abs = os.path.abspath(img_dir) if not os.path.isabs(img_dir) else img_dir
            cache_files_to_remove.extend([
                os.path.join(img_dir_abs, "cache"),
                os.path.join(img_dir_abs, f"{os.path.basename(img_dir_abs)}.cache"),
            ])
            # Also check parent directory
            parent_dir = os.path.dirname(img_dir_abs)
            if parent_dir:
                cache_files_to_remove.extend([
                    os.path.join(parent_dir, "cache"),
                    os.path.join(parent_dir, f"{os.path.basename(img_dir_abs)}.cache"),
                ])
    
    # Remove unique cache files
    removed_count = 0
    for cache_file in set(cache_files_to_remove):
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
                removed_count += 1
                print(f"   🗑️  Removed cache file: {cache_file}")
            except Exception as e:
                print(f"   ⚠️  Could not remove cache file {cache_file}: {e}")
    
    if removed_count == 0:
        print(f"   ℹ️  No cache files found to remove (will be created on first run)")
    
    # Verify label files are accessible
    print(f"\n   🔍 Verifying label files are accessible:")
    if os.path.exists(train_labels_dir):
        train_label_files = [f for f in os.listdir(train_labels_dir) if f.endswith('.txt')]
        print(f"      Train labels: {len(train_label_files)} files in {train_labels_dir}")
        if len(train_label_files) > 0:
            # Check a sample file
            sample = os.path.join(train_labels_dir, train_label_files[0])
            if os.path.exists(sample):
                with open(sample, 'r') as f:
                    lines = [l for l in f if l.strip()]
                    print(f"      Sample file '{train_label_files[0]}' has {len(lines)} annotations")
    
    if os.path.exists(val_labels_dir):
        val_label_files = [f for f in os.listdir(val_labels_dir) if f.endswith('.txt')]
        print(f"      Val labels: {len(val_label_files)} files in {val_labels_dir}")
        if len(val_label_files) > 0:
            # Check a sample file
            sample = os.path.join(val_labels_dir, val_label_files[0])
            if os.path.exists(sample):
                with open(sample, 'r') as f:
                    lines = [l for l in f if l.strip()]
                    print(f"      Sample file '{val_label_files[0]}' has {len(lines)} annotations")
    
    # Print YAML content for debugging
    print(f"\n   📋 YAML file content:")
    with open(coco_yaml_path, 'r') as f:
        print(f"   {f.read()}")
    
    return coco_yaml_path, train_images_dir, val_images_dir


def load_coco_samples(images_dir, labels_dir, max_samples=None):
    """Load COCO samples in YOLO format."""
    samples = []
    image_paths = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    
    if max_samples is not None:
        image_paths = image_paths[:max_samples]
    
    print(f"   📥 Loading {len(image_paths)} samples from {images_dir}...")
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base + ".txt")
        
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        labels.append({
                            "class_id": class_id,
                            "x_center": float(parts[1]),
                            "y_center": float(parts[2]),
                            "width": float(parts[3]),
                            "height": float(parts[4])
                        })
        
        samples.append({
            "image": img,
            "label": labels,
            "image_path": img_path,
            "label_path": label_path
        })
    
    return samples


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
        # Use BOTH matched and unmatched objects for activation extraction
        # Even if predictions don't match well, we can still extract activations from GT locations
        all_train_objs = train_matched_objs + train_unmatched_objs
        print(f"   📊 Using {len(train_matched_objs)} matched + {len(train_unmatched_objs)} unmatched = {len(all_train_objs)} total objects for activations")
        train_activations = aggregate_activations_from_matches(all_train_objs, classes)
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
        # Use BOTH matched and unmatched objects for activation extraction
        # Even if predictions don't match well, we can still extract activations from GT locations
        all_train_objs = train_matched_objs + train_unmatched_objs
        print(f"   📊 Using {len(train_matched_objs)} matched + {len(train_unmatched_objs)} unmatched = {len(all_train_objs)} total objects for activations")
        train_activations = aggregate_activations_from_matches(all_train_objs, classes)
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


def calculate_model_flops(model, input_size=(640, 640)):
    """
    Calculate FLOPs (Floating Point Operations) for a YOLO model.
    
    Args:
        model: YOLO model
        input_size: Input image size (height, width)
    
    Returns:
        FLOPs in GFLOPs (Giga FLOPs), or None if calculation fails
    """
    if not THOP_AVAILABLE:
        return None
    
    try:
        torch_model = model.model
        device = next(torch_model.parameters()).device
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
        
        # Calculate FLOPs
        flops, params = profile(torch_model, inputs=(dummy_input,), verbose=False)
        
        # Convert to GFLOPs (FLOPs are already in operations, multiply by 2 for MAC operations)
        gflops = flops / 1e9 * 2
        
        return gflops
    except Exception as e:
        print(f"⚠️  FLOPs calculation failed: {e}")
        return None


def calculate_model_sparsity(model):
    """
    Calculate sparsity percentage (ratio of zero weights) for a model.
    
    Args:
        model: YOLO model
    
    Returns:
        Dictionary with:
        - 'sparsity': Sparsity percentage (0-100)
        - 'total_params': Total number of parameters
        - 'zero_params': Number of zero parameters
        - 'non_zero_params': Number of non-zero parameters
    """
    try:
        # YOLO model structure: model.model is the torch model
        torch_model = model.model
        total_params = 0
        zero_params = 0
        
        # Count all parameters in the model
        for name, param in torch_model.named_parameters():
            param_flat = param.data.flatten()
            param_count = param_flat.numel()
            total_params += param_count
            zero_count = (param_flat == 0).sum().item()
            zero_params += zero_count
        
        # Also count buffers (like BatchNorm running stats) if needed
        for name, buffer in torch_model.named_buffers():
            if buffer.requires_grad:  # Only count learnable buffers
                buffer_flat = buffer.data.flatten()
                buffer_count = buffer_flat.numel()
                total_params += buffer_count
                zero_count = (buffer_flat == 0).sum().item()
                zero_params += zero_count
        
        non_zero_params = total_params - zero_params
        sparsity = (zero_params / total_params * 100) if total_params > 0 else 0.0
        
        return {
            'sparsity': sparsity,
            'total_params': total_params,
            'zero_params': zero_params,
            'non_zero_params': non_zero_params
        }
    except Exception as e:
        print(f"⚠️  Sparsity calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return {
            'sparsity': None,
            'total_params': None,
            'zero_params': None,
            'non_zero_params': None
        }


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================
def validate_labels_exist(images_dir, labels_dir, split_name='dataset'):
    """
    Validate that labels exist and match images.
    
    Returns:
        (is_valid, error_message) - (True, None) if valid, (False, error_msg) if invalid
    """
    if not os.path.exists(images_dir):
        return False, f"Images directory does not exist: {images_dir}"
    
    if not os.path.exists(labels_dir):
        return False, f"Labels directory does not exist: {labels_dir}"
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(glob.glob(os.path.join(images_dir, f"*{ext}")))
    
    if len(image_files) == 0:
        return False, f"No image files found in: {images_dir}"
    
    # Get all label files
    label_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    
    if len(label_files) == 0:
        return False, f"No label files found in: {labels_dir}"
    
    # Check that labels match images
    image_basenames = {os.path.splitext(os.path.basename(img))[0] for img in image_files}
    label_basenames = {os.path.splitext(os.path.basename(lbl))[0] for lbl in label_files}
    
    # Check how many images have corresponding labels
    images_with_labels = image_basenames & label_basenames
    images_without_labels = image_basenames - label_basenames
    
    if len(images_with_labels) == 0:
        return False, f"No matching labels found for images in {split_name}. Images: {len(image_basenames)}, Labels: {len(label_basenames)}"
    
    # Check that label files have content
    empty_labels = 0
    labels_with_content = 0
    for label_file in label_files[:100]:  # Check first 100 labels
        try:
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if content:
                    labels_with_content += 1
                else:
                    empty_labels += 1
        except Exception as e:
            return False, f"Error reading label file {label_file}: {e}"
    
    if labels_with_content == 0:
        return False, f"All checked label files are empty in {split_name}"
    
    # Summary
    match_ratio = len(images_with_labels) / len(image_basenames) * 100
    
    # For COCO dataset, it's normal to have many background images without labels
    # COCO typically has 40-50% labeled images (the rest are background/negative samples)
    # If we have at least 30% with labels, that's acceptable for COCO
    # Also, if we have a reasonable absolute number of labels (e.g., > 1000), that's also OK
    min_match_ratio = 30.0
    min_absolute_labels = 1000
    
    # Check both percentage and absolute count
    has_enough_labels = (match_ratio >= min_match_ratio) or (len(images_with_labels) >= min_absolute_labels)
    
    if not has_enough_labels:
        return False, f"Only {match_ratio:.1f}% of images have labels ({len(images_with_labels)}/{len(image_basenames)}). " \
                     f"Minimum required: {min_match_ratio}% OR at least {min_absolute_labels} labeled images. " \
                     f"For COCO, this is normal - many images are background without annotations."
    
    print(f"   ✅ {split_name}: {len(images_with_labels)}/{len(image_basenames)} images have labels ({match_ratio:.1f}%)")
    if len(images_without_labels) > 0:
        print(f"   ℹ️  {len(images_without_labels)} images without labels (normal for COCO - these are background images)")
    
    return True, None


def validate_coco_dataset_setup(coco_yaml_path, train_images_dir, val_images_dir):
    """
    Validate that COCO dataset is set up correctly before starting experiment.
    Also test that YOLO can actually find and load the labels.
    
    Returns:
        (is_valid, error_message) - (True, None) if valid, (False, error_msg) if invalid
    """
    print(f"\n{'='*70}")
    print("VALIDATING COCO DATASET SETUP")
    print(f"{'='*70}")
    
    if not os.path.exists(coco_yaml_path):
        return False, f"COCO YAML file not found: {coco_yaml_path}"
    
    # Print YAML content for debugging
    print(f"\n   📋 YAML file content:")
    print(f"   {'-'*60}")
    try:
        with open(coco_yaml_path, 'r') as f:
            yaml_content = f.read()
            for line in yaml_content.split('\n'):
                print(f"   {line}")
        print(f"   {'-'*60}")
    except Exception as e:
        print(f"   ⚠️  Could not read YAML file: {e}")
    
    # Read YAML to get paths
    try:
        with open(coco_yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        train_path = yaml_data.get('train', '')
        val_path = yaml_data.get('val', '')
        yaml_path_base = yaml_data.get('path', '')
        
        if not train_path or not val_path:
            return False, f"YAML file missing 'train' or 'val' paths"
        
        # Convert to absolute paths
        # If 'path' field exists, resolve relative to it; otherwise use YAML file location
        if yaml_path_base:
            # YAML has 'path' field - resolve relative to it
            if not os.path.isabs(train_path):
                train_path = os.path.join(yaml_path_base, train_path)
            if not os.path.isabs(val_path):
                val_path = os.path.join(yaml_path_base, val_path)
        else:
            # No 'path' field - paths should be absolute or relative to YAML file
            if not os.path.isabs(train_path):
                train_path = os.path.abspath(os.path.join(os.path.dirname(coco_yaml_path), train_path))
            if not os.path.isabs(val_path):
                val_path = os.path.abspath(os.path.join(os.path.dirname(coco_yaml_path), val_path))
        
        # Ensure paths are absolute
        train_path = os.path.abspath(train_path)
        val_path = os.path.abspath(val_path)
        
    except Exception as e:
        return False, f"Error reading YAML file: {e}"
    
    # Find actual image directories (handle nested structure)
    actual_train_img_dir = train_path
    actual_val_img_dir = val_path
    
    if os.path.exists(train_path):
        for item in os.listdir(train_path):
            item_path = os.path.join(train_path, item)
            if os.path.isdir(item_path) and item != "labels":
                files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(files) > 0:
                    actual_train_img_dir = item_path
                    break
    
    if os.path.exists(val_path):
        for item in os.listdir(val_path):
            item_path = os.path.join(val_path, item)
            if os.path.isdir(item_path) and item != "labels":
                files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(files) > 0:
                    actual_val_img_dir = item_path
                    break
    
    # Labels location: Check if standard YOLO format (labels/ separate from images/)
    # Standard format: images/train/ and labels/train/ are siblings under data/coco/
    # Old format: images/train/ and images/train/labels/ (labels inside image dir)
    
    # Find the base directory (data/coco/)
    # If path is like /path/to/data/coco/images/train, base is /path/to/data/coco
    # If path is like /path/to/data/coco/train2017/train2017, base is /path/to/data/coco
    if 'images' in actual_train_img_dir:
        # Standard format: go up from images/train to get base (data/coco)
        # actual_train_img_dir = /path/to/data/coco/images/train
        # dirname once = /path/to/data/coco/images
        # dirname twice = /path/to/data/coco
        base_dir = os.path.dirname(os.path.dirname(actual_train_img_dir))
    elif 'train2017' in actual_train_img_dir or 'val2017' in actual_train_img_dir:
        # Old format: go up to find data/coco
        base_dir = actual_train_img_dir
        while 'coco' not in os.path.basename(base_dir) and len(base_dir) > 1:
            base_dir = os.path.dirname(base_dir)
        base_dir = os.path.dirname(base_dir) if 'coco' in os.path.basename(base_dir) else base_dir
    else:
        # Fallback: try to find coco in path
        base_dir = actual_train_img_dir
        parts = actual_train_img_dir.split(os.sep)
        if 'coco' in parts:
            coco_idx = parts.index('coco')
            base_dir = os.sep.join(parts[:coco_idx + 1])
        else:
            # Last resort: go up two levels
            base_dir = os.path.dirname(os.path.dirname(actual_train_img_dir))
    
    # Try standard YOLO format first (labels/train, labels/val) - siblings of images/
    train_labels_standard = os.path.join(base_dir, "labels", "train")
    val_labels_standard = os.path.join(base_dir, "labels", "val")
    
    # Try old format (images/train/labels, images/val/labels)
    train_labels_old = os.path.join(actual_train_img_dir, "labels")
    val_labels_old = os.path.join(actual_val_img_dir, "labels")
    
    # Use whichever exists
    if os.path.exists(train_labels_standard) and os.path.exists(val_labels_standard):
        train_labels_dir = train_labels_standard
        val_labels_dir = val_labels_standard
        print(f"   📁 Using standard YOLO format labels (labels/train, labels/val)")
    elif os.path.exists(train_labels_old) and os.path.exists(val_labels_old):
        train_labels_dir = train_labels_old
        val_labels_dir = val_labels_old
        print(f"   📁 Using old format labels (inside image directories)")
    else:
        # Default to standard format and create if needed
        train_labels_dir = train_labels_standard
        val_labels_dir = val_labels_standard
        print(f"   📁 Labels directory not found at: {train_labels_dir}")
        print(f"   📁 Will check if labels exist in standard location: {train_labels_dir}")
        print(f"   📁 Base directory determined as: {base_dir}")
    
    print(f"   📁 Train images: {actual_train_img_dir}")
    print(f"   📁 Train labels: {train_labels_dir}")
    print(f"   📁 Val images:   {actual_val_img_dir}")
    print(f"   📁 Val labels:   {val_labels_dir}")
    
    # Validate train labels
    # For COCO dataset, it's normal to have many background images without labels
    # The setup script creates labels only for images that have annotations in the JSON
    is_valid, error_msg = validate_labels_exist(actual_train_img_dir, train_labels_dir, 'Train')
    if not is_valid:
        # Check if this is a COCO dataset (has many images, typical COCO structure)
        # If we have a reasonable number of labels (>1000), it's probably OK
        train_label_files = glob.glob(os.path.join(train_labels_dir, "*.txt"))
        if len(train_label_files) >= 1000:
            print(f"   ⚠️  Train validation warning (but continuing): {error_msg}")
            print(f"   ℹ️  However, found {len(train_label_files)} label files, which is sufficient for COCO")
        else:
            return False, f"Train dataset validation failed: {error_msg}"
    
    # Validate val labels
    is_valid, error_msg = validate_labels_exist(actual_val_img_dir, val_labels_dir, 'Val')
    if not is_valid:
        # Check if we have a reasonable number of labels
        val_label_files = glob.glob(os.path.join(val_labels_dir, "*.txt"))
        if len(val_label_files) >= 100:
            print(f"   ⚠️  Val validation warning (but continuing): {error_msg}")
            print(f"   ℹ️  However, found {len(val_label_files)} label files, which is sufficient for COCO")
        else:
            return False, f"Val dataset validation failed: {error_msg}"
    
    # CRITICAL: Verify label files match image files exactly
    print(f"\n   🔍 Verifying label-image filename matching...")
    val_image_files = {os.path.splitext(f)[0] for f in os.listdir(actual_val_img_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))}
    val_label_files = {os.path.splitext(f)[0] for f in os.listdir(val_labels_dir) 
                       if f.endswith('.txt')}
    
    matching = val_image_files & val_label_files
    images_without_labels = val_image_files - val_label_files
    labels_without_images = val_label_files - val_image_files
    
    print(f"      Images: {len(val_image_files)}, Labels: {len(val_label_files)}, Matching: {len(matching)}")
    if images_without_labels:
        print(f"      ⚠️  {len(images_without_labels)} images without labels (showing first 5):")
        for img in list(images_without_labels)[:5]:
            print(f"         - {img}")
    if labels_without_images:
        print(f"      ⚠️  {len(labels_without_images)} labels without images (showing first 5):")
        for lbl in list(labels_without_images)[:5]:
            print(f"         - {lbl}")
    
    if len(matching) == 0:
        return False, f"❌ CRITICAL: No matching label files found for images! " \
                     f"Images: {len(val_image_files)}, Labels: {len(val_label_files)}, Matching: 0. " \
                     f"This will NOT work. Check label conversion."
    
    # For COCO, it's normal to have many background images without labels
    # Check if we have a reasonable number of labels (at least 30% or 1000+ labels)
    match_ratio = len(matching) / len(val_image_files) * 100 if len(val_image_files) > 0 else 0
    min_match_ratio = 30.0
    min_absolute_labels = 1000
    
    if match_ratio < min_match_ratio and len(matching) < min_absolute_labels:
        return False, f"❌ CRITICAL: Only {len(matching)}/{len(val_image_files)} images have matching labels " \
                     f"({match_ratio:.1f}%). " \
                     f"Minimum required: {min_match_ratio}% OR at least {min_absolute_labels} labeled images. " \
                     f"For COCO, many images are background without annotations, but we need enough labeled images to train."
    
    # Check that label files are non-empty and have valid format
    print(f"   🔍 Checking label file contents...")
    empty_labels = 0
    invalid_labels = 0
    valid_labels = 0
    sample_checked = 0
    sample_files = []
    
    for label_file in sorted(os.listdir(val_labels_dir)):
        if label_file.endswith('.txt'):
            label_path = os.path.join(val_labels_dir, label_file)
            sample_files.append(label_file)
            
            if os.path.getsize(label_path) == 0:
                empty_labels += 1
            else:
                # Check if file has valid YOLO format (at least one line with 5 numbers)
                try:
                    with open(label_path, 'r') as f:
                        lines = [l.strip() for l in f if l.strip()]
                        if len(lines) > 0:
                            # Check first line format: class_id x_center y_center width height
                            parts = lines[0].split()
                            if len(parts) >= 5:
                                # Try to parse as numbers
                                try:
                                    class_id = int(parts[0])
                                    x_center = float(parts[1])
                                    y_center = float(parts[2])
                                    width = float(parts[3])
                                    height = float(parts[4])
                                    # Check if values are in valid range (0-1 for normalized)
                                    if 0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 <= width <= 1 and 0 <= height <= 1:
                                        valid_labels += 1
                                    else:
                                        invalid_labels += 1
                                except ValueError:
                                    invalid_labels += 1
                            else:
                                invalid_labels += 1
                        else:
                            empty_labels += 1
                except Exception as e:
                    invalid_labels += 1
            
            sample_checked += 1
            if sample_checked >= 100:  # Check first 100
                break
    
    print(f"      Checked {sample_checked} label files:")
    print(f"         ✅ Valid: {valid_labels}")
    print(f"         ⚠️  Empty: {empty_labels}")
    print(f"         ❌ Invalid format: {invalid_labels}")
    
    if empty_labels + invalid_labels > sample_checked * 0.1:  # More than 10% problematic
        print(f"      ⚠️  WARNING: {empty_labels + invalid_labels}/{sample_checked} label files have issues!")
        if sample_files:
            print(f"      Sample files checked: {sample_files[:5]}")
    
    # Test that YOLO can actually find labels by doing a quick validation
    print(f"\n   🧪 Testing YOLO label detection with actual model...")
    try:
        # Clear cache first - be very thorough
        cache_paths = [
            # Cache in image directory
            os.path.join(actual_val_img_dir, f"{os.path.basename(actual_val_img_dir)}.cache"),
            os.path.join(actual_val_img_dir, ".cache"),
            # Cache in parent directory
            os.path.join(os.path.dirname(actual_val_img_dir), f"{os.path.basename(actual_val_img_dir)}.cache"),
            os.path.join(os.path.dirname(actual_val_img_dir), ".cache"),
            # Cache in grandparent directory
            os.path.join(os.path.dirname(os.path.dirname(actual_val_img_dir)), f"{os.path.basename(actual_val_img_dir)}.cache"),
            # Also clear train cache
            os.path.join(actual_train_img_dir, f"{os.path.basename(actual_train_img_dir)}.cache"),
            os.path.join(actual_train_img_dir, ".cache"),
            os.path.join(os.path.dirname(actual_train_img_dir), f"{os.path.basename(actual_train_img_dir)}.cache"),
        ]
        cleared_count = 0
        for cache_path in cache_paths:
            if os.path.exists(cache_path):
                try:
                    os.remove(cache_path)
                    cleared_count += 1
                    print(f"      🗑️  Cleared cache: {cache_path}")
                except Exception as e:
                    print(f"      ⚠️  Could not clear cache {cache_path}: {e}")
        if cleared_count == 0:
            print(f"      ℹ️  No cache files found to clear")
        else:
            print(f"      ✅ Cleared {cleared_count} cache file(s)")
        
        # Verify label directory structure before testing
        print(f"      🔍 Verifying label directory structure...")
        print(f"         Expected labels dir: {val_labels_dir}")
        print(f"         Labels dir exists: {os.path.exists(val_labels_dir)}")
        if os.path.exists(val_labels_dir):
            label_files = [f for f in os.listdir(val_labels_dir) if f.endswith('.txt')]
            print(f"         Label files found: {len(label_files)}")
            if len(label_files) > 0:
                print(f"         Sample label files: {label_files[:3]}")
        
        # Verify image directory structure
        print(f"      🔍 Verifying image directory structure...")
        print(f"         Image dir: {actual_val_img_dir}")
        image_files = [f for f in os.listdir(actual_val_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"         Image files found: {len(image_files)}")
        if len(image_files) > 0:
            print(f"         Sample image files: {image_files[:3]}")
        
        # Check filename matching
        if os.path.exists(val_labels_dir) and len(image_files) > 0:
            image_basenames = {os.path.splitext(f)[0] for f in image_files}
            label_basenames = {os.path.splitext(f)[0] for f in label_files}
            matching = image_basenames & label_basenames
            print(f"         Matching image-label pairs: {len(matching)}/{len(image_files)}")
            if len(matching) < len(image_files) * 0.9:
                print(f"         ⚠️  WARNING: Only {len(matching)}/{len(image_files)} images have matching labels!")
        
        # Verify the resolved paths match what YOLO will use
        print(f"      🔍 Verifying YAML path resolution...")
        with open(coco_yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
            yaml_path = yaml_data.get('path', '')
            yaml_train = yaml_data.get('train', '')
            yaml_val = yaml_data.get('val', '')
            
            if yaml_path:
                # YOLO will resolve: <path>/<train> and <path>/<val>
                resolved_train = os.path.join(yaml_path, yaml_train) if not os.path.isabs(yaml_train) else yaml_train
                resolved_val = os.path.join(yaml_path, yaml_val) if not os.path.isabs(yaml_val) else yaml_val
                print(f"         YAML path: {yaml_path}")
                print(f"         YAML train: {yaml_train} → resolves to: {resolved_train}")
                print(f"         YAML val: {yaml_val} → resolves to: {resolved_val}")
                print(f"         Expected train labels: {os.path.join(resolved_train, 'labels')}")
                print(f"         Expected val labels: {os.path.join(resolved_val, 'labels')}")
                print(f"         Actual train labels: {train_labels_dir}")
                print(f"         Actual val labels: {val_labels_dir}")
                
                # Check if resolved paths match actual paths
                if os.path.abspath(resolved_train) != os.path.abspath(actual_train_img_dir):
                    print(f"         ⚠️  WARNING: Resolved train path doesn't match actual path!")
                if os.path.abspath(resolved_val) != os.path.abspath(actual_val_img_dir):
                    print(f"         ⚠️  WARNING: Resolved val path doesn't match actual path!")
        
        # CRITICAL: Inspect what YOLO actually sees before validation
        print(f"      🔍 Inspecting YOLO's dataset object before validation...")
        try:
            # Create a temporary model to inspect dataset loading
            inspect_model = YOLO("yolov8s.pt")
            # Load dataset without running validation
            from ultralytics.data import YOLODataset
            # Try to import yaml_load, but it might not exist in all versions
            try:
                from ultralytics.utils import yaml_load
            except ImportError:
                # Fallback: use yaml directly
                import yaml as yaml_module
                def yaml_load(path):
                    with open(path, 'r') as f:
                        return yaml_module.safe_load(f)
            
            # Load YAML to see what YOLO will read
            yaml_data = yaml_load(coco_yaml_path)
            print(f"         YAML data loaded:")
            print(f"            path: {yaml_data.get('path', 'NOT SET')}")
            print(f"            train: {yaml_data.get('train', 'NOT SET')}")
            print(f"            val: {yaml_data.get('val', 'NOT SET')}")
            
            # Try to manually create dataset to see where it looks
            try:
                # This is how YOLO internally resolves paths
                val_path_from_yaml = yaml_data.get('val', '')
                path_base = yaml_data.get('path', '')
                
                if path_base and not os.path.isabs(val_path_from_yaml):
                    # YOLO resolves: path + val
                    resolved_val_path = os.path.join(path_base, val_path_from_yaml)
                else:
                    resolved_val_path = val_path_from_yaml
                
                resolved_val_path = os.path.abspath(resolved_val_path)
                print(f"         YOLO will resolve val path to: {resolved_val_path}")
                
                # Check where YOLO will look for labels
                # YOLO logic: if path contains "images", replace with "labels", else append "/labels"
                if "/images/" in resolved_val_path:
                    yolo_label_path = resolved_val_path.replace("/images/", "/labels/")
                else:
                    yolo_label_path = os.path.join(resolved_val_path, "labels")
                
                print(f"         YOLO will look for labels in: {yolo_label_path}")
                print(f"         Actual labels are in: {val_labels_dir}")
                print(f"         Paths match: {os.path.abspath(yolo_label_path) == os.path.abspath(val_labels_dir)}")
                
                if os.path.abspath(yolo_label_path) != os.path.abspath(val_labels_dir):
                    print(f"         ❌ MISMATCH! YOLO is looking in the wrong place!")
                    print(f"         💡 YOLO expects labels in: {yolo_label_path}")
                    print(f"         💡 But labels are actually in: {val_labels_dir}")
                    print(f"         💡 This is likely the root cause!")
                    
                    # Test: Can YOLO actually read from the expected location?
                    if os.path.exists(yolo_label_path):
                        test_labels = [f for f in os.listdir(yolo_label_path) if f.endswith('.txt')]
                        print(f"         📊 YOLO's expected location exists: {len(test_labels)} label files")
                    else:
                        print(f"         ❌ YOLO's expected location does NOT exist!")
                        print(f"         💡 We need to either:")
                        print(f"            1. Move labels to: {yolo_label_path}")
                        print(f"            2. Or fix YAML to point to: {os.path.dirname(val_labels_dir)}")
                else:
                    print(f"         ✅ Paths match! YOLO should find labels.")
                    
                    # But if paths match and YOLO still doesn't find labels, there's another issue
                    # Test: Can we manually read a label file from where YOLO expects it?
                    if os.path.exists(yolo_label_path):
                        sample_labels = [f for f in os.listdir(yolo_label_path) if f.endswith('.txt')]
                        if sample_labels:
                            sample_label_path = os.path.join(yolo_label_path, sample_labels[0])
                            try:
                                with open(sample_label_path, 'r') as f:
                                    content = f.read().strip()
                                    if content:
                                        print(f"         ✅ Test: Can read label file '{sample_labels[0]}' ({len(content)} chars)")
                                    else:
                                        print(f"         ⚠️  Test: Label file '{sample_labels[0]}' is EMPTY!")
                            except Exception as e:
                                print(f"         ❌ Test: Cannot read label file: {e}")
                    
            except Exception as e:
                print(f"         ⚠️  Could not inspect path resolution: {e}")
                import traceback
                traceback.print_exc()
        except Exception as e:
            print(f"         ⚠️  Could not inspect dataset: {e}")
            import traceback
            traceback.print_exc()
        
        test_model = YOLO("yolov8s.pt")  # Use YOLOv8s for testing
        print(f"      🔍 Running quick validation test (this may take a moment)...")
        test_results = test_model.val(data=coco_yaml_path, imgsz=640, verbose=False, plots=False)
        
        # Check if labels were found - also check the dataset object
        instances = 0
        map50 = 0
        map50_95 = 0
        
        # Try to get from results_dict (most common)
        if hasattr(test_results, 'results_dict'):
            instances = test_results.results_dict.get('metrics/instances', 0)
            if instances == 0:
                instances = test_results.results_dict.get('instances', 0)
            
            map50 = test_results.results_dict.get('metrics/mAP50(B)', 0)
            if map50 == 0:
                map50 = test_results.results_dict.get('metrics/mAP50', 0)
            
            map50_95 = test_results.results_dict.get('metrics/mAP50-95(B)', 0)
            if map50_95 == 0:
                map50_95 = test_results.results_dict.get('metrics/mAP50-95', 0)
                if map50_95 == 0:
                    map50_95 = test_results.results_dict.get('metrics/mAP', 0)
        
        # Also try to get from metrics directly
        if instances == 0 and hasattr(test_results, 'metrics'):
            try:
                instances = getattr(test_results.metrics, 'instances', 0)
                if map50 == 0:
                    map50 = getattr(test_results.metrics, 'map50', 0)
                if map50_95 == 0:
                    map50_95 = getattr(test_results.metrics, 'map', 0)
            except:
                pass
        
        # Check dataset object if available
        non_empty_labels_count = 0
        if hasattr(test_model, 'validator') and hasattr(test_model.validator, 'dataset'):
            dataset = test_model.validator.dataset
            if hasattr(dataset, 'labels') and dataset.labels is not None:
                if isinstance(dataset.labels, list):
                    non_empty_labels = [lbl for lbl in dataset.labels if lbl is not None and len(lbl) > 0]
                    non_empty_labels_count = len(non_empty_labels)
                    print(f"      📊 Dataset has {non_empty_labels_count}/{len(dataset.labels)} non-empty labels")
                    if len(non_empty_labels) == 0 and map50 == 0 and map50_95 == 0:
                        # Try fallback: recreate YAML without path field using absolute paths
                        print(f"      ⚠️  No labels found with relative paths. Trying fallback: absolute paths without 'path' field...")
                        try:
                            import shutil
                            # Create backup of current YAML
                            backup_yaml = coco_yaml_path + ".backup"
                            shutil.copy(coco_yaml_path, backup_yaml)
                            
                            # Get paths and names from YAML
                            with open(coco_yaml_path, 'r') as f:
                                yaml_data_fallback = yaml.safe_load(f)
                            
                            base_path_fallback = yaml_data_fallback.get('path', os.path.dirname(coco_yaml_path))
                            train_path_fallback = os.path.abspath(os.path.join(base_path_fallback, yaml_data_fallback.get('train', 'images/train')))
                            val_path_fallback = os.path.abspath(os.path.join(base_path_fallback, yaml_data_fallback.get('val', 'images/val')))
                            coco_names_fallback = yaml_data_fallback.get('names', [])
                            
                            # Create new YAML with absolute paths, no path field
                            fallback_yaml_content = {
                                'train': train_path_fallback,  # Absolute path
                                'val': val_path_fallback,       # Absolute path
                                'names': coco_names_fallback,
                                'nc': len(coco_names_fallback)
                            }
                            
                            with open(coco_yaml_path, 'w') as f:
                                yaml.dump(fallback_yaml_content, f, default_flow_style=False)
                            
                            print(f"      🔄 Created fallback YAML (absolute paths, no 'path' field)")
                            print(f"         Train: {train_path_fallback}")
                            print(f"         Val: {val_path_fallback}")
                            
                            # Clear cache again
                            for cache_path in cache_paths:
                                if os.path.exists(cache_path):
                                    try:
                                        os.remove(cache_path)
                                    except:
                                        pass
                            
                            # Test again with fallback YAML
                            print(f"      🔍 Testing with fallback YAML...")
                            test_model_fallback = YOLO("yolov8s.pt")
                            test_results_fallback = test_model_fallback.val(data=coco_yaml_path, imgsz=640, verbose=False, plots=False)
                            
                            instances_fallback = 0
                            map50_fallback = 0
                            map50_95_fallback = 0
                            
                            if hasattr(test_results_fallback, 'results_dict'):
                                instances_fallback = test_results_fallback.results_dict.get('metrics/instances', 0)
                                map50_fallback = test_results_fallback.results_dict.get('metrics/mAP50(B)', 0)
                                if map50_fallback == 0:
                                    map50_fallback = test_results_fallback.results_dict.get('metrics/mAP50', 0)
                                map50_95_fallback = test_results_fallback.results_dict.get('metrics/mAP50-95(B)', 0)
                                if map50_95_fallback == 0:
                                    map50_95_fallback = test_results_fallback.results_dict.get('metrics/mAP50-95', 0)
                            
                            if instances_fallback > 0 or map50_fallback > 0 or map50_95_fallback > 0:
                                print(f"      ✅ Fallback YAML worked!")
                                if instances_fallback > 0:
                                    print(f"         Found {instances_fallback} instances")
                                if map50_fallback > 0 or map50_95_fallback > 0:
                                    print(f"         mAP50={map50_fallback:.4f}, mAP50-95={map50_95_fallback:.4f}")
                                instances = instances_fallback
                                map50 = map50_fallback
                                map50_95 = map50_95_fallback
                            else:
                                # Restore original YAML
                                shutil.copy(backup_yaml, coco_yaml_path)
                                os.remove(backup_yaml)
                                print(f"      ⚠️  Fallback also failed, restored original YAML")
                        except Exception as e:
                            print(f"      ⚠️  Fallback attempt failed: {e}")
                            # Try to restore original if backup exists
                            backup_yaml = coco_yaml_path + ".backup"
                            if os.path.exists(backup_yaml):
                                try:
                                    shutil.copy(backup_yaml, coco_yaml_path)
                                    os.remove(backup_yaml)
                                except:
                                    pass
        
        # Success criteria: mAP values > 0 means labels were found and processed
        # The validation output shows "all" row with instances count
        print(f"      📊 Validation metrics:")
        if instances > 0:
            print(f"         Instances found: {instances}")
        else:
            print(f"         Instances: (check output above - YOLO prints 'all' row with total instances)")
        print(f"         mAP50: {map50:.4f}")
        print(f"         mAP50-95: {map50_95:.4f}")
        
        if map50 > 0 or map50_95 > 0:
            print(f"      ✅ YOLO test validation SUCCESS! Labels are accessible!")
            print(f"         mAP50={map50:.4f} and mAP50-95={map50_95:.4f} indicate labels are working!")
            if instances > 0:
                print(f"         Total instances detected: {instances}")
            return True, None
        elif instances > 0:
            print(f"      ✅ YOLO test validation found {instances} instances - labels are accessible!")
            return True, None
        elif non_empty_labels_count > 0:
            print(f"      ✅ YOLO test validation found {non_empty_labels_count} non-empty labels - labels are accessible!")
            return True, None
        else:
            return False, f"❌ CRITICAL: YOLO validation test found 0 instances/boxes and mAP is 0. " \
                         f"Labels exist but YOLO cannot find them. " \
                         f"This means the dataset structure is NOT compatible with YOLO. " \
                         f"Check: (1) Labels in {val_labels_dir}, (2) YAML points to {val_path}, " \
                         f"(3) Label filenames match image filenames exactly, (4) Label files are not empty. " \
                         f"Tried both relative paths (with 'path' field) and absolute paths (without 'path' field)."
        # Additional check: verify YOLO can actually load the dataset
        print(f"      🔍 Verifying YOLO dataset loading...")
        try:
            from ultralytics.data import YOLODataset
            # Try to manually check if YOLO can find labels
            # This is a diagnostic check
            pass  # We already checked above
        except Exception as e:
            print(f"      ⚠️  Could not perform additional dataset check: {e}")
            
    except Exception as e:
        return False, f"❌ CRITICAL: YOLO validation test failed: {e}. " \
                     f"Labels may not be in the correct location for YOLO to find them. " \
                     f"This will NOT work."
    
    print(f"\n   ✅ Dataset validation passed! Ready to proceed.")
    return True, None


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================
def main():
    # Try to use COCO-trained model, fallback to yolov8s.pt (pre-trained on COCO)
    model_path = "data/best_coco.pt"
    if not os.path.exists(model_path):
        print(f"⚠️  COCO fine-tuned model not found: {model_path}")
        model_path = "yolov8s.pt"  # Use pre-trained COCO model
        print(f"   Using pre-trained COCO model: {model_path}")
        print(f"   💡 For better results, fine-tune first: python finetune_yolov8s_coco.py")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print(f"   Please ensure either 'data/best_coco.pt' or 'yolov8s.pt' exists")
        return False
    
    print(f"📦 Using model: {model_path}")
    epochs_final = 20
    epochs_per_finetune = 5
    
    # Check if COCO is already set up in standard YOLO format
    coco_yaml_standard = "data/coco/coco.yaml"
    if os.path.exists(coco_yaml_standard):
        # Read YAML to check format
        with open(coco_yaml_standard, 'r') as f:
            yaml_data = yaml.safe_load(f)
            train_path = yaml_data.get('train', '')
            val_path = yaml_data.get('val', '')
        
        base_path = yaml_data.get('path', os.path.dirname(coco_yaml_standard))
        
        # Check if it's standard format
        if train_path == 'images/train' and val_path == 'images/val':
            print(f"\n✅ Found COCO dataset in standard YOLO format: {coco_yaml_standard}")
            coco_yaml_path = coco_yaml_standard
            train_images_dir = os.path.join(base_path, 'images', 'train')
            val_images_dir = os.path.join(base_path, 'images', 'val')
        else:
            # Old format detected - check if standard format directories exist
            print(f"\n⚠️  COCO YAML found but uses old format (train: {train_path}, val: {val_path})")
            
            train_images_std = os.path.join(base_path, "images", "train")
            val_images_std = os.path.join(base_path, "images", "val")
            train_labels_std = os.path.join(base_path, "labels", "train")
            val_labels_std = os.path.join(base_path, "labels", "val")
            
            if os.path.exists(train_images_std) and os.path.exists(val_images_std):
                # Standard format directories exist - auto-fix YAML
                print(f"   ✅ Standard format directories found!")
                print(f"   Auto-updating YAML to standard format...")
                
                import shutil
                # Create backup
                backup_yaml = coco_yaml_standard + ".backup"
                shutil.copy(coco_yaml_standard, backup_yaml)
                
                # Update YAML to standard format
                yaml_data['train'] = 'images/train'
                yaml_data['val'] = 'images/val'
                
                with open(coco_yaml_standard, 'w') as f:
                    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
                
                print(f"   ✅ YAML updated to standard format!")
                print(f"   Backup saved to: {backup_yaml}")
                
                coco_yaml_path = coco_yaml_standard
                train_images_dir = train_images_std
                val_images_dir = val_images_std
            else:
                # Old format - use old paths
                print(f"   ⚠️  Standard format not found, using old format")
                print(f"   💡 For better compatibility, run: python adva_yolo_pruning/setup_coco_yolo_format.py")
                coco_yaml_path = coco_yaml_standard
                train_images_dir = os.path.join(base_path, train_path)
                val_images_dir = os.path.join(base_path, val_path)
    else:
        print(f"\n⚠️  Standard YOLO format not found. Setting up COCO dataset...")
        print(f"   (This will use the old nested structure)")
        coco_yaml_path, train_images_dir, val_images_dir = setup_coco_dataset()
        if coco_yaml_path is None:
            print("❌ Failed to setup COCO dataset")
            print("\n💡 Tip: Run 'python setup_coco_yolo_format.py' first for standard YOLO format")
            return False
    
    # Validate dataset setup before proceeding - CRITICAL CHECK
    print(f"\n{'='*70}")
    print("CRITICAL: VALIDATING DATASET BEFORE PROCEEDING")
    print(f"{'='*70}")
    print("⚠️  If labels are not found or accessible, the experiment will NOT work.")
    print("⚠️  This validation will test if YOLO can actually find and use the labels.")
    print(f"{'='*70}\n")
    
    is_valid, error_msg = validate_coco_dataset_setup(coco_yaml_path, train_images_dir, val_images_dir)
    if not is_valid:
        print(f"\n{'='*70}")
        print("❌❌❌ CRITICAL ERROR: DATASET VALIDATION FAILED ❌❌❌")
        print(f"{'='*70}")
        print(f"\n{error_msg}")
        print(f"\n{'='*70}")
        print("💡 TROUBLESHOOTING STEPS:")
        print(f"{'='*70}")
        print(f"   1. Verify labels exist: Check that label files (.txt) are in the labels/ subdirectory")
        print(f"   2. Check filename matching: Label files must have the same base name as image files")
        print(f"   3. Verify label content: Label files must not be empty")
        print(f"   4. Check YAML paths: YAML file must point to correct image directories")
        print(f"   5. Check directory structure: Labels should be in: <image_dir>/labels/")
        print(f"   6. Clear cache: Delete any .cache files in dataset directories")
        print(f"\n{'='*70}")
        print("❌ EXITING NOW - This experiment will NOT work with invalid labels.")
        print(f"{'='*70}\n")
        sys.exit(1)  # Exit with error code
    
    data_yaml = coco_yaml_path
    
    # Load COCO data - use ALL available data
    print("\n📥 Loading ALL COCO training data for activation extraction...")
    
    # Check if we're using standard YOLO format (images/train, labels/train)
    # or old format (train2017/train2017 with labels inside)
    actual_train_img_dir = train_images_dir
    
    # For standard format: images are directly in train_images_dir
    # For old format: images might be nested (train2017/train2017/)
    if os.path.exists(train_images_dir):
        # Check if images are directly here (standard format)
        direct_images = [f for f in os.listdir(train_images_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(direct_images) > 0:
            # Standard format: images directly in train_images_dir
            actual_train_img_dir = train_images_dir
            print(f"   📁 Using standard format: images directly in {actual_train_img_dir}")
        else:
            # Old format: look for nested directory
            for item in os.listdir(train_images_dir):
                item_path = os.path.join(train_images_dir, item)
                if os.path.isdir(item_path) and item != "labels":
                    files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(files) > 0:
                        actual_train_img_dir = item_path
                        print(f"   📁 Using old format: found nested directory {actual_train_img_dir}")
                        break
    
    # Find labels directory - check both standard YOLO format and old format
    # Standard format: labels/train/ (sibling of images/train/)
    # Old format: images/train/labels/ (inside image directory)
    if 'images' in actual_train_img_dir:
        # Standard format: go up from images/train to get base (data/coco)
        base_dir = os.path.dirname(os.path.dirname(actual_train_img_dir))
        train_labels_standard = os.path.join(base_dir, "labels", "train")
    else:
        # Old format: labels might be inside image directory
        base_dir = os.path.dirname(os.path.dirname(actual_train_img_dir)) if 'train2017' in actual_train_img_dir else os.path.dirname(actual_train_img_dir)
        train_labels_standard = os.path.join(base_dir, "labels", "train")
    
    train_labels_old = os.path.join(actual_train_img_dir, "labels")
    
    if os.path.exists(train_labels_standard):
        train_labels_dir = train_labels_standard
        print(f"   📁 Using standard YOLO format labels: {train_labels_dir}")
    elif os.path.exists(train_labels_old):
        train_labels_dir = train_labels_old
        print(f"   📁 Using old format labels: {train_labels_dir}")
    else:
        train_labels_dir = train_labels_standard
        print(f"   ⚠️  Labels directory not found, will try: {train_labels_dir}")
    
    train_data = load_coco_samples(actual_train_img_dir, train_labels_dir, max_samples=None)
    if len(train_data) == 0:
        print("❌ No training data loaded.")
        print(f"   Images dir: {actual_train_img_dir}")
        print(f"   Labels dir: {train_labels_dir}")
        return False
    
    # Check how many samples have labels
    samples_with_labels = sum(1 for s in train_data if len(s.get('label', [])) > 0)
    print(f"   ✅ Loaded {len(train_data)} training samples, {samples_with_labels} have labels")
    if samples_with_labels == 0:
        print(f"   ⚠️  WARNING: No samples have labels! Activation extraction will fail!")
        print(f"   💡 Check that labels exist in: {train_labels_dir}")
    
    print("📥 Loading ALL COCO validation data for activation extraction...")
    
    # Check if we're using standard YOLO format (images/val, labels/val)
    # or old format (val2017/val2017 with labels inside)
    actual_val_img_dir = val_images_dir
    
    # For standard format: images are directly in val_images_dir
    # For old format: images might be nested (val2017/val2017/)
    if os.path.exists(val_images_dir):
        # Check if images are directly here (standard format)
        direct_images = [f for f in os.listdir(val_images_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(direct_images) > 0:
            # Standard format: images directly in val_images_dir
            actual_val_img_dir = val_images_dir
            print(f"   📁 Using standard format: images directly in {actual_val_img_dir}")
        else:
            # Old format: look for nested directory
            for item in os.listdir(val_images_dir):
                item_path = os.path.join(val_images_dir, item)
                if os.path.isdir(item_path) and item != "labels":
                    files = [f for f in os.listdir(item_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if len(files) > 0:
                        actual_val_img_dir = item_path
                        print(f"   📁 Using old format: found nested directory {actual_val_img_dir}")
                        break
    
    # Find labels directory - check both standard YOLO format and old format
    if 'images' in actual_val_img_dir:
        # Standard format: use same base_dir from train
        val_labels_standard = os.path.join(base_dir, "labels", "val")
    else:
        # Old format: labels might be inside image directory
        val_labels_standard = os.path.join(base_dir, "labels", "val")
    
    val_labels_old = os.path.join(actual_val_img_dir, "labels")
    
    if os.path.exists(val_labels_standard):
        val_labels_dir = val_labels_standard
        print(f"   📁 Using standard YOLO format labels: {val_labels_dir}")
    elif os.path.exists(val_labels_old):
        val_labels_dir = val_labels_old
        print(f"   📁 Using old format labels: {val_labels_dir}")
    else:
        val_labels_dir = val_labels_standard
        print(f"   ⚠️  Labels directory not found, will try: {val_labels_dir}")
    
    valid_data = load_coco_samples(actual_val_img_dir, val_labels_dir, max_samples=None)
    if len(valid_data) == 0:
        print("⚠️  No validation data loaded, using training data only")
        all_activation_data = train_data
        valid_data = []
    else:
        all_activation_data = train_data + valid_data
        val_samples_with_labels = sum(1 for s in valid_data if len(s.get('label', [])) > 0)
        print(f"   ✅ Loaded {len(valid_data)} validation samples, {val_samples_with_labels} have labels")
        print(f"   ✅ Total samples for activation extraction: {len(all_activation_data)}")
        
        # Check total samples with labels
        total_with_labels = samples_with_labels + val_samples_with_labels
        print(f"   ✅ Total samples with labels: {total_with_labels}/{len(all_activation_data)}")
        if total_with_labels == 0:
            print(f"   ❌ CRITICAL: No samples have labels! Activation extraction will fail!")
            return False
    
    activation_data = all_activation_data
    
    # Load classes from COCO (80 classes)
    with open(data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
        classes = list(range(data_config['nc']))

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
    # MEASURE ORIGINAL MODEL METRICS (INFERENCE TIME, FLOPs, SPARSITY)
    # ========================================================================
    print(f"\n{'='*70}")
    print("MEASURING ORIGINAL MODEL METRICS")
    print(f"{'='*70}")
    
    original_model = YOLO(model_path)
    print("   Running validation on original model...")
    try:
        original_metrics = original_model.val(data=data_yaml, verbose=False)
        # Extract inference time from metrics
        original_inference_time = None
        if hasattr(original_metrics, 'speed') and original_metrics.speed is not None:
            if isinstance(original_metrics.speed, dict):
                original_inference_time = original_metrics.speed.get('inference', None)
            elif isinstance(original_metrics.speed, (int, float)):
                original_inference_time = original_metrics.speed
        
        if original_inference_time is not None:
            print(f"✅ Original model inference time: {original_inference_time:.2f} ms/image")
        else:
            print(f"⚠️  Could not extract inference time from metrics, using fallback...")
            original_inference_time = measure_inference_time(original_model, data_yaml, num_runs=1)
            if original_inference_time is not None:
                print(f"✅ Original model inference time (fallback): {original_inference_time:.2f} ms/image")
            else:
                print(f"⚠️  Could not measure original model inference time")
                original_inference_time = None
    except Exception as e:
        print(f"⚠️  Error measuring original model inference time: {e}")
        original_inference_time = None
    
    # Calculate original model FLOPs
    print("   Calculating original model FLOPs...")
    original_flops = calculate_model_flops(original_model)
    if original_flops is not None:
        print(f"✅ Original model FLOPs: {original_flops:.2f} GFLOPs")
    else:
        print(f"⚠️  Could not calculate original model FLOPs")
    
    # Calculate original model sparsity
    print("   Calculating original model sparsity...")
    original_sparsity = calculate_model_sparsity(original_model)
    if original_sparsity['sparsity'] is not None:
        print(f"✅ Original model sparsity: {original_sparsity['sparsity']:.2f}%")
        print(f"   Total parameters: {original_sparsity['total_params']:,}")
        print(f"   Zero parameters: {original_sparsity['zero_params']:,}")
    else:
        print(f"⚠️  Could not calculate original model sparsity")

    # Select target layers (blocks 1, 3, 5, 7)
    targets = select_layers_blocks_1_3_5_7(detection_models['method1_activation_max_weight'])
    if len(targets) == 0:
        print("❌ No eligible layers found.")
        return False

    # Validate that only expected blocks are selected
    expected_blocks = {1, 3, 5, 7}
    actual_blocks = {t['block_idx'] for t in targets}
    unexpected_blocks = actual_blocks - expected_blocks
    if unexpected_blocks:
        print(f"⚠️  WARNING: Unexpected blocks found: {unexpected_blocks}")
        print(f"   Expected blocks: {expected_blocks}")
        print(f"   Actual blocks: {actual_blocks}")
    
    missing_blocks = expected_blocks - actual_blocks
    if missing_blocks:
        print(f"⚠️  WARNING: Some expected blocks are missing: {missing_blocks}")
        print(f"   These blocks either don't exist or don't have Conv layers")

    print("\n" + "="*70)
    print("4-METHOD PRUNING COMPARISON EXPERIMENT (BLOCKS 1, 3, 5, 7) - COCO DATASET")
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
            if has_concatenation(block):
                print(f"⚠️  Skipping {t['name']} (block has concatenation operations)")
                print(f"   💡 Concatenation blocks cannot be processed with Sequential mini-net")
                print(f"   💡 This block requires special handling with forward hooks")
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
                error_msg = str(e)
                # Check if this is the concatenation error
                if "cat()" in error_msg or "concatenation" in error_msg.lower():
                    print(f"❌ Pruning failed: {t['name']} has concatenation operations")
                    print(f"   💡 Error: {error_msg}")
                    print(f"   💡 This block cannot be processed with Sequential mini-net")
                    print(f"   💡 Solution: Block should have been skipped by has_concatenation() check")
                    print(f"   💡 This block requires special handling with forward hooks")
                else:
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
                
                # Clear cache before training to ensure fresh label loading
                # YOLO creates cache files in the image directories, so we need to clear those
                with open(data_yaml, 'r') as f:
                    yaml_check = yaml.safe_load(f)
                    base_path = yaml_check.get('path', os.path.dirname(data_yaml))
                    train_path = yaml_check.get('train', '')
                    val_path = yaml_check.get('val', '')
                    
                    # Build full paths
                    train_full = os.path.join(base_path, train_path) if train_path else None
                    val_full = os.path.join(base_path, val_path) if val_path else None
                    
                    # Clear cache files
                    cache_files = []
                    if train_full:
                        cache_files.append(train_full + ".cache")
                    if val_full:
                        cache_files.append(val_full + ".cache")
                    # Also check base directory
                    cache_files.extend([
                        os.path.join(base_path, "train.cache"),
                        os.path.join(base_path, "val.cache"),
                    ])
                    
                    for cache_file in cache_files:
                        if os.path.exists(cache_file):
                            try:
                                os.remove(cache_file)
                                print(f"   🗑️  Cleared cache: {cache_file}")
                            except:
                                pass
                
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
            # Clear cache before final fine-tuning
            with open(data_yaml, 'r') as f:
                yaml_check = yaml.safe_load(f)
                base_path = yaml_check.get('path', os.path.dirname(data_yaml))
                train_path = yaml_check.get('train', '')
                val_path = yaml_check.get('val', '')
                
                # Build full paths
                train_full = os.path.join(base_path, train_path) if train_path else None
                val_full = os.path.join(base_path, val_path) if val_path else None
                
                # Clear cache files
                cache_files = []
                if train_full:
                    cache_files.append(train_full + ".cache")
                if val_full:
                    cache_files.append(val_full + ".cache")
                # Also check base directory
                cache_files.extend([
                    os.path.join(base_path, "train.cache"),
                    os.path.join(base_path, "val.cache"),
                ])
                
                for cache_file in cache_files:
                    if os.path.exists(cache_file):
                        try:
                            os.remove(cache_file)
                            print(f"   🗑️  Cleared cache: {cache_file}")
                        except:
                            pass
            
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
    flops_results = {}
    sparsity_results = {}
    
    for method_name, model in models.items():
        print(f"\n--- Evaluating {method_names[method_name]} ---")
        try:
            # Run validation - this also provides speed metrics
            metrics = model.val(data=data_yaml, verbose=False)
            metrics_results[method_name] = {
                'mAP50-95': metrics.results_dict.get('metrics/mAP50-95(B)', 0),
                'mAP50': metrics.results_dict.get('metrics/mAP50(B)', 0),
                'precision': metrics.results_dict.get('metrics/precision(B)', 0),
                'recall': metrics.results_dict.get('metrics/recall(B)', 0)
            }
            
            # Extract inference time from metrics
            inference_time = None
            if hasattr(metrics, 'speed') and metrics.speed is not None:
                if isinstance(metrics.speed, dict):
                    inference_time = metrics.speed.get('inference', None)
                elif isinstance(metrics.speed, (int, float)):
                    inference_time = metrics.speed
            
            if inference_time is not None:
                inference_times[method_name] = inference_time
                print(f"   ✅ Inference time: {inference_time:.2f} ms/image")
            else:
                inference_times[method_name] = None
                print(f"   ⚠️  Could not extract inference time from metrics")
            
            # Calculate FLOPs
            print(f"   Calculating FLOPs...")
            flops = calculate_model_flops(model)
            if flops is not None:
                flops_results[method_name] = flops
                print(f"   ✅ FLOPs: {flops:.2f} GFLOPs")
                if original_flops is not None and original_flops > 0:
                    flops_reduction = ((original_flops - flops) / original_flops) * 100
                    print(f"   📉 FLOPs reduction: {flops_reduction:.2f}% ({original_flops:.2f} → {flops:.2f} GFLOPs)")
            else:
                flops_results[method_name] = None
                print(f"   ⚠️  Could not calculate FLOPs")
            
            # Calculate sparsity
            print(f"   Calculating sparsity...")
            sparsity = calculate_model_sparsity(model)
            if sparsity['sparsity'] is not None:
                sparsity_results[method_name] = sparsity
                print(f"   ✅ Sparsity: {sparsity['sparsity']:.2f}%")
                if original_sparsity['sparsity'] is not None:
                    sparsity_increase = sparsity['sparsity'] - original_sparsity['sparsity']
                    print(f"   📈 Sparsity increase: {sparsity_increase:.2f}% (from {original_sparsity['sparsity']:.2f}%)")
            else:
                sparsity_results[method_name] = None
                print(f"   ⚠️  Could not calculate sparsity")
                
        except Exception as e:
            print(f"⚠️  Evaluation failed for {method_names[method_name]}: {e}")
            metrics_results[method_name] = None
            inference_times[method_name] = None
            flops_results[method_name] = None
            sparsity_results[method_name] = None

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
    
    # FLOPs comparison
    print("\n🔢 FLOPs Comparison:")
    print(f"{'Method':<35} {'FLOPs (GFLOPs)':<20} {'Reduction %':<15} {'Reduction (GFLOPs)':<20}")
    print("-" * 90)
    
    if original_flops is not None:
        print(f"{'Original Model':<35} {original_flops:<20.2f} {'0.00% (baseline)':<15} {'0.00 (baseline)':<20}")
    
    for method_name in method_names.keys():
        if flops_results.get(method_name) is not None:
            flops = flops_results[method_name]
            if original_flops is not None and original_flops > 0:
                reduction_pct = ((original_flops - flops) / original_flops) * 100
                reduction_abs = original_flops - flops
                print(f"{method_names[method_name]:<35} {flops:<20.2f} {reduction_pct:<15.2f} {reduction_abs:<20.2f}")
            else:
                print(f"{method_names[method_name]:<35} {flops:<20.2f} {'N/A':<15} {'N/A':<20}")
        else:
            print(f"{method_names[method_name]:<35} {'N/A':<20} {'N/A':<15} {'N/A':<20}")
    
    # Sparsity comparison
    print("\n📊 Sparsity Comparison:")
    print(f"{'Method':<35} {'Sparsity %':<15} {'Increase %':<15} {'Zero Params':<20} {'Total Params':<20}")
    print("-" * 105)
    
    if original_sparsity['sparsity'] is not None:
        print(f"{'Original Model':<35} {original_sparsity['sparsity']:<15.2f} {'0.00% (baseline)':<15} "
              f"{original_sparsity['zero_params']:<20,} {original_sparsity['total_params']:<20,}")
    
    for method_name in method_names.keys():
        if sparsity_results.get(method_name) is not None:
            sparsity = sparsity_results[method_name]
            if sparsity['sparsity'] is not None:
                if original_sparsity['sparsity'] is not None:
                    increase = sparsity['sparsity'] - original_sparsity['sparsity']
                    print(f"{method_names[method_name]:<35} {sparsity['sparsity']:<15.2f} {increase:<15.2f} "
                          f"{sparsity['zero_params']:<20,} {sparsity['total_params']:<20,}")
                else:
                    print(f"{method_names[method_name]:<35} {sparsity['sparsity']:<15.2f} {'N/A':<15} "
                          f"{sparsity['zero_params']:<20,} {sparsity['total_params']:<20,}")
            else:
                print(f"{method_names[method_name]:<35} {'N/A':<15} {'N/A':<15} {'N/A':<20} {'N/A':<20}")
        else:
            print(f"{method_names[method_name]:<35} {'N/A':<15} {'N/A':<15} {'N/A':<20} {'N/A':<20}")

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

    print(f"\n✅ 4-method comparison experiment completed on COCO dataset!")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

