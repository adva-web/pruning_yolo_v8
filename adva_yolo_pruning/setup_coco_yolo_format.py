#!/usr/bin/env python3
"""
Setup COCO dataset in standard YOLO format:
- data/coco/images/train/
- data/coco/images/val/
- data/coco/labels/train/
- data/coco/labels/val/
- data/coco/coco.yaml

This matches the standard YOLO dataset structure that YOLO expects.
"""

import os
import sys
import yaml
import json
import zipfile
import shutil
from pathlib import Path

# Try to import Ultralytics COCO mapping function
try:
    from ultralytics.data.converter import coco91_to_coco80_class
    USE_ULTRALYTICS_MAPPING = True
except ImportError:
    # Fallback: use our own mapping
    USE_ULTRALYTICS_MAPPING = False
    print("   ⚠️  Ultralytics not available, using custom mapping")

def extract_zip_if_needed(zip_path, extract_to):
    """Extract zip file if it exists and target doesn't."""
    if not os.path.exists(zip_path):
        print(f"   ⚠️  Zip file not found: {zip_path}")
        return False
    
    if os.path.exists(extract_to) and len(os.listdir(extract_to)) > 0:
        print(f"   ✅ Already extracted: {extract_to}")
        return True
    
    print(f"   📦 Extracting {zip_path} to {extract_to}...")
    try:
        os.makedirs(extract_to, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"   ✅ Extracted successfully")
        return True
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        return False

def find_images_in_directory(directory):
    """Find the directory containing images (handle nested structures)."""
    if not os.path.exists(directory):
        return None
    
    # Check if directory itself has images
    files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(files) > 0:
        return directory
    
    # Check subdirectories
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            result = find_images_in_directory(item_path)
            if result:
                return result
    
    return None

def convert_coco_to_yolo_format(json_path, images_dir, labels_dir, split_name):
    """Convert COCO JSON annotations to YOLO format."""
    print(f"\n   🔄 Converting {split_name} annotations to YOLO format...")
    
    if not os.path.exists(json_path):
        print(f"   ❌ JSON file not found: {json_path}")
        return False
    
    # Load COCO JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)
    
    # CRITICAL: Create mapping from COCO category_id to YOLO class index (0-79)
    # Use Ultralytics' official mapping if available, otherwise use custom mapping
    if USE_ULTRALYTICS_MAPPING:
        # Use Ultralytics' coco91_to_coco80_class mapping
        # This returns a list where index is COCO category_id (1-90) and value is YOLO class (0-79) or None
        coco80_map = coco91_to_coco80_class()
        category_id_to_class_idx = {}
        
        # Build reverse mapping: category_id -> class_idx
        for coco_id in range(1, 91):  # COCO IDs are 1-90
            yolo_class = coco80_map[coco_id - 1]  # coco80_map is 0-indexed for COCO IDs 1-90
            if yolo_class is not None:
                category_id_to_class_idx[coco_id] = yolo_class
        
        print(f"   📊 Using Ultralytics COCO mapping")
        print(f"   📊 Mapped {len(category_id_to_class_idx)} categories to class indices 0-79")
    else:
        # Fallback: Create mapping from COCO category_id to YOLO class index (0-79)
        # COCO has 80 classes, but category IDs are 1-90 (non-sequential)
        categories = coco_data.get('categories', [])
        if len(categories) != 80:
            print(f"   ⚠️  Warning: Expected 80 categories, found {len(categories)}")
        
        category_id_to_class_idx = {}
        
        # Sort categories by id to ensure consistent mapping
        sorted_categories = sorted(categories, key=lambda x: x['id'])
        for idx, cat in enumerate(sorted_categories):
            category_id_to_class_idx[cat['id']] = idx
        
        if len(category_id_to_class_idx) != 80:
            print(f"   ❌ ERROR: Expected 80 categories in mapping, got {len(category_id_to_class_idx)}")
            return False
        
        print(f"   📊 Found {len(category_id_to_class_idx)} categories")
        print(f"   📊 Category ID range: {min(category_id_to_class_idx.keys())} - {max(category_id_to_class_idx.keys())}")
        print(f"   📊 Mapped to class indices: 0 - {len(category_id_to_class_idx) - 1}")
        
        # Verify mapping: all class indices should be 0-79
        class_indices = set(category_id_to_class_idx.values())
        if min(class_indices) != 0 or max(class_indices) != 79:
            print(f"   ❌ ERROR: Class indices should be 0-79, got {min(class_indices)}-{max(class_indices)}")
            return False
    
    # Create mapping from image_id to image info
    images = {img['id']: img for img in coco_data['images']}
    
    # Create mapping from image_id to annotations
    image_annotations = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        if image_id not in image_annotations:
            image_annotations[image_id] = []
        image_annotations[image_id].append(ann)
    
    # Create labels directory
    os.makedirs(labels_dir, exist_ok=True)
    
    # Process each image
    processed = 0
    skipped = 0
    
    for image_id, image_info in images.items():
        filename = image_info['file_name']
        width = image_info['width']
        height = image_info['height']
        
        # Find image file (handle different naming)
        image_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = os.path.join(images_dir, filename)
            if os.path.exists(candidate):
                image_path = candidate
                break
            # Try with different case
            candidate_lower = os.path.join(images_dir, filename.lower())
            if os.path.exists(candidate_lower):
                image_path = candidate_lower
                break
        
        if image_path is None:
            # Try to find by ID
            base_name = os.path.splitext(filename)[0]
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = os.path.join(images_dir, f"{base_name}{ext}")
                if os.path.exists(candidate):
                    image_path = candidate
                    break
        
        if image_path is None:
            skipped += 1
            continue
        
        # Get annotations for this image
        annotations = image_annotations.get(image_id, [])
        
        # Create label file
        label_filename = os.path.splitext(os.path.basename(image_path))[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        with open(label_path, 'w') as f:
            for ann in annotations:
                # COCO bbox format: [x_min, y_min, width, height]
                bbox = ann['bbox']
                x_min, y_min, w, h = bbox
                
                # Convert to YOLO format: [class_id, x_center, y_center, width, height] (normalized)
                x_center = (x_min + w / 2) / width
                y_center = (y_min + h / 2) / height
                norm_w = w / width
                norm_h = h / height
                
                # CRITICAL: Map COCO category_id (1-90, non-sequential) to YOLO class index (0-79, sequential)
                coco_category_id = ann['category_id']
                if coco_category_id not in category_id_to_class_idx:
                    # Skip if category not in mapping (shouldn't happen, but be safe)
                    continue
                
                class_id = category_id_to_class_idx[coco_category_id]
                
                # Validate class_id is in range 0-79
                if class_id < 0 or class_id >= 80:
                    print(f"   ⚠️  Warning: Invalid class_id {class_id} for category_id {coco_category_id}")
                    continue
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        processed += 1
        if processed % 1000 == 0:
            print(f"      Processed {processed} images...")
    
    print(f"   ✅ Converted {processed} images, skipped {skipped}")
    return True

def setup_coco_yolo_format():
    """Setup COCO dataset in standard YOLO format."""
    print(f"\n{'='*70}")
    print("SETTING UP COCO DATASET IN STANDARD YOLO FORMAT")
    print(f"{'='*70}")
    
    base_dir = "data/coco"
    train_zip = os.path.join(base_dir, "train2017.zip")
    val_zip = os.path.join(base_dir, "val2017.zip")
    annotations_zip = os.path.join(base_dir, "annotations_trainval2017.zip")
    
    # Create standard YOLO directory structure
    images_train_dir = os.path.join(base_dir, "images", "train")
    images_val_dir = os.path.join(base_dir, "images", "val")
    labels_train_dir = os.path.join(base_dir, "labels", "train")
    labels_val_dir = os.path.join(base_dir, "labels", "val")
    
    # Extract images to temporary location first
    temp_train_dir = os.path.join(base_dir, "temp_train2017")
    temp_val_dir = os.path.join(base_dir, "temp_val2017")
    
    if not extract_zip_if_needed(train_zip, temp_train_dir):
        return False
    
    if not extract_zip_if_needed(val_zip, temp_val_dir):
        return False
    
    # Find actual image directories (handle nested structure)
    actual_train_images = find_images_in_directory(temp_train_dir)
    actual_val_images = find_images_in_directory(temp_val_dir)
    
    if actual_train_images is None:
        print(f"   ❌ Could not find train images in {temp_train_dir}")
        return False
    
    if actual_val_images is None:
        print(f"   ❌ Could not find val images in {temp_val_dir}")
        return False
    
    print(f"   📁 Found train images in: {actual_train_images}")
    print(f"   📁 Found val images in: {actual_val_images}")
    
    # Copy images to standard YOLO structure
    print(f"\n   📋 Copying images to standard YOLO structure...")
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    
    # Copy train images
    train_count = 0
    for img_file in os.listdir(actual_train_images):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(actual_train_images, img_file)
            dst = os.path.join(images_train_dir, img_file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            train_count += 1
    print(f"   ✅ Copied {train_count} train images to {images_train_dir}")
    
    # Copy val images
    val_count = 0
    for img_file in os.listdir(actual_val_images):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(actual_val_images, img_file)
            dst = os.path.join(images_val_dir, img_file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            val_count += 1
    print(f"   ✅ Copied {val_count} val images to {images_val_dir}")
    
    # Extract annotations
    annotations_dir = os.path.join(base_dir, "temp_annotations")
    if not extract_zip_if_needed(annotations_zip, annotations_dir):
        return False
    
    # Find JSON files
    train_json = None
    val_json = None
    
    for root, dirs, files in os.walk(annotations_dir):
        for file in files:
            if file == "instances_train2017.json":
                train_json = os.path.join(root, file)
            elif file == "instances_val2017.json":
                val_json = os.path.join(root, file)
    
    if train_json is None:
        print(f"   ❌ Could not find instances_train2017.json")
        return False
    
    if val_json is None:
        print(f"   ❌ Could not find instances_val2017.json")
        return False
    
    print(f"   ✅ Found train JSON: {train_json}")
    print(f"   ✅ Found val JSON: {val_json}")
    
    # Convert annotations to YOLO format
    if not convert_coco_to_yolo_format(train_json, images_train_dir, labels_train_dir, 'train'):
        return False
    
    if not convert_coco_to_yolo_format(val_json, images_val_dir, labels_val_dir, 'val'):
        return False
    
    # Create YAML file
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
    
    # Use standard YOLO format: path field with relative paths
    base_abs = os.path.abspath(base_dir)
    coco_yaml_content = {
        'path': base_abs,
        'train': 'images/train',  # Relative to path
        'val': 'images/val',      # Relative to path
        'names': coco_names,
        'nc': len(coco_names)
    }
    
    with open(coco_yaml_path, 'w') as f:
        yaml.dump(coco_yaml_content, f, default_flow_style=False)
    
    print(f"\n   ✅ Created YAML file: {coco_yaml_path}")
    print(f"\n   📋 YAML content:")
    print(f"   {'-'*60}")
    with open(coco_yaml_path, 'r') as f:
        for line in f:
            print(f"   {line.rstrip()}")
    print(f"   {'-'*60}")
    
    print(f"\n   📊 Final structure:")
    print(f"      Images train: {os.path.abspath(images_train_dir)} ({train_count} images)")
    print(f"      Images val:   {os.path.abspath(images_val_dir)} ({val_count} images)")
    print(f"      Labels train: {os.path.abspath(labels_train_dir)}")
    print(f"      Labels val:   {os.path.abspath(labels_val_dir)}")
    print(f"      YAML:         {os.path.abspath(coco_yaml_path)}")
    
    print(f"\n   ✅ COCO dataset setup complete in standard YOLO format!")
    print(f"   💡 YOLO will look for:")
    print(f"      Train images: {os.path.join(base_abs, 'images', 'train')}")
    print(f"      Train labels: {os.path.join(base_abs, 'labels', 'train')}")
    print(f"      Val images:   {os.path.join(base_abs, 'images', 'val')}")
    print(f"      Val labels:   {os.path.join(base_abs, 'labels', 'val')}")
    
    # Clean up temporary directories
    print(f"\n   🗑️  Cleaning up temporary directories...")
    for temp_dir in [temp_train_dir, temp_val_dir, annotations_dir]:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"      ✅ Removed {temp_dir}")
            except Exception as e:
                print(f"      ⚠️  Could not remove {temp_dir}: {e}")
    
    return True

if __name__ == "__main__":
    success = setup_coco_yolo_format()
    sys.exit(0 if success else 1)

