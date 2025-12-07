#!/usr/bin/env python3
"""
Fix COCO YAML to use standard YOLO format.
This script converts the old nested structure (train2017/train2017) to standard format (images/train, labels/train).
"""

import os
import yaml
import shutil
from pathlib import Path

def fix_coco_yaml():
    """Fix COCO YAML to use standard YOLO format."""
    coco_yaml = "data/coco/coco.yaml"
    
    if not os.path.exists(coco_yaml):
        print(f"❌ YAML file not found: {coco_yaml}")
        return False
    
    # Read current YAML
    with open(coco_yaml, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    print(f"📋 Current YAML structure:")
    print(f"   path: {yaml_data.get('path', 'NOT SET')}")
    print(f"   train: {yaml_data.get('train', 'NOT SET')}")
    print(f"   val: {yaml_data.get('val', 'NOT SET')}")
    
    base_path = yaml_data.get('path', 'data/coco')
    train_path = yaml_data.get('train', '')
    val_path = yaml_data.get('val', '')
    
    # Check if already in standard format
    if train_path == 'images/train' and val_path == 'images/val':
        print("✅ YAML already in standard format!")
        return True
    
    # Check if standard format directories exist
    train_images_standard = os.path.join(base_path, "images", "train")
    val_images_standard = os.path.join(base_path, "images", "val")
    train_labels_standard = os.path.join(base_path, "labels", "train")
    val_labels_standard = os.path.join(base_path, "labels", "val")
    
    if os.path.exists(train_images_standard) and os.path.exists(val_images_standard):
        print("✅ Standard format directories found!")
        print("   Updating YAML to use standard format...")
        
        # Create backup
        backup_yaml = coco_yaml + ".backup"
        shutil.copy(coco_yaml, backup_yaml)
        print(f"   Created backup: {backup_yaml}")
        
        # Update YAML to standard format
        yaml_data['train'] = 'images/train'
        yaml_data['val'] = 'images/val'
        
        with open(coco_yaml, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ YAML updated to standard format!")
        print(f"   train: images/train")
        print(f"   val: images/val")
        return True
    
    # If standard format doesn't exist, check old format and update YAML to work with it
    print("⚠️  Standard format not found. Checking old format...")
    
    train_path_old = os.path.join(base_path, train_path)
    val_path_old = os.path.join(base_path, val_path)
    
    train_labels_old = os.path.join(train_path_old, "labels")
    val_labels_old = os.path.join(val_path_old, "labels")
    
    if os.path.exists(train_labels_old) and os.path.exists(val_labels_old):
        print("✅ Old format labels found!")
        print("   The YAML should work, but YOLO might not find labels.")
        print("   💡 Recommendation: Run 'python setup_coco_yolo_format.py' to convert to standard format")
        return False
    
    print("❌ Neither standard nor old format labels found!")
    return False

if __name__ == "__main__":
    success = fix_coco_yaml()
    if success:
        print("\n✅ YAML fixed! You can now run the experiment.")
    else:
        print("\n⚠️  YAML not fixed. Please run 'python setup_coco_yolo_format.py' to convert to standard format.")

