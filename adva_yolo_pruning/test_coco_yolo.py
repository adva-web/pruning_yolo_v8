#!/usr/bin/env python3
"""
Test YOLO validation on COCO dataset in standard YOLO format.
This script verifies that YOLO can correctly find and load labels.
"""

import os
import sys
from ultralytics import YOLO

def test_coco_yolo():
    """Test YOLO on COCO dataset."""
    print(f"\n{'='*70}")
    print("TESTING YOLO ON COCO DATASET")
    print(f"{'='*70}")
    
    coco_yaml = "data/coco/coco.yaml"
    
    if not os.path.exists(coco_yaml):
        print(f"❌ YAML file not found: {coco_yaml}")
        print(f"   Please run setup_coco_yolo_format.py first!")
        return False
    
    print(f"\n📋 Loading YAML: {coco_yaml}")
    import yaml
    import shutil
    
    with open(coco_yaml, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    train_path_yaml = yaml_data.get('train', '')
    val_path_yaml = yaml_data.get('val', '')
    
    print(f"   path: {yaml_data.get('path', 'NOT SET')}")
    print(f"   train: {train_path_yaml}")
    print(f"   val: {val_path_yaml}")
    print(f"   nc: {yaml_data.get('nc', 'NOT SET')} classes")
    
    # Check if YAML needs to be updated to standard format
    base_path = yaml_data.get('path', '')
    train_images_std = os.path.join(base_path, "images", "train")
    val_images_std = os.path.join(base_path, "images", "val")
    train_labels_std = os.path.join(base_path, "labels", "train")
    val_labels_std = os.path.join(base_path, "labels", "val")
    
    if train_path_yaml != 'images/train' or val_path_yaml != 'images/val':
        # Old format detected - check if standard format exists
        if os.path.exists(train_images_std) and os.path.exists(val_images_std):
            print(f"\n⚠️  YAML uses old format, but standard format directories exist!")
            print(f"   Auto-updating YAML to standard format...")
            
            # Create backup
            backup_yaml = coco_yaml + ".backup"
            shutil.copy(coco_yaml, backup_yaml)
            
            # Update YAML to standard format
            yaml_data['train'] = 'images/train'
            yaml_data['val'] = 'images/val'
            
            with open(coco_yaml, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            
            print(f"   ✅ YAML updated to standard format!")
            print(f"   Backup saved to: {backup_yaml}")
            train_path_yaml = 'images/train'
            val_path_yaml = 'images/val'
        else:
            print(f"\n⚠️  YAML uses old format and standard format directories don't exist!")
            print(f"   💡 Run 'python setup_coco_yolo_format.py' to convert to standard format")
    
    # Verify directory structure
    train_path = os.path.join(base_path, train_path_yaml)
    val_path = os.path.join(base_path, val_path_yaml)
    train_labels = train_labels_std
    val_labels = val_labels_std
    
    print(f"\n🔍 Verifying directory structure...")
    print(f"   Train images: {train_path} - {'✅' if os.path.exists(train_path) else '❌'}")
    print(f"   Val images:   {val_path} - {'✅' if os.path.exists(val_path) else '❌'}")
    print(f"   Train labels: {train_labels} - {'✅' if os.path.exists(train_labels) else '❌'}")
    print(f"   Val labels:   {val_labels} - {'✅' if os.path.exists(val_labels) else '❌'}")
    
    if not all([os.path.exists(train_path), os.path.exists(val_path), 
                os.path.exists(train_labels), os.path.exists(val_labels)]):
        print(f"\n❌ Directory structure incomplete!")
        return False
    
    # Count files
    train_images = [f for f in os.listdir(train_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    val_images = [f for f in os.listdir(val_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    train_label_files = [f for f in os.listdir(train_labels) if f.endswith('.txt')]
    val_label_files = [f for f in os.listdir(val_labels) if f.endswith('.txt')]
    
    print(f"\n📊 File counts:")
    print(f"   Train images: {len(train_images)}")
    print(f"   Train labels: {len(train_label_files)}")
    print(f"   Val images:   {len(val_images)}")
    print(f"   Val labels:   {len(val_label_files)}")
    
    # Check filename matching
    train_image_basenames = {os.path.splitext(f)[0] for f in train_images}
    train_label_basenames = {os.path.splitext(f)[0] for f in train_label_files}
    train_matching = train_image_basenames & train_label_basenames
    
    val_image_basenames = {os.path.splitext(f)[0] for f in val_images}
    val_label_basenames = {os.path.splitext(f)[0] for f in val_label_files}
    val_matching = val_image_basenames & val_label_basenames
    
    print(f"\n🔍 Filename matching:")
    print(f"   Train: {len(train_matching)}/{len(train_images)} images have labels")
    print(f"   Val:   {len(val_matching)}/{len(val_images)} images have labels")
    
    if len(train_matching) < len(train_images) * 0.9:
        print(f"   ⚠️  WARNING: Less than 90% of train images have labels!")
    
    if len(val_matching) < len(val_images) * 0.9:
        print(f"   ⚠️  WARNING: Less than 90% of val images have labels!")
    
    # Clear cache
    print(f"\n🗑️  Clearing YOLO cache...")
    cache_paths = [
        os.path.join(train_path, f"{os.path.basename(train_path)}.cache"),
        os.path.join(val_path, f"{os.path.basename(val_path)}.cache"),
        os.path.join(base_path, "images", "train.cache"),
        os.path.join(base_path, "images", "val.cache"),
    ]
    
    for cache_path in cache_paths:
        if os.path.exists(cache_path):
            try:
                os.remove(cache_path)
                print(f"   ✅ Cleared: {cache_path}")
            except Exception as e:
                print(f"   ⚠️  Could not clear {cache_path}: {e}")
    
    # Test YOLO validation
    print(f"\n🧪 Running YOLO validation test...")
    print(f"   This may take a few minutes...")
    
    try:
        model = YOLO("yolov8s.pt")  # Use YOLOv8s for testing
        results = model.val(data=coco_yaml, imgsz=640, verbose=True, plots=False)
        
        # Check results - YOLO validation returns results in different formats
        instances = 0
        map50 = 0
        map50_95 = 0
        
        # Try to get from results_dict (most common)
        if hasattr(results, 'results_dict'):
            # Try different possible keys
            instances = results.results_dict.get('metrics/instances', 0)
            if instances == 0:
                # Try alternative keys
                instances = results.results_dict.get('instances', 0)
            
            map50 = results.results_dict.get('metrics/mAP50(B)', 0)
            if map50 == 0:
                map50 = results.results_dict.get('metrics/mAP50', 0)
            
            map50_95 = results.results_dict.get('metrics/mAP50-95(B)', 0)
            if map50_95 == 0:
                map50_95 = results.results_dict.get('metrics/mAP50-95', 0)
                if map50_95 == 0:
                    map50_95 = results.results_dict.get('metrics/mAP', 0)
        
        # If not found, try to get from metrics object
        if instances == 0 and hasattr(results, 'metrics'):
            try:
                instances = getattr(results.metrics, 'instances', 0)
                if map50 == 0:
                    map50 = getattr(results.metrics, 'map50', 0)
                if map50_95 == 0:
                    map50_95 = getattr(results.metrics, 'map', 0)
            except:
                pass
        
        # Print results
        print(f"\n📊 Validation Results:")
        if instances > 0:
            print(f"   Instances found: {instances}")
        else:
            print(f"   Instances: (check output above - YOLO prints 'all' row with total instances)")
        
        print(f"   mAP50: {map50:.4f}")
        print(f"   mAP50-95: {map50_95:.4f}")
        
        # Success criteria: mAP values > 0 means labels were found and processed
        # The validation output shows "all" row with instances count
        if map50 > 0 or map50_95 > 0:
            print(f"\n✅ SUCCESS! YOLO found and processed labels!")
            print(f"   mAP50={map50:.4f} and mAP50-95={map50_95:.4f} indicate labels are working!")
            if instances > 0:
                print(f"   Total instances detected: {instances}")
            else:
                print(f"   Check the validation output above - look for 'all' row showing total instances")
            return True
        elif instances > 0:
            print(f"\n✅ SUCCESS! YOLO found {instances} instances - labels are working!")
            return True
        else:
            print(f"\n❌ FAILED! YOLO found 0 instances and mAP is 0 - labels not detected!")
            print(f"   Check the validation output above for errors")
            return False
            
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_coco_yolo()
    sys.exit(0 if success else 1)

