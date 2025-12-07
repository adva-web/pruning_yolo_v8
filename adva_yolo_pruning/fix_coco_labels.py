#!/usr/bin/env python3
"""
Fix COCO label files by removing annotations with class IDs > 79.
Keeps only classes 0-79 in all label files.
"""

import os
import sys
from pathlib import Path

def fix_label_file(label_path):
    """Fix a single label file by removing classes > 79."""
    if not os.path.exists(label_path):
        return 0, 0
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        valid_lines = []
        removed_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) >= 5:
                try:
                    class_id = int(parts[0])
                    if class_id >= 0 and class_id <= 79:
                        valid_lines.append(line + '\n')
                    else:
                        removed_count += 1
                except ValueError:
                    # Invalid format, skip
                    removed_count += 1
            else:
                # Invalid format, skip
                removed_count += 1
        
        # Write back the cleaned file
        with open(label_path, 'w') as f:
            f.writelines(valid_lines)
        
        return len(valid_lines), removed_count
    
    except Exception as e:
        print(f"   ⚠️  Error processing {label_path}: {e}")
        return 0, 0

def fix_coco_labels():
    """Fix all COCO label files."""
    print(f"\n{'='*70}")
    print("FIXING COCO LABEL FILES")
    print(f"{'='*70}")
    
    base_dir = "data/coco"
    labels_train_dir = os.path.join(base_dir, "labels", "train")
    labels_val_dir = os.path.join(base_dir, "labels", "val")
    
    if not os.path.exists(labels_train_dir):
        print(f"❌ Train labels directory not found: {labels_train_dir}")
        return False
    
    if not os.path.exists(labels_val_dir):
        print(f"❌ Val labels directory not found: {labels_val_dir}")
        return False
    
    # Process train labels
    print(f"\n📝 Processing train labels: {labels_train_dir}")
    train_files = [f for f in os.listdir(labels_train_dir) if f.endswith('.txt')]
    print(f"   Found {len(train_files)} label files")
    
    train_total_valid = 0
    train_total_removed = 0
    train_files_fixed = 0
    
    for label_file in train_files:
        label_path = os.path.join(labels_train_dir, label_file)
        valid_count, removed_count = fix_label_file(label_path)
        
        if removed_count > 0:
            train_files_fixed += 1
            train_total_valid += valid_count
            train_total_removed += removed_count
            if train_files_fixed <= 10:  # Show first 10
                print(f"   Fixed {label_file}: kept {valid_count}, removed {removed_count} annotations")
    
    if train_files_fixed > 10:
        print(f"   ... and {train_files_fixed - 10} more files")
    
    print(f"\n   ✅ Train: Fixed {train_files_fixed} files")
    print(f"      Kept: {train_total_valid} annotations")
    print(f"      Removed: {train_total_removed} annotations (class > 79)")
    
    # Process val labels
    print(f"\n📝 Processing val labels: {labels_val_dir}")
    val_files = [f for f in os.listdir(labels_val_dir) if f.endswith('.txt')]
    print(f"   Found {len(val_files)} label files")
    
    val_total_valid = 0
    val_total_removed = 0
    val_files_fixed = 0
    
    for label_file in val_files:
        label_path = os.path.join(labels_val_dir, label_file)
        valid_count, removed_count = fix_label_file(label_path)
        
        if removed_count > 0:
            val_files_fixed += 1
            val_total_valid += valid_count
            val_total_removed += removed_count
            if val_files_fixed <= 10:  # Show first 10
                print(f"   Fixed {label_file}: kept {valid_count}, removed {removed_count} annotations")
    
    if val_files_fixed > 10:
        print(f"   ... and {val_files_fixed - 10} more files")
    
    print(f"\n   ✅ Val: Fixed {val_files_fixed} files")
    print(f"      Kept: {val_total_valid} annotations")
    print(f"      Removed: {val_total_removed} annotations (class > 79)")
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"   Total files fixed: {train_files_fixed + val_files_fixed}")
    print(f"   Total annotations kept: {train_total_valid + val_total_valid}")
    print(f"   Total annotations removed: {train_total_removed + val_total_removed}")
    print(f"\n✅ All label files now contain only classes 0-79")
    
    return True

if __name__ == "__main__":
    success = fix_coco_labels()
    sys.exit(0 if success else 1)

