#!/usr/bin/env python3
"""
Test script for the channel adjustment fix.
This script tests the new activation pruning function that includes channel mismatch fixes.
"""

import sys
import os
import yaml

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def load_samples(img_dir, label_dir, max_samples=100):
    """Load samples for activation extraction."""
    import cv2
    import torch
    from pathlib import Path
    
    samples = []
    img_path = Path(img_dir)
    label_path = Path(label_dir)
    
    if not img_path.exists():
        print(f"❌ Training images directory not found: {img_dir}")
        return []
    
    if not label_path.exists():
        print(f"❌ Training labels directory not found: {label_dir}")
        return []
    
    img_files = list(img_path.glob("*.jpg"))[:max_samples]
    print(f"📥 Loading {len(img_files)} samples from {img_dir}")
    
    for img_file in img_files:
        label_file = label_path / (img_file.stem + ".txt")
        if label_file.exists():
            # Load image
            img = cv2.imread(str(img_file))
            if img is not None:
                samples.append({
                    'image_path': str(img_file),
                    'label_path': str(label_file)
                })
    
    print(f"✅ Loaded {len(samples)} valid samples")
    return samples

def test_channel_fix():
    """Test the channel adjustment fix with 3 layers."""
    print("🧪 Testing Channel Adjustment Fix")
    print("=" * 50)
    
    try:
        # Import the new function
        from channel_adjustment_fix import apply_activation_pruning_with_channel_fix
        
        # Load data configuration
        data_yaml = "data/VOC_adva.yaml"
        with open(data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)
        
        classes_names = data_cfg["names"]
        classes = list(range(len(classes_names)))
        
        # Load samples
        train_img_dir = data_cfg["train"]
        val_img_dir = data_cfg["val"]
        
        # Convert relative paths to absolute paths
        if not train_img_dir.startswith("/"):
            train_img_dir = os.path.join("data", train_img_dir)
        if not val_img_dir.startswith("/"):
            val_img_dir = os.path.join("data", val_img_dir)
            
        train_label_dir = train_img_dir.replace("/images", "/labels")
        val_label_dir = val_img_dir.replace("/images", "/labels")
        
        train_data = load_samples(train_img_dir, train_label_dir, max_samples=100)
        valid_data = load_samples(val_img_dir, val_label_dir, max_samples=50)
        
        print(f"📥 Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
        
        if len(train_data) == 0:
            print("❌ No training data loaded. Cannot proceed.")
            return False
        
        # Test with 3 layers (the problematic case)
        print(f"\n🚀 Testing with 3 layers...")
        pruned_model = apply_activation_pruning_with_channel_fix(
            model_path="data/best.pt",
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=3,
            data_yaml=data_yaml
        )
        
        print(f"✅ Channel fix test completed successfully!")
        
        # Evaluate the pruned model
        print(f"\n📊 Evaluating pruned model...")
        metrics = pruned_model.val(data=data_yaml, verbose=False)
        
        print(f"📈 Results:")
        print(f"  mAP@0.5:0.95: {metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  mAP@0.5: {metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  Precision: {metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall: {metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Channel fix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_channel_fix()
    if success:
        print(f"\n🎉 Channel adjustment fix test PASSED!")
        print(f"The fix successfully handles multi-layer pruning without channel mismatches.")
    else:
        print(f"\n💥 Channel adjustment fix test FAILED!")
        print(f"Further debugging is needed.")
