#!/usr/bin/env python3
"""
Test script for activation pruning to verify it works properly.
"""

import os
import sys
from pathlib import Path
import cv2
import glob

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from pruning_yolo_v8 import apply_activation_pruning_blocks_3_4
import yaml

def load_samples(image_dir: str, label_dir: str, max_samples=100):
    """Load dataset samples for activation pruning."""
    samples = []
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Limit to max_samples for quick testing
    image_paths = image_paths[:max_samples]
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
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

def test_activation_pruning():
    """Test the activation pruning function."""
    
    print("🧪 Testing Activation Pruning Method")
    print("=" * 50)
    
    # Check if required files exist
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML not found: {data_yaml}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"✅ Data YAML found: {data_yaml}")
    
    # Load class names
    try:
        with open(data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)
        classes_names = data_cfg["names"]
        classes = list(range(len(classes_names)))
        print(f"✅ Classes loaded: {len(classes)} classes")
    except Exception as e:
        print(f"❌ Error loading classes: {e}")
        return False
    
    # Load samples (simplified for testing)
    try:
        train_img_dir = data_cfg["train"]
        val_img_dir = data_cfg["val"]
        
        # Convert relative paths to absolute paths
        if not train_img_dir.startswith("/"):
            train_img_dir = os.path.join("data", train_img_dir)
        if not val_img_dir.startswith("/"):
            val_img_dir = os.path.join("data", val_img_dir)
            
        train_label_dir = train_img_dir.replace("/images", "/labels")
        val_label_dir = val_img_dir.replace("/images", "/labels")
        
        # Check if directories exist
        if not os.path.exists(train_img_dir):
            print(f"❌ Training images directory not found: {train_img_dir}")
            print(f"   Current working directory: {os.getcwd()}")
            print(f"   Available directories in data/: {os.listdir('data') if os.path.exists('data') else 'data/ not found'}")
            return False
        if not os.path.exists(val_img_dir):
            print(f"❌ Validation images directory not found: {val_img_dir}")
            return False
            
        print(f"✅ Training images directory found: {train_img_dir}")
        print(f"✅ Validation images directory found: {val_img_dir}")
        
        # Load actual training data for activation pruning (limited for quick testing)
        print("📥 Loading training data for activation extraction...")
        train_data = load_samples(train_img_dir, train_label_dir, max_samples=50)
        valid_data = load_samples(val_img_dir, val_label_dir, max_samples=20)
        
        print(f"✅ Loaded {len(train_data)} training samples (limited for testing)")
        print(f"✅ Loaded {len(valid_data)} validation samples (limited for testing)")
        
    except Exception as e:
        print(f"❌ Error setting up data directories: {e}")
        return False
    
    # Test activation pruning
    try:
        print("\n🚀 Starting Activation Pruning Test...")
        print("   Method: Original activation extraction algorithm")
        print("   Layers: 2 (for quick testing)")
        print("   Blocks: 1-5")
        
        pruned_model = apply_activation_pruning_blocks_3_4(
            model_path=model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=2,  # Small number for quick testing
            data_yaml=data_yaml
        )
        
        print("\n✅ Activation pruning test completed successfully!")
        
        # Check if pruned model has layer details
        if hasattr(pruned_model, 'pruned_layers_details'):
            layer_details = pruned_model.pruned_layers_details
            print(f"✅ Layer details captured: {len(layer_details)} layers")
            
            for i, detail in enumerate(layer_details):
                print(f"   Layer {i+1}: Block {detail.get('block_idx')}, "
                      f"Conv #{detail.get('conv_in_block_idx')}, "
                      f"Channels: {detail.get('original_channels')} → {detail.get('remaining_channels')}")
        else:
            print("⚠️  No layer details found in pruned model")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during activation pruning test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_activation_pruning()
    
    if success:
        print("\n🎉 Original activation pruning algorithm is working properly!")
        print("   You can now use it in your experiments.")
    else:
        print("\n💥 Original activation pruning algorithm has issues.")
        print("   Please check the error messages above.")
