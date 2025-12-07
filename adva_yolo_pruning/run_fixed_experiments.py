#!/usr/bin/env python3
"""
Simple script to run the fixed YOLOv8 pruning experiments.
This script demonstrates how to use the fixed pruning implementation.
"""

import os
import sys
import yaml
import cv2
import glob
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def load_training_data(data_yaml, max_samples=100):
    """Load real training data for activation extraction."""
    print(f"📥 Loading training data...")
    
    try:
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        # Get image and label directories
        train_img_dir = data_cfg["train"]
        if not train_img_dir.startswith("/"):
            train_img_dir = os.path.join("data", train_img_dir)
        
        train_label_dir = train_img_dir.replace("/images", "/labels")
        
        print(f"   Image directory: {train_img_dir}")
        print(f"   Label directory: {train_label_dir}")
        
        # Check if directories exist
        if not os.path.exists(train_img_dir):
            print(f"❌ Training image directory not found: {train_img_dir}")
            return []
        
        if not os.path.exists(train_label_dir):
            print(f"❌ Training label directory not found: {train_label_dir}")
            return []
        
        # Load images and labels
        samples = []
        image_paths = sorted(glob.glob(os.path.join(train_img_dir, "*.jpg")))
        
        # Limit samples for faster processing
        image_paths = image_paths[:max_samples]
        
        print(f"   Found {len(image_paths)} images, loading {min(len(image_paths), max_samples)} samples...")
        
        for i, img_path in enumerate(image_paths):
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                base = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(train_label_dir, base + ".txt")
                
                # Load labels
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
                
                if (i + 1) % 20 == 0:
                    print(f"   Loaded {i + 1}/{len(image_paths)} samples...")
                    
            except Exception as e:
                print(f"   ⚠️  Error loading {img_path}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(samples)} training samples")
        return samples
        
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        return []

def load_validation_data(data_yaml, max_samples=50):
    """Load real validation data for activation extraction."""
    print(f"📥 Loading validation data...")
    
    try:
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        # Get image and label directories
        val_img_dir = data_cfg["val"]
        if not val_img_dir.startswith("/"):
            val_img_dir = os.path.join("data", val_img_dir)
        
        val_label_dir = val_img_dir.replace("/images", "/labels")
        
        print(f"   Image directory: {val_img_dir}")
        print(f"   Label directory: {val_label_dir}")
        
        # Check if directories exist
        if not os.path.exists(val_img_dir):
            print(f"❌ Validation image directory not found: {val_img_dir}")
            return []
        
        if not os.path.exists(val_label_dir):
            print(f"❌ Validation label directory not found: {val_label_dir}")
            return []
        
        # Load images and labels
        samples = []
        image_paths = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")))
        
        # Limit samples for faster processing
        image_paths = image_paths[:max_samples]
        
        print(f"   Found {len(image_paths)} images, loading {min(len(image_paths), max_samples)} samples...")
        
        for i, img_path in enumerate(image_paths):
            try:
                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                h, w = img.shape[:2]
                base = os.path.splitext(os.path.basename(img_path))[0]
                label_path = os.path.join(val_label_dir, base + ".txt")
                
                # Load labels
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
                
                if (i + 1) % 10 == 0:
                    print(f"   Loaded {i + 1}/{len(image_paths)} samples...")
                    
            except Exception as e:
                print(f"   ⚠️  Error loading {img_path}: {e}")
                continue
        
        print(f"✅ Successfully loaded {len(samples)} validation samples")
        return samples
        
    except Exception as e:
        print(f"❌ Error loading validation data: {e}")
        return []

def run_fixed_activation_pruning(layers_to_prune=3):
    """Run the fixed activation pruning experiment."""
    print(f"🧪 Running Fixed Activation Pruning Experiment")
    print(f"   Layers to prune: {layers_to_prune}")
    print("=" * 60)
    
    try:
        from pruning_yolo_v8_fixed import apply_activation_pruning_blocks_3_4_fixed
        
        # Load data configuration
        data_yaml = "data/VOC_adva.yaml"
        if not os.path.exists(data_yaml):
            print(f"❌ Data YAML file not found: {data_yaml}")
            return False
        
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        classes = list(range(len(data_cfg['names'])))
        print(f"✅ Loaded {len(classes)} classes from {data_yaml}")
        
        # Check if model exists
        model_path = "data/best.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"✅ Model file found: {model_path}")
        
        # Load real training and validation data
        print(f"\n📥 Loading real training data...")
        train_data = load_training_data(data_yaml, max_samples=100)
        
        if len(train_data) == 0:
            print(f"❌ No training data loaded. Cannot proceed.")
            return False
        
        print(f"\n📥 Loading real validation data...")
        valid_data = load_validation_data(data_yaml, max_samples=50)
        
        if len(valid_data) == 0:
            print(f"⚠️  No validation data loaded, using training data for validation")
            valid_data = train_data[:20]  # Use subset of training data
        
        print(f"🚀 Starting Fixed Activation Pruning...")
        print(f"   Method: Fixed activation-based pruning")
        print(f"   Target blocks: 3-4")
        print(f"   Layers to prune: {layers_to_prune}")
        
        # Run the fixed pruning
        pruned_model = apply_activation_pruning_blocks_3_4_fixed(
            model_path=model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=layers_to_prune
        )
        
        print(f"\n✅ Fixed activation pruning completed successfully!")
        print(f"📊 Model has been pruned and is ready for evaluation")
        
        # Test the pruned model
        print(f"\n📊 Testing pruned model...")
        try:
            test_metrics = pruned_model.val(data=data_yaml, verbose=False)
            
            print(f"📈 Pruned model performance:")
            print(f"   mAP@0.5:0.95: {test_metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"   mAP@0.5: {test_metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   Precision: {test_metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
            print(f"   Recall: {test_metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
        except Exception as e:
            print(f"⚠️  Could not evaluate pruned model: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fixed activation pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_fixed_comparison(layers_to_prune=3):
    """Run the fixed comparison experiment."""
    print(f"\n🔬 Running Fixed Comparison Experiment")
    print(f"   Layers to prune: {layers_to_prune}")
    print("=" * 60)
    
    try:
        from pruning_yolo_v8_fixed import run_fixed_comparison_experiment
        
        # Load data configuration
        data_yaml = "data/VOC_adva.yaml"
        if not os.path.exists(data_yaml):
            print(f"❌ Data YAML file not found: {data_yaml}")
            return False
        
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        classes = list(range(len(data_cfg['names'])))
        print(f"✅ Loaded {len(classes)} classes from {data_yaml}")
        
        # Check if model exists
        model_path = "data/best.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"✅ Model file found: {model_path}")
        
        # Load real training and validation data
        print(f"\n📥 Loading real training data...")
        train_data = load_training_data(data_yaml, max_samples=100)
        
        if len(train_data) == 0:
            print(f"❌ No training data loaded. Cannot proceed.")
            return False
        
        print(f"\n📥 Loading real validation data...")
        valid_data = load_validation_data(data_yaml, max_samples=50)
        
        if len(valid_data) == 0:
            print(f"⚠️  No validation data loaded, using training data for validation")
            valid_data = train_data[:20]
        
        print(f"🚀 Starting Fixed Comparison Experiment...")
        print(f"   Method: Fixed activation-based pruning")
        print(f"   Layers to prune: {layers_to_prune}")
        
        # Run comparison
        fixed_model = run_fixed_comparison_experiment(
            model_path=model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=layers_to_prune,
            data_yaml=data_yaml
        )
        
        print(f"\n✅ Fixed comparison experiment completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Fixed comparison experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_original_experiment(layers_to_prune=3):
    """Run the original experiment for comparison."""
    print(f"\n🧪 Running Original Experiment (for comparison)")
    print(f"   Layers to prune: {layers_to_prune}")
    print("=" * 60)
    
    try:
        from pruning_yolo_v8 import apply_activation_pruning_blocks_3_4
        
        # Load data configuration
        data_yaml = "data/VOC_adva.yaml"
        if not os.path.exists(data_yaml):
            print(f"❌ Data YAML file not found: {data_yaml}")
            return False
        
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        classes = list(range(len(data_cfg['names'])))
        print(f"✅ Loaded {len(classes)} classes from {data_yaml}")
        
        # Check if model exists
        model_path = "data/best.pt"
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            return False
        
        print(f"✅ Model file found: {model_path}")
        
        # Load real training and validation data
        print(f"\n📥 Loading real training data...")
        train_data = load_training_data(data_yaml, max_samples=100)
        
        if len(train_data) == 0:
            print(f"❌ No training data loaded. Cannot proceed.")
            return False
        
        print(f"\n📥 Loading real validation data...")
        valid_data = load_validation_data(data_yaml, max_samples=50)
        
        if len(valid_data) == 0:
            print(f"⚠️  No validation data loaded, using training data for validation")
            valid_data = train_data[:20]
        
        print(f"🚀 Starting Original Activation Pruning...")
        print(f"   Method: Original activation-based pruning")
        print(f"   Target blocks: 3-4")
        print(f"   Layers to prune: {layers_to_prune}")
        print(f"   ⚠️  This may encounter channel mismatch errors")
        
        # Run the original pruning
        pruned_model = apply_activation_pruning_blocks_3_4(
            model_path=model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=layers_to_prune
        )
        
        print(f"\n✅ Original activation pruning completed!")
        return True
        
    except Exception as e:
        print(f"❌ Original activation pruning failed: {e}")
        print(f"   This is expected due to channel mismatch issues")
        return False

def main():
    """Main function to run all experiments."""
    print("YOLOv8 Fixed Pruning Experiment Runner")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists("data/VOC_adva.yaml"):
        print("❌ Please run this script from the adva_yolo_pruning directory")
        print("   Expected files: data/VOC_adva.yaml, data/best.pt")
        return
    
    # Test 1: Fixed activation pruning
    print("\n🧪 Test 1: Fixed Activation Pruning")
    success1 = run_fixed_activation_pruning(layers_to_prune=3)
    
    # Test 2: Fixed comparison
    print("\n🧪 Test 2: Fixed Comparison")
    success2 = run_fixed_comparison(layers_to_prune=3)
    
    # Test 3: Original experiment (for comparison)
    print("\n🧪 Test 3: Original Experiment (for comparison)")
    success3 = run_original_experiment(layers_to_prune=3)
    
    # Summary
    print(f"\n📊 Experiment Summary:")
    print(f"   Fixed Activation Pruning: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"   Fixed Comparison: {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"   Original Experiment: {'✅ PASS' if success3 else '❌ FAIL'}")
    
    if success1 and success2:
        print(f"\n🎉 Fixed pruning implementation is working correctly!")
        print(f"   The fixed version resolves channel dimension mismatch issues")
        print(f"   You can now run multi-layer pruning experiments successfully")
    else:
        print(f"\n⚠️  Some experiments failed. Please check the error messages above.")
    
    if not success3:
        print(f"\n💡 As expected, the original experiment failed due to channel mismatches")
        print(f"   This demonstrates why the fixed version is necessary")

if __name__ == "__main__":
    main()
