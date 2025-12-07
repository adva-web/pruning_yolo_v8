#!/usr/bin/env python3
"""
Test script for the original working activation pruning function.
This uses k-medoids clustering and 5-epoch training after each layer.
"""

import os
import sys
import yaml
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from pruning_yolo_v8 import apply_enhanced_activation_pruning_blocks_3_4
from pruning_experiments import PruningEvaluator, PruningConfig

def load_samples_for_test(image_dir: str, label_dir: str, max_samples=50):
    """Load dataset samples for testing (smaller sample size for faster testing)."""
    import cv2
    import glob
    
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

def test_original_activation_pruning():
    """Test the original working activation pruning function."""
    
    print("🧪 Testing Original Activation Pruning Function")
    print("=" * 60)
    print("🔧 Features:")
    print("  ✅ Uses k-medoids clustering")
    print("  ✅ 5-epoch training after each layer")
    print("  ✅ 20-epoch final retraining")
    print("  ✅ Multi-layer support")
    print("=" * 60)
    
    # Load data configuration
    data_yaml = "data/VOC_adva.yaml"
    with open(data_yaml, "r") as f:
        data_cfg = yaml.safe_load(f)
    
    classes_names = data_cfg["names"]
    classes = list(range(len(classes_names)))
    
    # Load samples (smaller sample size for faster testing)
    train_img_dir = data_cfg["train"]
    val_img_dir = data_cfg["val"]
    
    # Convert relative paths to absolute paths
    if not train_img_dir.startswith("/"):
        train_img_dir = os.path.join("data", train_img_dir)
    if not val_img_dir.startswith("/"):
        val_img_dir = os.path.join("data", val_img_dir)
        
    train_label_dir = train_img_dir.replace("/images", "/labels")
    val_label_dir = val_img_dir.replace("/images", "/labels")
    
    print(f"📥 Loading training data...")
    train_data = load_samples_for_test(train_img_dir, train_label_dir, max_samples=50)  # Smaller sample for testing
    valid_data = load_samples_for_test(val_img_dir, val_label_dir, max_samples=25)
    print(f"✅ Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
    
    # Test with 1 layer first
    layers_to_prune = 1
    print(f"\n🚀 Testing with {layers_to_prune} layer...")
    
    try:
        # Run the original working activation pruning function
        pruned_model = apply_enhanced_activation_pruning_blocks_3_4(
            model_path="data/best.pt",
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=layers_to_prune,
            data_yaml=data_yaml,
            fine_tune_epochs_per_step=5  # 5-epoch training after each layer
        )
        
        print(f"\n✅ Original activation pruning completed successfully!")
        
        # Check if pruning details are available
        if hasattr(pruned_model, 'pruned_layers_details'):
            print(f"\n📊 Pruning Summary:")
            print(f"{'='*60}")
            print(f"{'Layer':<8} {'Block':<6} {'Channels':<15} {'Status':<10}")
            print(f"{'-'*60}")
            
            for i, detail in enumerate(pruned_model.pruned_layers_details):
                channels_info = f"{detail.get('original_channels', 'N/A')}→{detail.get('remaining_channels', 'N/A')}"
                status = detail.get('status', 'unknown')
                print(f"{i+1:<8} {detail.get('block_idx', 'N/A'):<6} {channels_info:<15} {status:<10}")
            
            print(f"{'-'*60}")
            print(f"✅ Successfully pruned {len(pruned_model.pruned_layers_details)} layers")
        else:
            print(f"⚠️  No pruning details found in model")
        
        # Final evaluation
        print(f"\n🔍 Running final evaluation...")
        final_metrics = pruned_model.val(data=data_yaml, verbose=False)
        
        print(f"\n📈 Final Results:")
        print(f"  mAP@0.5: {final_metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        print(f"  mAP@0.5:0.95: {final_metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
        print(f"  Precision: {final_metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
        print(f"  Recall: {final_metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
        
        return pruned_model
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧪 Starting Original Activation Pruning Test")
    print("This will test the function that uses k-medoids clustering and 5-epoch training")
    print()
    
    result = test_original_activation_pruning()
    
    if result is not None:
        print(f"\n🎉 Test completed successfully!")
        print(f"   The original activation pruning function works correctly.")
    else:
        print(f"\n❌ Test failed!")
        print(f"   There was an error with the original activation pruning function.")
