#!/usr/bin/env python3
"""
Test experiment for apply_activation_pruning_blocks_3_4 function.
This tests the activation pruning with clustering algorithm.
"""

import os
import sys
import yaml
from pathlib import Path

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from pruning_yolo_v8 import apply_activation_pruning_blocks_3_4

def load_samples_for_test(image_dir: str, label_dir: str, max_samples=100):
    """Load dataset samples for testing."""
    import cv2
    import glob
    
    samples = []
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    # Limit to max_samples for testing
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

def test_activation_pruning_experiment():
    """Test the apply_activation_pruning_blocks_3_4 function."""
    
    print("🧪 Testing Activation Pruning Experiment")
    print("=" * 60)
    print("🔧 Features:")
    print("  ✅ Uses k-medoids clustering")
    print("  ✅ Targets blocks 3-6")
    print("  ✅ No fine-tuning (faster testing)")
    print("  ✅ Detailed logging and summary")
    print("=" * 60)
    
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
    
    print(f"📥 Loading training data...")
    train_data = load_samples_for_test(train_img_dir, train_label_dir, max_samples=100)
    valid_data = load_samples_for_test(val_img_dir, val_label_dir, max_samples=50)
    print(f"✅ Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
    
    # Test with different numbers of layers
    test_cases = [1, 2, 3]
    
    for layers_to_prune in test_cases:
        print(f"\n🚀 Testing with {layers_to_prune} layer(s)...")
        print("-" * 50)
        
        try:
            # Run the activation pruning experiment
            pruned_model = apply_activation_pruning_blocks_3_4(
                model_path="data/best.pt",
                train_data=train_data,
                valid_data=valid_data,
                classes=classes,
                layers_to_prune=layers_to_prune
            )
            
            print(f"\n✅ Activation pruning completed successfully for {layers_to_prune} layer(s)!")
            
            # Final evaluation
            print(f"\n🔍 Running final evaluation...")
            final_metrics = pruned_model.val(data=data_yaml, verbose=False)
            
            print(f"\n📈 Final Results for {layers_to_prune} layer(s):")
            print(f"  mAP@0.5: {final_metrics.results_dict.get('metrics/mAP50(B)', 0):.4f}")
            print(f"  mAP@0.5:0.95: {final_metrics.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")
            print(f"  Precision: {final_metrics.results_dict.get('metrics/precision(B)', 0):.4f}")
            print(f"  Recall: {final_metrics.results_dict.get('metrics/recall(B)', 0):.4f}")
            
            # Check if pruning details are available
            if hasattr(pruned_model, 'pruned_layers_details'):
                print(f"\n📊 Pruning Summary for {layers_to_prune} layer(s):")
                print(f"{'='*60}")
                print(f"{'Layer':<8} {'Block':<6} {'Channels':<15} {'Status':<10}")
                print(f"{'-'*60}")
                
                for i, detail in enumerate(pruned_model.pruned_layers_details):
                    channels_info = f"{detail.get('original_channels', 'N/A')}→{detail.get('remaining_channels', 'N/A')}"
                    status = detail.get('status', 'unknown')
                    print(f"{i+1:<8} {detail.get('block_idx', 'N/A'):<6} {channels_info:<15} {status:<10}")
                
                print(f"{'-'*60}")
                print(f"✅ Successfully processed {len(pruned_model.pruned_layers_details)} layers")
            else:
                print(f"⚠️  No pruning details found in model")
            
            print(f"\n✅ Test case {layers_to_prune} layer(s) completed successfully!")
            
        except Exception as e:
            print(f"❌ Test case {layers_to_prune} layer(s) failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n🎉 All test cases completed!")
    print(f"   Check the results above to see the performance of activation pruning.")

if __name__ == "__main__":
    print("🧪 Starting Activation Pruning Experiment")
    print("This will test the apply_activation_pruning_blocks_3_4 function")
    print("with k-medoids clustering algorithm")
    print()
    
    test_activation_pruning_experiment()
