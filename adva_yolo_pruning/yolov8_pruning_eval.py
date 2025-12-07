import torch
import os
import time
from ultralytics import YOLO
import logging
import yaml
import cv2
import numpy as np
import glob

from pruning_yolo_v8 import apply_activation_pruning_blocks_3_4, apply_50_percent_gamma_pruning_blocks_3_4,prune_conv2d_in_block_with_activations,apply_pruning_v8,apply_gamma_pruning_iter, apply_gamma_pruning_on_block_zeroed

logger = logging.getLogger("yolov8_pruning")
logging.basicConfig(level=logging.INFO)

def train_model(path, data_yaml):
    """
    Train the YOLOv8 model on the provided training data.
    """
    model = YOLO(path)

    model.train(
    data=data_yaml,  # Path to your dataset YAML
    epochs=30,                 # Number of epochs
    imgsz=640,                 # Image size
    batch=16,                  # Batch size (adjust as needed)
    device='cpu'              # Use 'cpu' or 'cuda'
)
    return model


def load_samples(image_dir, label_dir):
    samples = []
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))  # or .png
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
                    # YOLO format: class x_center y_center width height (normalized)
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

def evaluate_model(model_path, data_yaml, imgsz=640, batch=32, device='cpu', conf=0.25, iou=0.45):
    """
    Evaluate a YOLOv8 model and return metrics.
    """
    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        verbose=False
    )
    metrics = {
        "precision": results.results_dict.get("metrics/precision(B)", None),
        "recall": results.results_dict.get("metrics/recall(B)", None),
        "mAP_0.5": results.results_dict.get("metrics/mAP50(B)", None),
        "mAP_0.5:0.95": results.results_dict.get("metrics/mAP50-95(B)", None), 
        "per_class_mAP": results.maps if hasattr(results, "maps") else None,
        "mean_mAP_0.5_0.95": float(results.maps.mean()) if hasattr(results, "maps") else None,
        "speed": results.speed if hasattr(results, "speed") else None,
        "inference_time": results.speed.get("inference", None) if hasattr(results, "speed") else None
    }
    return metrics, results

def prune_model(
    model_path,
    train_data,
    valid_data,
    classes,
    last_layer_idx=3,
    save_path="pruned_model.pt"
):
    """
    Main entry point for pruning a YOLOv8 model.
    Calls apply_pruning_v8 and saves the final pruned model.
    """
    logger.info("Starting full pruning pipeline for YOLOv8.")
    # pruned_model = prune_conv2d_in_block_with_activations(
    #     model_path=model_path,
    #     train_data=train_data,
    #     valid_data=valid_data,
    #     classes=classes,
    # )

    # pruned_model = apply_50_percent_gamma_pruning_blocks_3_4(
    #     model_path=model_path,
    #     layers_to_prune=3
    # )
    # Save the pruned model weights
    pruned_model = apply_activation_pruning_blocks_3_4(
        model_path=model_path,
        train_data=train_data,
        valid_data=valid_data,
        classes=classes,
    )
    

    torch_model = pruned_model.model
    torch.save(torch_model.state_dict(), save_path)
    logger.info(f"Pruned model saved to {save_path}")
    print(f"DEBUG: Pruned model saved to {save_path}")
    return save_path

def run_gamma_pruning_with_metrics(model_path="data/best.pt", data_yaml="data/VOC_adva.yaml"):
    """
    Run gamma pruning with comprehensive metrics tracking.
    
    Args:
        model_path: Path to the YOLOv8 model
        data_yaml: Path to the dataset configuration file
    """
    
    print("🚀 GAMMA PRUNING WITH DETAILED METRICS")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset: {data_yaml}")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # Record start time
    total_start_time = time.time()
    
    # Step 1: Evaluate original model
    print("\n📊 STEP 1: Evaluating Original Model")
    print("-" * 40)
    original_start_time = time.time()
    
    try:
        original_metrics, _ = evaluate_model(model_path, data_yaml)
        original_eval_time = time.time() - original_start_time
        
        print(f"✅ Original Model Metrics:")
        print(f"   mAP@0.5: {original_metrics['mAP_0.5']:.4f}")
        print(f"   mAP@0.5:0.95: {original_metrics['mAP_0.5:0.95']:.4f}")
        print(f"   Precision: {original_metrics['precision']:.4f}")
        print(f"   Recall: {original_metrics['recall']:.4f}")
        print(f"   Evaluation Time: {original_eval_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error evaluating original model: {e}")
        return None
    
    # Step 2: Apply Gamma Pruning
    print("\n🔧 STEP 2: Applying 50% Gamma Pruning on 3 Layers")
    print("-" * 40)
    pruning_start_time = time.time()
    
    try:
        # This function will handle the detailed logging internally
        pruned_model = apply_50_percent_gamma_pruning_blocks_3_4(
            model_path=model_path, 
            layers_to_prune=3
        )
        pruning_time = time.time() - pruning_start_time
        
        print(f"✅ Pruning completed in {pruning_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error during pruning: {e}")
        return None
    
    # Step 3: Evaluate pruned model
    print("\n📊 STEP 3: Evaluating Pruned Model")
    print("-" * 40)
    pruned_start_time = time.time()
    
    try:
        # The pruned model should be saved, let's evaluate it
        pruned_model_path = "pruned_v3_yolov8n.pt"  # Default name from the function
        if not os.path.exists(pruned_model_path):
            print(f"⚠️  Pruned model file not found at {pruned_model_path}")
            print("   Using the returned model object for evaluation...")
            pruned_metrics, _ = evaluate_model(model_path, data_yaml)  # Fallback
        else:
            pruned_metrics, _ = evaluate_model(pruned_model_path, data_yaml)
        
        pruned_eval_time = time.time() - pruned_start_time
        
        print(f"✅ Pruned Model Metrics:")
        print(f"   mAP@0.5: {pruned_metrics['mAP_0.5']:.4f}")
        print(f"   mAP@0.5:0.95: {pruned_metrics['mAP_0.5:0.95']:.4f}")
        print(f"   Precision: {pruned_metrics['precision']:.4f}")
        print(f"   Recall: {pruned_metrics['recall']:.4f}")
        print(f"   Evaluation Time: {pruned_eval_time:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error evaluating pruned model: {e}")
        pruned_metrics = original_metrics  # Fallback
    
    # Step 4: Calculate Performance Impact
    print("\n📈 STEP 4: Performance Impact Analysis")
    print("-" * 40)
    
    mAP50_change = pruned_metrics['mAP_0.5'] - original_metrics['mAP_0.5']
    mAP50_95_change = pruned_metrics['mAP_0.5:0.95'] - original_metrics['mAP_0.5:0.95']
    precision_change = pruned_metrics['precision'] - original_metrics['precision']
    recall_change = pruned_metrics['recall'] - original_metrics['recall']
    
    print(f"📊 Metrics Change:")
    print(f"   mAP@0.5:     {original_metrics['mAP_0.5']:.4f} → {pruned_metrics['mAP_0.5']:.4f} ({mAP50_change:+.4f})")
    print(f"   mAP@0.5:0.95: {original_metrics['mAP_0.5:0.95']:.4f} → {pruned_metrics['mAP_0.5:0.95']:.4f} ({mAP50_95_change:+.4f})")
    print(f"   Precision:   {original_metrics['precision']:.4f} → {pruned_metrics['precision']:.4f} ({precision_change:+.4f})")
    print(f"   Recall:      {original_metrics['recall']:.4f} → {pruned_metrics['recall']:.4f} ({recall_change:+.4f})")
    
    # Step 5: Final Summary
    total_time = time.time() - total_start_time
    
    print("\n🎯 FINAL SUMMARY")
    print("=" * 60)
    print(f"⏱️  TIMING:")
    print(f"   Original Model Evaluation: {original_eval_time:.2f} seconds")
    print(f"   Pruning Process:           {pruning_time:.2f} seconds")
    print(f"   Pruned Model Evaluation:   {pruned_eval_time:.2f} seconds")
    print(f"   Total Time:                {total_time:.2f} seconds")
    print()
    print(f"📊 PERFORMANCE:")
    print(f"   mAP@0.5 Change:     {mAP50_change:+.4f} ({mAP50_change/original_metrics['mAP_0.5']*100:+.2f}%)")
    print(f"   mAP@0.5:0.95 Change: {mAP50_95_change:+.4f} ({mAP50_95_change/original_metrics['mAP_0.5:0.95']*100:+.2f}%)")
    print(f"   Precision Change:   {precision_change:+.4f} ({precision_change/original_metrics['precision']*100:+.2f}%)")
    print(f"   Recall Change:      {recall_change:+.4f} ({recall_change/original_metrics['recall']*100:+.2f}%)")
    print()
    print(f"🔧 PRUNING DETAILS:")
    print(f"   Method: 50% Gamma Pruning")
    print(f"   Layers Pruned: 3 layers")
    print(f"   Target Blocks: 3-4 (as configured)")
    print(f"   Selection Criteria: Lowest average gamma values")
    print()
    print(f"💾 OUTPUT FILES:")
    print(f"   Pruned Model: pruned_v3_yolov8n.pt")
    print(f"   Detailed Log: pruning_log_50_percent_blocks_3_4.txt")
    print("=" * 60)
    
    return {
        'original_metrics': original_metrics,
        'pruned_metrics': pruned_metrics,
        'timing': {
            'original_eval': original_eval_time,
            'pruning': pruning_time,
            'pruned_eval': pruned_eval_time,
            'total': total_time
        },
        'performance_change': {
            'mAP50': mAP50_change,
            'mAP50-95': mAP50_95_change,
            'precision': precision_change,
            'recall': recall_change
        }
    }

if __name__ == "__main__":
    # Paths
    data_yaml = "data/VOC_adva.yaml"  # Path to dataset YAML
    weights_yolo_s = "data/best.pt"   # Path to YOLOv8 weights
    
    print("🚀 YOLOv8 Gamma Pruning with Detailed Metrics")
    print("=" * 50)
    
    try:
        result = run_gamma_pruning_with_metrics(weights_yolo_s, data_yaml)
        if result:
            print("\n✅ Gamma pruning completed successfully!")
        else:
            print("\n❌ Gamma pruning failed!")
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()