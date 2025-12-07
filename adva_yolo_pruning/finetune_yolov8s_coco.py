#!/usr/bin/env python3
"""
Fine-tune YOLOv8s on COCO dataset.
This will create a model trained on COCO that can be used for activation-based pruning.
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO

def main():
    # Paths
    coco_yaml = "data/coco/coco.yaml"
    base_model = "yolov8s.pt"  # Pre-trained COCO model
    output_model = "data/best_coco.pt"  # Output fine-tuned model
    
    # Check if COCO YAML exists
    if not os.path.exists(coco_yaml):
        print(f"❌ COCO YAML not found: {coco_yaml}")
        print("💡 Run 'python setup_coco_yolo_format.py' first to set up COCO dataset")
        return False
    
    # Check if base model exists
    if not os.path.exists(base_model):
        print(f"⚠️  Base model not found: {base_model}")
        print("💡 Downloading YOLOv8s pre-trained model...")
        # YOLO will auto-download if not found
        try:
            model = YOLO(base_model)
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    else:
        model = YOLO(base_model)
    
    print(f"\n{'='*70}")
    print("FINE-TUNING YOLOv8s ON COCO DATASET")
    print(f"{'='*70}")
    print(f"📦 Base model: {base_model}")
    print(f"📁 Dataset: {coco_yaml}")
    print(f"💾 Output: {output_model}")
    print(f"{'='*70}\n")
    
    # Fine-tune on COCO
    # Note: This will take a while depending on your hardware
    try:
        results = model.train(
            data=coco_yaml,
            epochs=50,              # Number of epochs (adjust as needed)
            imgsz=640,             # Image size
            batch=16,               # Batch size (adjust based on GPU memory)
            device=0,               # GPU device (use 'cpu' if no GPU)
            project="runs/detect",  # Project directory
            name="coco_finetune",   # Experiment name
            save=True,              # Save checkpoints
            save_period=10,         # Save checkpoint every N epochs
            val=True,               # Validate during training
            plots=True,             # Generate plots
            verbose=True            # Verbose output
        )
        
        # Save the best model
        best_model_path = results.save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            import shutil
            os.makedirs(os.path.dirname(output_model), exist_ok=True)
            shutil.copy(best_model_path, output_model)
            print(f"\n✅ Fine-tuning complete!")
            print(f"   Best model saved to: {output_model}")
            print(f"   You can now use this model for pruning experiments")
            return True
        else:
            print(f"\n⚠️  Fine-tuning complete, but best.pt not found at expected location")
            print(f"   Check: {results.save_dir}/weights/")
            return False
            
    except Exception as e:
        print(f"\n❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

