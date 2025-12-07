# Files to Upload to Git

This document lists exactly which files will be committed to git for the 4-methods pruning experiments.

## ✅ Main Experiment Files (Will be uploaded)

### Core 4-Methods Comparison Scripts
- `adva_yolo_pruning/run_4_methods_comparison_blocks_1357_coco.py` - **COCO experiments** (main file)
- `adva_yolo_pruning/run_4_methods_comparison_blocks_1357.py` - **VOC experiments** (main file)

### Setup and Utility Scripts
- `adva_yolo_pruning/finetune_yolov8s_coco.py` - Fine-tune YOLOv8s on COCO
- `adva_yolo_pruning/setup_coco_yolo_format.py` - Setup COCO dataset in YOLO format
- `adva_yolo_pruning/test_coco_yolo.py` - Test COCO dataset setup

### Core Pruning Modules (Required dependencies)
- `adva_yolo_pruning/pruning_yolo_v8_sequential_fix.py` - Sequential pruning utilities
- `adva_yolo_pruning/yolov8_utils.py` - YOLO utility functions
- `adva_yolo_pruning/yolo_layer_pruner.py` - Layer pruning logic
- `adva_yolo_pruning/clustering_variants.py` - K-medoid and gamma variants
- `adva_yolo_pruning/pruning_yolo_v8.py` - Base pruning functions
- `adva_yolo_pruning/c2f_utils.py` - C2f block utilities

### Configuration Files
- `adva_yolo_pruning/VOC_adva.yaml` - VOC dataset configuration

### Documentation
- `README.md` - Main documentation
- `GIT_SETUP.md` - Git setup guide
- `FILES_TO_UPLOAD.md` - This file
- `requirements.txt` - Python dependencies
- `.gitignore` - Git ignore rules

### Other Supporting Files (if they exist)
- Any other `.py` files in `adva_yolo_pruning/` that are imported by the main scripts
- Any `.md` documentation files in `adva_yolo_pruning/` that explain the methods

## ❌ Files NOT Uploaded (Excluded by .gitignore)

### YOLOv7 Related (Excluded)
- `yolov7/` - Entire YOLOv7 directory
- `inspect_yolov7.py` - YOLOv7 inspection script

### Model Files (Too large)
- `*.pt` - All PyTorch model files (yolov7.pt, yolov8*.pt, yolov9*.pt, yolo11*.pt, etc.)
- `*.pth`, `*.onnx`, `*.engine`, `*.trt` - Other model formats

### Data Directories (Too large)
- `data/dataset_voc/` - VOC dataset
- `data/coco/` - COCO dataset
- `data/images/`, `data/labels/` - Image and label directories

### Output Directories
- `runs/` - Training runs
- `experiment_results/` - Experiment outputs
- `pruned_models/` - Saved pruned models
- `model/` - Model storage
- `pruning/` - Pruning outputs

### Logs and Temporary Files
- `*.log`, `*.txt` - Log files
- `pruning_log*.txt` - Pruning logs
- `__pycache__/` - Python cache
- `*.pyc`, `*.pyo` - Compiled Python files

## 📊 Summary

**What WILL be uploaded:**
- ✅ 2 main experiment scripts (COCO and VOC)
- ✅ ~10-15 supporting Python modules
- ✅ Configuration files (YAML)
- ✅ Documentation files (README, guides)
- ✅ Requirements file

**What will NOT be uploaded:**
- ❌ YOLOv7 directory and related files
- ❌ Model files (all .pt files)
- ❌ Dataset directories
- ❌ Log files and outputs
- ❌ Cache and temporary files

## 🔍 Verify Before Committing

Run this to see what will be committed:
```bash
git status
```

You should see:
- ✅ Python scripts in `adva_yolo_pruning/`
- ✅ README.md, requirements.txt, .gitignore
- ❌ NO yolov7/ directory
- ❌ NO .pt model files
- ❌ NO data/ directories
- ❌ NO log files

