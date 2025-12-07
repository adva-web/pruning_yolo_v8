# Git Setup Guide

This document outlines what files should be committed to git for the 4-methods pruning experiments.

## ✅ Files to Commit

### Core Experiment Scripts
- `adva_yolo_pruning/run_4_methods_comparison_blocks_1357_coco.py` - COCO experiments
- `adva_yolo_pruning/run_4_methods_comparison_blocks_1357.py` - VOC experiments

### Setup and Utility Scripts
- `adva_yolo_pruning/finetune_yolov8s_coco.py` - COCO fine-tuning
- `adva_yolo_pruning/setup_coco_yolo_format.py` - COCO dataset setup
- `adva_yolo_pruning/test_coco_yolo.py` - COCO validation test

### Core Pruning Modules
- `adva_yolo_pruning/pruning_yolo_v8_sequential_fix.py` - Sequential pruning utilities
- `adva_yolo_pruning/yolov8_utils.py` - YOLO utility functions
- `adva_yolo_pruning/yolo_layer_pruner.py` - Layer pruning logic
- `adva_yolo_pruning/clustering_variants.py` - K-medoid and gamma variants
- `adva_yolo_pruning/pruning_yolo_v8.py` - Base pruning functions
- `adva_yolo_pruning/c2f_utils.py` - C2f block utilities

### Configuration Files
- `adva_yolo_pruning/VOC_adva.yaml` - VOC dataset configuration (users need to update paths)
- `requirements.txt` - Python dependencies

### Documentation
- `README.md` - Main documentation
- `GIT_SETUP.md` - This file
- `.gitignore` - Git ignore rules

## ❌ Files NOT to Commit (handled by .gitignore)

### Model Files
- `*.pt` - All PyTorch model files
- `*.pth`, `*.onnx`, `*.engine`, `*.trt` - Other model formats

### Data Directories
- `data/dataset_voc/` - VOC dataset (too large)
- `data/coco/` - COCO dataset (too large)
- `data/images/`, `data/labels/` - Image and label directories
- `*.cache` - Cache files

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

## 📝 Setup Instructions for New Users

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd pruning
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare VOC dataset**
   - Download VOC dataset
   - Organize in `data/dataset_voc/` structure
   - Update `adva_yolo_pruning/VOC_adva.yaml` with your paths

4. **Prepare COCO dataset**
   - Download COCO dataset zip files
   - Place in `data/coco/` directory
   - Run `python adva_yolo_pruning/setup_coco_yolo_format.py`

5. **Prepare models**
   - For VOC: Place trained model at `data/best.pt`
   - For COCO: Run `python adva_yolo_pruning/finetune_yolov8s_coco.py` or use `yolov8s.pt`

6. **Run experiments**
   - COCO: `python adva_yolo_pruning/run_4_methods_comparison_blocks_1357_coco.py`
   - VOC: `python adva_yolo_pruning/run_4_methods_comparison_blocks_1357.py`

## 🔍 Verifying What Will Be Committed

Before committing, check what files will be added:

```bash
# Check git status
git status

# See what would be committed
git add -n .

# Review the .gitignore is working
git check-ignore -v <file-path>
```

## 📦 Initial Commit Checklist

- [ ] `.gitignore` is in place
- [ ] `README.md` is complete
- [ ] `requirements.txt` is up to date
- [ ] Core experiment scripts are included
- [ ] Supporting utility modules are included
- [ ] Configuration files are included (with note about path updates)
- [ ] No model files (`.pt`) are included
- [ ] No dataset directories are included
- [ ] No log files are included
- [ ] No cache files are included

## 🚀 First Commit Command

```bash
# Add all files (respecting .gitignore)
git add .

# Review what will be committed
git status

# Commit
git commit -m "Initial commit: 4-methods pruning experiments for VOC and COCO"

# Push to remote
git push origin main
```

