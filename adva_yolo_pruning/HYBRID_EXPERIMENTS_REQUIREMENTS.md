# Hybrid Pruning Experiments - Required Files

## Overview
Two hybrid pruning experiments that combine activation-based pruning (for regular Conv blocks) and gamma-based pruning (for C2f blocks).

## Experiment Files

### 1. `run_hybrid_pruning_blocks_123.py`
- Prunes blocks 1, 2, 3
- Block 1, 3: Activation pruning
- Block 2: Gamma pruning

### 2. `run_hybrid_pruning_blocks_12345.py`
- Prunes blocks 1, 2, 3, 4, 5
- Blocks 1, 3, 5: Activation pruning
- Blocks 2, 4: Gamma pruning

## Required Python Files (Dependencies)

All these files must be in the same directory (`adva_yolo_pruning/`):

### Core Utility Files
1. **`pruning_yolo_v8_sequential_fix.py`**
   - Provides: `load_training_data`, `load_validation_data`, `count_active_channels`

2. **`yolov8_utils.py`**
   - Provides: `get_all_conv2d_layers`, `build_mini_net`, `get_raw_objects_debug_v8`, `aggregate_activations_from_matches`

3. **`yolo_layer_pruner.py`**
   - Provides: `YoloLayerPruner` class

4. **`pruning_yolo_v8.py`**
   - Provides: `prune_conv2d_in_block_with_activations`, `select_optimal_components`

5. **`c2f_utils.py`**
   - Provides: `is_c2f_block` function

### Supporting Files (Indirect Dependencies)
6. **`clustering.py`** or **`clustering_variants.py`**
   - Used by `prune_conv2d_in_block_with_activations` for channel selection

## Required Data Files

### Model File
- **`data/best.pt`** - Pre-trained YOLOv8 model
  - Path used in scripts: `data/best.pt`
  - Alternative location: `data/model/best.pt` (may need path adjustment)

### Dataset Configuration
- **`data/VOC_adva.yaml`** - Dataset configuration file
  - Must contain:
    - `train`: path to training images
    - `val`: path to validation images
    - `names`: list of class names

### Dataset Directory Structure
```
data/
├── best.pt
├── VOC_adva.yaml
└── dataset_voc/
    ├── images/
    │   ├── train/  (training images)
    │   └── val/    (validation images)
    └── labels/
        ├── train/  (training labels in YOLO format)
        └── val/    (validation labels in YOLO format)
```

## Required Python Packages

Install via `pip install`:

```bash
pip install ultralytics
pip install torch
pip install torchvision
pip install numpy
pip install pyyaml
```

Or ensure you have a `requirements.txt` with:
```
ultralytics>=8.0.0
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
pyyaml>=5.4.0
```

## File Structure

```
adva_yolo_pruning/
├── run_hybrid_pruning_blocks_123.py      # Experiment 1
├── run_hybrid_pruning_blocks_12345.py    # Experiment 2
├── pruning_yolo_v8_sequential_fix.py     # Required
├── yolov8_utils.py                       # Required
├── yolo_layer_pruner.py                  # Required
├── pruning_yolo_v8.py                    # Required
├── c2f_utils.py                          # Required
├── clustering.py                         # Required (or clustering_variants.py)
├── data/
│   ├── best.pt                          # Required
│   ├── VOC_adva.yaml                    # Required
│   └── dataset_voc/                     # Required
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
└── runs/
    └── detect/                          # Created automatically
        ├── hybrid_pruning_blocks_123.pt    # Output from Exp 1
        └── hybrid_pruning_blocks_12345.pt  # Output from Exp 2
```

## Quick Start

1. **Verify all files exist:**
   ```bash
   cd /Users/advahelman/Code/pruning/adva_yolo_pruning
   ls -la *.py  # Check all Python dependencies exist
   ls -la data/best.pt  # Check model exists
   ls -la data/VOC_adva.yaml  # Check config exists
   ```

2. **Run Experiment 1:**
   ```bash
   python run_hybrid_pruning_blocks_123.py
   ```

3. **Run Experiment 2:**
   ```bash
   python run_hybrid_pruning_blocks_12345.py
   ```

## Output Files

Both experiments create:
- **Pruned model**: `runs/detect/hybrid_pruning_blocks_XXX.pt`
- **Console output**: Detailed pruning results, inference time comparison, summary statistics

## What Gets Created Automatically

- `runs/detect/` directory (if it doesn't exist)
- `temp_activation_pruning.pt` (temporary file, deleted automatically)
- Fine-tuning checkpoints in `runs/detect/train*/`

## Troubleshooting

### Missing Import Error
If you get `ModuleNotFoundError`, check:
- All Python files are in the same directory
- `sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))` is working

### Missing Data Error
If you get file not found errors:
- Verify `data/best.pt` exists
- Verify `data/VOC_adva.yaml` exists
- Check paths in the scripts match your directory structure

### C2f Detection Issues
- Ensure `c2f_utils.py` exists and contains `is_c2f_block` function
- The function should detect C2f blocks by class name

## Notes

- Both experiments use **50% pruning ratio** by default
- Fine-tuning: **3 epochs after each layer** + **20 epochs at the end**
- Inference time is measured **before pruning** and **after fine-tuning**
- Models are saved automatically after completion

