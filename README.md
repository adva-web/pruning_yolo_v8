# YOLOv8 Pruning: 4 Methods Comparison Experiments

This repository contains comprehensive experiments comparing 4 different pruning methods on YOLOv8 models using both VOC and COCO datasets.

## 📋 Overview

The experiments compare 4 pruning methods:
1. **Activation with Max Weight** - Original activation-based method using maximum weight selection
2. **Activation with K-Medoid** - Geometric center selection using k-medoid clustering
3. **Activation with Max Gamma** - BatchNorm gamma-based channel selection combined with activations
4. **Pure Gamma Pruning** - BatchNorm gamma magnitude-based pruning only

All methods prune Conv 0 from blocks 1, 3, 5, 7 (4 layers total) and include fine-tuning for 20 epochs at the end, and 5 epochs between each pruning layer

## 🎯 Features

- **4 Pruning Methods**: Comprehensive comparison of different channel selection strategies
- **2 Datasets**: Experiments on both VOC and COCO datasets
- **Comprehensive Metrics**: 
  - mAP@0.5:0.95 and mAP@0.5
  - Precision and Recall
  - Inference time and speedup
  - FLOPs reduction
  - Model sparsity
- **Automatic Setup**: Scripts to prepare datasets and models

## 📁 Project Structure

```
pruning/
├── adva_yolo_pruning/
│   ├── run_4_methods_comparison_blocks_1357_coco.py  # COCO experiments
│   ├── run_4_methods_comparison_blocks_1357.py      # VOC experiments
│   ├── finetune_yolov8s_coco.py                     # COCO fine-tuning
│   ├── setup_coco_yolo_format.py                    # COCO dataset setup
│   ├── test_coco_yolo.py                            # COCO validation test
│   ├── pruning_yolo_v8_sequential_fix.py            # Core pruning utilities
│   ├── yolov8_utils.py                              # YOLO utilities
│   ├── yolo_layer_pruner.py                         # Layer pruning logic
│   ├── clustering_variants.py                       # K-medoid and gamma variants
│   ├── pruning_yolo_v8.py                           # Base pruning functions
│   ├── c2f_utils.py                                 # C2f block utilities
│   └── VOC_adva.yaml                                # VOC dataset config
└── README.md                                         # This file
```

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install torch torchvision torchaudio
pip install ultralytics
pip install opencv-python
pip install numpy
pip install pyyaml
pip install thop  # For FLOPs calculation (optional)
```

### 2. Prepare Datasets

#### For VOC Dataset:

1. Organize your VOC dataset in the following structure:
```
data/
└── dataset_voc/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

2. Update `adva_yolo_pruning/VOC_adva.yaml` with your dataset paths:
```yaml
train: /path/to/data/dataset_voc/images/train
val: /path/to/data/dataset_voc/images/val
nc: 20  # number of classes
names: [aeroplane, bicycle, bird, ...]
```

**Note**: The `VOC_adva.yaml` file in the repository contains example paths. You must update them to match your local dataset location.

#### For COCO Dataset:

1. Download COCO dataset (or use existing):
   - `train2017.zip` - Training images
   - `val2017.zip` - Validation images
   - `annotations_trainval2017.zip` - Annotations

2. Place zip files in `data/coco/` directory

3. Run setup script:
```bash
cd adva_yolo_pruning
python setup_coco_yolo_format.py
```

4. Verify COCO setup:
```bash
python test_coco_yolo.py
```

### 3. Prepare Models

#### For VOC:
- Place your trained model at `data/best.pt` or use a pre-trained YOLOv8 model

#### For COCO:
- Option 1: Use pre-trained YOLOv8s (auto-downloaded):
  ```bash
  # The script will use yolov8s.pt if fine-tuned model not found
  ```

- Option 2: Fine-tune on COCO first (not tested yet 07/12/25):
  ```bash
  cd adva_yolo_pruning
  python finetune_yolov8s_coco.py
  ```
  This creates `data/best_coco.pt` for better results.

### 4. Run Experiments

#### Run COCO Experiments:
```bash
cd adva_yolo_pruning
python run_4_methods_comparison_blocks_1357_coco.py
```

#### Run VOC Experiments:
```bash
cd adva_yolo_pruning
python run_4_methods_comparison_blocks_1357.py
```

## 📊 Understanding the Results

Each experiment outputs:

### 1. Pruning Statistics
- Success rate for each method
- Total channels removed
- Pruning ratio percentage

### 2. Performance Metrics
- **mAP@0.5:0.95**: Mean Average Precision at IoU 0.5:0.95
- **mAP@0.5**: Mean Average Precision at IoU 0.5
- **Precision**: Detection precision
- **Recall**: Detection recall

### 3. Inference Time
- Inference time per image (ms)
- Speedup compared to original model

### 4. FLOPs Analysis
- FLOPs in GFLOPs
- Reduction percentage
- Absolute reduction in GFLOPs

### 5. Sparsity Analysis
- Model sparsity percentage
- Increase in sparsity
- Zero parameters count
- Total parameters count

### 6. Detailed Layer Information
- Per-layer pruning results for each method
- Channels removed per layer
- Pruning ratio per layer

## 🔧 Configuration

### Experiment Parameters

You can modify the following in the experiment scripts:

- **Target Blocks**: Currently set to blocks 1, 3, 5, 7
- **Fine-tuning Epochs**: Default 20 epochs (final) + 5 epochs (per-layer)
- **Pruning Ratio**: Determined by each method's selection criteria

### Method Details

1. **Activation with Max Weight** (`activation_max_weight`)
   - Uses activation patterns to identify important channels
   - Selects channels with maximum activation weights

2. **Activation with K-Medoid** (`activation_k_medoid`)
   - Uses k-medoid clustering on activation patterns
   - Selects geometric centers of activation clusters

3. **Activation with Max Gamma** (`activation_max_gamma`)
   - Combines activation patterns with BatchNorm gamma values
   - Selects channels with high activation AND high gamma

4. **Pure Gamma Pruning** (`pure_gamma`)
   - Uses only BatchNorm gamma magnitude
   - Selects channels with highest gamma values

## 📝 Output Files

Experiments generate:
- Pruned model files (`.pt` format)
- Evaluation metrics (printed to console)
- Training logs (if verbose mode enabled)

## 🐛 Troubleshooting

### Common Issues

1. **Model Not Found**
   ```
   ❌ Model file not found: data/best_coco.pt
   ```
   **Solution**: 
   - For COCO: Run `python finetune_yolov8s_coco.py` or ensure `yolov8s.pt` is available
   - For VOC: Ensure `data/best.pt` exists or update model path in script

2. **Dataset Not Found**
   ```
   ❌ Dataset validation failed
   ```
   **Solution**: 
   - For COCO: Run `python setup_coco_yolo_format.py` first
   - For VOC: Check `VOC_adva.yaml` paths and dataset structure

3. **CUDA Out of Memory**
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: 
   - Reduce batch size in fine-tuning
   - Use CPU instead of GPU (slower but works)
   - Use smaller model (yolov8n.pt instead of yolov8s.pt)

4. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'ultralytics'
   ```
   **Solution**: Install missing dependencies:
   ```bash
   pip install ultralytics torch opencv-python numpy pyyaml
   ```

5. **COCO Labels Not Found**
   ```
   ⚠️  Could not find labels for image
   ```
   **Solution**: 
   - Verify COCO setup: `python test_coco_yolo.py`
   - Check that labels are in `data/coco/labels/train/` and `data/coco/labels/val/`
   - Ensure YAML uses standard format: `train: images/train`, `val: images/val`

## 📚 Dependencies

### Required
- `torch` >= 1.9.0
- `ultralytics` >= 8.0.0
- `opencv-python` >= 4.5.0
- `numpy` >= 1.19.0
- `pyyaml` >= 5.4.0

### Optional
- `thop` - For FLOPs calculation (install with `pip install thop`)

## 🎓 Citation

If you use this code in your research, please cite:

```bibtex
@software{yolov8_pruning_4methods,
  title = {YOLOv8 Pruning: 4 Methods Comparison},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/pruning}
}
```

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines if applicable]

## 📧 Contact

[Add contact information if desired]

---

**Note**: This repository focuses on the 4-methods comparison experiments. For other pruning experiments and utilities, see the individual script files in the `adva_yolo_pruning/` directory.

