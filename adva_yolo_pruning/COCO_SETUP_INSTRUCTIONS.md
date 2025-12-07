# COCO Dataset Setup Instructions

## Step 1: Set up COCO in Standard YOLO Format

First, run the setup script to extract and organize COCO data:

```bash
cd /Users/advahelman/Code/pruning
python adva_yolo_pruning/setup_coco_yolo_format.py
```

This will:
- Extract COCO zip files (train2017.zip, val2017.zip, annotations_trainval2017.zip)
- Organize into standard YOLO structure:
  - `data/coco/images/train/`
  - `data/coco/images/val/`
  - `data/coco/labels/train/`
  - `data/coco/labels/val/`
- Create `data/coco/coco.yaml` with correct format

**Prerequisites:**
- Make sure you have the COCO zip files in `data/coco/`:
  - `train2017.zip`
  - `val2017.zip`
  - `annotations_trainval2017.zip`

## Step 2: Test YOLO on the Dataset

After setup, test that YOLO can find labels:

```bash
python adva_yolo_pruning/test_coco_yolo.py
```

This will:
- Verify directory structure
- Check filename matching
- Clear cache
- Run YOLO validation
- Report if labels are found correctly

**Expected output:**
- ✅ Instances found > 0
- ✅ Validation metrics (mAP50, mAP50-95)

## Step 3: Run Your Main Experiment

Once setup and test pass, run your main comparison:

```bash
python adva_yolo_pruning/run_4_methods_comparison_blocks_1357_coco.py
```

The script will automatically detect the standard YOLO format and use it.

## Troubleshooting

### If setup fails:
1. Check that zip files exist in `data/coco/`
2. Check disk space (COCO is large)
3. Check file permissions

### If test fails:
1. Verify directory structure matches expected format
2. Check that label files exist and are not empty
3. Check filename matching (image.jpg → image.txt)

### If main experiment fails:
1. Make sure you ran setup first
2. Check that `data/coco/coco.yaml` exists
3. Verify paths in YAML are correct

## Directory Structure After Setup

```
data/coco/
├── coco.yaml              # YAML config file
├── images/
│   ├── train/            # Training images
│   └── val/              # Validation images
└── labels/
    ├── train/            # Training labels (.txt files)
    └── val/              # Validation labels (.txt files)
```

## YAML Format

The created `coco.yaml` will look like:

```yaml
path: /absolute/path/to/data/coco
train: images/train
val: images/val
nc: 80
names: ['person', 'bicycle', ...]
```

This is the standard YOLO format that YOLO expects!

