# How YOLOv8 (Ultralytics) Loads Data

## Overview
YOLOv8 uses a YAML configuration file to locate images and automatically finds corresponding label files based on the image directory structure.

## YAML File Format

### Option 1: With `path` field (relative paths)
```yaml
path: /absolute/path/to/dataset/root
train: train/images          # Relative to 'path'
val: val/images              # Relative to 'path'
nc: 80
names: ['class1', 'class2', ...]
```

**How YOLO resolves paths:**
- Train images: `/absolute/path/to/dataset/root/train/images/`
- Val images: `/absolute/path/to/dataset/root/val/images/`
- Train labels: `/absolute/path/to/dataset/root/train/labels/` (replaces "images" with "labels")
- Val labels: `/absolute/path/to/dataset/root/val/labels/` (replaces "images" with "labels")

### Option 2: Absolute paths (no `path` field)
```yaml
train: /absolute/path/to/train/images
val: /absolute/path/to/val/images
nc: 80
names: ['class1', 'class2', ...]
```

**How YOLO resolves paths:**
- Train images: `/absolute/path/to/train/images/`
- Val images: `/absolute/path/to/val/images/`
- Train labels: `/absolute/path/to/train/labels/` (replaces "images" with "labels")
- Val labels: `/absolute/path/to/val/labels/` (replaces "images" with "labels")

### Option 3: Labels in same directory as images
If images are in `/path/to/images/` (no "images" subdirectory), YOLO looks for labels in:
- `/path/to/images/labels/` (same directory, with "labels" subdirectory)

## Key Rules for Label Location

1. **Standard structure (with "images" in path):**
   - Images: `/dataset/train/images/`
   - Labels: `/dataset/train/labels/` (YOLO replaces "images" → "labels")

2. **Alternative structure (no "images" in path):**
   - Images: `/dataset/train/`
   - Labels: `/dataset/train/labels/` (YOLO adds "labels" subdirectory)

3. **Label file naming:**
   - Image: `image_001.jpg`
   - Label: `image_001.txt` (same base name, `.txt` extension)
   - Must be in the labels directory

## How YOLO Finds Labels

When YOLO processes a YAML file:

1. **Reads YAML** to get train/val paths
2. **Resolves paths** (absolute or relative to `path` field)
3. **Scans image directory** for image files (`.jpg`, `.png`, etc.)
4. **Determines label directory:**
   - If path contains `/images/`, replaces it with `/labels/`
   - Otherwise, appends `/labels/` to the image directory path
5. **Matches labels to images:**
   - For each image `image_001.jpg`, looks for `image_001.txt` in labels directory
   - If label file exists and has content → image has annotations
   - If label file missing or empty → image is "background" (no objects)

## Common Issues

### Issue 1: Labels not found
**Symptom:** `WARNING ⚠️ No labels found in ...cache`

**Causes:**
- Label directory doesn't exist
- Label files don't match image filenames
- Label files are empty
- Path resolution is incorrect

**Solution:**
- Verify label directory exists: `<image_dir>/labels/`
- Check filename matching: `image.jpg` → `image.txt`
- Ensure label files have content (not empty)
- Use absolute paths in YAML for clarity

### Issue 2: All images marked as "backgrounds"
**Symptom:** `0 images, 5000 backgrounds`

**Causes:**
- Labels exist but YOLO can't find them (path issue)
- Label files are empty
- Label files have invalid format

**Solution:**
- Check YAML path resolution
- Verify labels are in correct location
- Check label file format (YOLO format: `class_id x_center y_center width height`)
- Clear cache files (`.cache` files)

### Issue 3: Path resolution confusion
**Symptom:** YOLO looks in wrong directory

**Solution:**
- Use absolute paths in YAML (most reliable)
- Or ensure `path` field is correct and train/val are relative to it
- Print YAML content to verify paths

## Example: COCO Dataset Structure

### Directory Structure:
```
data/coco/
├── train2017/
│   └── train2017/          # Images here
│       ├── 000000000139.jpg
│       ├── 000000000285.jpg
│       └── labels/         # Labels here
│           ├── 000000000139.txt
│           └── 000000000285.txt
└── val2017/
    └── val2017/            # Images here
        ├── 000000000632.jpg
        └── labels/         # Labels here
            └── 000000000632.txt
```

### YAML Configuration:
```yaml
train: /absolute/path/to/data/coco/train2017/train2017
val: /absolute/path/to/data/coco/val2017/val2017
nc: 80
names: ['person', 'bicycle', ...]
```

**YOLO will look for labels in:**
- Train: `/absolute/path/to/data/coco/train2017/train2017/labels/`
- Val: `/absolute/path/to/data/coco/val2017/val2017/labels/`

## Debugging Tips

1. **Print YAML content** to see what YOLO reads
2. **Check cache files** - YOLO caches dataset info in `.cache` files
3. **Verify label directory** exists and has `.txt` files
4. **Check filename matching** - image and label must have same base name
5. **Test with small subset** - verify with few images first
6. **Clear cache** - delete `.cache` files to force re-scan

## Cache Files

YOLO creates cache files (`.cache`) to speed up dataset loading. These cache:
- List of images found
- List of labels found
- Image/label matching

**Location:** Usually in the same directory as images, named `<directory_name>.cache`

**To clear:** Delete `.cache` files to force YOLO to re-scan the dataset


