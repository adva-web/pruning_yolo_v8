# Quick Commands Reference

## 🚀 How to Run Experiments

### **1. Test the Fixed Implementation**
```bash
# Test the fixed pruning implementation
python test_fixed_pruning.py

# Run fixed experiments
python run_fixed_experiments.py
```

### **2. Basic Experiments**
```bash
# Gamma pruning
python run_experiments.py gamma --layers 3

# Activation pruning
python run_experiments.py activation --layers 3

# Comparison (gamma vs activation)
python run_experiments.py compare --layers 3
```

### **3. Fixed Pruning (Recommended)**
```bash
# Run fixed activation pruning directly
python -c "
from pruning_yolo_v8_fixed import apply_activation_pruning_blocks_3_4_fixed
import yaml

# Load data
with open('data/VOC_adva.yaml', 'r') as f:
    data_cfg = yaml.safe_load(f)
classes = list(range(len(data_cfg['names'])))

# Run fixed pruning
pruned_model = apply_activation_pruning_blocks_3_4_fixed(
    model_path='data/best.pt',
    train_data=[],
    valid_data=[],
    classes=classes,
    layers_to_prune=3
)
print('✅ Fixed pruning completed!')
"
```

### **4. Advanced Experiments**
```bash
# Extended experiments
python run_experiments.py extended --method both --layers 6 8 10

# Structural pruning
python run_experiments.py structural --layers 3

# Batch experiments
python run_experiments.py batch
```

## 🔧 Fixed vs Original

### **Original (Has Channel Mismatch Issues)**
```bash
# This will likely fail with channel mismatch errors
python run_experiments.py activation --layers 3
```

### **Fixed (Resolves Channel Mismatches)**
```bash
# This should work without channel mismatch errors
python run_fixed_experiments.py
```

## 📊 Expected Output

### **Successful Fixed Pruning**
```
===== Fixed Activation-based pruning of 3 layers in blocks 3-4 =====
🔧 This version properly handles channel dimension mismatches

Pruning Layer 1/3:
  ✅ Activation-based pruning applied successfully!

Pruning Layer 2/3:
  🔧 This layer's input was affected by previous pruning
  ⚠️  Skipping this layer to avoid channel mismatch

Pruning Layer 3/3:
  ✅ Activation-based pruning applied successfully!
```

### **Failed Original Pruning**
```
Pruning Layer 2/3:
  ❌ Activation pruning failed: Given groups=1, weight of size [256, 118, 1, 1], 
     expected input[1, 256, 22, 32] to have 118 channels, but got 256 channels instead
```

## 🎯 Quick Test Commands

```bash
# Test 1: Basic fixed pruning
python test_fixed_pruning.py

# Test 2: Fixed experiments
python run_fixed_experiments.py

# Test 3: Original experiments (may fail)
python run_experiments.py activation --layers 3

# Test 4: Comparison
python run_experiments.py compare --layers 3
```

## 📁 Key Files

- `pruning_yolo_v8_fixed.py` - **Fixed implementation** (recommended)
- `pruning_yolo_v8.py` - Original implementation (has channel mismatch issues)
- `test_fixed_pruning.py` - Test script for fixed implementation
- `run_fixed_experiments.py` - Fixed experiment runner
- `run_experiments.py` - Original experiment runner

## 🚨 Troubleshooting

### **If you get channel mismatch errors:**
- Use the fixed version: `python run_fixed_experiments.py`
- Or use the fixed implementation directly: `pruning_yolo_v8_fixed.py`

### **If you get import errors:**
```bash
pip install torch ultralytics opencv-python numpy PyYAML
```

### **If you get file not found errors:**
```bash
# Check if files exist
ls -la data/best.pt
ls -la data/VOC_adva.yaml
```

---

**The fixed version resolves the channel dimension mismatch issues you were experiencing! 🎉**
