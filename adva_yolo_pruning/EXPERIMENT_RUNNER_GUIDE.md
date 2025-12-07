# YOLOv8 Pruning Experiments - Complete Runner Guide

## 🚀 Quick Start - How to Run Experiments

### 1. **Basic Fixed Pruning (Recommended)**

```bash
# Test the fixed pruning implementation
python test_fixed_pruning.py

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
    train_data=[],  # Will be loaded internally
    valid_data=[],
    classes=classes,
    layers_to_prune=3
)
print('✅ Fixed pruning completed!')
"
```

### 2. **Using the Experiment Runner**

```bash
# Run gamma pruning experiment
python run_experiments.py gamma --layers 3

# Run activation pruning experiment  
python run_experiments.py activation --layers 3

# Run comparison experiment (gamma vs activation)
python run_experiments.py compare --layers 3

# Run extended experiments (6, 8, 10, 12 layers)
python run_experiments.py extended --method both --layers 6 8 10

# Run batch experiments (comprehensive)
python run_experiments.py batch
```

### 3. **Fixed Pruning Experiments**

```bash
# Test the fixed implementation
python test_fixed_pruning.py

# Run fixed comparison
python -c "
from pruning_yolo_v8_fixed import run_fixed_comparison_experiment
import yaml

with open('data/VOC_adva.yaml', 'r') as f:
    data_cfg = yaml.safe_load(f)
classes = list(range(len(data_cfg['names'])))

result = run_fixed_comparison_experiment(
    model_path='data/best.pt',
    train_data=[],
    valid_data=[],
    classes=classes,
    layers_to_prune=3,
    data_yaml='data/VOC_adva.yaml'
)
"
```

## 📋 Available Experiment Commands

### **Basic Experiments**

```bash
# Gamma pruning
python run_experiments.py gamma --layers 3 --blocks 3 4 5

# Activation pruning
python run_experiments.py activation --layers 3 --blocks 3 4 5

# Comparison (gamma vs activation)
python run_experiments.py compare --layers 3 --blocks 3 4 5
```

### **Advanced Experiments**

```bash
# Extended layer experiments
python run_experiments.py extended --method gamma --layers 6 8 10 12
python run_experiments.py extended --method activation --layers 6 8 10 12
python run_experiments.py extended --method both --layers 6 8 10 12

# Structural pruning
python run_experiments.py structural --layers 3

# Structural comparison (gamma soft vs activation structural)
python run_experiments.py structural-compare --layers 3

# Channel fix experiments
python run_experiments.py channel-fix --layers 3
python run_experiments.py robust-fix --layers 3

# Batch experiments (comprehensive)
python run_experiments.py batch
```

### **Custom Experiments**

```bash
# Custom experiment names
python run_experiments.py gamma --layers 4 --name "my_gamma_experiment"

# Different block combinations
python run_experiments.py gamma --layers 3 --blocks 1 2 3
python run_experiments.py gamma --layers 3 --blocks 2 4 5
python run_experiments.py gamma --layers 3 --blocks 1 3 5

# Extended block combinations
python run_experiments.py gamma --layers 6 --blocks 1 2 3 4
python run_experiments.py gamma --layers 8 --blocks 1 2 3 4 5
```

## 🔧 Fixed Pruning Implementation

### **Using the Fixed Version**

The fixed version resolves channel dimension mismatches:

```python
from pruning_yolo_v8_fixed import apply_activation_pruning_blocks_3_4_fixed

# Run fixed pruning
pruned_model = apply_activation_pruning_blocks_3_4_fixed(
    model_path="data/best.pt",
    train_data=train_data,
    valid_data=valid_data,
    classes=classes,
    layers_to_prune=3
)
```

### **Key Benefits of Fixed Version**

1. **No Channel Mismatches** - Prevents "expected input to have X channels, but got Y channels" errors
2. **Smart Layer Selection** - Skips layers affected by previous pruning
3. **Proper Channel Adjustment** - Adjusts subsequent layers correctly
4. **Clear Status Reporting** - Shows which layers were pruned, skipped, or failed

## 📊 Expected Output Examples

### **Successful Fixed Pruning**

```
===== Fixed Activation-based pruning of 3 layers in blocks 3-4 =====
🔧 This version properly handles channel dimension mismatches

Pruning Layer 1/3:
  - Block: 3
  - Conv in block index: 0
  - Original channels: 256
  🔍 Extracting activations...
  📊 Activation analysis complete:
    - Total channels: 256
    - Channels to keep: 118
    - Channels to remove: 138
    - Pruning ratio: 53.9%
  🔧 Adjusting next Conv2d layer (Block 6, Conv 0)
    Input channels: 256 → 118
    ✅ Successfully adjusted input channels
  ✅ Activation-based pruning applied successfully!

Pruning Layer 2/3:
  - Block: 3
  - Conv in block index: 0
  - Original channels: 128
  🔧 This layer's input was affected by previous pruning
  ⚠️  Skipping this layer to avoid channel mismatch

Pruning Layer 3/3:
  - Block: 4
  - Conv in block index: 0
  - Original channels: 128
  🔍 Extracting activations...
  📊 Activation analysis complete:
    - Total channels: 128
    - Channels to keep: 64
    - Channels to remove: 64
    - Pruning ratio: 50.0%
  ✅ Activation-based pruning applied successfully!
```

### **Status Summary**

```
Detailed Fixed Activation-Based Pruning Summary:
================================================================================
Layer    Block  Conv#   Original#  Channels         Status    
--------------------------------------------------------------------------------
1         3      0       6         256→118         success   
2         3      0       6         128→128         skipped  
3         4      0       7         128→64          success   
--------------------------------------------------------------------------------
Overall Statistics:
  Layers pruned: 3
  Total channels before: 512
  Total channels after: 310
  Overall pruning ratio: 39.5%
```

## 🎯 Experiment Types Explained

### **1. Gamma Pruning**
- **Method**: Uses BatchNorm gamma values to identify less important channels
- **Best for**: Aggressive pruning with significant size reduction
- **Command**: `python run_experiments.py gamma --layers 3`

### **2. Activation Pruning**
- **Method**: Uses activation patterns to determine channel importance
- **Best for**: General purpose pruning with good performance retention
- **Command**: `python run_experiments.py activation --layers 3`

### **3. Fixed Activation Pruning**
- **Method**: Same as activation pruning but with channel mismatch fixes
- **Best for**: Multi-layer pruning without errors
- **Command**: Use `pruning_yolo_v8_fixed.py` directly

### **4. Structural Pruning**
- **Method**: True architectural pruning that modifies model structure
- **Best for**: When you need actual model size reduction
- **Command**: `python run_experiments.py structural --layers 3`

### **5. Comparison Experiments**
- **Method**: Runs both gamma and activation on the same configuration
- **Best for**: Determining which method works better for your use case
- **Command**: `python run_experiments.py compare --layers 3`

## 📁 Output Files

### **Results Files**
- `experiment_results/` - Main results directory
- `{experiment_name}_results.json` - Detailed JSON results
- `{experiment_name}_results.csv` - Tabular CSV results
- `pruned_models/` - Saved pruned model files

### **Log Files**
- `pruning_log_activation_blocks_3_4_fixed.txt` - Fixed pruning logs
- `pruning_log_50_percent_blocks_3_4.txt` - Gamma pruning logs
- `pruning_log_activation_blocks_3_4.txt` - Activation pruning logs

## 🚀 Quick Test Commands

### **Test 1: Basic Fixed Pruning**
```bash
python test_fixed_pruning.py
```

### **Test 2: Simple Gamma Experiment**
```bash
python run_experiments.py gamma --layers 3
```

### **Test 3: Simple Activation Experiment**
```bash
python run_experiments.py activation --layers 3
```

### **Test 4: Comparison Experiment**
```bash
python run_experiments.py compare --layers 3
```

### **Test 5: Extended Experiments**
```bash
python run_experiments.py extended --method gamma --layers 6 8
```

## 🔧 Troubleshooting

### **Common Issues**

1. **Model Not Found**
   ```bash
   # Check if model exists
   ls -la data/best.pt
   # If not, download or use correct path
   ```

2. **Dataset Not Found**
   ```bash
   # Check dataset structure
   ls -la data/dataset_voc/images/train/
   ls -la data/dataset_voc/labels/train/
   ```

3. **GPU Memory Issues**
   ```bash
   # Use smaller model or reduce batch size
   python run_experiments.py gamma --layers 2  # Fewer layers
   ```

4. **Import Errors**
   ```bash
   # Check dependencies
   pip install -r requirements.txt
   # Or install individually
   pip install torch ultralytics opencv-python numpy PyYAML
   ```

### **Getting Help**

1. **Check Logs**: Look at the generated log files for detailed error messages
2. **Test Imports**: Run `python -c "import torch; print(torch.cuda.is_available())"`
3. **Verify Files**: Ensure all required files exist in the correct locations
4. **Check GPU**: Make sure CUDA is available for faster processing

## 📈 Performance Tips

### **For Faster Experiments**
- Use fewer layers: `--layers 2` instead of `--layers 3`
- Use smaller models: `yolov8n.pt` instead of `yolov8s.pt`
- Reduce sample counts in the configuration

### **For Better Results**
- Use GPU if available
- Ensure sufficient RAM (16GB+ recommended)
- Use SSD storage for faster I/O
- Monitor GPU memory usage during training

## 🎉 Success Indicators

### **Successful Experiment Output**
```
✅ Experiment completed successfully!
📊 Results saved to experiment_results/
📈 Performance metrics calculated
💾 Pruned model saved
```

### **Failed Experiment Output**
```
❌ Experiment failed: [error message]
⚠️  Check logs for details
🔄 Try with fewer layers or different configuration
```

---

**Happy Experimenting! 🚀**

The fixed version should resolve your channel dimension mismatch issues and allow you to successfully run multi-layer pruning experiments.
