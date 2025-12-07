# YOLOv8 Pruning Experiments Package

A comprehensive package for running and comparing different YOLOv8 pruning methods with configurable parameters.

## Features

- **Configurable Pruning Methods**: Gamma pruning and Activation pruning
- **Flexible Layer Selection**: Choose how many layers to prune
- **Block Targeting**: Specify which blocks to target for pruning
- **Batch Experiments**: Run multiple experiments automatically
- **Detailed Metrics**: Comprehensive before/after performance comparison
- **Export Results**: Save results to CSV and JSON formats
- **Easy-to-Use Interface**: Simple command-line interface

## Quick Start

### 1. Single Gamma Pruning Experiment

```bash
# Run gamma pruning on 3 layers (default)
python run_experiments.py gamma

# Run gamma pruning on 4 layers
python run_experiments.py gamma --layers 4

# Run gamma pruning on specific blocks
python run_experiments.py gamma --layers 3 --blocks 3 4

# Run with custom experiment name
python run_experiments.py gamma --layers 3 --name "my_gamma_experiment"
```

### 2. Single Activation Pruning Experiment

```bash
# Run activation pruning on 3 layers (default)
python run_experiments.py activation

# Run activation pruning on 2 layers
python run_experiments.py activation --layers 2

# Run activation pruning on specific blocks
python run_experiments.py activation --layers 3 --blocks 4 5
```

### 3. Comparison Experiments

```bash
# Compare gamma vs activation on 3 layers
python run_experiments.py compare

# Compare gamma vs activation on 4 layers
python run_experiments.py compare --layers 4

# Compare on specific blocks
python run_experiments.py compare --layers 3 --blocks 3 4
```

### 4. Extended Experiments (6, 8, 10, 12 layers)

```bash
# Run extended experiments with both methods
python run_experiments.py extended

# Run extended experiments with gamma only
python run_experiments.py extended --method gamma

# Run extended experiments with activation only
python run_experiments.py extended --method activation

# Run specific layer counts
python run_experiments.py extended --layers 6 8 10

# Run extended experiments with custom layer counts
python run_experiments.py extended --method both --layers 6 10 12
```

### 5. Batch Experiments

```bash
# Run comprehensive batch of experiments (includes all configurations)
python run_experiments.py batch
```

## Advanced Usage

### Using the Full Package

```python
from pruning_experiments import PruningConfig, PruningEvaluator

# Create custom configuration
config = PruningConfig(
    method="gamma",
    layers_to_prune=3,
    target_blocks=[3, 4, 5],
    experiment_name="custom_experiment",
    model_path="data/best.pt",
    data_yaml="data/VOC_adva.yaml"
)

# Run experiment
evaluator = PruningEvaluator(config)
result = evaluator.run_single_experiment()
evaluator.save_results()
evaluator.print_summary()
```

### Running Multiple Experiments

```python
from pruning_experiments import PruningConfig, PruningEvaluator

# Create multiple configurations
configs = [
    PruningConfig(method="gamma", layers_to_prune=2, experiment_name="gamma_2"),
    PruningConfig(method="gamma", layers_to_prune=3, experiment_name="gamma_3"),
    PruningConfig(method="activation", layers_to_prune=2, experiment_name="activation_2"),
    PruningConfig(method="activation", layers_to_prune=3, experiment_name="activation_3"),
]

# Run batch experiments
evaluator = PruningEvaluator(configs[0])
results = evaluator.run_batch_experiments(configs)
evaluator.print_summary()
```

## Configuration

### PruningConfig Parameters

- `method`: "gamma" or "activation"
- `layers_to_prune`: Number of layers to prune (1-5)
- `target_blocks`: List of block indices to target (e.g., [3, 4, 5])
- `experiment_name`: Unique name for the experiment
- `model_path`: Path to the YOLOv8 model file
- `data_yaml`: Path to the dataset configuration file
- `output_dir`: Directory to save results
- `pruning_percentage`: Percentage of channels to prune (for gamma method)
- `evaluate_before`: Whether to evaluate original model
- `evaluate_after`: Whether to evaluate pruned model
- `save_model`: Whether to save the pruned model

### Example Configurations

```python
# Gamma pruning on 3 layers
config = PruningConfig(
    method="gamma",
    layers_to_prune=3,
    target_blocks=[3, 4, 5],
    experiment_name="gamma_3_layers"
)

# Activation pruning on 2 layers, specific blocks
config = PruningConfig(
    method="activation",
    layers_to_prune=2,
    target_blocks=[4, 5],
    experiment_name="activation_blocks_4_5"
)

# Custom experiment with different parameters
config = PruningConfig(
    method="gamma",
    layers_to_prune=4,
    target_blocks=[3, 4, 5],
    pruning_percentage=30.0,
    experiment_name="gamma_4_layers_30_percent"
)
```

## Output

### Results Structure

Each experiment produces:

1. **Detailed Console Output**: Real-time progress and metrics
2. **JSON Results**: Complete experiment data (`experiment_name_results.json`)
3. **CSV Results**: Tabular format for analysis (`experiment_name_results.csv`)
4. **Pruned Models**: Saved model files (if `save_model=True`)
5. **Log Files**: Detailed pruning logs

### Metrics Tracked

- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5-0.95
- **Precision**: Detection precision
- **Recall**: Detection recall
- **Timing**: Evaluation and pruning times
- **Pruning Details**: Layer information, channel counts, block details

### Sample Output

```
🚀 GAMMA PRUNING WITH DETAILED METRICS
============================================================
Model: data/best.pt
Dataset: data/VOC_adva.yaml
GPU Available: True
GPU Device: Quadro RTX 6000
============================================================

📊 STEP 1: Evaluating Original Model
----------------------------------------
✅ Original Model Metrics:
   mAP@0.5: 0.8542
   mAP@0.5:0.95: 0.7234
   Precision: 0.8912
   Recall: 0.8456
   Evaluation Time: 45.23 seconds

🔧 STEP 2: Applying 50% Gamma Pruning on 3 Layers
----------------------------------------
[Detailed pruning process output...]
✅ Pruning completed in 234.56 seconds

📊 STEP 3: Evaluating Pruned Model
----------------------------------------
✅ Pruned Model Metrics:
   mAP@0.5: 0.8234
   mAP@0.5:0.95: 0.6987
   Precision: 0.8734
   Recall: 0.8123
   Evaluation Time: 42.15 seconds

🎯 FINAL SUMMARY
============================================================
⏱️  TIMING:
   Original Model Evaluation: 45.23 seconds
   Pruning Process:           234.56 seconds
   Pruned Model Evaluation:   42.15 seconds
   Total Time:                322.94 seconds

📊 PERFORMANCE:
   mAP@0.5 Change:     -0.0308 (-3.61%)
   mAP@0.5:0.95 Change: -0.0247 (-3.41%)
   Precision Change:   -0.0178 (-2.00%)
   Recall Change:      -0.0333 (-3.94%)

🔧 PRUNING DETAILS:
   Method: 50% Gamma Pruning
   Layers Pruned: 3 layers
   Target Blocks: 3-4 (as configured)
   Selection Criteria: Lowest average gamma values

💾 OUTPUT FILES:
   Pruned Model: pruned_v3_yolov8n.pt
   Detailed Log: pruning_log_50_percent_blocks_3_4.txt
============================================================
```

## File Structure

```
adva_yolo_pruning/
├── pruning_experiments.py      # Main experiment package
├── run_experiments.py          # Simple runner script
├── experiment_config.yaml      # Configuration file
├── EXPERIMENTS_README.md       # This file
├── data/
│   ├── best.pt                 # YOLOv8 model
│   └── VOC_adva.yaml           # Dataset configuration
└── experiment_results/         # Output directory
    ├── experiment_name_results.json
    ├── experiment_name_results.csv
    └── pruned_models/
```

## Examples

### Example 1: Quick Comparison

```bash
# Compare gamma vs activation pruning on 3 layers
python run_experiments.py compare --layers 3
```

### Example 2: Systematic Layer Analysis

```bash
# Test different layer counts with gamma pruning
python run_experiments.py gamma --layers 2
python run_experiments.py gamma --layers 3
python run_experiments.py gamma --layers 4
```

### Example 3: Block-Specific Analysis

```bash
# Test different block combinations
python run_experiments.py gamma --layers 3 --blocks 3 4
python run_experiments.py gamma --layers 3 --blocks 4 5
python run_experiments.py gamma --layers 3 --blocks 3 5
```

### Example 4: Extended Layer Analysis

```bash
# Test higher layer counts (6, 8, 10, 12 layers)
python run_experiments.py extended --method gamma
python run_experiments.py extended --method activation

# Test specific layer combinations
python run_experiments.py extended --layers 6 10
```

### Example 5: Block Combination Analysis

```bash
# Test different block combinations with 3 layers
python run_experiments.py gamma --layers 3 --blocks 0 1 2
python run_experiments.py gamma --layers 3 --blocks 1 2 3
python run_experiments.py gamma --layers 3 --blocks 2 3 4
python run_experiments.py gamma --layers 3 --blocks 0 2 4
python run_experiments.py gamma --layers 3 --blocks 1 3 5
```

### Example 6: Comprehensive Analysis

```bash
# Run all experiments (includes extended layer counts and block combinations)
python run_experiments.py batch
```

## Tips

1. **Start Small**: Begin with single experiments to understand the system
2. **Use Comparison Mode**: Compare gamma vs activation on the same configuration
3. **Check Results**: Always review the CSV output for easy analysis
4. **GPU Usage**: Ensure CUDA is available for faster processing
5. **Storage**: Pruned models can be large, monitor disk space
6. **Logs**: Check detailed logs for any issues during pruning

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure `data/best.pt` exists
2. **Dataset Not Found**: Check `data/VOC_adva.yaml` and dataset paths
3. **GPU Memory**: Reduce batch size or use CPU if GPU memory is insufficient
4. **Import Errors**: Ensure all dependencies are installed

### Getting Help

- Check the console output for detailed error messages
- Review the generated log files
- Ensure all required files are in the correct locations
- Verify GPU availability with `torch.cuda.is_available()`
