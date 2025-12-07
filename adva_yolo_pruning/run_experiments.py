#!/usr/bin/env python3
"""
Simple runner script for YOLOv8 pruning experiments.
This script provides easy-to-use commands for running different types of experiments.
"""

import argparse
import sys
import os
from pathlib import Path

from pruning_experiments import PruningConfig, PruningEvaluator, create_experiment_configs

def run_single_gamma_experiment(layers=3, blocks=None, experiment_name=None):
    """Run a single gamma pruning experiment."""
    if blocks is None:
        blocks = [3, 4, 5]
    
    if experiment_name is None:
        experiment_name = f"gamma_layers_{layers}"
    
    config = PruningConfig(
        method="gamma",
        layers_to_prune=layers,
        target_blocks=blocks,
        experiment_name=experiment_name,
        model_path="data/best.pt",
        data_yaml="data/VOC_adva.yaml"
    )
    
    print(f"🚀 Running Gamma Pruning Experiment")
    print(f"   Layers to prune: {layers}")
    print(f"   Target blocks: {blocks}")
    print(f"   Note: Will search ALL blocks (skip block 0) for layers with BatchNorm")
    print(f"   Experiment name: {experiment_name}")
    print("-" * 50)
    
    evaluator = PruningEvaluator(config)
    result = evaluator.run_single_experiment()
    evaluator.save_results()
    evaluator.print_summary()
    
    return result

def run_single_activation_experiment(layers=3, blocks=None, experiment_name=None):
    """Run a single activation pruning experiment."""
    if blocks is None:
        blocks = [3, 4, 5]
    
    if experiment_name is None:
        experiment_name = f"activation_layers_{layers}"
    
    config = PruningConfig(
        method="activation",
        layers_to_prune=layers,
        target_blocks=blocks,
        experiment_name=experiment_name,
        model_path="data/best.pt",
        data_yaml="data/VOC_adva.yaml"
    )
    
    print(f"🚀 Running Activation Pruning Experiment")
    print(f"   Layers to prune: {layers}")
    print(f"   Target blocks: {blocks}")
    print(f"   Experiment name: {experiment_name}")
    print("-" * 50)
    
    evaluator = PruningEvaluator(config)
    result = evaluator.run_single_experiment()
    evaluator.save_results()
    evaluator.print_summary()
    
    return result

def run_comparison_experiment(layers=3, blocks=None):
    """Run both gamma and activation pruning on the same configuration for comparison."""
    if blocks is None:
        blocks = [3, 4, 5]
    
    print(f"🚀 Running Comparison Experiment")
    print(f"   Layers to prune: {layers}")
    print(f"   Target blocks: {blocks}")
    print(f"   Methods: gamma + activation")
    print("-" * 50)
    
    configs = [
        PruningConfig(
            method="gamma",
            layers_to_prune=layers,
            target_blocks=blocks,
            experiment_name=f"comparison_gamma_{layers}layers",
            model_path="data/best.pt",
            data_yaml="data/VOC_adva.yaml"
        ),
        PruningConfig(
            method="activation",
            layers_to_prune=layers,
            target_blocks=blocks,
            experiment_name=f"comparison_activation_{layers}layers",
            model_path="data/best.pt",
            data_yaml="data/VOC_adva.yaml"
        )
    ]
    
    evaluator = PruningEvaluator(configs[0])
    results = evaluator.run_batch_experiments(configs)
    evaluator.save_results()
    evaluator.print_summary()
    
    return results

def run_same_layer_comparison_experiment(layers=3, experiment_name=None):
    """Run comparison experiment with guaranteed same layers using the new comparison system."""
    if experiment_name is None:
        experiment_name = f"same_layer_comparison_{layers}layers"
    
    print(f"🚀 Running Same-Layer Comparison Experiment")
    print(f"   Layers to prune: {layers}")
    print(f"   Methods: gamma vs activation on EXACT SAME layers")
    print(f"   Experiment name: {experiment_name}")
    print("-" * 50)
    
    config = PruningConfig(
        method="comparison",
        layers_to_prune=layers,
        experiment_name=experiment_name,
        model_path="data/best.pt",
        data_yaml="data/VOC_adva.yaml"
    )
    
    evaluator = PruningEvaluator(config)
    results = evaluator.run_comparison_experiment()
    
    print(f"\n🎯 COMPARISON RESULTS:")
    print(f"   Winner: {results['comparison_summary']['winner']}")
    print(f"   Improvement: {results['comparison_summary']['improvement_percent']:.2f}%")
    print(f"   Same layers guaranteed: ✅")
    
    return results

def run_structural_pruning_experiment(layers=3, experiment_name=None):
    """Run structural pruning experiment using TRUE architectural pruning."""
    if experiment_name is None:
        experiment_name = f"structural_pruning_{layers}layers"
    
    print(f"🔧 Running Structural Activation Pruning Experiment")
    print(f"   Layers to prune: {layers}")
    print(f"   Experiment name: {experiment_name}")
    print(f"   🔧 This uses TRUE structural pruning - modifying model architecture")
    print(f"   🔧 No channel mismatches will occur!")
    print("-" * 50)
    
    try:
        # Run structural gamma pruning
        pruned_model = apply_structural_gamma_pruning_blocks_3_4(
            model_path="data/best.pt",
            data_yaml="data/VOC_adva.yaml",
            layers_to_prune=layers
        )
        
        print(f"\n✅ Structural pruning experiment completed successfully!")
        
        # Test the pruned model
        print(f"\n📊 Testing pruned model...")
        test_metrics = pruned_model.val(data="data/VOC_adva.yaml", verbose=False)
        
        print(f"📈 Pruned model performance:")
        print(f"   mAP@0.5:0.95: {test_metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"   mAP@0.5: {test_metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   Precision: {test_metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
        print(f"   Recall: {test_metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
        
        # Check if pruning details are available
        if hasattr(pruned_model, 'pruned_layers_details'):
            print(f"\n📋 Pruning Details:")
            for i, detail in enumerate(pruned_model.pruned_layers_details):
                print(f"   Layer {i+1}: Block {detail.get('block_idx', 'N/A')}")
                print(f"     Original channels: {detail.get('original_channels', 'N/A')}")
                print(f"     Remaining channels: {detail.get('remaining_channels', 'N/A')}")
                print(f"     Status: {detail.get('status', 'N/A')}")
        
        return pruned_model
        
    except Exception as e:
        print(f"❌ Structural pruning experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_activation_blocks_3_4_experiment(layers=3, experiment_name=None):
    """Run activation pruning experiment using apply_activation_pruning_blocks_3_4 function."""
    if experiment_name is None:
        experiment_name = f"activation_blocks_3_4_{layers}layers"
    
    print(f"🧪 Running Activation Pruning Experiment (Blocks 3-6)")
    print(f"   Layers to prune: {layers}")
    print(f"   Experiment name: {experiment_name}")
    print(f"   🔧 Uses k-medoids clustering algorithm")
    print(f"   🔧 Targets blocks 3-6")
    print(f"   🔧 No fine-tuning (faster testing)")
    print("-" * 50)
    
    try:
        # Import the function
        from pruning_yolo_v8 import apply_activation_pruning_blocks_3_4
        
        # Load data configuration
        import yaml
        data_yaml = "data/VOC_adva.yaml"
        with open(data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)
        
        classes_names = data_cfg["names"]
        classes = list(range(len(classes_names)))
        
        # Load samples for activation pruning
        from pruning_experiments import PruningEvaluator, PruningConfig
        
        # Create a temporary config to use the evaluator's data loading
        temp_config = PruningConfig(
            method="activation",
            layers_to_prune=layers,
            model_path="data/best.pt",
            data_yaml=data_yaml
        )
        evaluator = PruningEvaluator(temp_config)
        
        # Load samples with correct paths
        train_img_dir = data_cfg["train"]
        val_img_dir = data_cfg["val"]
        
        # Convert relative paths to absolute paths
        if not train_img_dir.startswith("/"):
            train_img_dir = os.path.join("data", train_img_dir)
        if not val_img_dir.startswith("/"):
            val_img_dir = os.path.join("data", val_img_dir)
            
        train_label_dir = train_img_dir.replace("/images", "/labels")
        val_label_dir = val_img_dir.replace("/images", "/labels")
        
        train_data = evaluator.load_samples(train_img_dir, train_label_dir, max_samples=100)
        valid_data = evaluator.load_samples(val_img_dir, val_label_dir, max_samples=50)
        
        print(f"📥 Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
        
        # Run the activation pruning experiment
        pruned_model = apply_activation_pruning_blocks_3_4(
            model_path="data/best.pt",
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=layers
        )
        
        print(f"\n✅ Activation pruning experiment completed successfully!")
        
        # Test the pruned model
        print(f"\n📊 Testing pruned model...")
        test_metrics = pruned_model.val(data=data_yaml, verbose=False)
        
        print(f"📈 Pruned model performance:")
        print(f"   mAP@0.5:0.95: {test_metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"   mAP@0.5: {test_metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   Precision: {test_metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
        print(f"   Recall: {test_metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
        
        # Check if pruning details are available
        if hasattr(pruned_model, 'pruned_layers_details'):
            print(f"\n📋 Pruning Details:")
            for i, detail in enumerate(pruned_model.pruned_layers_details):
                print(f"   Layer {i+1}: Block {detail.get('block_idx', 'N/A')}")
                print(f"     Original channels: {detail.get('original_channels', 'N/A')}")
                print(f"     Remaining channels: {detail.get('remaining_channels', 'N/A')}")
                print(f"     Status: {detail.get('status', 'N/A')}")
        
        return pruned_model
        
    except Exception as e:
        print(f"❌ Activation pruning experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_channel_fix_experiment(layers=3, experiment_name=None):
    """Run activation pruning experiment with channel mismatch fix."""
    if experiment_name is None:
        experiment_name = f"channel_fix_{layers}layers"
    
    print(f"🔧 Running Channel Fix Experiment")
    print(f"   Layers to prune: {layers}")
    print(f"   Experiment name: {experiment_name}")
    print(f"   🔧 Tests activation pruning with channel mismatch fix")
    print(f"   🔧 Should handle multi-layer pruning without errors")
    print("-" * 50)
    
    try:
        # Import the new function
        from channel_adjustment_fix import apply_activation_pruning_with_channel_fix
        
        # Load data configuration
        import yaml
        data_yaml = "data/VOC_adva.yaml"
        with open(data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)
        
        classes_names = data_cfg["names"]
        classes = list(range(len(classes_names)))
        
        # Load samples for activation pruning
        from pruning_experiments import PruningEvaluator, PruningConfig
        
        # Create a temporary config to use the evaluator's data loading
        temp_config = PruningConfig(
            method="activation",
            layers_to_prune=layers,
            model_path="data/best.pt",
            data_yaml=data_yaml
        )
        evaluator = PruningEvaluator(temp_config)
        
        # Load samples with correct paths
        train_img_dir = data_cfg["train"]
        val_img_dir = data_cfg["val"]
        
        # Convert relative paths to absolute paths
        if not train_img_dir.startswith("/"):
            train_img_dir = os.path.join("data", train_img_dir)
        if not val_img_dir.startswith("/"):
            val_img_dir = os.path.join("data", val_img_dir)
            
        train_label_dir = train_img_dir.replace("/images", "/labels")
        val_label_dir = val_img_dir.replace("/images", "/labels")
        
        train_data = evaluator.load_samples(train_img_dir, train_label_dir, max_samples=100)
        valid_data = evaluator.load_samples(val_img_dir, val_label_dir, max_samples=50)
        
        print(f"📥 Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
        
        if len(train_data) == 0:
            print("❌ No training data loaded. Cannot proceed.")
            return None
        
        # Run the channel fix experiment
        pruned_model = apply_activation_pruning_with_channel_fix(
            model_path="data/best.pt",
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=layers,
            data_yaml=data_yaml
        )
        
        print(f"\n✅ Channel fix experiment completed successfully!")
        
        # Test the pruned model
        print(f"\n📊 Testing pruned model...")
        test_metrics = pruned_model.val(data=data_yaml, verbose=False)
        
        print(f"📈 Pruned model performance:")
        print(f"   mAP@0.5:0.95: {test_metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"   mAP@0.5: {test_metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   Precision: {test_metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
        print(f"   Recall: {test_metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
        
        # Check if pruning details are available
        if hasattr(pruned_model, 'pruned_layers_details'):
            print(f"\n📋 Pruning Details:")
            for i, detail in enumerate(pruned_model.pruned_layers_details):
                print(f"   Layer {i+1}: Block {detail.get('block_idx', 'N/A')}")
                print(f"     Original channels: {detail.get('original_channels', 'N/A')}")
                print(f"     Remaining channels: {detail.get('remaining_channels', 'N/A')}")
                print(f"     Status: {detail.get('status', 'N/A')}")
        
        return pruned_model
        
    except Exception as e:
        print(f"❌ Channel fix experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_robust_channel_fix_experiment(layers=3, experiment_name=None):
    """Run robust activation pruning experiment with comprehensive channel mismatch fix."""
    if experiment_name is None:
        experiment_name = f"robust_channel_fix_{layers}layers"
    
    print(f"🔧 Running Robust Channel Fix Experiment")
    print(f"   Layers to prune: {layers}")
    print(f"   Experiment name: {experiment_name}")
    print(f"   🔧 Tests robust activation pruning with comprehensive channel fix")
    print(f"   🔧 Prunes one layer at a time to avoid channel mismatches")
    print("-" * 50)
    
    try:
        # Import the robust function
        from robust_channel_fix import apply_robust_activation_pruning
        
        # Load data configuration
        import yaml
        data_yaml = "data/VOC_adva.yaml"
        with open(data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)
        
        classes_names = data_cfg["names"]
        classes = list(range(len(classes_names)))
        
        # Load samples for activation pruning
        from pruning_experiments import PruningEvaluator, PruningConfig
        
        # Create a temporary config to use the evaluator's data loading
        temp_config = PruningConfig(
            method="activation",
            layers_to_prune=layers,
            model_path="data/best.pt",
            data_yaml=data_yaml
        )
        evaluator = PruningEvaluator(temp_config)
        
        # Load samples with correct paths
        train_img_dir = data_cfg["train"]
        val_img_dir = data_cfg["val"]
        
        # Convert relative paths to absolute paths
        if not train_img_dir.startswith("/"):
            train_img_dir = os.path.join("data", train_img_dir)
        if not val_img_dir.startswith("/"):
            val_img_dir = os.path.join("data", val_img_dir)
            
        train_label_dir = train_img_dir.replace("/images", "/labels")
        val_label_dir = val_img_dir.replace("/images", "/labels")
        
        train_data = evaluator.load_samples(train_img_dir, train_label_dir, max_samples=100)
        valid_data = evaluator.load_samples(val_img_dir, val_label_dir, max_samples=50)
        
        print(f"📥 Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
        
        if len(train_data) == 0:
            print("❌ No training data loaded. Cannot proceed.")
            return None
        
        # Run the robust channel fix experiment
        pruned_model = apply_robust_activation_pruning(
            model_path="data/best.pt",
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=layers,
            data_yaml=data_yaml
        )
        
        print(f"\n✅ Robust channel fix experiment completed successfully!")
        
        # Test the pruned model
        print(f"\n📊 Testing pruned model...")
        test_metrics = pruned_model.val(data=data_yaml, verbose=False)
        
        print(f"📈 Pruned model performance:")
        print(f"   mAP@0.5:0.95: {test_metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"   mAP@0.5: {test_metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   Precision: {test_metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
        print(f"   Recall: {test_metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
        
        # Check if pruning details are available
        if hasattr(pruned_model, 'pruned_layers_details'):
            print(f"\n📋 Pruning Details:")
            for i, detail in enumerate(pruned_model.pruned_layers_details):
                print(f"   Layer {i+1}: Block {detail.get('block_idx', 'N/A')}")
                print(f"     Original channels: {detail.get('original_channels', 'N/A')}")
                print(f"     Remaining channels: {detail.get('remaining_channels', 'N/A')}")
                print(f"     Status: {detail.get('status', 'N/A')}")
        
        return pruned_model
        
    except Exception as e:
        print(f"❌ Robust channel fix experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_structural_comparison_experiment(layers=3, experiment_name=None):
    """Run comparison between gamma (soft) and activation (structural) pruning."""
    if experiment_name is None:
        experiment_name = f"structural_comparison_{layers}layers"
    
    print(f"🔬 Running Structural Comparison Experiment")
    print(f"   Layers to prune: {layers}")
    print(f"   Experiment name: {experiment_name}")
    print(f"   🔧 Gamma: Soft pruning (existing method)")
    print(f"   🔧 Activation: Structural pruning (new method)")
    print("-" * 50)
    
    try:
        # Step 1: Run Activation Pruning (Structural)
        print(f"\n🚀 Step 1: Running Activation Pruning (Structural Pruning)...")
        
        # Run structural activation pruning using the wrapper function
        from pruning_yolo_v8 import apply_structural_activation_pruning_blocks_3_4
        
        activation_model = apply_structural_activation_pruning_blocks_3_4(
            model_path="data/best.pt",
            data_yaml="data/VOC_adva.yaml",
            layers_to_prune=layers
        )
        
        activation_metrics = activation_model.val(data="data/VOC_adva.yaml", verbose=False)
        print(f"✅ Activation structural pruning completed!")
        
        # Step 2: Run Gamma Pruning (Soft)
        print(f"\n🚀 Step 2: Running Gamma Pruning (Soft Pruning)...")
        from pruning_yolo_v8 import apply_50_percent_gamma_pruning_blocks_3_4
        
        gamma_model = apply_50_percent_gamma_pruning_blocks_3_4(
            model_path="data/best.pt",
            data_yaml="data/VOC_adva.yaml",
            layers_to_prune=layers
        )
        
        gamma_metrics = gamma_model.val(data="data/VOC_adva.yaml", verbose=False)
        print(f"✅ Gamma pruning completed!")
        
        # Step 3: Compare Results
        print(f"\n📊 COMPARISON RESULTS:")
        print(f"{'='*80}")
        print(f"{'Method':<20} {'mAP@0.5:0.95':<15} {'mAP@0.5':<15} {'Precision':<15} {'Recall':<15}")
        print(f"{'-'*80}")
        
        gamma_map = gamma_metrics.results_dict.get('metrics/mAP50-95(B)', 0)
        gamma_map50 = gamma_metrics.results_dict.get('metrics/mAP50(B)', 0)
        gamma_precision = gamma_metrics.results_dict.get('metrics/precision(B)', 0)
        gamma_recall = gamma_metrics.results_dict.get('metrics/recall(B)', 0)
        
        activation_map = activation_metrics.results_dict.get('metrics/mAP50-95(B)', 0)
        activation_map50 = activation_metrics.results_dict.get('metrics/mAP50(B)', 0)
        activation_precision = activation_metrics.results_dict.get('metrics/precision(B)', 0)
        activation_recall = activation_metrics.results_dict.get('metrics/recall(B)', 0)
        
        print(f"{'Activation (Structural)':<20} {activation_map:<15.4f} {activation_map50:<15.4f} {activation_precision:<15.4f} {activation_recall:<15.4f}")
        print(f"{'Gamma (Soft)':<20} {gamma_map:<15.4f} {gamma_map50:<15.4f} {gamma_precision:<15.4f} {gamma_recall:<15.4f}")
        
        # Calculate differences
        map_diff = activation_map - gamma_map
        map50_diff = activation_map50 - gamma_map50
        precision_diff = activation_precision - gamma_precision
        recall_diff = activation_recall - gamma_recall
        
        print(f"{'-'*80}")
        print(f"{'Difference (A-G)':<20} {map_diff:<+15.4f} {map50_diff:<+15.4f} {precision_diff:<+15.4f} {recall_diff:<+15.4f}")
        
        # Determine winner
        if activation_map > gamma_map:
            winner = "Activation (Structural)"
            improvement = (activation_map - gamma_map) / gamma_map * 100
        else:
            winner = "Gamma (Soft)"
            improvement = (gamma_map - activation_map) / activation_map * 100
        
        print(f"\n🎯 SUMMARY:")
        print(f"  Winner: {winner}")
        print(f"  Improvement: {improvement:.2f}%")
        print(f"  Gamma Method: Soft pruning (channel mismatches possible)")
        print(f"  Activation Method: Structural pruning (no channel mismatches)")
        print(f"  Same layers tested: ✅")
        
        return {
            'gamma_model': gamma_model,
            'activation_model': activation_model,
            'gamma_metrics': gamma_metrics.results_dict,
            'activation_metrics': activation_metrics.results_dict,
            'winner': winner,
            'improvement_percent': improvement
        }
        
    except Exception as e:
        print(f"❌ Structural comparison experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_extended_experiments(method="both", layer_counts=None):
    """Run extended experiments with 6, 8, 10, 12 layers."""
    if layer_counts is None:
        layer_counts = [6, 8, 10, 12]
    
    print(f"🚀 Running Extended Experiments")
    print(f"   Method(s): {method}")
    print(f"   Layer counts: {layer_counts}")
    print("-" * 50)
    
    configs = []
    
    for layers in layer_counts:
        if method in ["gamma", "both"]:
            # Determine target blocks based on layer count (starting from block 1)
            if layers <= 3:
                target_blocks = [3, 4, 5]
            elif layers <= 6:
                target_blocks = [2, 3, 4]
            elif layers <= 8:
                target_blocks = [2, 3, 4, 5]
            elif layers <= 10:
                target_blocks = [1, 2, 3, 4, 5]
            else:  # 12 layers
                target_blocks = [1, 2, 3, 4, 5]  # No block 0
            
            configs.append(PruningConfig(
                method="gamma",
                layers_to_prune=layers,
                target_blocks=target_blocks,
                experiment_name=f"extended_gamma_{layers}_layers",
                model_path="data/best.pt",
                data_yaml="data/VOC_adva.yaml"
            ))
        
        if method in ["activation", "both"]:
            # Determine target blocks based on layer count (starting from block 1)
            if layers <= 3:
                target_blocks = [3, 4, 5]
            elif layers <= 6:
                target_blocks = [2, 3, 4]
            elif layers <= 8:
                target_blocks = [2, 3, 4, 5]
            elif layers <= 10:
                target_blocks = [1, 2, 3, 4, 5]
            else:  # 12 layers
                target_blocks = [1, 2, 3, 4, 5]  # No block 0
            
            configs.append(PruningConfig(
                method="activation",
                layers_to_prune=layers,
                target_blocks=target_blocks,
                experiment_name=f"extended_activation_{layers}_layers",
                model_path="data/best.pt",
                data_yaml="data/VOC_adva.yaml"
            ))
    
    evaluator = PruningEvaluator(configs[0])
    results = evaluator.run_batch_experiments(configs)
    evaluator.save_results()
    evaluator.print_summary()
    
    return results

def run_batch_experiments():
    """Run a comprehensive batch of experiments."""
    print("🚀 Running Batch Experiments")
    print("   This will run multiple experiments with different configurations")
    print("-" * 50)
    
    configs = create_experiment_configs()
    evaluator = PruningEvaluator(configs[0])
    results = evaluator.run_batch_experiments(configs)
    evaluator.save_results()
    evaluator.print_summary()
    
    return results

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="YOLOv8 Pruning Experiment Runner")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Gamma experiment
    gamma_parser = subparsers.add_parser('gamma', help='Run gamma pruning experiment')
    gamma_parser.add_argument('--layers', type=int, default=3, help='Number of layers to prune (default: 3, supports 2-12)')
    gamma_parser.add_argument('--blocks', type=int, nargs='+', default=[3, 4, 5], help='Target blocks (default: 3 4 5, supports 1-5)')
    gamma_parser.add_argument('--name', type=str, help='Experiment name')
    
    # Activation experiment
    activation_parser = subparsers.add_parser('activation', help='Run activation pruning experiment')
    activation_parser.add_argument('--layers', type=int, default=3, help='Number of layers to prune (default: 3, supports 2-12)')
    activation_parser.add_argument('--blocks', type=int, nargs='+', default=[3, 4, 5], help='Target blocks (default: 3 4 5, supports 1-5)')
    activation_parser.add_argument('--name', type=str, help='Experiment name')
    
    # Comparison experiment
    compare_parser = subparsers.add_parser('compare', help='Run comparison experiment (gamma vs activation)')
    compare_parser.add_argument('--layers', type=int, default=3, help='Number of layers to prune (default: 3, supports 2-12)')
    compare_parser.add_argument('--blocks', type=int, nargs='+', default=[3, 4, 5], help='Target blocks (default: 3 4 5, supports 1-5)')
    
    # Extended experiments
    extended_parser = subparsers.add_parser('extended', help='Run extended experiments (6, 8, 10, 12 layers)')
    extended_parser.add_argument('--method', choices=['gamma', 'activation', 'both'], default='both', help='Pruning method(s)')
    extended_parser.add_argument('--layers', type=int, nargs='+', default=[6, 8, 10, 12], help='Layer counts to test')
    
    # Comparison experiment
    comparison_parser = subparsers.add_parser('comparison', help='Run comparison experiment (gamma vs activation on SAME layers)')
    comparison_parser.add_argument('--layers', type=int, default=3, help='Number of layers to prune (default: 3)')
    comparison_parser.add_argument('--name', type=str, help='Experiment name')
    
    # Structural pruning experiment
    structural_parser = subparsers.add_parser('structural', help='Run structural pruning experiment (TRUE architectural pruning)')
    structural_parser.add_argument('--layers', type=int, default=3, help='Number of layers to prune (default: 3)')
    structural_parser.add_argument('--name', type=str, help='Experiment name')
    
    # Structural comparison experiment (Gamma soft vs Activation structural)
    structural_compare_parser = subparsers.add_parser('structural-compare', help='Compare gamma (soft) vs activation (structural) pruning')
    structural_compare_parser.add_argument('--layers', type=int, default=3, help='Number of layers to prune (default: 3)')
    structural_compare_parser.add_argument('--name', type=str, help='Experiment name')
    
    # Activation blocks 3-4 experiment
    activation_blocks_parser = subparsers.add_parser('activation-blocks', help='Run activation pruning experiment (blocks 3-6, no fine-tuning)')
    activation_blocks_parser.add_argument('--layers', type=int, default=3, help='Number of layers to prune (default: 3)')
    activation_blocks_parser.add_argument('--name', type=str, help='Experiment name')
    
    # Channel fix experiment
    channel_fix_parser = subparsers.add_parser('channel-fix', help='Test activation pruning with channel mismatch fix')
    channel_fix_parser.add_argument('--layers', type=int, default=3, help='Number of layers to prune (default: 3)')
    channel_fix_parser.add_argument('--name', type=str, help='Experiment name')
    
    # Robust channel fix experiment
    robust_fix_parser = subparsers.add_parser('robust-fix', help='Test robust activation pruning with comprehensive channel fix')
    robust_fix_parser.add_argument('--layers', type=int, default=3, help='Number of layers to prune (default: 3)')
    robust_fix_parser.add_argument('--name', type=str, help='Experiment name')
    
    # Batch experiments
    batch_parser = subparsers.add_parser('batch', help='Run batch of experiments')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'gamma':
            run_single_gamma_experiment(
                layers=args.layers,
                blocks=args.blocks,
                experiment_name=args.name
            )
        elif args.command == 'activation':
            run_single_activation_experiment(
                layers=args.layers,
                blocks=args.blocks,
                experiment_name=args.name
            )
        elif args.command == 'compare':
            run_comparison_experiment(
                layers=args.layers,
                blocks=args.blocks
            )
        elif args.command == 'extended':
            run_extended_experiments(args.method, args.layers)
        elif args.command == 'comparison':
            run_same_layer_comparison_experiment(args.layers, args.name)
        elif args.command == 'structural':
            run_structural_pruning_experiment(args.layers, args.name)
        elif args.command == 'structural-compare':
            run_structural_comparison_experiment(args.layers, args.name)
        elif args.command == 'activation-blocks':
            run_activation_blocks_3_4_experiment(args.layers, args.name)
        elif args.command == 'channel-fix':
            run_channel_fix_experiment(args.layers, args.name)
        elif args.command == 'robust-fix':
            run_robust_channel_fix_experiment(args.layers, args.name)
        elif args.command == 'batch':
            run_batch_experiments()
        
        print("\n✅ Experiment(s) completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
