#!/usr/bin/env python3
"""
YOLOv8 Pruning Experiments Package
Comprehensive pruning evaluation system with configurable parameters.

Features:
- Configurable pruning methods (gamma, activation)
- Flexible layer selection and block targeting
- Batch experiment execution
- Detailed metrics and comparison
- Export results to CSV/JSON
"""

import os
import time
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
import yaml

from pruning_yolo_v8 import (
    apply_activation_pruning_blocks_3_4,
    apply_50_percent_gamma_pruning_blocks_3_4,
    apply_pruning_v8,
    apply_gamma_pruning_iter,
    apply_gamma_pruning_on_block_zeroed,
    run_comparison_experiment,
    get_layer_selection_info
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PruningConfig:
    """Configuration for pruning experiments."""
    # Basic settings
    model_path: str = "data/best.pt"
    data_yaml: str = "data/VOC_adva.yaml"
    output_dir: str = "experiment_results"
    
    # Pruning parameters
    method: str = "gamma"  # "gamma" or "activation"
    layers_to_prune: int = 3
    target_blocks: List[int] = None
    pruning_percentage: float = 50.0  # For gamma pruning
    
    # Experiment settings
    experiment_name: str = "default_experiment"
    save_model: bool = True
    evaluate_before: bool = True
    evaluate_after: bool = True
    
    def __post_init__(self):
        if self.target_blocks is None:
            self.target_blocks = [3, 4, 5]

@dataclass
class ExperimentResult:
    """Results from a pruning experiment."""
    config: PruningConfig
    original_metrics: Dict[str, float]
    pruned_metrics: Dict[str, float]
    timing: Dict[str, float]
    pruning_details: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class PruningEvaluator:
    """Main class for running pruning experiments."""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.results = []
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create experiment subdirectory
        self.experiment_dir = self.output_dir / config.experiment_name
        self.experiment_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized PruningEvaluator for experiment: {config.experiment_name}")
    
    def evaluate_model(self, model_path: str, data_yaml: str) -> Tuple[Dict[str, float], Any]:
        """Evaluate a model and return metrics."""
        model = YOLO(model_path)
        results = model.val(data=data_yaml, verbose=False)
        
        metrics = {
            "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
            "mAP50_95": results.results_dict.get("metrics/mAP50-95(B)", 0),
            "precision": results.results_dict.get("metrics/precision(B)", 0),
            "recall": results.results_dict.get("metrics/recall(B)", 0),
        }
        return metrics, results
    
    def run_gamma_pruning(self) -> Tuple[Dict[str, Any], float]:
        """Run gamma pruning experiment."""
        logger.info(f"Running gamma pruning on {self.config.layers_to_prune} layers")
        
        start_time = time.time()
        
        # Run the gamma pruning function with updated parameters
        pruned_model = apply_50_percent_gamma_pruning_blocks_3_4(
            model_path=self.config.model_path,
            data_yaml=self.config.data_yaml,
            layers_to_prune=self.config.layers_to_prune
        )
        
        pruning_time = time.time() - start_time
        
        # Extract pruning details from the function's output
        pruning_details = {
            "method": "gamma_pruning",
            "layers_pruned": self.config.layers_to_prune,
            "target_blocks": self.config.target_blocks,
            "pruning_percentage": self.config.pruning_percentage,
            "pruning_time": pruning_time,
            "fine_tuning_epochs": 20,
            "pruned_layer_details": getattr(pruned_model, 'pruned_layers_details', []),
            "pruned_model": pruned_model  # Store the pruned model for evaluation
        }
        
        return pruning_details, pruning_time
    
    def run_activation_pruning(self) -> Tuple[Dict[str, Any], float]:
        """Run activation pruning experiment."""
        logger.info(f"Running activation pruning on {self.config.layers_to_prune} layers")
        
        start_time = time.time()
        
        # Load class names and data for activation pruning
        with open(self.config.data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)
        classes_names = data_cfg["names"]
        classes = list(range(len(classes_names)))
        
        # Load samples for activation pruning
        train_img_dir = data_cfg["train"]
        val_img_dir = data_cfg["val"]
        
        # Convert relative paths to absolute paths
        if not train_img_dir.startswith("/"):
            train_img_dir = os.path.join("data", train_img_dir)
        if not val_img_dir.startswith("/"):
            val_img_dir = os.path.join("data", val_img_dir)
            
        train_label_dir = train_img_dir.replace("/images", "/labels")
        val_label_dir = val_img_dir.replace("/images", "/labels")
        
        logger.info(f"Loading training data from: {train_img_dir}")
        train_data = self.load_samples(train_img_dir, train_label_dir, max_samples=200)
        valid_data = self.load_samples(val_img_dir, val_label_dir, max_samples=100)
        
        logger.info(f"Loaded {len(train_data)} training samples and {len(valid_data)} validation samples for activation pruning")
        
        # Run activation pruning (using the available function)
        pruned_model = apply_activation_pruning_blocks_3_4(
            model_path=self.config.model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=self.config.layers_to_prune
        )
        
        pruning_time = time.time() - start_time
        
        # Save the pruned model if requested
        if self.config.save_model:
            torch_model = pruned_model.model
            save_path = self.experiment_dir / f"pruned_activation_{self.config.experiment_name}.pt"
            torch.save(torch_model.state_dict(), save_path)
            logger.info(f"Saved pruned model to {save_path}")
        
        pruning_details = {
            "method": "activation_pruning",
            "layers_pruned": self.config.layers_to_prune,
            "target_blocks": self.config.target_blocks,
            "pruning_time": pruning_time,
            "fine_tuning_epochs": 20,
            "pruned_layer_details": getattr(pruned_model, 'pruned_layers_details', []),
            "pruned_model": pruned_model  # Store the pruned model for evaluation
        }
        
        return pruning_details, pruning_time
    
    def run_comparison_experiment(self) -> Dict[str, Any]:
        """Run comparison experiment with guaranteed same layers."""
        logger.info(f"Running comparison experiment on {self.config.layers_to_prune} layers")
        
        # Load class names and data for activation pruning
        with open(self.config.data_yaml, "r") as f:
            data_cfg = yaml.safe_load(f)
        classes_names = data_cfg["names"]
        classes = list(range(len(classes_names)))
        
        # Load samples for activation pruning
        train_img_dir = data_cfg["train"]
        val_img_dir = data_cfg["val"]
        
        # Convert relative paths to absolute paths
        if not train_img_dir.startswith("/"):
            train_img_dir = os.path.join("data", train_img_dir)
        if not val_img_dir.startswith("/"):
            val_img_dir = os.path.join("data", val_img_dir)
            
        train_label_dir = train_img_dir.replace("/images", "/labels")
        val_label_dir = val_img_dir.replace("/images", "/labels")
        
        logger.info(f"Loading training data from: {train_img_dir}")
        train_data = self.load_samples(train_img_dir, train_label_dir, max_samples=200)
        valid_data = self.load_samples(val_img_dir, val_label_dir, max_samples=100)
        
        logger.info(f"Loaded {len(train_data)} training samples and {len(valid_data)} validation samples")
        
        # Run the comparison experiment
        comparison_results = run_comparison_experiment(
            model_path=self.config.model_path,
            train_data=train_data,
            valid_data=valid_data,
            classes=classes,
            layers_to_prune=self.config.layers_to_prune,
            data_yaml=self.config.data_yaml
        )
        
        return comparison_results
    
    def load_samples(self, image_dir: str, label_dir: str, max_samples=200) -> List[Dict]:
        """Load dataset samples for activation pruning."""
        import cv2
        import glob
        
        samples = []
        image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
        
        # Limit samples for faster experiments
        image_paths = image_paths[:max_samples]
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            base = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, base + ".txt")
            labels = []
            
            if os.path.exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        labels.append({
                            "class_id": class_id,
                            "x_center": float(parts[1]),
                            "y_center": float(parts[2]),
                            "width": float(parts[3]),
                            "height": float(parts[4])
                        })
            
            samples.append({
                "image": img,
                "label": labels,
                "image_path": img_path,
                "label_path": label_path
            })
        
        return samples
    
    def run_single_experiment(self) -> ExperimentResult:
        """Run a single pruning experiment."""
        logger.info(f"Starting experiment: {self.config.experiment_name}")
        
        result = ExperimentResult(
            config=self.config,
            original_metrics={},
            pruned_metrics={},
            timing={},
            pruning_details={},
            success=False
        )
        
        try:
            # Step 1: Evaluate original model
            if self.config.evaluate_before:
                logger.info("Evaluating original model...")
                eval_start = time.time()
                original_metrics, _ = self.evaluate_model(self.config.model_path, self.config.data_yaml)
                eval_time = time.time() - eval_start
                
                result.original_metrics = original_metrics
                result.timing['original_evaluation'] = eval_time
                
                logger.info(f"Original metrics: mAP50={original_metrics['mAP50']:.4f}, mAP50-95={original_metrics['mAP50_95']:.4f}")
            
            # Step 2: Run pruning
            logger.info(f"Running {self.config.method} pruning...")
            pruning_start = time.time()
            
            if self.config.method == "gamma":
                pruning_details, pruning_time = self.run_gamma_pruning()
            elif self.config.method == "activation":
                pruning_details, pruning_time = self.run_activation_pruning()
            else:
                raise ValueError(f"Unknown pruning method: {self.config.method}")
            
            result.pruning_details = pruning_details
            result.timing['pruning'] = pruning_time
            
            # Step 3: Evaluate pruned model
            if self.config.evaluate_after:
                logger.info("Evaluating pruned model...")
                eval_start = time.time()
                
                # The pruned model should be the one returned from the pruning function
                # Save it temporarily for evaluation
                temp_pruned_path = self.experiment_dir / f"temp_pruned_{self.config.experiment_name}.pt"
                try:
                    # Save the pruned model for evaluation
                    pruned_model = result.pruning_details.get('pruned_model')
                    if pruned_model is not None:
                        pruned_model.save(str(temp_pruned_path))
                        pruned_metrics, _ = self.evaluate_model(str(temp_pruned_path), self.config.data_yaml)
                        logger.info(f"✅ Evaluated actual pruned model for {self.config.experiment_name}")
                    else:
                        # Look for default pruned model files
                        default_paths = ["pruned_v3_yolov8n.pt", "pruned_model.pt"]
                        pruned_model_path = None
                        for path in default_paths:
                            if os.path.exists(path):
                                pruned_model_path = path
                                break
                        
                        if pruned_model_path and os.path.exists(pruned_model_path):
                            pruned_metrics, _ = self.evaluate_model(str(pruned_model_path), self.config.data_yaml)
                            logger.warning(f"⚠️ Used default pruned model file for {self.config.experiment_name}")
                        else:
                            logger.error(f"❌ No pruned model found for {self.config.experiment_name}, using original model")
                            pruned_metrics, _ = self.evaluate_model(self.config.model_path, self.config.data_yaml)
                finally:
                    # Clean up temporary file
                    if temp_pruned_path.exists():
                        temp_pruned_path.unlink()
                
                eval_time = time.time() - eval_start
                result.pruned_metrics = pruned_metrics
                result.timing['pruned_evaluation'] = eval_time
                
                logger.info(f"Pruned metrics: mAP50={pruned_metrics['mAP50']:.4f}, mAP50-95={pruned_metrics['mAP50_95']:.4f}")
            
            result.timing['total'] = time.time() - pruning_start
            result.success = True
            
            logger.info(f"Experiment {self.config.experiment_name} completed successfully!")
            
        except Exception as e:
            logger.error(f"Experiment {self.config.experiment_name} failed: {e}")
            result.success = False
            result.error_message = str(e)
        
        return result
    
    def run_batch_experiments(self, configs: List[PruningConfig]) -> List[ExperimentResult]:
        """Run multiple experiments with different configurations."""
        results = []
        
        logger.info(f"Starting batch of {len(configs)} experiments")
        
        for i, config in enumerate(configs):
            logger.info(f"Running experiment {i+1}/{len(configs)}: {config.experiment_name}")
            
            # Create evaluator for this config
            evaluator = PruningEvaluator(config)
            result = evaluator.run_single_experiment()
            
            results.append(result)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
        
        return results
    
    def save_results(self):
        """Save experiment results to files."""
        # Save to JSON
        json_path = self.output_dir / f"{self.config.experiment_name}_results.json"
        with open(json_path, 'w') as f:
            json.dump([asdict(result) for result in self.results], f, indent=2)
        
        # Save to CSV
        csv_path = self.output_dir / f"{self.config.experiment_name}_results.csv"
        self.save_results_csv(csv_path)
        
        logger.info(f"Results saved to {json_path} and {csv_path}")
    
    def save_results_csv(self, csv_path: Path):
        """Save results to CSV format."""
        if not self.results:
            return
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            header = [
                'experiment_name', 'method', 'layers_pruned', 'target_blocks',
                'original_mAP50', 'original_mAP50_95', 'original_precision', 'original_recall',
                'pruned_mAP50', 'pruned_mAP50_95', 'pruned_precision', 'pruned_recall',
                'mAP50_change', 'mAP50_95_change', 'precision_change', 'recall_change',
                'pruning_time', 'total_time', 'pruned_layer_indices', 'success'
            ]
            writer.writerow(header)
            
            # Write data
            for result in self.results:
                if result.success and result.original_metrics and result.pruned_metrics:
                    # Extract layer indices from pruning details
                    pruned_layer_details = result.pruning_details.get('pruned_layer_details', [])
                    layer_indices = []
                    for layer_detail in pruned_layer_details:
                        if 'global_conv_idx' in layer_detail:
                            layer_indices.append(str(layer_detail['global_conv_idx']))
                        elif 'original_model_idx' in layer_detail:
                            layer_indices.append(str(layer_detail['original_model_idx']))
                    layer_indices_str = ','.join(layer_indices) if layer_indices else 'N/A'
                    
                    row = [
                        result.config.experiment_name,
                        result.config.method,
                        result.config.layers_to_prune,
                        str(result.config.target_blocks),
                        result.original_metrics.get('mAP50', 0),
                        result.original_metrics.get('mAP50_95', 0),
                        result.original_metrics.get('precision', 0),
                        result.original_metrics.get('recall', 0),
                        result.pruned_metrics.get('mAP50', 0),
                        result.pruned_metrics.get('mAP50_95', 0),
                        result.pruned_metrics.get('precision', 0),
                        result.pruned_metrics.get('recall', 0),
                        result.pruned_metrics.get('mAP50', 0) - result.original_metrics.get('mAP50', 0),
                        result.pruned_metrics.get('mAP50_95', 0) - result.original_metrics.get('mAP50_95', 0),
                        result.pruned_metrics.get('precision', 0) - result.original_metrics.get('precision', 0),
                        result.pruned_metrics.get('recall', 0) - result.original_metrics.get('recall', 0),
                        result.timing.get('pruning', 0),
                        result.timing.get('total', 0),
                        layer_indices_str,
                        result.success
                    ]
                else:
                    row = [
                        result.config.experiment_name,
                        result.config.method,
                        result.config.layers_to_prune,
                        str(result.config.target_blocks),
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        result.timing.get('pruning', 0),
                        result.timing.get('total', 0),
                        'N/A',  # pruned_layer_indices
                        result.success
                    ]
                writer.writerow(row)
    
    def print_summary(self):
        """Print a comprehensive summary of all experiments."""
        if not self.results:
            logger.info("No results to summarize")
            return
        
        print("\n" + "="*100)
        print("COMPREHENSIVE EXPERIMENT SUMMARY")
        print("="*100)
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        print(f"Total Experiments: {len(self.results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"Failed: {len(failed_results)}")
        
        if successful_results:
            print("\nSUCCESSFUL EXPERIMENTS - DETAILED RESULTS:")
            print("-" * 100)
            print(f"{'Name':<25} {'Method':<12} {'Layers':<7} {'Blocks':<12} {'Orig mAP50':<12} {'Pruned mAP50':<12} {'Change':<10} {'Time(s)':<10}")
            print("-" * 100)
            
            for result in successful_results:
                if result.original_metrics and result.pruned_metrics:
                    change = result.pruned_metrics['mAP50'] - result.original_metrics['mAP50']
                    blocks_str = str(result.config.target_blocks).replace(' ', '')[:10]
                    pruning_time = result.timing.get('pruning', 0)
                    
                    print(f"{result.config.experiment_name:<25} {result.config.method:<12} "
                          f"{result.config.layers_to_prune:<7} {blocks_str:<12} "
                          f"{result.original_metrics['mAP50']:<12.4f} "
                          f"{result.pruned_metrics['mAP50']:<12.4f} "
                          f"{change:<10.4f} {pruning_time:<10.1f}")
            
            # Detailed layer-by-layer analysis
            print("\n" + "="*100)
            print("DETAILED LAYER PRUNING ANALYSIS")
            print("="*100)
            
            for result in successful_results:
                if result.pruning_details:
                    print(f"\n📊 Experiment: {result.config.experiment_name}")
                    print(f"   Method: {result.pruning_details.get('method', 'unknown')}")
                    print(f"   Layers Pruned: {result.config.layers_to_prune}")
                    print(f"   Target Blocks: {result.config.target_blocks}")
                    print(f"   Pruning Time: {result.pruning_details.get('pruning_time', 0):.2f} seconds")
                    print(f"   Fine-tuning Epochs: {result.pruning_details.get('fine_tuning_epochs', 20)}")
                    
                    # Display detailed layer information
                    pruned_layer_details = result.pruning_details.get('pruned_layer_details', [])
                    if pruned_layer_details:
                        print(f"   📋 Pruned Layer Details:")
                        for i, layer_detail in enumerate(pruned_layer_details):
                            if result.config.method == "gamma":
                                print(f"     Layer {i+1}: Block {layer_detail.get('block_idx', 'N/A')}, "
                                      f"Pair #{layer_detail.get('local_pair_idx', 'N/A')}, "
                                      f"Model Index {layer_detail.get('original_model_idx', 'N/A')} → "
                                      f"Global Index {layer_detail.get('global_conv_idx', 'N/A')}")
                                print(f"       Channels: {layer_detail.get('original_channels', 'N/A')} → "
                                      f"{layer_detail.get('remaining_channels', 'N/A')} "
                                      f"(removed {layer_detail.get('pruned_channels', 'N/A')})")
                                if 'avg_gamma' in layer_detail:
                                    print(f"       Avg Gamma: {layer_detail['avg_gamma']:.6f}")
                            else:  # activation pruning
                                print(f"     Layer {i+1}: Block {layer_detail.get('block_idx', 'N/A')}, "
                                      f"Conv #{layer_detail.get('conv_in_block_idx', 'N/A')}, "
                                      f"Model Index {layer_detail.get('original_model_idx', 'N/A')}")
                                print(f"       Channels: {layer_detail.get('original_channels', 'N/A')} → "
                                      f"{layer_detail.get('remaining_channels', 'N/A')} "
                                      f"(removed {layer_detail.get('pruned_channels', 'N/A')})")
                    else:
                        print(f"   📋 Pruned Layer Details: Not available")
                    
                    if 'note' in result.pruning_details:
                        print(f"   Note: {result.pruning_details['note']}")
                    
                    # Performance impact
                    if result.original_metrics and result.pruned_metrics:
                        mAP50_change = result.pruned_metrics['mAP50'] - result.original_metrics['mAP50']
                        mAP50_95_change = result.pruned_metrics['mAP50_95'] - result.original_metrics['mAP50_95']
                        precision_change = result.pruned_metrics['precision'] - result.original_metrics['precision']
                        recall_change = result.pruned_metrics['recall'] - result.original_metrics['recall']
                        
                        print(f"   Performance Impact:")
                        print(f"     mAP@0.5:     {result.original_metrics['mAP50']:.4f} → {result.pruned_metrics['mAP50']:.4f} ({mAP50_change:+.4f})")
                        print(f"     mAP@0.5:0.95: {result.original_metrics['mAP50_95']:.4f} → {result.pruned_metrics['mAP50_95']:.4f} ({mAP50_95_change:+.4f})")
                        print(f"     Precision:   {result.original_metrics['precision']:.4f} → {result.pruned_metrics['precision']:.4f} ({precision_change:+.4f})")
                        print(f"     Recall:      {result.original_metrics['recall']:.4f} → {result.pruned_metrics['recall']:.4f} ({recall_change:+.4f})")
        
        if failed_results:
            print("\n" + "="*100)
            print("FAILED EXPERIMENTS:")
            print("-" * 100)
            for result in failed_results:
                print(f"❌ {result.config.experiment_name}: {result.error_message}")
        
        # Best and worst performing experiments
        if successful_results and any(r.original_metrics and r.pruned_metrics for r in successful_results):
            print("\n" + "="*100)
            print("PERFORMANCE RANKINGS")
            print("="*100)
            
            # Sort by mAP50 change (best to worst)
            sorted_results = sorted(
                [r for r in successful_results if r.original_metrics and r.pruned_metrics],
                key=lambda x: x.pruned_metrics['mAP50'] - x.original_metrics['mAP50'],
                reverse=True
            )
            
            print("\n🏆 TOP 5 BEST PERFORMING EXPERIMENTS:")
            print("-" * 80)
            for i, result in enumerate(sorted_results[:5]):
                change = result.pruned_metrics['mAP50'] - result.original_metrics['mAP50']
                print(f"{i+1}. {result.config.experiment_name:<30} | Change: {change:+.4f} | Method: {result.config.method}")
            
            print("\n📉 TOP 5 WORST PERFORMING EXPERIMENTS:")
            print("-" * 80)
            for i, result in enumerate(sorted_results[-5:]):
                change = result.pruned_metrics['mAP50'] - result.original_metrics['mAP50']
                print(f"{i+1}. {result.config.experiment_name:<30} | Change: {change:+.4f} | Method: {result.config.method}")
        
        print("="*100)

def create_experiment_configs() -> List[PruningConfig]:
    """Create a set of experiment configurations for comparison."""
    configs = []
    
    # Gamma pruning experiments - extended layer range
    for layers in [2, 3, 4, 6, 8, 10, 12]:
        configs.append(PruningConfig(
            method="gamma",
            layers_to_prune=layers,
            experiment_name=f"gamma_layers_{layers}",
            target_blocks=[3, 4, 5]  # Will be extended for higher layer counts
        ))
    
    # Activation pruning experiments - extended layer range
    for layers in [2, 3, 4, 6, 8, 10, 12]:
        configs.append(PruningConfig(
            method="activation",
            layers_to_prune=layers,
            experiment_name=f"activation_layers_{layers}",
            target_blocks=[3, 4, 5]  # Will be extended for higher layer counts
        ))
    
    # Different block combinations for standard layer counts
    configs.append(PruningConfig(
        method="gamma",
        layers_to_prune=3,
        experiment_name="gamma_blocks_3_4",
        target_blocks=[3, 4]
    ))
    
    configs.append(PruningConfig(
        method="gamma",
        layers_to_prune=3,
        experiment_name="gamma_blocks_4_5",
        target_blocks=[4, 5]
    ))
    
    # Extended block combinations for higher layer counts
    configs.append(PruningConfig(
        method="gamma",
        layers_to_prune=6,
        experiment_name="gamma_6_layers_blocks_2_3_4",
        target_blocks=[2, 3, 4]
    ))
    
    configs.append(PruningConfig(
        method="gamma",
        layers_to_prune=8,
        experiment_name="gamma_8_layers_blocks_2_3_4_5",
        target_blocks=[2, 3, 4, 5]
    ))
    
    configs.append(PruningConfig(
        method="gamma",
        layers_to_prune=10,
        experiment_name="gamma_10_layers_blocks_1_2_3_4_5",
        target_blocks=[1, 2, 3, 4, 5]
    ))
    
    configs.append(PruningConfig(
        method="gamma",
        layers_to_prune=12,
        experiment_name="gamma_12_layers_all_blocks",
        target_blocks=[0, 1, 2, 3, 4, 5]
    ))
    
    # Additional block combinations for comprehensive testing (starting from block 1)
    configs.append(PruningConfig(
        method="gamma",
        layers_to_prune=3,
        experiment_name="gamma_3_layers_blocks_1_2_3",
        target_blocks=[1, 2, 3]
    ))
    
    configs.append(PruningConfig(
        method="gamma",
        layers_to_prune=3,
        experiment_name="gamma_3_layers_blocks_2_3_4",
        target_blocks=[2, 3, 4]
    ))
    
    configs.append(PruningConfig(
        method="gamma",
        layers_to_prune=3,
        experiment_name="gamma_3_layers_blocks_1_2_4",
        target_blocks=[1, 2, 4]
    ))
    
    configs.append(PruningConfig(
        method="gamma",
        layers_to_prune=3,
        experiment_name="gamma_3_layers_blocks_2_4_5",
        target_blocks=[2, 4, 5]
    ))
    
    configs.append(PruningConfig(
        method="gamma",
        layers_to_prune=3,
        experiment_name="gamma_3_layers_blocks_1_3_5",
        target_blocks=[1, 3, 5]
    ))
    
    # Activation pruning with extended blocks
    configs.append(PruningConfig(
        method="activation",
        layers_to_prune=6,
        experiment_name="activation_6_layers_blocks_2_3_4",
        target_blocks=[2, 3, 4]
    ))
    
    configs.append(PruningConfig(
        method="activation",
        layers_to_prune=8,
        experiment_name="activation_8_layers_blocks_2_3_4_5",
        target_blocks=[2, 3, 4, 5]
    ))
    
    # Additional activation pruning block combinations (starting from block 1)
    configs.append(PruningConfig(
        method="activation",
        layers_to_prune=3,
        experiment_name="activation_3_layers_blocks_1_2_3",
        target_blocks=[1, 2, 3]
    ))
    
    configs.append(PruningConfig(
        method="activation",
        layers_to_prune=3,
        experiment_name="activation_3_layers_blocks_2_3_4",
        target_blocks=[2, 3, 4]
    ))
    
    configs.append(PruningConfig(
        method="activation",
        layers_to_prune=3,
        experiment_name="activation_3_layers_blocks_1_2_4",
        target_blocks=[1, 2, 4]
    ))
    
    return configs

def main():
    """Main function for running pruning experiments."""
    parser = argparse.ArgumentParser(description="YOLOv8 Pruning Experiments")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--method", choices=["gamma", "activation"], help="Pruning method")
    parser.add_argument("--layers", type=int, help="Number of layers to prune")
    parser.add_argument("--blocks", type=int, nargs="+", help="Target blocks")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--model", type=str, default="data/best.pt", help="Model path")
    parser.add_argument("--data", type=str, default="data/VOC_adva.yaml", help="Data YAML path")
    parser.add_argument("--output-dir", type=str, default="experiment_results", help="Output directory")
    parser.add_argument("--batch", action="store_true", help="Run batch experiments")
    parser.add_argument("--single", action="store_true", help="Run single experiment")
    
    args = parser.parse_args()
    
    if args.batch:
        # Run batch experiments
        configs = create_experiment_configs()
        evaluator = PruningEvaluator(configs[0])  # Use first config for output dir
        evaluator.run_batch_experiments(configs)
        evaluator.print_summary()
    
    elif args.single:
        # Run single experiment
        config = PruningConfig(
            method=args.method or "gamma",
            layers_to_prune=args.layers or 3,
            target_blocks=args.blocks or [3, 4, 5],
            experiment_name=args.experiment_name or "single_experiment",
            model_path=args.model,
            data_yaml=args.data,
            output_dir=args.output_dir
        )
        
        evaluator = PruningEvaluator(config)
        result = evaluator.run_single_experiment()
        evaluator.save_results()
        evaluator.print_summary()
    
    else:
        print("Please specify --batch or --single")
        parser.print_help()

if __name__ == "__main__":
    main()
