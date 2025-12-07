#!/usr/bin/env python3
"""
Structural Multi-Block Pruning
This implementation uses TRUE structural pruning to modify the model architecture,
allowing us to prune multiple layers across multiple blocks without channel mismatches.
"""

import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from structural_pruning import StructuralPruner
from pruning_yolo_v8_sequential_fix import load_training_data, load_validation_data
import torch.nn as nn

def main():
    print("=" * 80)
    print("Structural Multi-Block Pruning")
    print("=" * 80)
    
    # Configuration
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    blocks_to_prune = [1, 2, 3]  # Blocks to prune from
    conv_in_block_idx_to_prune = 0  # Which conv layer in each block to prune (0 = first, 1 = second, etc.)
    epochs_per_finetune = 5
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False
    
    print(f"✅ Configuration:")
    print(f"   Model: {model_path}")
    print(f"   Data: {data_yaml}")
    print(f"   Blocks to prune: {blocks_to_prune}")
    print(f"   Conv index in block: {conv_in_block_idx_to_prune} (0=first, 1=second, etc.)")
    print(f"   Epochs per fine-tune: {epochs_per_finetune}")
    
    try:
        # Load data
        train_data = load_training_data(data_yaml, max_samples=50)
        if len(train_data) == 0:
            print(f"❌ No training data loaded.")
            return False
        
        valid_data = load_validation_data(data_yaml, max_samples=30)
        if len(valid_data) == 0:
            valid_data = train_data[:20]
        
        # Get classes
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        classes = list(range(len(data_cfg['names'])))
        
        print(f"\n📊 Data loaded:")
        print(f"   - Training samples: {len(train_data)}")
        print(f"   - Validation samples: {len(valid_data)}")
        print(f"   - Classes: {len(classes)}")
        
        # Initialize structural pruner
        print(f"\n🔧 Initializing structural pruner...")
        pruner = StructuralPruner(model_path, train_data, valid_data, classes)
        
        # Get all Conv2d layers from blocks 1-5
        print(f"\n🔍 Analyzing model architecture...")
        detection_model = pruner.detection_model
        all_conv_layers = pruner.get_all_conv2d_layers(detection_model)
        print(f"   Found {len(all_conv_layers)} Conv2d layers total")
        
        # Organize by block
        target_blocks = [1, 2, 3, 4, 5]
        block_layers = {}
        
        for i, conv_layer in enumerate(all_conv_layers):
            for block_idx in target_blocks:
                if block_idx < len(detection_model):
                    block = detection_model[block_idx]
                    block_convs = pruner.get_all_conv2d_layers(block)
                    if conv_layer in block_convs:
                        conv_in_block_idx = block_convs.index(conv_layer)
                        if block_idx not in block_layers:
                            block_layers[block_idx] = []
                        block_layers[block_idx].append({
                            'conv_layer': conv_layer,
                            'block_idx': block_idx,
                            'conv_in_block_idx': conv_in_block_idx,
                            'global_idx': i,
                            'num_channels': conv_layer.weight.shape[0]
                        })
                        break
        
        # Select specified conv layer from each block
        selected_layers = []
        for block_idx in blocks_to_prune:
            if block_idx in block_layers:
                # Filter to get the specific conv layer index we want
                matching_layers = [l for l in block_layers[block_idx] if l['conv_in_block_idx'] == conv_in_block_idx_to_prune]
                
                if matching_layers:
                    # Take the layer with most channels if multiple matches
                    best_layer = max(matching_layers, key=lambda x: x['num_channels'])
                    selected_layers.append(best_layer)
                    print(f"   Block {block_idx}: Selected Conv {best_layer['conv_in_block_idx']}, {best_layer['num_channels']} channels")
                else:
                    print(f"   ⚠️  Block {block_idx} has no Conv layer with index {conv_in_block_idx_to_prune}")
                    print(f"       Available conv indices in block {block_idx}: {set(l['conv_in_block_idx'] for l in block_layers[block_idx])}")
            else:
                print(f"   ⚠️  Block {block_idx} not found, skipping")
        
        print(f"\n✅ Selected {len(selected_layers)} layers for structural pruning across {len(blocks_to_prune)} blocks")
        
        # Run structural pruning using the StructuralPruner class
        print(f"\n{'='*80}")
        print(f"STARTING STRUCTURAL PRUNING")
        print(f"{'='*80}")
        
        # Prune one layer at a time with fine-tuning between each
        results = []
        
        for idx, conv_info in enumerate(selected_layers):
            print(f"\n{'='*80}")
            print(f"ITERATION {idx + 1}/{len(selected_layers)}")
            print(f"  Block: {conv_info['block_idx']}")
            print(f"  Conv in block: {conv_info['conv_in_block_idx']}")
            print(f"  Channels: {conv_info['num_channels']}")
            print(f"{'='*80}")
            
            try:
                # Prune ONE layer at a time from specific block
                single_result = pruner.structural_activation_pruning(
                    layers_to_prune=1,
                    target_blocks=[conv_info['block_idx']]
                )
                
                print(f"✅ Layer {idx + 1} pruned successfully")
                results.append({
                    'block_idx': conv_info['block_idx'],
                    'conv_in_block_idx': conv_info['conv_in_block_idx'],
                    'global_idx': conv_info['global_idx'],
                    'original_channels': conv_info['num_channels'],
                    'status': 'success'
                })
                
                # Fine-tune after EACH layer
                print(f"\n📝 Fine-tuning for {epochs_per_finetune} epochs...")
                pruner.model.train(data=data_yaml, epochs=epochs_per_finetune, verbose=True)
                print(f"✅ Fine-tuning completed")
                
            except Exception as e:
                print(f"❌ Layer {idx + 1} pruning failed: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    'block_idx': conv_info['block_idx'],
                    'conv_in_block_idx': conv_info['conv_in_block_idx'],
                    'global_idx': conv_info['global_idx'],
                    'original_channels': conv_info['num_channels'],
                    'status': 'failed',
                    'error': str(e)
                })
                # Continue to next layer even if this one failed
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"FINAL SUMMARY")
        print(f"{'='*80}")
        
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'failed')
        
        print(f"Total layers pruned: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        # Test final model
        print(f"\n{'='*80}")
        print(f"FINAL MODEL PERFORMANCE")
        print(f"{'='*80}")
        
        try:
            test_metrics = pruner.model.val(data=data_yaml, verbose=False)
            print(f"📈 Final model performance:")
            print(f"   mAP@0.5:0.95: {test_metrics.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
            print(f"   mAP@0.5: {test_metrics.results_dict.get('metrics/mAP50(B)', 'N/A')}")
            print(f"   Precision: {test_metrics.results_dict.get('metrics/precision(B)', 'N/A')}")
            print(f"   Recall: {test_metrics.results_dict.get('metrics/recall(B)', 'N/A')}")
        except Exception as e:
            print(f"⚠️  Performance testing failed: {e}")
        
        # Save final model
        final_model_path = "pruned_structural_multi_block.pt"
        pruner.model.save(final_model_path)
        print(f"\n💾 Saved final model: {final_model_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Structural pruning failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

