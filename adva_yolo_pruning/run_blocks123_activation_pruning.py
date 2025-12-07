#!/usr/bin/env python3
"""
Activation-based pruning (hybrid): prune Conv 0 from blocks 1, 2, 3 in order.
 - Conv blocks use original sliced_block activation algorithm
 - C2f blocks use C2f-aware mini-net (Solution 3)
Fine-tune after each prune and accumulate weights.
"""

import os
import sys
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ultralytics import YOLO
import torch
import torch.nn as nn

from pruning_yolo_v8_sequential_fix import (
    load_training_data,
    load_validation_data,
    count_active_channels
)
from yolov8_utils import get_all_conv2d_layers
from pruning_yolo_v8 import prune_conv2d_in_block_with_activations
from pruning_c2f_activation import prune_conv2d_in_c2f_with_activations
from c2f_utils import is_c2f_block


def select_layers_blocks_1_2_3(detection_model, total_layers=3):
    target_blocks = [1, 2, 3]
    layers = []
    for b in target_blocks:
        if b < len(detection_model):
            block = detection_model[b]
            convs = get_all_conv2d_layers(block)
            if len(convs) >= 1:
                conv0 = convs[0]
                layers.append({
                    'block_idx': b,
                    'conv_in_block_idx': 0,
                    'name': f"Block {b}, Conv 0",
                    'num_channels': conv0.weight.shape[0]
                })
        if len(layers) >= total_layers:
            break
    return layers[:total_layers]


def is_c2f_block(block: nn.Module) -> bool:
    return block.__class__.__name__.lower() == 'c2f'


def main():
    model_path = "data/best.pt"
    data_yaml = "data/VOC_adva.yaml"
    epochs_per_finetune = 5
    total_layers_to_prune = 3

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    if not os.path.exists(data_yaml):
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False

    train_data = load_training_data(data_yaml, max_samples=50)
    if len(train_data) == 0:
        print("❌ No training data loaded."); return False
    try:
        valid_data = load_validation_data(data_yaml, max_samples=30)
    except Exception:
        valid_data = train_data[:20]

    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model

    targets = select_layers_blocks_1_2_3(detection_model, total_layers=total_layers_to_prune)
    if len(targets) == 0:
        print("❌ No eligible layers found in blocks 1, 2, 3."); return False

    print("\n🎯 Target layers (in order):")
    for i, t in enumerate(targets):
        print(f"  {i+1}. {t['name']} ({t['num_channels']} channels)")

    pruned_details = []
    for i, t in enumerate(targets):
        print(f"\n{'='*70}\nITERATION {i+1}/{len(targets)}\n{'='*70}")
        block = detection_model[t['block_idx']]
        if is_c2f_block(block):
            print("   ℹ️  C2f block detected → using C2f-aware mini-net")
            # Use new C2f-aware pruning function
            temp_path_c2f = f"temp_iter_{i+1}.pt"
            model.save(temp_path_c2f)
            
            try:
                updated_model = prune_conv2d_in_c2f_with_activations(
                    model_path=temp_path_c2f,
                    train_data=train_data,
                    valid_data=valid_data,
                    classes=list(range(len(yaml.safe_load(open(data_yaml))['names']))),
                    block_idx=t['block_idx'],
                    conv_in_block_idx=t['conv_in_block_idx'],
                    log_file=f"iter_{i+1}.txt",
                    data_yaml=data_yaml
                )
            except Exception as e:
                print(f"❌ Pruning failed for {t['name']}: {e}")
                if os.path.exists(temp_path_c2f):
                    os.remove(temp_path_c2f)
                pruned_details.append({'name': t['name'], 'status': 'failed', 'original': t['num_channels'], 'remaining': t['num_channels']})
                continue
            
            if updated_model is None:
                print(f"❌ Pruning returned None for {t['name']}")
                if os.path.exists(temp_path_c2f):
                    os.remove(temp_path_c2f)
                pruned_details.append({'name': t['name'], 'status': 'failed', 'original': t['num_channels'], 'remaining': t['num_channels']})
                continue
            
            # Copy pruned weights back
            block_main = detection_model[t['block_idx']]
            conv_main = get_all_conv2d_layers(block_main)[t['conv_in_block_idx']]
            block_upd = updated_model.model.model[t['block_idx']]
            conv_upd = get_all_conv2d_layers(block_upd)[t['conv_in_block_idx']]
            
            with torch.no_grad():
                conv_main.weight.copy_(conv_upd.weight)
                if conv_main.bias is not None and conv_upd.bias is not None:
                    conv_main.bias.copy_(conv_upd.bias)
            
            remaining = count_active_channels(conv_main)
            print(f"✅ Pruned {t['name']}: {t['num_channels']} → {remaining}")
            pruned_details.append({'name': t['name'], 'status': 'success', 'original': t['num_channels'], 'remaining': remaining})
            
            if os.path.exists(temp_path_c2f):
                os.remove(temp_path_c2f)
        else:
            print("   ℹ️  Conv block detected → using original activation algorithm")
            temp_path = f"temp_iter_{i+1}.pt"; model.save(temp_path)
            try:
                updated_model = prune_conv2d_in_block_with_activations(
                    model_path=temp_path,
                    train_data=train_data,
                    valid_data=valid_data,
                    classes=list(range(len(yaml.safe_load(open(data_yaml))['names']))),
                    block_idx=t['block_idx'],
                    conv_in_block_idx=t['conv_in_block_idx'],
                    log_file=f"iter_{i+1}.txt",
                    data_yaml=data_yaml
                )
            except Exception as e:
                print(f"❌ Pruning failed for {t['name']}: {e}")
                if os.path.exists(temp_path): os.remove(temp_path)
                pruned_details.append({'name': t['name'], 'status': 'failed', 'original': t['num_channels'], 'remaining': t['num_channels']});
                continue
            if updated_model is None:
                print(f"❌ Pruning returned None for {t['name']}")
                if os.path.exists(temp_path): os.remove(temp_path)
                pruned_details.append({'name': t['name'], 'status': 'failed', 'original': t['num_channels'], 'remaining': t['num_channels']});
                continue
            # Accumulate weights
            # Copy pruned weights back
            block_main = detection_model[t['block_idx']]
            conv_main = get_all_conv2d_layers(block_main)[t['conv_in_block_idx']]
            block_upd = updated_model.model.model[t['block_idx']]
            conv_upd = get_all_conv2d_layers(block_upd)[t['conv_in_block_idx']]
            with torch.no_grad():
                conv_main.weight.copy_(conv_upd.weight)
                if conv_main.bias is not None and conv_upd.bias is not None:
                    conv_main.bias.copy_(conv_upd.bias)
            remaining = count_active_channels(conv_main)
            print(f"✅ Pruned {t['name']}: {t['num_channels']} → {remaining}")
            pruned_details.append({'name': t['name'], 'status': 'success', 'original': t['num_channels'], 'remaining': remaining})
            if os.path.exists(temp_path): os.remove(temp_path)

        # Fine-tune
        try:
            print("🔄 Fine-tuning for 5 epochs...")
            model.train(data=data_yaml, epochs=5, verbose=True)
            print("✅ Fine-tuning done")
        except Exception as e:
            print(f"⚠️  Fine-tuning failed: {e}")

    print("\n"+"="*70)
    print("FINAL SUMMARY (Blocks 1,2,3)")
    print("="*70)
    ok = sum(1 for d in pruned_details if d['status'] == 'success')
    print(f"Successfully pruned: {ok}/{len(pruned_details)}")
    for d in pruned_details:
        print(f" - {d['name']}: {d['status']} ({d['original']} → {d.get('remaining', d['original'])})")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


