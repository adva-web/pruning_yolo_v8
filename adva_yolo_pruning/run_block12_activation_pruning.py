#!/usr/bin/env python3
"""
Prune exactly 3 layers using activation-based pruning:
  - Two highest-channel Conv2d layers from Block 1
  - Conv 0 from Block 2

Order: Block 1 → Block 1 → Block 2
After each prune: fine-tune and accumulate weights into the main model.
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
from pruning_yolo_v8 import (
    prune_conv2d_in_block_with_activations,
    YoloLayerPruner,
    select_optimal_components
)
from pruning_c2f_activation import prune_conv2d_in_c2f_with_activations
from c2f_utils import is_c2f_block


def select_layers_blocks_1_3_5(detection_model, total_layers=3):
    """
    Select Conv 0 from blocks 1, 3, and 5 (one conv layer per block),
    in ascending block order, up to total_layers.
    Returns list of dicts with keys: block_idx, conv_in_block_idx, name, num_channels
    """
    target_blocks = [1, 3, 5]
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


def apply_pruned_weights(main_model, pruned_model, block_idx, conv_in_block_idx):
    """Copy pruned conv (and following BN if exists) from pruned_model into main_model."""
    tm_main = main_model.model
    dm_main = tm_main.model
    tm_pruned = pruned_model.model
    dm_pruned = tm_pruned.model

    block_main = dm_main[block_idx]
    block_pruned = dm_pruned[block_idx]

    convs_main = get_all_conv2d_layers(block_main)
    convs_pruned = get_all_conv2d_layers(block_pruned)

    target_conv_main = convs_main[conv_in_block_idx]
    target_conv_pruned = convs_pruned[conv_in_block_idx]

    with torch.no_grad():
        target_conv_main.weight.copy_(target_conv_pruned.weight)
        if target_conv_main.bias is not None and target_conv_pruned.bias is not None:
            target_conv_main.bias.copy_(target_conv_pruned.bias)

    # Copy the next BatchNorm if present
    def find_next_bn(block):
        hit = -1
        for i, m in enumerate(block.children()):
            if isinstance(m, nn.Conv2d):
                hit += 1
                if hit == conv_in_block_idx:
                    # from here, find next BN
                    for n in block.children():
                        if isinstance(n, nn.BatchNorm2d):
                            return n
                    return None
        return None

    bn_main = find_next_bn(block_main)
    bn_pruned = find_next_bn(block_pruned)
    if bn_main is not None and bn_pruned is not None:
        with torch.no_grad():
            bn_main.weight.copy_(bn_pruned.weight)
            bn_main.bias.copy_(bn_pruned.bias)
            bn_main.running_mean.copy_(bn_pruned.running_mean)
            bn_main.running_var.copy_(bn_pruned.running_var)


def main():
    # Config
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

    # Load data
    train_data = load_training_data(data_yaml, max_samples=50)
    if len(train_data) == 0:
        print("❌ No training data loaded.")
        return False
    try:
        valid_data = load_validation_data(data_yaml, max_samples=30)
    except Exception:
        valid_data = train_data[:20]

    # Load model
    model = YOLO(model_path)
    torch_model = model.model
    detection_model = torch_model.model

    # Select target layers: Conv 0 from blocks 1, 3, 5
    targets = select_layers_blocks_1_3_5(detection_model, total_layers=total_layers_to_prune)
    if len(targets) == 0:
        print("❌ No eligible layers found in blocks 1, 3, 5.")
        return False

    print("\n🎯 Target layers (in order):")
    for i, t in enumerate(targets):
        print(f"  {i+1}. {t['name']} ({t['num_channels']} channels)")

    pruned_details = []

    # Iterate and prune: use original algorithm for Conv blocks; use hook for C2f
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
            # Save current state to prune independently on this state
            temp_path = f"temp_iter_{i+1}.pt"
            model.save(temp_path)

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
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                pruned_details.append({'name': t['name'], 'status': 'failed', 'original': t['num_channels'], 'remaining': t['num_channels']})
                continue

            if updated_model is None:
                print(f"❌ Pruning returned None for {t['name']}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                pruned_details.append({'name': t['name'], 'status': 'failed', 'original': t['num_channels'], 'remaining': t['num_channels']})
                continue

            # Accumulate pruned weights back to the main model
            apply_pruned_weights(model, updated_model, t['block_idx'], t['conv_in_block_idx'])

            # Stats after accumulation
            conv_after = get_all_conv2d_layers(model.model.model[t['block_idx']])[t['conv_in_block_idx']]
            remaining = count_active_channels(conv_after)
            print(f"✅ Pruned {t['name']}: {t['num_channels']} → {remaining}")
            pruned_details.append({'name': t['name'], 'status': 'success', 'original': t['num_channels'], 'remaining': remaining})

            if os.path.exists(temp_path):
                os.remove(temp_path)

        # Fine-tune
        try:
            print(f"🔄 Fine-tuning for {epochs_per_finetune} epochs...")
            model.train(data=data_yaml, epochs=epochs_per_finetune, verbose=True)
            print("✅ Fine-tuning done")
        except Exception as e:
            print(f"⚠️  Fine-tuning failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    ok = sum(1 for d in pruned_details if d['status'] == 'success')
    print(f"Successfully pruned: {ok}/{len(pruned_details)}")
    for d in pruned_details:
        print(f" - {d['name']}: {d['status']} ({d['original']} → {d.get('remaining', d['original'])})")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


