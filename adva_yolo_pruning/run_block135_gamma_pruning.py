#!/usr/bin/env python3
"""
Prune exactly 3 layers using BN-gamma-based pruning:
  - Conv 0 from blocks 1, 3, and 5 (in that order)

After each prune: fine-tune and accumulate changes. Mirrors the activation
experiment for apples-to-apples comparison.
"""

import os
import sys

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


def select_layers_blocks_1_3_5(detection_model, total_layers=3):
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


def find_following_bn(block: nn.Module, conv_in_block_idx: int):
    hit = -1
    for m in block.children():
        if isinstance(m, nn.Conv2d):
            hit += 1
            if hit == conv_in_block_idx:
                # from here, return first BN encountered
                for n in block.children():
                    if isinstance(n, nn.BatchNorm2d):
                        return n
                return None
    return None


def gamma_prune_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d, keep_ratio: float = 0.5) -> int:
    """Soft-prune output channels using BN gamma magnitudes (abs). Returns removed count."""
    out_ch = conv.weight.shape[0]
    k_keep = max(int(out_ch * keep_ratio), max(1, out_ch // 4))
    if bn is None:
        # Fallback to conv weight magnitude if no BN
        scores = conv.weight.detach().abs().mean(dim=(1, 2, 3))
    else:
        scores = bn.weight.detach().abs()
    order = torch.argsort(scores, descending=True)
    keep_idx = set(order[:k_keep].tolist())
    remove_idx = [i for i in range(out_ch) if i not in keep_idx]
    if not remove_idx:
        return 0
    with torch.no_grad():
        conv.weight[remove_idx] = 0
        if conv.bias is not None:
            conv.bias[remove_idx] = 0
        if bn is not None:
            bn.weight[remove_idx] = 0
            bn.bias[remove_idx] = 0
            # keep running stats as-is to avoid destabilizing immediately
    return len(remove_idx)


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

    # Select targets
    targets = select_layers_blocks_1_3_5(detection_model, total_layers=total_layers_to_prune)
    if len(targets) == 0:
        print("❌ No eligible layers found in blocks 1, 3, 5.")
        return False

    print("\n🎯 Target layers (in order):")
    for i, t in enumerate(targets):
        print(f"  {i+1}. {t['name']} ({t['num_channels']} channels)")

    pruned_details = []

    # Iterate and gamma-prune
    for i, t in enumerate(targets):
        print(f"\n{'='*70}\nITERATION {i+1}/{len(targets)}\n{'='*70}")
        block = detection_model[t['block_idx']]
        conv = get_all_conv2d_layers(block)[t['conv_in_block_idx']]
        bn = find_following_bn(block, t['conv_in_block_idx'])

        removed = gamma_prune_conv_and_bn(conv, bn, keep_ratio=0.5)
        remaining = count_active_channels(conv)
        print(f"✅ Gamma-pruned {t['name']}: {t['num_channels']} → {remaining} (removed {removed})")
        pruned_details.append({'name': t['name'], 'status': 'success', 'original': t['num_channels'], 'remaining': remaining})

        # Fine-tune
        try:
            print(f"🔄 Fine-tuning for {epochs_per_finetune} epochs...")
            model.train(data=data_yaml, epochs=epochs_per_finetune, verbose=True)
            print("✅ Fine-tuning done")
        except Exception as e:
            print(f"⚠️  Fine-tuning failed: {e}")

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY (Gamma)")
    print("="*70)
    ok = sum(1 for d in pruned_details if d['status'] == 'success')
    print(f"Successfully pruned: {ok}/{len(pruned_details)}")
    for d in pruned_details:
        print(f" - {d['name']}: {d['status']} ({d['original']} → {d.get('remaining', d['original'])})")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


