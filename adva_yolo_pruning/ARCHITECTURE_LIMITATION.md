# Architecture Limitation: Why Sequential Pruning Fails Across Blocks

## The Core Problem

When pruning multiple layers sequentially, we encounter a fundamental architectural limitation:

**We cannot update `sliced_block` to reflect pruned channels.**

## Why This Happens

### The `sliced_block` Construction

When pruning a layer, the code builds `sliced_block` like this:

```python
# Build sliced block for this layer
blocks_up_to = list(detection_model[:block_idx])  # Includes ALL previous blocks
block = detection_model[block_idx]
partial_block = nn.Sequential(*submodules)
sliced_block = nn.Sequential(*(blocks_up_to + [partial_block]))
```

### The Dependency Chain

In a typical YOLOv8 architecture:

```
Block 0 → Block 1 → Block 2 → Block 3 → Block 4 → Block 5
```

When pruning Layer 2 in Block 3:
- `sliced_block` includes Blocks 0, 1, 2, and partial Block 3
- If Block 1 was pruned earlier, its channels are now reduced (e.g., 128→64)
- But `sliced_block` still expects the original architecture (128 channels)
- This causes: `expected input[1, 128, 88, 128] to have 64 channels, but got 128 channels instead`

## Why We Can't Fix It

### Option 1: Update sliced_block After Pruning? ❌ NO

We cannot dynamically modify the architecture of `sliced_block` after pruning because:

1. **PyTorch nn.Sequential is immutable**: Once created, you can't change its architecture
2. **Channel dimensions are fixed at creation**: Can't resize Conv2d layers dynamically
3. **Would require reconstructing the entire sliced_block**: Too complex and error-prone

### Option 2: Build sliced_block with Awareness of Pruning? ❌ NO

Even if we build `sliced_block` after pruning, the problem remains:

```python
# After pruning Block 1: 128→64 channels
blocks_up_to = list(detection_model[:block_idx])  # Block 1 now has 64 channels
sliced_block = nn.Sequential(*(blocks_up_to + [partial_block]))

# But Block 2 still expects 128 channels as input!
# Mismatch: Block 1 output is 64, Block 2 input expects 128
```

The architecture is fixed, so subsequent layers expect original channel dimensions.

## Current Workaround

### Solution: Prune Only Independent Layers

We select layers from **Block 1 only** because:

1. Block 1 doesn't depend on any previous blocks (it's the first block)
2. All Block 1 layers are independent of each other
3. `sliced_block` for Block 1 doesn't include pruned layers

### Code

```python
# Select ALL layers from Block 1 only (different conv indices in same block)
selected_convs = []

if 1 in block_layers:
    block_layers[1].sort(key=lambda x: x['num_channels'], reverse=True)
    
    # Take up to layers_to_prune layers from Block 1
    for i in range(min(layers_to_prune, len(block_layers[1]))):
        best_layer = block_layers[1][i]
        selected_convs.append(best_layer)
```

### Why This Works

- **All layers from Block 1**: No cross-block dependencies
- **Same sliced_block for all**: All use `detection_model[:1]` which doesn't include pruned layers
- **Independent pruning**: Each layer can be pruned without affecting others

## Limitations

### What We CAN'T Do

1. ❌ Prune layers from multiple blocks (e.g., Block 1, 2, 3)
2. ❌ Prune subsequent layers that depend on pruned earlier layers
3. ❌ Update `sliced_block` to reflect pruned channels dynamically

### What We CAN Do

1. ✅ Prune multiple independent layers from Block 1 only
2. ✅ Prune layers that don't have dependencies on pruned layers
3. ✅ Use the iterative pruning + fine-tuning approach

## Future Work

### Possible Solutions (Not Implemented)

1. **True Structural Pruning**: Actually remove channels from the architecture (more complex)
2. **Channel Mapping Layer**: Add a layer that maps pruned channels to expected dimensions
3. **Separate Activation Extraction**: Build activations for each layer independently

But these would require significant changes to the existing algorithm.

## Summary

The sequential pruning limitation is due to **architectural constraints**, not the pruning algorithm itself:

- We can't modify `sliced_block` after pruning
- Pruned layers change channel dimensions that subsequent layers expect
- The only reliable way to prune multiple layers is to use **independent layers** from the same block

This is why we limit to Block 1 only—it's the simplest and most reliable approach given the architectural constraints.

