# How the Activation Algorithm Chooses Channels to Keep

## Overview

The activation-based pruning algorithm uses a sophisticated multi-step process that combines **activation analysis**, **class separability**, **clustering**, and **weight importance** to select which channels to keep.

---

## Step-by-Step Process

### **Step 1: Activation Extraction** 📊

**What happens:**
- For each training image, the algorithm runs a forward pass through a **mini-net** (blocks up to the target Conv layer)
- Extracts feature map activations at the spatial location of each ground-truth object
- Records activations per channel, per class, per matched detection (IoU > 0.5)

**Code location:** `yolov8_utils.py:process_sample_for_layer_idx_v8()`

**Result:**
```python
train_activations = {
    channel_0: {class_0: [act1, act2, ...], class_1: [act3, ...], ...},
    channel_1: {class_0: [act5, ...], ...},
    ...
}
```

---

### **Step 2: Separability Matrix Construction** 🔍

**What happens:**
- For each channel, compute **Jeffries-Matusita (JM) distance** between all pairs of classes
- JM distance measures how well a channel separates two classes (higher = better separation)
- Builds a matrix: **Rows = Channels**, **Columns = Class pairs**

**Code location:** `yolo_layer_pruner.py:_calculate_separation_matrix()`

**Example:**
- If you have 20 classes, there are C(20,2) = 190 class pairs
- For 128 channels, you get a **128 × 190** separability matrix
- Each value tells you: "How well does channel X separate class pair (A, B)?"

**Why this matters:** Channels that strongly separate different classes are more important for detection.

---

### **Step 3: Dimensionality Reduction** 📉

**What happens:**
- Apply **t-SNE** (t-Distributed Stochastic Neighbor Embedding) to reduce the separability matrix from high dimensions to **2D**
- This creates a compact feature space where similar channels cluster together

**Code location:** `yolo_layer_pruner.py:_reduce_dimension()`

**Result:** 
- Each channel is now represented as a point in 2D space
- Channels with similar class-separation patterns are nearby
- Channels with unique separation patterns are far apart

---

### **Step 4: K-Medoids Clustering (Multiple K Values)** 🎯

**What happens:**
- Try different numbers of clusters `k` from 2 to `num_channels - 1`
- For each `k`, perform **k-medoids clustering** on the reduced 2D space
- k-medoids finds representative channels (medoids) and groups similar channels together

**Code location:** `clustering.py:select_optimal_components()`

**Why multiple k values?** We don't know ahead of time how many channel groups exist. Testing different k values helps find the optimal grouping.

---

### **Step 5: MSS (Mean Simplified Silhouette) Evaluation** 📈

**What happens:**
- For each clustering (each k value), compute the **MSS score**
- MSS measures clustering quality:
  - **High MSS**: Channels within clusters are similar; clusters are well-separated
  - **Low MSS**: Poor clustering quality

**Formula:**
```
MSS = mean( (nearest_cluster_distance - intra_cluster_distance) / max(intra_cluster_distance, nearest_cluster_distance) )
```

**Code location:** `distance.py:calc_mss_value()`

**Result:** A curve of MSS scores vs. k values:
```
k=2  → MSS=0.45
k=3  → MSS=0.52
k=4  → MSS=0.58
k=5  → MSS=0.61  ← peak
k=6  → MSS=0.59
...
```

---

### **Step 6: Knee Point Detection (Optimal K Selection)** 🔪

**What happens:**
- Use the **Kneedle algorithm** to find the "knee" point in the MSS curve
- The knee point represents the optimal number of clusters: **the sweet spot between too few and too many clusters**

**Code location:** `clustering.py:get_knee()`

**Why knee point?** 
- Too few clusters (small k): Loses important distinctions between channels
- Too many clusters (large k): Over-fits, doesn't generalize well
- **Knee point = optimal balance**

**Result:** `optimal_k = knee_point` (e.g., k=5 means we want 5 representative channels)

---

### **Step 7: Weighted Selection (Max Weight per Cluster)** ⚖️

**What happens:**
- For the optimal clustering (from Step 6), we have clusters and their medoids
- **Important**: We don't actually keep the medoids! Instead:
  - Calculate weights for each channel (L1 norm of Conv layer weights OR activation magnitude)
  - For each cluster, find ALL channels in that cluster
  - Within each cluster, select the channel with the **highest weight** (most important)
  - The medoid is only used to identify which cluster we're working with

**Code location:** `clustering.py:find_optimal_weighted_medoids()`

**Key code logic:**
```python
for each medoid:
    cluster_indices = find_all_channels_in_same_cluster(medoid)
    cluster_weights = weights[cluster_indices]
    max_weight_channel = cluster_indices[argmax(cluster_weights)]  # ← Selected, not medoid!
    keep_channels.append(max_weight_channel)
```

**Why max weight instead of medoid?** 
- Two channels might cluster together (similar separation patterns)
- But one might have much stronger weights → more important for the network
- The medoid represents the "center" of similar patterns, but the **max-weight channel is more important**
- **We keep the most important one, not necessarily the representative center!**

**Result:** A list of channel indices to keep (one per cluster, selected by **highest weight**, not medoid)

---

### **Step 8: Pruning Ratio Application** ✂️

**What happens:**
- The algorithm aims for a target pruning ratio (typically 50-75%)
- Target: `max(num_channels // 2, num_channels // 4)` (25-50% kept)

**Code location:** `pruning_yolo_v8.py:prune_conv2d_in_block_with_activations()`

**Example:**
- If you have 128 channels and optimal clustering found 32 clusters
- But target is 64 channels (50% kept)
- The algorithm might need additional selection or adjustment

**Note:** The current implementation uses `select_optimal_components()` which may return fewer channels than the target if the knee point suggests fewer clusters are optimal.

---

### **Step 9: Channel Zeroing** 🗑️

**What happens:**
- Zero out weights and biases for all channels **NOT** in the keep list
- Channels in the keep list remain unchanged

**Code location:** `pruning_yolo_v8.py:prune_conv2d_in_block_with_activations()`

```python
indices_to_remove = [i for i in all_indices if i not in optimal_components]
target_conv_layer.weight[indices_to_remove] = 0
target_conv_layer.bias[indices_to_remove] = 0
```

---

## Visual Summary

```
Training Images
    ↓
[Forward Pass] → Extract activations per channel, per class
    ↓
[Build Separability Matrix] → JM distances (channels × class pairs)
    ↓
[t-SNE Reduction] → 2D embedding (each channel = point)
    ↓
[K-Medoids Clustering] → Test k=2,3,4,...,N
    ↓
[MSS Evaluation] → Quality score for each k
    ↓
[Knee Detection] → Find optimal k*
    ↓
[Weighted Selection] → For each cluster, pick channel with MAX WEIGHT (not medoid!)
    ↓
[Keep Channels] → Zero out all others
```

---

## Key Insights

1. **Class Separation Matters**: Channels that better separate classes (higher JM distance) are favored.

2. **Clustering Groups Similar Channels**: Similar separation patterns → similar channels → can prune redundant ones.

3. **Weight Importance**: Even within a cluster, we prioritize channels with stronger weights.

4. **Data-Driven**: The algorithm learns which channels are important from actual activations on your dataset.

5. **Adaptive**: The optimal number of channels is automatically determined (not fixed at 50%).

---

## Comparison with Gamma Pruning

| Aspect | Activation Pruning | Gamma Pruning |
|--------|-------------------|---------------|
| **Selection Criterion** | Class separability + weights | BN gamma magnitude |
| **Data Dependency** | Requires training data | Model weights only |
| **Computational Cost** | High (forward passes + clustering) | Low (sorting) |
| **Quality** | Considers task-specific importance | Generic importance |

---

## Code References

- **Activation extraction**: `yolov8_utils.py:process_sample_for_layer_idx_v8()`, `aggregate_activations_from_matches()`
- **Separability matrix**: `yolo_layer_pruner.py:create_layer_space()`, `_calculate_separation_matrix()`
- **Clustering & selection**: `clustering.py:select_optimal_components()`, `find_optimal_weighted_medoids()`
- **MSS calculation**: `distance.py:calc_mss_value()`
- **Main pruning function**: `pruning_yolo_v8.py:prune_conv2d_in_block_with_activations()`
