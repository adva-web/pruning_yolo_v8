================================================================================
ACTIVATION-BASED CHANNEL PRUNING ALGORITHM FOR YOLO: 
A COMPREHENSIVE TECHNICAL EXPLANATION
================================================================================

ABSTRACT
--------
This document provides a detailed technical explanation of the activation-based 
channel pruning algorithm used for compressing YOLO object detection models. 
The algorithm leverages feature activation patterns extracted from training data 
to identify and prune redundant channels in convolutional layers while maintaining 
detection accuracy.


1. INTRODUCTION AND MOTIVATION
==============================

1.1 Problem Statement
---------------------
Deep convolutional neural networks, particularly YOLO (You Only Look Once) object 
detection models, contain millions of parameters and require significant 
computational resources for inference. Channel pruning aims to reduce model 
complexity by removing entire channels (feature maps) from convolutional layers, 
resulting in:
- Reduced model size
- Faster inference time
- Lower memory consumption
- Lower computational requirements

However, indiscriminate pruning can severely degrade model performance. The 
challenge is to identify which channels are most critical for maintaining 
detection accuracy.

1.2 Activation-Based Approach
------------------------------
Unlike weight-magnitude-based pruning methods, our activation-based approach 
analyzes how channels respond to actual training data. Channels that contribute 
significantly to distinguishing between different object classes are considered 
more important and retained during pruning.


2. ALGORITHM OVERVIEW
=====================

The activation-based pruning algorithm consists of six main phases:

1. Activation Extraction: Extract feature activations from target layers using 
   training data
2. Separation Matrix Construction: Compute inter-class separability for each 
   channel
3. Feature Space Embedding: Create a low-dimensional representation of channel 
   characteristics
4. Optimal Channel Selection: Use clustering and importance metrics to select 
   critical channels
5. Weight Zeroing: Prune selected channels by zeroing their weights
6. Fine-tuning: Retrain the model to recover performance


3. DETAILED ALGORITHM STEPS
===========================

3.1 Phase 1: Activation Extraction
-----------------------------------

3.1.1 Mini-Network Construction

For each target convolutional layer Conv_lj in block l at position j, we construct 
a "mini-network" that terminates at this layer:

Mini-Net = [Block_0, Block_1, ..., Block_{l-1}, Partial_Block_l]

Where:
- Block_0 through Block_{l-1}: All blocks preceding the target block
- Partial_Block_l: All layers in block l up to (but not including) Conv_lj

Implementation Details:
# Build sliced block: all blocks before, plus partial block up to target Conv2d
blocks_up_to = list(detection_model[:block_idx])
block = detection_model[block_idx]
submodules = []
conv_count = 0
for sublayer in block.children():
    submodules.append(sublayer)
    if isinstance(sublayer, nn.Conv2d):
        if conv_count == conv_in_block_idx:
            break
        conv_count += 1
partial_block = nn.Sequential(*submodules)
sliced_block = nn.Sequential(*(blocks_up_to + [partial_block]))
mini_net = build_mini_net(sliced_block, target_conv_layer)

3.1.2 Feature Map Extraction

For each training sample x_i:
1. Forward pass through the mini-network to obtain feature map F_{lj}(x_i)
2. Extract activations at spatial locations corresponding to ground truth 
   bounding boxes
3. Match ground truth boxes with predicted boxes using IoU threshold 
   (typically 0.5)

Patch Extraction:
Given a ground truth bounding box bbox_{gt} = (x1, y1, x2, y2) and class c, we:
1. Compute the corresponding spatial location in the feature map:
   patch_row = int(y_center / stride_h)
   patch_col = int(x_center / stride_w)

2. Extract activation vector: a_{c,i} = F_{lj}(x_i)[:, patch_row, patch_col]
   - Shape: [C] where C is the number of channels
   - Each element a_{c,i}[k] is the activation of channel k at this location

3.1.3 Activation Aggregation

We aggregate activations by channel and class:
A = {
    channel_0: {
        class_0: [a_0, a_1, ..., a_n],
        class_1: [a_0, a_1, ..., a_m],
        ...
    },
    channel_1: { ... },
    ...
}

Mathematical Formulation:
For channel k and class c, we collect all activations:
A[k][c] = {a_{c,i}[k] : (x_i, bbox_i, c) matches with predictions, IoU > 0.5}


3.2 Phase 2: Separation Matrix Construction
--------------------------------------------

3.2.1 Channel-Class Separability

For each channel k, we compute its ability to separate different object classes 
using the Jeffries-Matusita (JM) distance metric.

Class Pair Separation:
For each pair of classes (c_i, c_j) where c_i != c_j:
1. Extract activation distributions:
   - Class c_i: D_{k,i} = A[k][c_i] (all activations for channel k and class c_i)
   - Class c_j: D_{k,j} = A[k][c_j]
2. Compute JM distance: JM(D_{k,i}, D_{k,j})

Jeffries-Matusita Distance:
The JM distance measures the separability between two probability distributions:

JM(p, q) = 2(1 - exp(-B(p, q)))

Where B(p, q) is the Bhattacharyya coefficient:

B(p, q) = (1/8) * (μ_p - μ_q)² * (2/(σ_p² + σ_q²)) + (1/2) * ln((σ_p² + σ_q²)/(2σ_p σ_q))

With:
- μ_p, μ_q: Means of distributions p and q
- σ_p, σ_q: Standard deviations of distributions p and q

Properties:
- Range: [0, 2] where 2 indicates perfect separation
- Symmetric: JM(p, q) = JM(q, p)
- Interpretable: Higher values indicate better class separability

3.2.2 Separation Matrix

We construct a separation matrix S where:
- Rows: Channels (indices 0 to C-1)
- Columns: Class pairs (all combinations of 2 classes from N total classes)

S[k, (i,j)] = JM(A[k][c_i], A[k][c_j])

The total number of columns = C(N,2) = N(N-1)/2 for N classes.

Example for 3 classes:
- Column 0: JM(channel_k, class_0 vs class_1)
- Column 1: JM(channel_k, class_0 vs class_2)
- Column 2: JM(channel_k, class_1 vs class_2)


3.3 Phase 3: Feature Space Embedding
-------------------------------------

3.3.1 Dimensionality Reduction

The separation matrix S has dimensions C × (N(N-1)/2), which can be 
high-dimensional. We apply t-SNE (t-Distributed Stochastic Neighbor Embedding) 
to reduce it to 2D:

S_reduced = t-SNE(S, n_components=2, perplexity=10)

t-SNE Parameters:
- n_components=2: Embed into 2D space for visualization and clustering
- perplexity=10: Balances local and global structure preservation
- max_iter=1000: Maximum iterations for convergence

Why t-SNE?
- Preserves local neighborhood structure
- Groups similar channels together in the embedding space
- Helps identify clusters of channels with similar separation patterns

3.3.2 Reduced Feature Space

Each channel k is now represented as a 2D point:
P_k = (x_k, y_k) ∈ ℝ²

Where channels with similar class separation patterns are close together.


3.4 Phase 4: Optimal Channel Selection
--------------------------------------

4.1 K-Medoids Clustering

We use K-Medoids clustering on the reduced feature space to group similar channels:

Algorithm:
For each candidate number of clusters k ∈ [2, C-1]:
1. Perform K-Medoids clustering: clusters_k = KMedoids(P, k)
2. Compute Mean Simplified Silhouette (MSS) score: MSS_k = compute_MSS(clusters_k)

K-Medoids vs K-Means:
- Uses actual data points (medoids) as cluster centers
- More robust to outliers
- Computationally efficient with FasterPAM algorithm

4.2 Mean Simplified Silhouette (MSS)

The MSS measures clustering quality by evaluating intra-cluster and 
inter-cluster distances:

For each point i in cluster C_j:
- Intra-cluster distance: a_i = distance(P_i, medoid(C_j))
- Nearest-cluster distance: b_i = mean(distance(P_i, medoid(C_k))) for all k != j

Silhouette score:
s_i = (b_i - a_i) / max(a_i, b_i)

MSS value:
MSS = mean({s_i : i = 1, ..., C})

Properties:
- Range: [-1, 1], with higher values indicating better clustering
- Perfect clustering: MSS = 1.0 (all points at cluster centers)

4.3 Optimal Cluster Count Selection (Knee Detection)

We identify the optimal number of clusters using the "elbow method" or 
"knee detection":

Knee Detection Algorithm:
1. Plot MSS_k vs k for k ∈ [2, C-1]
2. Find the "knee point" where MSS improvement plateaus
3. Use the KneeLocator algorithm:
   knee = KneeLocator(
       x=k_values,
       y=mss_values,
       curve='concave',      # MSS curve is concave
       direction='increasing',  # Higher MSS is better
       interp_method='polynomial',
       polynomial_degree=2
   )
   optimal_k = knee.knee

Rationale:
- Too few clusters: Lose channel diversity
- Too many clusters: Over-segmentation, similar channels split
- Knee point: Optimal balance between clustering quality and compactness

4.4 Weight-Based Channel Selection

After determining optimal cluster count k*, we select representative channels 
from each cluster:

Within-Cluster Selection:
For each cluster C_i with medoid m_i:
1. Identify all channels in cluster: channels_in_cluster = {j : label[j] == i}
2. Compute channel importance using weight magnitude:
   weight[j] = ||W_j||_1 = Σ |W_j[u,v]|
   Where W_j is the weight tensor for channel j
3. Select channel with maximum weight:
   selected[i] = argmax_{j ∈ channels_in_cluster} weight[j]

Final Selection:
channels_to_keep = {selected[i] : i = 1, ..., k*}
channels_to_remove = {0, 1, ..., C-1} - channels_to_keep


3.5 Phase 5: Weight Zeroing (Pruning)
--------------------------------------

5.1 Convolutional Layer Pruning

For each channel k in channels_to_remove:
conv_layer.weight[k, :, :, :] = 0  # Zero output channels
if conv_layer.bias is not None:
    conv_layer.bias[k] = 0

Weight Tensor Structure:
- Shape: [out_channels, in_channels, kernel_h, kernel_w]
- Pruning zeroes entire output channels

5.2 Batch Normalization Pruning

For corresponding BatchNorm layers:
bn_layer.weight[k] = 0      # Scale parameter
bn_layer.bias[k] = 0        # Shift parameter
bn_layer.running_mean[k] = 0
bn_layer.running_var[k] = 1

Rationale:
- Zeroed BN parameters ensure zeroed channels remain inactive
- running_var = 1 maintains numerical stability


3.6 Phase 6: Fine-tuning
-------------------------

After pruning, we retrain the model on the original training dataset:
model = train(model, train_data, epochs=E)

Purpose:
- Recover accuracy lost during pruning
- Allow remaining channels to adapt
- Optimize weights of kept channels


4. MATHEMATICAL FORMULATION
===========================

4.1 Complete Algorithm

Given:
- Model M with convolutional layer Conv_{l,j} at block l, position j
- Training dataset D = {(x_i, y_i)}
- Target pruning ratio ρ (e.g., 0.5 for 50% pruning)

Step 1: Extract Activations
∀ (x_i, y_i) ∈ D:
    F_i = MiniNet(x_i)  # Feature map at Conv_{l,j}
    A[k][c] = {F_i[k, r, c] : y_i matches prediction, IoU > 0.5}

Step 2: Compute Separation Matrix
∀ k ∈ [0, C-1], ∀ (c_i, c_j) pairs:
    S[k, (i,j)] = JM(A[k][c_i], A[k][c_j])

Step 3: Embed to Low-Dimensional Space
P = t-SNE(S)  # P ∈ ℝ^{C × 2}

Step 4: Find Optimal Channels
∀ k ∈ [2, C-1]:
    clusters_k = KMedoids(P, k)
    MSS_k = compute_MSS(clusters_k, P)
k* = KneeLocator({k}, {MSS_k})
clusters_optimal = KMedoids(P, k*)

Step 5: Select Channels
∀ cluster C_i in clusters_optimal:
    selected[i] = argmax_{j ∈ C_i} ||W_j||_1
channels_to_keep = {selected[i] : i = 1, ..., k*}

Step 6: Prune
∀ k ∈ channels_to_remove:
    W[k, :, :, :] = 0
    BN_parameters[k] = 0


5. ALGORITHM VARIANTS
=====================

5.1 Medoid-Based Selection

Instead of selecting max-weight channels per cluster, return actual medoids:
channels_to_keep = {medoids[i] : i = 1, ..., k*}

Advantage: Preserves geometric center of clusters

5.2 Gamma-Based Selection

Use BatchNorm gamma values instead of weight magnitudes:
importance[j] = |γ_j|  # BN gamma (scale parameter)
selected[i] = argmax_{j ∈ C_i} importance[j]

Advantage: BN gamma often more indicative of channel importance

5.3 Hybrid Approaches

- Activation + Max Weight: Standard approach (clustering + weight selection)
- Activation + Medoid: Clustering + medoid selection
- Activation + Max Gamma: Clustering + BN gamma selection
- Pure Gamma: No clustering, just prune based on gamma magnitude


6. IMPLEMENTATION CONSIDERATIONS
================================

6.1 Computational Complexity

- Activation Extraction: O(N × H × W × C) per sample
- JM Distance: O(K) where K is number of activation values
- Separation Matrix: O(C × P × K) where P is number of class pairs
- t-SNE: O(C² × iterations)
- K-Medoids: O(C² × k_max)
- Total: Approximately O(C² × max_iterations)

6.2 Memory Requirements

- Separation matrix: C × (N(N-1)/2) floats
- Reduced space: C × 2 floats
- Activation storage: Variable, depends on number of matched objects

6.3 Parallelization Opportunities

- Activation extraction: Process different samples in parallel
- JM distance computation: Compute for different class pairs in parallel
- Clustering: Use GPU-accelerated t-SNE and distance computations


7. ADVANTAGES AND LIMITATIONS
=============================

7.1 Advantages

1. Data-Driven: Uses actual feature patterns rather than static weight values
2. Class-Aware: Considers which channels distinguish between object classes
3. Adaptive: Automatically determines optimal pruning ratio via knee detection
4. Robust: Clustering handles channel redundancy effectively

7.2 Limitations

1. Sequential Dependencies: Cannot easily prune layers with architectural 
   dependencies
2. Concatenation Blocks: Struggles with C2f/Concat blocks due to branching 
   structure
3. Computational Cost: Requires forward passes on training data for activation 
   extraction
4. Memory Intensive: Must store activations for all matched detections


8. EXPERIMENTAL RESULTS SUMMARY
================================

(Include your experimental results here: pruning ratios, mAP scores, inference 
times, etc.)


9. CONCLUSION
=============

The activation-based channel pruning algorithm provides an effective method for 
compressing YOLO models while maintaining detection accuracy. By leveraging 
feature activation patterns and class separability metrics, it identifies and 
retains the most critical channels for object detection tasks.


REFERENCES
==========

1. Jeffries-Matusita Distance: A statistical measure of class separability
2. K-Medoids Clustering: Partitioning Around Medoids (PAM) algorithm
3. t-SNE: t-distributed Stochastic Neighbor Embedding for dimensionality reduction
4. Knee Detection: Elbow method for optimal parameter selection


================================================================================
Document Version: 1.0
Last Updated: [Date]
Author: [Your Name/Team]
================================================================================


4.2 COMPLETE ALGORITHM PSEUDOCODE
----------------------------------

Function ACTIVATION_BASED_PRUNING(M, D_train, Conv_lj, classes):
    Input:
        M: YOLO model
        D_train: Training dataset {(x_i, y_i)}
        Conv_lj: Target convolutional layer at block l, position j
        classes: List of class labels [c_0, c_1, ..., c_{N-1}]
    Output:
        M_pruned: Pruned model
    
    // ============================================================
    // PHASE 1: ACTIVATION EXTRACTION
    // ============================================================
    
    // Step 1.1: Build mini-network
    blocks_before = [M.block_0, M.block_1, ..., M.block_{l-1}]
    partial_block_l = GetPartialBlock(M.block_l, j)  // Up to but not including Conv_lj
    mini_net = Sequential([blocks_before, partial_block_l])
    
    // Step 1.2: Extract activations for each training sample
    A = {}  // Dictionary: A[channel][class] = [activation_values]
    
    FOR each (x_i, y_i) in D_train:
        // Forward pass through mini-network
        F_i = mini_net.forward(x_i)  // Feature map at Conv_lj
        
        // Run full model to get predictions
        predictions = M.forward(x_i)
        
        // Match ground truth boxes with predictions
        FOR each gt_box in y_i:
            gt_class = gt_box.class_id
            gt_bbox = gt_box.coordinates
            
            // Find best matching prediction
            best_pred = None
            max_iou = 0
            FOR each pred_box in predictions:
                iou = CalculateIoU(gt_bbox, pred_box.bbox)
                IF iou > max_iou AND iou > 0.5:
                    max_iou = iou
                    best_pred = pred_box
            
            // If matched, extract activation patch
            IF best_pred != None:
                // Map bounding box center to feature map coordinates
                (patch_row, patch_col) = MapToFeatureSpace(gt_bbox, F_i.stride)
                
                // Extract activation vector for all channels at this location
                activation_vector = F_i[:, patch_row, patch_col]  // Shape: [C]
                
                // Store activations by channel and class
                FOR channel_k = 0 to C-1:
                    IF A[channel_k] does not exist:
                        A[channel_k] = {}
                    IF A[channel_k][gt_class] does not exist:
                        A[channel_k][gt_class] = []
                    
                    A[channel_k][gt_class].append(activation_vector[channel_k])
                END FOR
            END IF
        END FOR
    END FOR
    
    // ============================================================
    // PHASE 2: SEPARATION MATRIX CONSTRUCTION
    // ============================================================
    
    // Step 2.1: Generate all class pairs
    class_pairs = []
    FOR i = 0 to N-1:
        FOR j = i+1 to N-1:
            class_pairs.append((i, j))
        END FOR
    END FOR
    
    // Step 2.2: Compute separation matrix
    C = Conv_lj.num_output_channels
    num_pairs = |class_pairs|  // = N*(N-1)/2
    S = zeros(C, num_pairs)  // Separation matrix
    
    FOR channel_k = 0 to C-1:
        FOR pair_idx = 0 to num_pairs-1:
            (c_i, c_j) = class_pairs[pair_idx]
            
            // Get activation distributions for both classes
            D_i = A[channel_k][c_i]  // Array of activations for class c_i
            D_j = A[channel_k][c_j]  // Array of activations for class c_j
            
            // Compute Jeffries-Matusita distance
            S[channel_k, pair_idx] = JM_DISTANCE(D_i, D_j)
        END FOR
    END FOR
    
    // ============================================================
    // PHASE 3: FEATURE SPACE EMBEDDING
    // ============================================================
    
    // Apply t-SNE to reduce dimensionality
    P = TSNE(S, n_components=2, perplexity=10, max_iter=1000)
    // P is now C × 2 matrix where each row is a 2D point representing a channel
    
    // ============================================================
    // PHASE 4: OPTIMAL CHANNEL SELECTION
    // ============================================================
    
    // Step 4.1: Find optimal number of clusters using knee detection
    mss_values = []
    k_values = []
    
    FOR k = 2 to C-1:
        // Perform K-Medoids clustering
        clusters_k = KMEDOIDS(P, k)
        
        // Compute Mean Simplified Silhouette score
        mss_k = COMPUTE_MSS(clusters_k, P)
        mss_values.append(mss_k)
        k_values.append(k)
        
        // Early termination if perfect clustering achieved
        IF mss_k >= 1.0:
            BREAK
        END IF
    END FOR
    
    // Find knee point using KneeLocator
    optimal_k = KNEE_DETECTOR(k_values, mss_values, 
                              curve='concave', 
                              direction='increasing')
    
    // Perform final clustering with optimal k
    clusters_optimal = KMEDOIDS(P, optimal_k)
    
    // Step 4.2: Select representative channel from each cluster
    channels_to_keep = []
    weights = Conv_lj.weight  // Shape: [C, in_channels, kernel_h, kernel_w]
    
    FOR cluster_i = 0 to optimal_k-1:
        // Get all channels in this cluster
        cluster_channels = []
        FOR channel_j = 0 to C-1:
            IF clusters_optimal.labels[channel_j] == cluster_i:
                cluster_channels.append(channel_j)
            END IF
        END FOR
        
        // Compute weight magnitude for each channel in cluster
        max_weight = -infinity
        best_channel = -1
        
        FOR each channel_j in cluster_channels:
            // Compute L1 norm of channel weights
            weight_magnitude = SUM(ABS(weights[channel_j, :, :, :]))
            
            IF weight_magnitude > max_weight:
                max_weight = weight_magnitude
                best_channel = channel_j
            END IF
        END FOR
        
        channels_to_keep.append(best_channel)
    END FOR
    
    // Determine channels to remove
    channels_to_remove = []
    FOR channel_k = 0 to C-1:
        IF channel_k NOT IN channels_to_keep:
            channels_to_remove.append(channel_k)
        END IF
    END FOR
    
    // ============================================================
    // PHASE 5: WEIGHT ZEROING (PRUNING)
    // ============================================================
    
    // Step 5.1: Zero out convolutional layer weights
    FOR each channel_k in channels_to_remove:
        Conv_lj.weight[channel_k, :, :, :] = 0
        IF Conv_lj.bias exists:
            Conv_lj.bias[channel_k] = 0
        END IF
    END FOR
    
    // Step 5.2: Zero out corresponding BatchNorm parameters
    bn_layer = FindBatchNormAfter(Conv_lj)
    IF bn_layer != None:
        FOR each channel_k in channels_to_remove:
            bn_layer.weight[channel_k] = 0      // Scale parameter (gamma)
            bn_layer.bias[channel_k] = 0        // Shift parameter (beta)
            bn_layer.running_mean[channel_k] = 0
            bn_layer.running_var[channel_k] = 1
        END FOR
    END IF
    
    // ============================================================
    // PHASE 6: FINE-TUNING
    // ============================================================
    
    M_pruned = TRAIN(M, D_train, epochs=E)
    
    RETURN M_pruned
END FUNCTION


// ============================================================
// HELPER FUNCTIONS
// ============================================================

Function JM_DISTANCE(p, q):
    // Jeffries-Matusita distance between two distributions
    // Input: p, q are arrays of values representing distributions
    
    μ_p = MEAN(p)
    μ_q = MEAN(q)
    σ_p = STD(p)
    σ_q = STD(q)
    
    // Handle case where std is zero
    IF σ_p == 0:
        σ_p = 1e-10
    END IF
    IF σ_q == 0:
        σ_q = 1e-10
    END IF
    
    // Bhattacharyya coefficient
    B = (1/8) * (μ_p - μ_q)² * (2 / (σ_p² + σ_q²)) +
        (1/2) * LN((σ_p² + σ_q²) / (2 * σ_p * σ_q))
    
    // Jeffries-Matusita distance
    JM = 2 * (1 - EXP(-B))
    
    RETURN JM
END FUNCTION


Function KMEDOIDS(P, k):
    // K-Medoids clustering using FasterPAM algorithm
    // Input: P is C × 2 matrix (2D points), k is number of clusters
    // Output: Dictionary with 'labels', 'medoids', 'medoids_loc'
    
    distances = PAIRWISE_DISTANCES(P)  // C × C distance matrix
    clusters = FASTERPAM(distances, k)
    
    RETURN {
        'labels': clusters.labels,           // Cluster assignment for each point
        'medoids': clusters.medoids,         // Indices of medoid points
        'medoids_loc': P[clusters.medoids]   // Coordinates of medoids
    }
END FUNCTION


Function COMPUTE_MSS(clusters, P):
    // Mean Simplified Silhouette score
    // Input: clusters dictionary from KMEDOIDS, P is C × 2 matrix
    // Output: MSS score (float)
    
    labels = clusters['labels']
    medoids_loc = clusters['medoids_loc']
    C = number of rows in P
    
    silhouette_scores = []
    
    FOR point_i = 0 to C-1:
        cluster_id = labels[point_i]
        point = P[point_i]
        medoid = medoids_loc[cluster_id]
        
        // Intra-cluster distance
        a_i = EUCLIDEAN_DISTANCE(point, medoid)
        
        // Nearest-cluster distance (mean distance to other cluster medoids)
        other_medoids = EXCLUDE(medoids_loc, cluster_id)
        distances_to_others = []
        FOR each other_medoid in other_medoids:
            distances_to_others.append(EUCLIDEAN_DISTANCE(point, other_medoid))
        END FOR
        b_i = MEAN(distances_to_others)
        
        // Silhouette score for this point
        IF a_i != 0:
            s_i = (b_i - a_i) / MAX(a_i, b_i)
        ELSE:
            s_i = 1.0  // Perfect clustering at medoid
        END IF
        
        silhouette_scores.append(s_i)
    END FOR
    
    // If all intra-cluster distances are zero, return 1
    IF all a_i == 0:
        RETURN 1.0
    END IF
    
    MSS = MEAN(silhouette_scores)
    RETURN MSS
END FUNCTION


Function KNEE_DETECTOR(x, y, curve='concave', direction='increasing'):
    // Find knee point using KneeLocator algorithm
    // Input: x (list of k values), y (list of MSS values)
    // Output: Optimal k value (knee point)
    
    // Use polynomial interpolation
    knee_locator = KneeLocator(
        x_values=x,
        y_values=y,
        curve=curve,
        direction=direction,
        interp_method='polynomial',
        polynomial_degree=2
    )
    
    optimal_k = knee_locator.knee
    
    // Fallback if knee not found
    IF optimal_k == None:
        // Use target pruning ratio as fallback
        optimal_k = ROUND(C * target_pruning_ratio)
    END IF
    
    RETURN optimal_k
END FUNCTION


Function TSNE(S, n_components=2, perplexity=10, max_iter=1000):
    // t-SNE dimensionality reduction
    // Input: S is C × D matrix (separation matrix)
    // Output: P is C × 2 matrix (reduced 2D space)
    
    tsne_model = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        max_iter=max_iter,
        random_state=42
    )
    
    P = tsne_model.fit_transform(S)
    
    RETURN P
END FUNCTION


// ============================================================
// MAIN EXECUTION FLOW
// ============================================================

Function MAIN():
    // Load model and data
    M = LOAD_MODEL("yolo_model.pt")
    D_train = LOAD_DATASET("train_data.yaml")
    classes = [0, 1, 2, ..., N-1]  // Class labels
    
    // Select target layers to prune
    target_layers = SELECT_TARGET_LAYERS(M, strategy="highest_channels")
    
    // Prune each layer sequentially
    FOR each Conv_lj in target_layers:
        PRINT("Pruning layer: Block {}, Conv {}".format(l, j))
        
        // Apply activation-based pruning
        M = ACTIVATION_BASED_PRUNING(M, D_train, Conv_lj, classes)
        
        // Fine-tune after each layer (optional)
        M = TRAIN(M, D_train, epochs=5)
    END FOR
    
    // Final fine-tuning
    M = TRAIN(M, D_train, epochs=20)
    
    // Evaluate pruned model
    metrics = EVALUATE(M, D_validation)
    PRINT("Final metrics: mAP={}, Precision={}, Recall={}".format(
        metrics.mAP, metrics.precision, metrics.recall))
    
    // Save pruned model
    SAVE_MODEL(M, "pruned_model.pt")
END FUNCTION
