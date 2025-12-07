#!/usr/bin/env python3
"""
Alternative channel selection strategies for activation-based pruning:
1. Medoid-based selection (returns medoids directly, not max weight)
2. Gamma-based selection (uses BN gamma instead of weight norms)
"""

import numpy as np
from typing import Dict, List
from loguru import logger
from clustering import get_knee, calc_mss_value, kmedoids_fasterpam


def select_optimal_components_medoid(graph_space: Dict[str, np.ndarray], weights: np.ndarray, 
                                     num_components: int, k_value: int) -> List[int]:
    """
    Select optimal components using MEDOID-based selection (not weighted).
    
    This is the same as select_optimal_components but with weight_form=False,
    meaning it returns the actual medoids from clustering, not max-weight channels.
    
    Args:
        graph_space: Graph space containing the reduced matrix
        weights: Weights (ignored in this variant)
        num_components: Total number of components
        k_value: Target number of clusters (may be adjusted by knee detection)
    
    Returns:
        List of channel indices (medoids) to keep
    """
    logger.info("Starting medoid-based selection (using actual medoids, not max weight)")
    mss_values = []
    k_values = range(2, num_components)

    for i, k in enumerate(k_values):
        k_medoids = kmedoids_fasterpam(graph_space['reduced_matrix'], k)
        mss_value = calc_mss_value(clustering=k_medoids, space=graph_space['reduced_matrix'])
        mss_values.append(mss_value)

        if mss_value == 1.0 or mss_value == 1:
            extension_length = len(k_values) - i - 1
            mss_values.extend([1.0] * extension_length)
            break

    knee = get_knee(list(k_values), y=mss_values)
    if knee is None:
        logger.warning("Knee detection failed, using k_value")
        knee = k_value
    
    optimal_kmedoids = kmedoids_fasterpam(graph_space['reduced_matrix'], int(knee))
    
    # Return medoids directly (no weight-based selection)
    return optimal_kmedoids['medoids'].tolist()


def find_optimal_gamma_medoids(gamma_values: np.ndarray, labels: np.ndarray, medoids: np.ndarray) -> List[int]:
    """
    Find optimal channels by selecting the channel with MAX GAMMA (BN gamma value)
    within each cluster, instead of max weight.
    
    Args:
        gamma_values: BN gamma values for all channels (absolute values)
        labels: Cluster labels for each channel
        medoids: Initial medoid indices
    
    Returns:
        List of channel indices with highest gamma in each cluster
    """
    try:
        highest_gamma_indices = np.empty_like(medoids)

        for i, medoid in enumerate(medoids):
            # Find all channels in the same cluster as this medoid
            cluster_indices = np.where(labels == labels[medoid])[0]

            # Extract gamma values for channels in this cluster
            cluster_gammas = gamma_values[cluster_indices]

            # Find the index of the maximum gamma in this cluster
            max_gamma_index = cluster_indices[np.argmax(cluster_gammas)]

            # Store the index of the channel with the highest gamma
            highest_gamma_indices[i] = max_gamma_index

        return highest_gamma_indices.tolist()
    
    except Exception as e:
        logger.error(f"Error in find_optimal_gamma_medoids: {e}")
        # Fallback: return medoids directly
        return medoids.tolist()


def select_optimal_components_max_gamma(graph_space: Dict[str, np.ndarray], gamma_values: np.ndarray, 
                                        num_components: int, k_value: int) -> List[int]:
    """
    Select optimal components using MAX GAMMA selection (instead of max weight).
    
    Uses the same clustering and MSS approach, but selects channels based on
    BN gamma magnitudes instead of weight magnitudes.
    
    Args:
        graph_space: Graph space containing the reduced matrix
        gamma_values: BN gamma values (absolute) for all channels
        num_components: Total number of components
        k_value: Target number of clusters (may be adjusted by knee detection)
    
    Returns:
        List of channel indices (max gamma per cluster) to keep
    """
    logger.info("Starting max-gamma-based selection (using BN gamma instead of weights)")
    mss_values = []
    k_values = range(2, num_components)

    for i, k in enumerate(k_values):
        k_medoids = kmedoids_fasterpam(graph_space['reduced_matrix'], k)
        mss_value = calc_mss_value(clustering=k_medoids, space=graph_space['reduced_matrix'])
        mss_values.append(mss_value)

        if mss_value == 1.0 or mss_value == 1:
            extension_length = len(k_values) - i - 1
            mss_values.extend([1.0] * extension_length)
            break

    knee = get_knee(list(k_values), y=mss_values)
    if knee is None:
        logger.warning("Knee detection failed, using k_value")
        knee = k_value
    
    optimal_kmedoids = kmedoids_fasterpam(graph_space['reduced_matrix'], int(knee))
    
    # Use gamma-based selection instead of weight-based
    return find_optimal_gamma_medoids(gamma_values, optimal_kmedoids['labels'], optimal_kmedoids['medoids'])
