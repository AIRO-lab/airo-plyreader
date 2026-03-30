"""
GPU-accelerated point clustering using cuML DBSCAN.

Groups colored points into spatial clusters as pillar candidates.
"""

import cupy as cp
import time
from typing import Tuple
from cuml.cluster import DBSCAN
from ..config import DBSCAN_EPS, DBSCAN_MIN_SAMPLES


def cluster_colored_points(
    colored_points: cp.ndarray,
    colored_indices: cp.ndarray = None,
) -> Tuple[list[cp.ndarray], list[int], list[cp.ndarray]]:
    """
    Cluster colored points using cuML DBSCAN on GPU.

    Args:
        colored_points: CuPy array of shape (N, 3), float32
        colored_indices: CuPy array of shape (N,) with original point indices

    Returns:
        Tuple of (clusters, cluster_labels, cluster_indices) where:
        - clusters: List of CuPy arrays, each containing points for one cluster
        - cluster_labels: List of cluster label IDs (Python ints)
        - cluster_indices: List of CuPy arrays with original indices per cluster
    """
    if len(colored_points) == 0:
        return [], [], []

    print("Clustering colored points using DBSCAN (GPU)...")
    start_time = time.time()

    # cuML DBSCAN (no n_jobs — GPU parallelism is implicit)
    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    cluster_labels = dbscan.fit_predict(colored_points)

    # Convert labels to CuPy array if needed
    if not isinstance(cluster_labels, cp.ndarray):
        cluster_labels = cp.asarray(cluster_labels)

    # Extract valid clusters (exclude noise labeled as -1)
    unique_labels = cp.unique(cluster_labels)
    valid_labels = unique_labels[unique_labels != -1]

    clusters = []
    cluster_ids = []
    cluster_indices = []

    for label in valid_labels:
        label_val = int(label)
        cluster_mask = cluster_labels == label
        cluster_points = colored_points[cluster_mask]

        if len(cluster_points) >= DBSCAN_MIN_SAMPLES:
            clusters.append(cluster_points)
            cluster_ids.append(label_val)

            if colored_indices is not None:
                cluster_indices.append(colored_indices[cluster_mask])
            else:
                cluster_indices.append(cp.arange(len(cluster_points)))

    noise_points = int(cp.sum(cluster_labels == -1))
    cluster_time = time.time() - start_time

    print(
        f"Found {len(clusters)} valid clusters and {noise_points:,} noise points "
        f"in {cluster_time:.2f} seconds"
    )

    for i, cluster in enumerate(clusters):
        print(f"  Cluster {cluster_ids[i]}: {len(cluster):,} points")

    return clusters, cluster_ids, cluster_indices
