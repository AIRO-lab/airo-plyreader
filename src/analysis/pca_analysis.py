"""
GPU-accelerated PCA-based pillar detection using cuML.

Analyzes point clusters to detect cylindrical structures via eigenvalue
ratios from Principal Component Analysis.
"""

import cupy as cp
import numpy as np
import time
from typing import List, Optional, Dict, Any
from cuml.decomposition import PCA
from ..config import (
    PILLAR_RADIUS_MIN, PILLAR_RADIUS_MAX, PILLAR_HEIGHT_MIN,
    PILLAR_HEIGHT_MAX, PILLAR_AXIS_MAX_ANGLE_DEG,
    MAX_POINTS_PER_CLUSTER, PCA_CYLINDER_THRESHOLD,
    PCA_CROSS_SECTION_RATIO_THRESHOLD, PCA_MIN_SECONDARY_VARIANCE,
    PCA_MAX_TERTIARY_VARIANCE, TOP_CLUSTERS_TO_ANALYZE,
)


def validate_pca_geometry(
    center: cp.ndarray,
    axis: cp.ndarray,
    radius: float,
    cluster_points: cp.ndarray,
) -> bool:
    """
    Validate PCA-derived cylinder geometry against pillar constraints.

    All operations on GPU via CuPy.
    """
    if radius < PILLAR_RADIUS_MIN or radius > PILLAR_RADIUS_MAX:
        print(f"    PCA cylinder rejected (radius={radius:.3f}m)")
        return False

    axis_normalized = axis / cp.linalg.norm(axis)
    projections = cp.dot(cluster_points - center, axis_normalized)
    height = float(cp.max(projections) - cp.min(projections))

    if height < PILLAR_HEIGHT_MIN:
        print(f"    PCA cylinder rejected (height={height:.3f}m)")
        return False

    if height > PILLAR_HEIGHT_MAX:
        print(f"    PCA cylinder rejected (height={height:.3f}m, max={PILLAR_HEIGHT_MAX}m)")
        return False

    if PILLAR_AXIS_MAX_ANGLE_DEG is not None:
        cos_angle = float(cp.abs(axis_normalized[2]))
        angle_deg = float(cp.degrees(cp.arccos(cp.minimum(cos_angle, 1.0))))
        if angle_deg > PILLAR_AXIS_MAX_ANGLE_DEG:
            print(f"    PCA cylinder rejected (angle={angle_deg:.1f}° from Z, max={PILLAR_AXIS_MAX_ANGLE_DEG}°)")
            return False

    return True


def analyze_cluster_with_pca(
    cluster_points: cp.ndarray,
    cluster_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Analyze point cluster using cuML PCA to detect cylindrical structures.

    Args:
        cluster_points: CuPy array of shape (N, 3), float32
        cluster_id: Cluster identifier for logging

    Returns:
        Dictionary with NumPy-converted pillar parameters, or None
    """
    if len(cluster_points) < 10:
        return None

    print(f"  Analyzing cluster {cluster_id} with PCA ({len(cluster_points):,} points)...")

    # Limit cluster size
    if len(cluster_points) > MAX_POINTS_PER_CLUSTER:
        rng = cp.random.RandomState(42)
        indices = rng.choice(len(cluster_points), MAX_POINTS_PER_CLUSTER, replace=False)
        cluster_points = cluster_points[indices]
        print(f"    Downsampled to {len(cluster_points):,} points")

    try:
        start_time = time.time()

        # Center the data
        cluster_center = cp.mean(cluster_points, axis=0)
        centered_points = cluster_points - cluster_center

        # cuML PCA
        pca = PCA(n_components=3)
        pca.fit(centered_points)

        components = cp.asarray(pca.components_)
        explained_variance_ratios = cp.asarray(pca.explained_variance_ratio_)
        explained_variance = cp.asarray(pca.explained_variance_)

        analysis_time = time.time() - start_time

        # Extract eigenvalue ratios (as Python floats for comparison)
        ev1 = float(explained_variance_ratios[0])
        ev2 = float(explained_variance_ratios[1])
        ev3 = float(explained_variance_ratios[2])

        # Cylindrical structure criteria
        is_cylindrical = (
            ev1 > PCA_CYLINDER_THRESHOLD
            and ev2 > PCA_MIN_SECONDARY_VARIANCE
            and ev3 < PCA_MAX_TERTIARY_VARIANCE
            and abs(ev2 - ev3) < PCA_CROSS_SECTION_RATIO_THRESHOLD
        )

        if not is_cylindrical:
            print(f"    PCA rejected (not cylindrical): ev1={ev1:.3f}, ev2={ev2:.3f}, ev3={ev3:.3f}")
            return None

        # Extract cylinder properties
        cylinder_axis = components[0]

        # Estimate radius from cross-sectional variance
        cross_sectional_variance = (float(explained_variance[1]) + float(explained_variance[2])) / 2
        estimated_radius = float(cp.sqrt(cp.float32(cross_sectional_variance)))

        # Validate geometry
        if not validate_pca_geometry(cluster_center, cylinder_axis, estimated_radius, cluster_points):
            return None

        # Confidence score
        cylindrical_quality = ev1 * (1 - abs(ev2 - ev3))
        confidence = min(cylindrical_quality, 1.0)

        print(
            f"    PCA cylinder detected: radius={estimated_radius:.3f}m, "
            f"confidence={confidence:.3f}, eigenvalues=[{ev1:.3f}, {ev2:.3f}, {ev3:.3f}], "
            f"time={analysis_time:.2f}s"
        )

        # GPU→CPU boundary: convert all results to NumPy/Python types
        return {
            'center': cp.asnumpy(cluster_center),
            'axis': cp.asnumpy(cylinder_axis),
            'radius': estimated_radius,
            'inlier_points': cp.asnumpy(cluster_points),
            'confidence': confidence,
            'cluster_id': cluster_id,
            'eigenvalue_ratios': np.array([ev1, ev2, ev3]),
            'analysis_method': 'PCA',
        }

    except Exception as e:
        print(f"    PCA analysis error: {str(e)}")
        return None


def analyze_cluster_with_traditional_pca(
    cluster_points: cp.ndarray,
    cluster_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Analyze point cluster using traditional PCA — PC1 is the reference axis.

    No cylindrical structure criteria. Height validated inline.

    Args:
        cluster_points: CuPy array of shape (N, 3), float32
        cluster_id: Cluster identifier for logging

    Returns:
        Dictionary with NumPy-converted pillar parameters, or None
    """
    if len(cluster_points) < 10:
        return None

    print(f"  Analyzing cluster {cluster_id} with traditional PCA ({len(cluster_points):,} points)...")

    # Limit cluster size
    if len(cluster_points) > MAX_POINTS_PER_CLUSTER:
        rng = cp.random.RandomState(42)
        indices = rng.choice(len(cluster_points), MAX_POINTS_PER_CLUSTER, replace=False)
        cluster_points = cluster_points[indices]
        print(f"    Downsampled to {len(cluster_points):,} points")

    try:
        start_time = time.time()

        # Center the data
        cluster_center = cp.mean(cluster_points, axis=0)
        centered_points = cluster_points - cluster_center

        # cuML PCA
        pca = PCA(n_components=3)
        pca.fit(centered_points)

        components = cp.asarray(pca.components_)
        explained_variance_ratios = cp.asarray(pca.explained_variance_ratio_)

        analysis_time = time.time() - start_time

        ev1 = float(explained_variance_ratios[0])
        ev2 = float(explained_variance_ratios[1])
        ev3 = float(explained_variance_ratios[2])

        # PC1 is the reference axis — no cylindrical criteria
        reference_axis = components[0]

        # Inline height validation
        axis_normalized = reference_axis / cp.linalg.norm(reference_axis)
        projections = cp.dot(cluster_points - cluster_center, axis_normalized)
        height = float(cp.max(projections) - cp.min(projections))

        if height < PILLAR_HEIGHT_MIN:
            print(f"    Traditional PCA rejected (height={height:.3f}m, min={PILLAR_HEIGHT_MIN}m)")
            return None

        if height > PILLAR_HEIGHT_MAX:
            print(f"    Traditional PCA rejected (height={height:.3f}m, max={PILLAR_HEIGHT_MAX}m)")
            return None

        print(
            f"    Traditional PCA accepted: height={height:.3f}m, "
            f"eigenvalues=[{ev1:.3f}, {ev2:.3f}, {ev3:.3f}], "
            f"time={analysis_time:.2f}s"
        )

        return {
            'center': cp.asnumpy(cluster_center),
            'axis': cp.asnumpy(reference_axis),
            'inlier_points': cp.asnumpy(cluster_points),
            'cluster_id': cluster_id,
            'eigenvalue_ratios': np.array([ev1, ev2, ev3]),
            'analysis_method': 'traditional_PCA',
        }

    except Exception as e:
        print(f"    Traditional PCA analysis error: {str(e)}")
        return None


def detect_pillars_with_pca(
    clusters: List[cp.ndarray],
    cluster_ids: List[int],
    method: str = 'cylinder',
) -> List[Dict[str, Any]]:
    """
    Detect pillars by analyzing clusters with cuML PCA.

    Args:
        clusters: List of CuPy point arrays
        cluster_ids: List of cluster identifiers
        method: PCA method to use — 'cylinder' (default) or 'traditional'

    Returns:
        List of detected pillar dicts (NumPy/Python types)
    """
    print(f"Detecting pillars using {method} PCA analysis (GPU)...")
    start_time = time.time()

    # Sort clusters by point count (largest first)
    cluster_data = list(zip(clusters, cluster_ids))
    cluster_data.sort(key=lambda x: len(x[0]), reverse=True)

    # Limit to top N clusters
    if TOP_CLUSTERS_TO_ANALYZE > 0:
        clusters_to_analyze = cluster_data[:TOP_CLUSTERS_TO_ANALYZE]
        print(f"Analyzing top {TOP_CLUSTERS_TO_ANALYZE} clusters by point count:")
        if len(cluster_data) > TOP_CLUSTERS_TO_ANALYZE:
            print(f"Skipping {len(cluster_data) - TOP_CLUSTERS_TO_ANALYZE} smaller clusters")
    else:
        clusters_to_analyze = cluster_data
        print(f"Analyzing all {len(cluster_data)} clusters:")

    for i, (cluster, cid) in enumerate(clusters_to_analyze):
        print(f"  {i + 1}. Cluster {cid}: {len(cluster):,} points")

    detected_pillars = []
    for cluster, cid in clusters_to_analyze:
        if method == 'traditional':
            pillar = analyze_cluster_with_traditional_pca(cluster, cid)
        else:
            pillar = analyze_cluster_with_pca(cluster, cid)
        if pillar is not None:
            detected_pillars.append(pillar)

    detection_time = time.time() - start_time
    print(f"Detected {len(detected_pillars)} pillars in {detection_time:.2f} seconds")

    if detected_pillars:
        if method == 'traditional':
            point_counts = [len(p['inlier_points']) for p in detected_pillars]
            print(f"  Point count range: {min(point_counts):,} - {max(point_counts):,}")
            ev_first = [p['eigenvalue_ratios'][0] for p in detected_pillars]
            print(f"  PC1 variance ratio range: {min(ev_first):.3f} - {max(ev_first):.3f}")
        else:
            confidences = [p['confidence'] for p in detected_pillars]
            radii = [p['radius'] for p in detected_pillars]
            print(f"  Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"  Radius range: {min(radii):.3f}m - {max(radii):.3f}m")
    else:
        print("  No pillars were detected")

    if method == 'traditional':
        detected_pillars.sort(key=lambda p: len(p['inlier_points']), reverse=True)
    else:
        detected_pillars.sort(key=lambda p: p['confidence'], reverse=True)
    return detected_pillars
