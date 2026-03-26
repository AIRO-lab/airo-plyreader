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
from cuml.neighbors import NearestNeighbors
from ..config import (
    PILLAR_RADIUS_MIN, PILLAR_RADIUS_MAX, PILLAR_HEIGHT_MIN,
    PILLAR_HEIGHT_MAX, PILLAR_AXIS_MAX_ANGLE_DEG,
    MAX_POINTS_PER_CLUSTER, PCA_CYLINDER_THRESHOLD,
    PCA_CROSS_SECTION_RATIO_THRESHOLD, PCA_MIN_SECONDARY_VARIANCE,
    PCA_MAX_TERTIARY_VARIANCE, TOP_CLUSTERS_TO_ANALYZE,
    TRIPLANE_K_NEIGHBORS, TRIPLANE_SQUARE_SIZE, TRIPLANE_ANGLE_THRESHOLD,
    TRIPLANE_MAX_ATTEMPTS, TRIPLANE_FLATNESS_WARN,
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


def analyze_rectangular_pca(
    points: cp.ndarray,
) -> Optional[Dict[str, Any]]:
    """Detect a rectangular box shape using PCA on the given points.

    Computes PCA to find 3 principal axes, then derives the oriented
    bounding box (OBB) dimensions and 8 vertices.

    Args:
        points: CuPy array of shape (N, 3), float32.

    Returns:
        Dictionary with box parameters (NumPy/Python types), or None.
    """
    if len(points) < 4:
        print("  Rectangular PCA: too few points (<4), skipping")
        return None

    print(f"  Analyzing rectangular PCA ({len(points):,} points)...")

    # Downsample if too large
    if len(points) > MAX_POINTS_PER_CLUSTER:
        rng = cp.random.RandomState(42)
        indices = rng.choice(len(points), MAX_POINTS_PER_CLUSTER, replace=False)
        analysis_points = points[indices]
        print(f"    Downsampled to {len(analysis_points):,} points for PCA")
    else:
        analysis_points = points

    try:
        start_time = time.time()

        # Center the data
        center = cp.mean(analysis_points, axis=0)
        centered = analysis_points - center

        # cuML PCA — components sorted by explained variance descending
        pca = PCA(n_components=3)
        pca.fit(centered)

        components = cp.asarray(pca.components_)
        explained_variance_ratios = cp.asarray(pca.explained_variance_ratio_)

        ev1 = float(explained_variance_ratios[0])
        ev2 = float(explained_variance_ratios[1])
        ev3 = float(explained_variance_ratios[2])

        # Compute dimensions along each principal axis
        # Use ALL points (not downsampled) for accurate dimensions
        centered_all = points - center
        axes_np = cp.asnumpy(components)  # (3, 3)
        center_np = cp.asnumpy(center)

        dimensions = np.zeros(3)
        geo_center_np = center_np.copy()
        for i in range(3):
            axis_vec = components[i]
            axis_vec = axis_vec / cp.linalg.norm(axis_vec)
            proj = cp.dot(centered_all, axis_vec)
            proj_min = float(cp.min(proj))
            proj_max = float(cp.max(proj))
            dimensions[i] = proj_max - proj_min
            # Offset from mean to geometric midpoint along this axis
            geo_center_np += axes_np[i] * (proj_min + proj_max) / 2.0

        # Compute 8 vertices using geometric center (not mean)
        # Vertex ordering:
        #   0-3: one face, 4-7: opposite face
        #   Edges: bottom(0-1,1-2,2-3,3-0), top(4-5,5-6,6-7,7-4), vertical(0-4,1-5,2-6,3-7)
        half_extents = []
        for i in range(3):
            half_extents.append(axes_np[i] * dimensions[i] / 2.0)

        vertices = np.zeros((8, 3))
        vertices[0] = geo_center_np - half_extents[0] - half_extents[1] - half_extents[2]
        vertices[1] = geo_center_np + half_extents[0] - half_extents[1] - half_extents[2]
        vertices[2] = geo_center_np + half_extents[0] + half_extents[1] - half_extents[2]
        vertices[3] = geo_center_np - half_extents[0] + half_extents[1] - half_extents[2]
        vertices[4] = geo_center_np - half_extents[0] - half_extents[1] + half_extents[2]
        vertices[5] = geo_center_np + half_extents[0] - half_extents[1] + half_extents[2]
        vertices[6] = geo_center_np + half_extents[0] + half_extents[1] + half_extents[2]
        vertices[7] = geo_center_np - half_extents[0] + half_extents[1] + half_extents[2]

        analysis_time = time.time() - start_time

        print(
            f"    Rectangular PCA detected: "
            f"dims=[{dimensions[0]:.3f}, {dimensions[1]:.3f}, {dimensions[2]:.3f}]m, "
            f"eigenvalues=[{ev1:.3f}, {ev2:.3f}, {ev3:.3f}], "
            f"time={analysis_time:.2f}s"
        )

        return {
            'center': geo_center_np,
            'axes': axes_np,
            'dimensions': dimensions,
            'vertices': vertices,
            'inlier_points': cp.asnumpy(points),
            'point_count': len(points),
            'eigenvalue_ratios': np.array([ev1, ev2, ev3]),
            'analysis_method': 'rectangular_PCA',
        }

    except Exception as e:
        print(f"    Rectangular PCA analysis error: {str(e)}")
        return None


def analyze_triplane_pca(
    points: cp.ndarray,
) -> Optional[Dict[str, Any]]:
    """Detect 3 distinct planes via random point sampling + KNN + local PCA.

    For each random seed point, finds K nearest neighbors, computes the
    local covariance matrix, and extracts the surface normal (smallest
    eigenvalue's eigenvector). Repeats until 3 planes with sufficiently
    different normals are found.

    Args:
        points: CuPy array of shape (N, 3), float32. ROI-cropped points.

    Returns:
        Dictionary with planes, angles, point_count, analysis_method.
        Returns None if fewer than 3 distinct planes are found.
    """
    n_points = len(points)
    if n_points < TRIPLANE_K_NEIGHBORS:
        print(f"  Triplane PCA: too few points ({n_points} < {TRIPLANE_K_NEIGHBORS}), skipping")
        return None

    print(f"  Analyzing triplane PCA ({n_points:,} points)...")
    start_time = time.time()

    # Fit KNN model once on full ROI
    k = min(TRIPLANE_K_NEIGHBORS, n_points)
    nn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    nn_model.fit(points)

    planes = []
    excluded_indices = set()
    attempts = 0
    all_indices = np.arange(n_points)

    while len(planes) < 3 and attempts < TRIPLANE_MAX_ATTEMPTS:
        attempts += 1

        # Pick a random index not in excluded set
        available = np.setdiff1d(all_indices, np.array(list(excluded_indices), dtype=np.int64))
        if len(available) == 0:
            print(f"    No more available points after {attempts} attempts")
            break
        seed_idx = int(np.random.choice(available))

        # Query KNN on GPU
        seed_point = points[seed_idx:seed_idx+1]  # shape (1, 3)
        distances, neighbor_indices = nn_model.kneighbors(seed_point)
        neighbor_indices_np = cp.asnumpy(neighbor_indices[0]).astype(int)

        # Transfer neighbors to CPU for local PCA
        neighbor_points = cp.asnumpy(points[neighbor_indices_np])  # (K, 3)

        # Compute covariance matrix and eigendecomposition
        centroid = np.mean(neighbor_points, axis=0)
        centered = neighbor_points - centroid
        cov_matrix = np.dot(centered.T, centered) / (len(centered) - 1)

        # eigh returns ascending order: λ_small first
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Reverse to get λ1 >= λ2 >= λ3 convention
        eigenvalues = eigenvalues[::-1]
        eigenvectors = eigenvectors[:, ::-1]

        ev_sum = np.sum(eigenvalues)
        if ev_sum < 1e-12:
            excluded_indices.add(seed_idx)
            continue

        # v1, v2 = plane axes; v3 = normal (smallest eigenvalue)
        v1 = eigenvectors[:, 0]
        v2 = eigenvectors[:, 1]
        normal = eigenvectors[:, 2]

        # Normalize
        normal = normal / np.linalg.norm(normal)
        v1 = v1 / np.linalg.norm(v1)
        v2 = v2 / np.linalg.norm(v2)

        # Compute flatness
        flatness = float(eigenvalues[2] / ev_sum)
        if flatness > TRIPLANE_FLATNESS_WARN:
            print(f"    Warning: plane {len(planes)} flatness={flatness:.4f} (threshold={TRIPLANE_FLATNESS_WARN})")

        # Check angle against existing planes
        is_distinct = True
        for existing in planes:
            existing_normal = np.array(existing['normal'])
            dot = np.abs(np.dot(normal, existing_normal))
            angle = float(np.degrees(np.arccos(np.clip(dot, 0.0, 1.0))))
            if angle < TRIPLANE_ANGLE_THRESHOLD:
                is_distinct = False
                break

        if is_distinct:
            # Compute square vertices
            seed_center = cp.asnumpy(points[seed_idx])
            half = TRIPLANE_SQUARE_SIZE / 2.0
            sq_v1 = seed_center + half * v1 + half * v2
            sq_v2 = seed_center - half * v1 + half * v2
            sq_v3 = seed_center - half * v1 - half * v2
            sq_v4 = seed_center + half * v1 - half * v2

            ev_ratios = (eigenvalues / ev_sum).tolist()

            planes.append({
                'center': seed_center.tolist(),
                'normal': normal.tolist(),
                'axis1': v1.tolist(),
                'axis2': v2.tolist(),
                'flatness': flatness,
                'eigenvalue_ratios': ev_ratios,
                'num_neighbors': k,
                'square_vertices': [sq_v1.tolist(), sq_v2.tolist(), sq_v3.tolist(), sq_v4.tolist()],
            })
            print(f"    Plane {len(planes)}/3 found (attempt {attempts}): "
                  f"flatness={flatness:.4f}, normal=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]")

        # Exclude seed + neighbors
        excluded_indices.add(seed_idx)
        excluded_indices.update(neighbor_indices_np.tolist())

    if len(planes) < 3:
        print(f"  Triplane PCA failed: only {len(planes)}/3 planes found after {attempts} attempts")
        return None

    # Compute pairwise angles
    angles = {}
    for i in range(3):
        for j in range(i + 1, 3):
            ni = np.array(planes[i]['normal'])
            nj = np.array(planes[j]['normal'])
            dot = np.abs(np.dot(ni, nj))
            angle = float(np.degrees(np.arccos(np.clip(dot, 0.0, 1.0))))
            angles[f'plane_{i}_{j}'] = round(angle, 3)

    analysis_time = time.time() - start_time
    print(f"    Triplane PCA completed in {analysis_time:.2f}s")
    print(f"    Angles: {angles}")

    return {
        'planes': planes,
        'angles': angles,
        'point_count': n_points,
        'analysis_method': 'triplane_PCA',
    }


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
