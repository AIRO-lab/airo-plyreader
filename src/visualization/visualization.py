"""
Visualization functions for the multi-color pillar detection pipeline.

This module handles the creation of visualization output including cylinder
sample point generation, color-coded point cloud creation, and output
visualization preparation.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any
from ..config import GRAY_COLOR, RED_COLOR


def generate_pca_axes_points(pillar: Dict[str, Any], axis_length_factor: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate cylinder axis line for a detected pillar (main axis only).

    Args:
        pillar: Detected pillar dictionary containing analysis results
        axis_length_factor: Factor to scale axis length relative to pillar height

    Returns:
        Tuple of (axis_points, axis_colors) for cylinder main axis only
    """
    center = pillar['center']
    axis = pillar['axis']  # Cylinder axis

    # Calculate axis length based on pillar dimensions
    if len(pillar['inlier_points']) > 0:
        # Estimate height from inlier points
        axis_norm = axis / np.linalg.norm(axis)
        projections = np.dot(pillar['inlier_points'] - center, axis_norm)
        height = np.max(projections) - np.min(projections)
        axis_length = height * axis_length_factor
    else:
        axis_length = 1.0  # Fixed fallback when inlier_points is empty

    # Generate only the main axis (cylinder direction)
    primary_axis = axis / np.linalg.norm(axis)
    primary_start = center - axis_length * 0.5 * primary_axis
    primary_end = center + axis_length * 0.5 * primary_axis

    # Create axis line with more points for visibility
    axis_points = np.linspace(primary_start, primary_end, 30)
    axis_colors = np.full((len(axis_points), 3), [
                          255, 255, 0], dtype=np.uint8)  # Yellow for visibility

    return axis_points, axis_colors


def create_visualization_output(points: np.ndarray, colors: np.ndarray,
                                detected_pillars: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create visualization with color-coded results including cylinder axes.

    Args:
        points: Original point cloud coordinates
        colors: Original point cloud colors
        detected_pillars: List of detected pillar dictionaries

    Returns:
        Tuple of (output_points, output_colors) for visualization
    """
    print("Creating visualization output with cylinder axes...")
    start_time = time.time()

    # Start with all points in gray
    output_points = points.copy()
    output_colors = np.full_like(colors, GRAY_COLOR)

    # Collect all pillar inlier points
    all_pillar_points = []
    all_axes_points = []
    all_axes_colors = []

    # Find the pillar with the most points
    largest_pillar = max(detected_pillars, key=lambda p: len(p['inlier_points']), default=None)

    for pillar in detected_pillars:
        if len(pillar['inlier_points']) > 0:
            all_pillar_points.append(pillar['inlier_points'])

            # Generate axis only for the largest pillar
            if pillar is largest_pillar:
                axes_points, axes_colors = generate_pca_axes_points(pillar)
                all_axes_points.append(axes_points)
                all_axes_colors.append(axes_colors)
                print(
                    f"Generating axis for largest cluster: {len(pillar['inlier_points']):,} points")

    if all_pillar_points:
        # Combine all pillar points
        pillar_points = np.vstack(all_pillar_points)

        # Color pillar points in red
        output_points = np.vstack([output_points, pillar_points])
        pillar_colors = np.full((len(pillar_points), 3),
                                RED_COLOR, dtype=np.uint8)
        output_colors = np.vstack([output_colors, pillar_colors])

        # Add PCA-based coordinate axes
        if all_axes_points:
            combined_axes_points = np.vstack(all_axes_points)
            combined_axes_colors = np.vstack(all_axes_colors)

            output_points = np.vstack([output_points, combined_axes_points])
            output_colors = np.vstack([output_colors, combined_axes_colors])

            print(
                f"Added cylinder axes for {len(detected_pillars)} pillars ({len(combined_axes_points):,} axis points)")

    viz_time = time.time() - start_time
    print(
        f"Created visualization with {len(output_points):,} points in {viz_time:.2f} seconds")

    return output_points, output_colors


def create_clustering_visualization(
    original_points: np.ndarray,
    original_colors: np.ndarray,
    clusters: List[np.ndarray],
    cluster_ids: List[int],
    cluster_indices: List[np.ndarray] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a visualization of clustering results with color-coded clusters.

    Args:
        original_points: Original point cloud array of shape (N, 3)
        original_colors: Original color array of shape (N, 3) with values 0-255
        clusters: List of cluster point arrays
        cluster_ids: List of cluster IDs
        cluster_indices: List of arrays containing original indices for each cluster (optional)

    Returns:
        Tuple of (visualization_points, visualization_colors) for saving
    """
    print("Creating clustering visualization...")

    # Start with gray background for all original points
    viz_points = original_points.copy()
    viz_colors = np.full_like(original_colors, GRAY_COLOR,
                              dtype=np.uint8)

    # Generate distinctive colors for each cluster
    cluster_colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [255, 128, 0],  # Orange
        [128, 0, 255],  # Purple
        [255, 128, 128],  # Light Red
        [128, 255, 128],  # Light Green
    ]

    # Efficiently color cluster points using indices
    total_cluster_points = 0

    for i, (cluster_points, cluster_id) in enumerate(zip(clusters, cluster_ids)):
        if len(cluster_points) == 0:
            continue

        # Use cycling colors if we have more clusters than predefined colors
        color_idx = i % len(cluster_colors)
        cluster_color = np.array(cluster_colors[color_idx], dtype=np.uint8)

        # Use direct indexing if cluster_indices is provided (FAST path)
        if cluster_indices is not None and i < len(cluster_indices):
            # Direct index-based coloring - O(M) instead of O(N×M)
            indices = cluster_indices[i]
            viz_colors[indices] = cluster_color
        else:
            raise ValueError(
                f"cluster_indices required for cluster {cluster_id}. "
                "Distance-based fallback has been removed."
            )

        total_cluster_points += len(cluster_points)

    print(
        f"  Colored {total_cluster_points:,} cluster points across {len(clusters)} clusters")

    return viz_points, viz_colors


def show_viewer(ply_path: str, title: str, left: int, top: int) -> None:
    """
    Open a PLY file in an Open3D interactive viewer window.

    Designed to run as a multiprocessing.Process target.
    Imports Open3D inside the function to avoid inheriting CUDA context.
    """
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            print(f"[Viewer] '{title}': empty point cloud, skipping")
            return
        print(f"[Viewer] '{title}': {len(pcd.points):,} points")
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=title,
            width=960,
            height=540,
            left=left,
            top=top,
        )
    except Exception as e:
        print(f"[Viewer] '{title}' error: {e}")


def show_overlay_viewer(
    ply_path: str,
    pillars_data: list[dict],
    title: str,
    left: int,
    top: int,
) -> None:
    """Open a downsampled PLY with pillar axis overlay in an Open3D viewer.

    Designed to run as a multiprocessing.Process target.
    Builds LineSet inside the subprocess (Open3D geometries are not picklable).

    Args:
        ply_path: Path to the downsampled PLY file.
        pillars_data: List of pillar dicts from JSON (each has center, axis, height).
        title: Window title.
        left: Window X position.
        top: Window Y position.
    """
    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            print(f"[Viewer] '{title}': empty point cloud, skipping")
            return

        geometries = [pcd]

        # Build cylinder meshes for pillar axes
        if pillars_data:
            for p in pillars_data:
                center = np.array(p["center"])
                axis = np.array(p["axis"])
                height = p["height"]
                axis_length = height * 3.0

                # Normalize target axis
                axis_norm = np.linalg.norm(axis)
                if axis_norm < 1e-9:
                    continue
                target_axis = axis / axis_norm

                # Create cylinder along Z-axis
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=0.01, height=axis_length, resolution=20, split=4
                )
                cylinder.compute_vertex_normals()
                cylinder.paint_uniform_color([0.0, 1.0, 1.0])

                # Rotate from Z-axis to target axis
                z_axis = np.array([0.0, 0.0, 1.0])
                rot_axis = np.cross(z_axis, target_axis)
                rot_axis_len = np.linalg.norm(rot_axis)
                dot = np.clip(np.dot(z_axis, target_axis), -1.0, 1.0)

                if rot_axis_len < 1e-6:
                    if dot < 0:
                        # Anti-parallel: rotate 180 degrees around X-axis
                        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                            np.array([np.pi, 0.0, 0.0])
                        )
                        cylinder.rotate(R, center=np.array([0.0, 0.0, 0.0]))
                    # else: parallel to Z, no rotation needed
                else:
                    angle = np.arccos(dot)
                    rot_axis_normalized = rot_axis / rot_axis_len
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                        rot_axis_normalized * angle
                    )
                    cylinder.rotate(R, center=np.array([0.0, 0.0, 0.0]))

                # Translate to pillar center
                cylinder.translate(center)
                geometries.append(cylinder)

        print(f"[Viewer] '{title}': {len(pcd.points):,} points, {len(pillars_data)} axes")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=title,
            width=960,
            height=540,
            left=left,
            top=top,
        )
    except Exception as e:
        print(f"[Viewer] '{title}' error: {e}")


def launch_all_viewers(
    targets: list[tuple[str, str]],
    overlay: tuple[str, str, list[dict]] | None = None,
) -> None:
    """Launch Open3D viewer windows for given PLY files.

    Args:
        targets: List of (title, ply_path) tuples to display.
        overlay: Optional (title, ply_path, pillars_data) for axis overlay viewer.
    """
    import os
    import multiprocessing

    if not os.environ.get("DISPLAY"):
        print("Warning: DISPLAY not set, skipping visualization viewers")
        return

    # Filter to existing files only
    targets = [(t, p) for t, p in targets if os.path.isfile(p)]

    if not targets and overlay is None:
        print("No result files found for visualization")
        return

    # 2x2 tiled positions
    positions = [(0, 0), (960, 0), (0, 540), (960, 540)]

    ctx = multiprocessing.get_context("spawn")
    processes = []

    for i, (title, path) in enumerate(targets):
        left, top = positions[i % len(positions)]
        p = ctx.Process(target=show_viewer, args=(path, title, left, top))
        processes.append(p)

    # Add overlay viewer if provided
    if overlay is not None:
        ov_title, ov_path, pillars_data = overlay
        if os.path.isfile(ov_path):
            idx = len(targets)
            left, top = positions[idx % len(positions)]
            p = ctx.Process(
                target=show_overlay_viewer,
                args=(ov_path, pillars_data, ov_title, left, top),
            )
            processes.append(p)

    if not processes:
        print("No result files found for visualization")
        return

    for p in processes:
        p.start()

    print(f"Launched {len(processes)} viewer(s). Close all windows to continue...")

    for p in processes:
        p.join()
