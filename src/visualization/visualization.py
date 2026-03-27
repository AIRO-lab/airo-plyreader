"""
Visualization functions for the multi-color pillar detection pipeline.

This module handles the creation of visualization output including cylinder
sample point generation, color-coded point cloud creation, and output
visualization preparation.
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Any
from ..config import (
    GRAY_COLOR, RED_COLOR, CYAN_COLOR,
    TRIPLANE_COLOR_BLUE, TRIPLANE_COLOR_MAGENTA, TRIPLANE_COLOR_GREEN,
    TRIPLANE_SQUARE_SIZE,
)


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


# 12 edges of a box: bottom face, top face, vertical connections
_BOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
]

_POINTS_PER_EDGE = 30


def generate_rectangular_wireframe(
    vertices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate point-based wireframe for a rectangular box.

    Creates 30 points per edge along each of the 12 box edges.
    Used for PLY-based visualization (point representation).

    Args:
        vertices: Box vertices of shape (8, 3).

    Returns:
        Tuple of (wireframe_points, wireframe_colors).
    """
    all_points = []
    for i, j in _BOX_EDGES:
        edge_points = np.linspace(vertices[i], vertices[j], _POINTS_PER_EDGE)
        all_points.append(edge_points)

    wireframe_points = np.vstack(all_points)
    wireframe_colors = np.full(
        (len(wireframe_points), 3), (255, 255, 0), dtype=np.uint8  # yellow
    )

    return wireframe_points, wireframe_colors


def create_rectangular_visualization(
    points: np.ndarray,
    colors: np.ndarray,
    rectangular_result: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create visualization for rectangular PCA result.

    Colors original points gray, ROI points red, adds a red Z-axis
    line from the center, and cyan wireframe edges.

    Args:
        points: Original point cloud (N, 3) NumPy array.
        colors: Original colors (N, 3) NumPy uint8 array.
        rectangular_result: Dict from analyze_rectangular_pca().

    Returns:
        Tuple of (output_points, output_colors).
    """
    print("Creating rectangular PCA visualization...")
    start_time = time.time()

    # All original points in gray
    output_points = points.copy()
    output_colors = np.full_like(colors, GRAY_COLOR)

    # ROI inlier points in red
    inlier_points = rectangular_result['inlier_points']
    if len(inlier_points) > 0:
        output_points = np.vstack([output_points, inlier_points])
        inlier_colors = np.full(
            (len(inlier_points), 3), RED_COLOR, dtype=np.uint8
        )
        output_colors = np.vstack([output_colors, inlier_colors])

    # Red Z-axis line from center
    center = rectangular_result['center']
    dimensions = rectangular_result['dimensions']
    axis_length = float(np.max(dimensions)) * 1.5
    z_axis = np.array([0.0, 0.0, 1.0])
    z_start = center - axis_length * 0.5 * z_axis
    z_end = center + axis_length * 0.5 * z_axis
    z_line_points = np.linspace(z_start, z_end, 30)
    z_line_colors = np.full((30, 3), RED_COLOR, dtype=np.uint8)
    output_points = np.vstack([output_points, z_line_points])
    output_colors = np.vstack([output_colors, z_line_colors])

    # Wireframe
    wireframe_points, wireframe_colors = generate_rectangular_wireframe(
        rectangular_result['vertices']
    )
    output_points = np.vstack([output_points, wireframe_points])
    output_colors = np.vstack([output_colors, wireframe_colors])

    viz_time = time.time() - start_time
    print(
        f"Created rectangular visualization with {len(output_points):,} points "
        f"({len(wireframe_points):,} wireframe points) in {viz_time:.2f}s"
    )

    return output_points, output_colors


# Grid density for triplane square visualization
_PLANE_GRID_SIZE = 30


def create_triplane_visualization(
    points: np.ndarray,
    colors: np.ndarray,
    triplane_result: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Create visualization for triplane PCA result.

    Colors original points gray, then draws 3 colored squares and
    their normal vectors as point-based lines.

    Args:
        points: Original point cloud (N, 3) NumPy array.
        colors: Original colors (N, 3) NumPy uint8 array.
        triplane_result: Dict from analyze_triplane_pca().

    Returns:
        Tuple of (output_points, output_colors).
    """
    print("Creating triplane PCA visualization...")
    start_time = time.time()

    # All original points in gray
    output_points = points.copy()
    output_colors = np.full_like(colors, GRAY_COLOR)

    plane_colors_rgb = [
        TRIPLANE_COLOR_BLUE,
        TRIPLANE_COLOR_MAGENTA,
        TRIPLANE_COLOR_GREEN,
    ]
    plane_labels = ["blue", "magenta", "green"]

    for i, plane in enumerate(triplane_result['planes']):
        center = np.array(plane['center'])
        axis1 = np.array(plane['axis1'])
        axis2 = np.array(plane['axis2'])
        normal = np.array(plane['normal'])

        # Generate grid points on the square
        half = TRIPLANE_SQUARE_SIZE / 2.0
        u = np.linspace(-half, half, _PLANE_GRID_SIZE)
        v = np.linspace(-half, half, _PLANE_GRID_SIZE)
        uu, vv = np.meshgrid(u, v)
        uu_flat = uu.flatten()
        vv_flat = vv.flatten()

        square_points = (
            center[np.newaxis, :]
            + uu_flat[:, np.newaxis] * axis1[np.newaxis, :]
            + vv_flat[:, np.newaxis] * axis2[np.newaxis, :]
        )

        # Color (convert 0-1 float to 0-255 uint8)
        color_uint8 = np.array([int(c * 255) for c in plane_colors_rgb[i]], dtype=np.uint8)
        square_colors = np.full((len(square_points), 3), color_uint8, dtype=np.uint8)

        output_points = np.vstack([output_points, square_points])
        output_colors = np.vstack([output_colors, square_colors])

        # Normal vector line from center (always Z-axis direction)
        normal_length = TRIPLANE_SQUARE_SIZE * 0.5
        z_axis = np.array([0.0, 0.0, 1.0])
        normal_start = center
        normal_end = center + normal_length * z_axis
        normal_line = np.linspace(normal_start, normal_end, _PLANE_GRID_SIZE)
        normal_colors = np.full((len(normal_line), 3), color_uint8, dtype=np.uint8)

        output_points = np.vstack([output_points, normal_line])
        output_colors = np.vstack([output_colors, normal_colors])

        print(f"    Plane {i} ({plane_labels[i]}): {len(square_points)} grid + {len(normal_line)} normal points")

    viz_time = time.time() - start_time
    print(f"Created triplane visualization with {len(output_points):,} points in {viz_time:.2f}s")

    return output_points, output_colors


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


def _show_with_capture(geometries, title, width, height, left, top, save_dir=None):
    """Show Open3D geometries in a viewer with 's' key screenshot capture.

    Replaces draw_geometries for overlay viewers. Press 'S' to save
    the current view as a PNG file.

    Args:
        geometries: List of Open3D geometry objects to display.
        title: Window title.
        width: Window width in pixels.
        height: Window height in pixels.
        left: Window X position.
        top: Window Y position.
        save_dir: Directory to save captures. Uses cwd if None.
    """
    import open3d as o3d
    import os
    from datetime import datetime

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=title, width=width, height=height, left=left, top=top)

    for geom in geometries:
        vis.add_geometry(geom)

    capture_count = [0]

    def on_key_s(vis):
        capture_count[0] += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}_{capture_count[0]:03d}.png"
        if save_dir:
            filepath = os.path.join(save_dir, filename)
        else:
            filepath = filename
        vis.capture_screen_image(filepath)
        print(f"[Capture] Saved: {filepath}")
        return False

    vis.register_key_callback(ord('S'), on_key_s)
    print(f"[Viewer] Press 'S' to capture screenshot")

    vis.run()
    vis.destroy_window()


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
    save_dir: str | None = None,
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
        _show_with_capture(geometries, title, 960, 540, left, top, save_dir)
    except Exception as e:
        print(f"[Viewer] '{title}' error: {e}")


def show_combined_overlay_viewer(
    ply_path: str,
    overlay_data: dict,
    title: str,
    left: int,
    top: int,
    save_dir: str | None = None,
) -> None:
    """Open a PLY with pillar axes, rectangular wireframe, and/or triplane overlay.

    Designed to run as a multiprocessing.Process target.
    Combines rendering from show_overlay_viewer (pillar axes) and
    show_rectangular_overlay_viewer (wireframe + Z-axis).

    Args:
        ply_path: Path to the downsampled PLY file.
        overlay_data: Dict with optional keys 'pillars' (list[dict]) and
            'rectangular' (dict). Missing or None keys are skipped.
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

        # --- Rectangular overlay: wireframe + red Z-axis ---
        rect_data = overlay_data.get("rectangular")
        if rect_data is not None:
            vertices = rect_data.get("vertices")
            if vertices is not None:
                verts = np.array(vertices)
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(verts)
                line_set.lines = o3d.utility.Vector2iVector(list(_BOX_EDGES))
                line_set.colors = o3d.utility.Vector3dVector(
                    [[1.0, 1.0, 0.0]] * len(_BOX_EDGES)  # yellow
                )
                geometries.append(line_set)

            center = rect_data.get("center")
            dimensions = rect_data.get("dimensions")
            if center is not None and dimensions is not None:
                center_np = np.array(center)
                axis_length = float(np.max(dimensions)) * 1.5
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=0.01, height=axis_length, resolution=20, split=4
                )
                cylinder.compute_vertex_normals()
                cylinder.paint_uniform_color([1.0, 0.0, 0.0])  # red
                cylinder.translate(center_np)
                geometries.append(cylinder)

        # --- Pillar overlay: cyan axis cylinders ---
        pillars_data = overlay_data.get("pillars")
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
                cylinder.paint_uniform_color([0.0, 1.0, 1.0])  # cyan

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

        # --- Triplane overlay: colored square meshes + normal cylinders ---
        triplane_data = overlay_data.get("triplane")
        if triplane_data is not None:
            plane_colors = [
                [0.0, 0.0, 1.0],   # blue
                [1.0, 0.0, 1.0],   # magenta
                [0.0, 1.0, 0.0],   # green
            ]
            tri_planes = triplane_data.get("planes", [])
            for i, plane in enumerate(tri_planes):
                color = plane_colors[i % len(plane_colors)]
                verts = plane.get("square_vertices")
                if verts is None or len(verts) < 4:
                    continue
                v = np.array(verts)

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(v)
                mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color(color)
                geometries.append(mesh)

                # Normal cylinder (always Z-axis direction)
                center_arr = plane.get("center")
                if center_arr is not None:
                    center_np = np.array(center_arr)

                    cyl_length = 0.3
                    cyl = o3d.geometry.TriangleMesh.create_cylinder(
                        radius=0.008, height=cyl_length, resolution=20, split=4
                    )
                    cyl.compute_vertex_normals()
                    cyl.paint_uniform_color(color)
                    # Cylinder is already along Z-axis, just translate
                    cyl.translate(center_np + np.array([0.0, 0.0, cyl_length / 2]))
                    geometries.append(cyl)

        overlay_parts = []
        if rect_data is not None:
            overlay_parts.append("wireframe")
        if pillars_data:
            overlay_parts.append(f"{len(pillars_data)} axes")
        if triplane_data is not None:
            overlay_parts.append(f"{len(triplane_data.get('planes', []))} planes")
        print(f"[Viewer] '{title}': {len(pcd.points):,} points, {' + '.join(overlay_parts)}")
        _show_with_capture(geometries, title, 960, 540, left, top, save_dir)
    except Exception as e:
        print(f"[Viewer] '{title}' error: {e}")


def show_rectangular_overlay_viewer(
    ply_path: str,
    rect_data: dict,
    title: str,
    left: int,
    top: int,
    save_dir: str | None = None,
) -> None:
    """Open a PLY with rectangular wireframe overlay in an Open3D viewer.

    Designed to run as a multiprocessing.Process target.
    Builds LineSet inside the subprocess (Open3D geometries are not picklable).

    Args:
        ply_path: Path to the visualization PLY file.
        rect_data: Rectangular PCA result dict from JSON (has 'vertices').
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

        # Build LineSet wireframe from vertices (reuse module-level _BOX_EDGES)
        vertices = rect_data.get("vertices")
        if vertices is not None:
            verts = np.array(vertices)
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(verts)
            line_set.lines = o3d.utility.Vector2iVector(list(_BOX_EDGES))
            line_set.colors = o3d.utility.Vector3dVector(
                [[1.0, 1.0, 0.0]] * len(_BOX_EDGES)  # yellow
            )
            geometries.append(line_set)

        # Build red Z-axis cylinder from center
        center = rect_data.get("center")
        dimensions = rect_data.get("dimensions")
        if center is not None and dimensions is not None:
            center_np = np.array(center)
            axis_length = float(np.max(dimensions)) * 1.5

            cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                radius=0.01, height=axis_length, resolution=20, split=4
            )
            cylinder.compute_vertex_normals()
            cylinder.paint_uniform_color([1.0, 0.0, 0.0])  # red
            # Cylinder is already along Z-axis, just translate to center
            cylinder.translate(center_np)
            geometries.append(cylinder)

        print(f"[Viewer] '{title}': {len(pcd.points):,} points, wireframe overlay")
        _show_with_capture(geometries, title, 960, 540, left, top, save_dir)
    except Exception as e:
        print(f"[Viewer] '{title}' error: {e}")


def show_triplane_overlay_viewer(
    ply_path: str,
    triplane_data: dict,
    title: str,
    left: int,
    top: int,
    save_dir: str | None = None,
) -> None:
    """Open a PLY with triplane square overlay in an Open3D viewer.

    Designed to run as a multiprocessing.Process target.
    Renders 3 colored squares as TriangleMesh and normal direction cylinders.

    Args:
        ply_path: Path to the visualization PLY file.
        triplane_data: Triplane result dict from JSON (has 'planes').
        title: Window title.
        left: Window X position.
        top: Window Y position.
        save_dir: Directory for screenshot captures.
    """
    try:
        import open3d as o3d

        pcd = o3d.io.read_point_cloud(ply_path)
        if pcd.is_empty():
            print(f"[Viewer] '{title}': empty point cloud, skipping")
            return

        geometries = [pcd]

        plane_colors = [
            [0.0, 0.0, 1.0],   # blue
            [1.0, 0.0, 1.0],   # magenta
            [0.0, 1.0, 0.0],   # green
        ]

        planes = triplane_data.get("planes", [])
        for i, plane in enumerate(planes):
            color = plane_colors[i % len(plane_colors)]
            verts = plane.get("square_vertices")
            if verts is None or len(verts) < 4:
                continue
            v = np.array(verts)

            # TriangleMesh: 4 vertices, 2 triangles
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(v)
            mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(color)
            geometries.append(mesh)

            # Normal cylinder (always Z-axis direction)
            center = plane.get("center")
            if center is not None:
                center_np = np.array(center)

                cyl_length = 0.3
                cylinder = o3d.geometry.TriangleMesh.create_cylinder(
                    radius=0.008, height=cyl_length, resolution=20, split=4
                )
                cylinder.compute_vertex_normals()
                cylinder.paint_uniform_color(color)
                # Cylinder is already along Z-axis, just translate
                cylinder.translate(center_np + np.array([0.0, 0.0, cyl_length / 2]))
                geometries.append(cylinder)

        print(f"[Viewer] '{title}': {len(pcd.points):,} points, {len(planes)} planes")
        _show_with_capture(geometries, title, 960, 540, left, top, save_dir)
    except Exception as e:
        print(f"[Viewer] '{title}' error: {e}")


def launch_all_viewers(
    targets: list[tuple[str, str]],
    overlay: tuple[str, str, list[dict] | dict] | None = None,
    overlay_viewer_fn=None,
    save_dir: str | None = None,
) -> None:
    """Launch Open3D viewer windows for given PLY files.

    Args:
        targets: List of (title, ply_path) tuples to display.
        overlay: Optional (title, ply_path, data) for overlay viewer.
        overlay_viewer_fn: Optional custom viewer function for overlay.
            Defaults to show_overlay_viewer if not specified.
        save_dir: Directory for overlay viewer screenshot captures ('S' key).
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
        ov_title, ov_path, overlay_data = overlay
        if os.path.isfile(ov_path):
            idx = len(targets)
            left, top = positions[idx % len(positions)]
            viewer_fn = overlay_viewer_fn or show_overlay_viewer
            p = ctx.Process(
                target=viewer_fn,
                args=(ov_path, overlay_data, ov_title, left, top),
                kwargs={"save_dir": save_dir},
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
