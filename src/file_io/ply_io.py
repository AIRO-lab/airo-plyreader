"""
PLY file input/output operations.

This module handles all PLY file reading and writing operations, including
coordinate and color extraction, data type conversions, and file format
management.
"""

import numpy as np
import time
from typing import Tuple
import open3d as o3d
from pathlib import Path


# =============================================================================
# OPEN3D-BASED PLY I/O FUNCTIONS
# =============================================================================

def load_ply_file_open3d(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PLY file using Open3D and extract coordinates and colors.

    Open3D provides optimized I/O with better memory management for large point clouds.

    Args:
        file_path: Path to PLY file

    Returns:
        Tuple of (points, colors) where:
        - points: numpy array of shape (N, 3) with [x, y, z] coordinates
        - colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)
    """
    print(f"Loading PLY file with Open3D: {file_path}")
    start_time = time.time()

    try:
        # Load point cloud using Open3D
        pcd = o3d.io.read_point_cloud(file_path)

        if len(pcd.points) == 0:
            raise ValueError(f"No points found in PLY file: {file_path}")

        # Extract points
        points = np.asarray(pcd.points, dtype=np.float64)

        # Extract colors (Open3D uses [0,1] range, convert to [0,255])
        if pcd.has_colors():
            colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
        else:
            # If no colors, create default white colors
            colors = np.full((len(points), 3), 255, dtype=np.uint8)
            print("Warning: No colors found in PLY file, using default white colors")

        load_time = time.time() - start_time
        print(
            f"Loaded {len(points):,} points with Open3D in {load_time:.2f} seconds")
        print(f"Point cloud bounds: X[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
              f"Y[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
              f"Z[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")

        return points, colors

    except Exception as e:
        raise RuntimeError(
            f"Failed to load PLY file with Open3D {file_path}: {str(e)}")


def save_ply_file_open3d(file_path: str, points: np.ndarray, colors: np.ndarray) -> None:
    """
    Save points and colors to PLY file using Open3D.

    Open3D provides optimized I/O with better compression and format options.

    Args:
        file_path: Output PLY file path
        points: numpy array of shape (N, 3) with [x, y, z] coordinates
        colors: numpy array of shape (N, 3) with [r, g, b] colors (0-255)
    """
    print(f"Saving output PLY file with Open3D: {file_path}")
    start_time = time.time()

    try:
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

        # Convert colors from [0,255] to [0,1] range for Open3D
        colors_normalized = colors.astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

        # Ensure output directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

        # Save point cloud
        success = o3d.io.write_point_cloud(file_path, pcd, write_ascii=True)

        if not success:
            raise RuntimeError("Open3D failed to write point cloud")

        save_time = time.time() - start_time
        print(
            f"Saved {len(points):,} points with Open3D in {save_time:.2f} seconds")

    except Exception as e:
        raise RuntimeError(
            f"Failed to save PLY file with Open3D {file_path}: {str(e)}")
