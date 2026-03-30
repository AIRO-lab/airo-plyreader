"""
Shared utilities for the pillar detection pipeline.
"""

import numpy as np

from ..config import ENABLE_INTERMEDIATE_SAVES
from ..file_io.ply_io import save_ply_file_open3d


def format_voxel_size(voxel_size: float) -> str:
    """Convert voxel size float to filename-safe string.

    Strips trailing zeros, converts to decimal form, replaces dots with underscores.
    Examples: 0.01 → "0_01", 0.010 → "0_01", 0.001 → "0_001"
    """
    s = f"{voxel_size:.10f}".rstrip("0").rstrip(".")
    return s.replace(".", "_")


def save_intermediate(points: np.ndarray, colors: np.ndarray, path: str, description: str) -> None:
    """Save intermediate pipeline result if ENABLE_INTERMEDIATE_SAVES is True."""
    if not ENABLE_INTERMEDIATE_SAVES:
        return
    try:
        print(f"Saving {description} to: {path}")
        save_ply_file_open3d(path, points, colors)
        print(f"{description} saved successfully")
    except Exception as e:
        print(f"Warning: Failed to save {description}: {str(e)}")
