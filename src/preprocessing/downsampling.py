"""
GPU-accelerated point cloud downsampling using CuPy.

Supports voxel grid downsampling on GPU. Points and colors are expected
as CuPy arrays and returned as CuPy arrays (no CPU transfer).

For very large point clouds, uses chunked processing to stay within
GPU memory limits. Each chunk is voxel-averaged independently, then
a final merge pass combines results from all chunks.
"""

import time
import cupy as cp
import cupyx

from ..config import (
    DOWNSAMPLING_VOXEL_SIZE,
    GPU_CHUNK_SIZE,
)

def _average_voxel_chunk(
    points: cp.ndarray,
    colors: cp.ndarray,
    voxel_size: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute voxel grid averages for a single chunk of points.

    Args:
        points: CuPy array of shape (N, 3), float32
        colors: CuPy array of shape (N, 3), uint8
        voxel_size: Voxel edge length in meters

    Returns:
        Tuple of (averaged_points, averaged_colors) as CuPy arrays
    """
    if len(points) == 0:
        return points, colors

    # Compute voxel indices per axis as int32
    vi = cp.floor(points[:, 0] / voxel_size).astype(cp.int32)
    vj = cp.floor(points[:, 1] / voxel_size).astype(cp.int32)
    vk = cp.floor(points[:, 2] / voxel_size).astype(cp.int32)

    # Shift to non-negative
    vi -= vi.min()
    vj -= vj.min()
    vk -= vk.min()

    # Compute dimensions
    ny = int(vj.max()) + 1
    nz = int(vk.max()) + 1

    # Encode to linear index (int64 for the product to avoid overflow)
    linear_idx = vi.astype(cp.int64) * (ny * nz) + vj.astype(cp.int64) * nz + vk.astype(cp.int64)
    del vi, vj, vk

    # Find unique voxels and inverse mapping
    unique_idx, inverse = cp.unique(linear_idx, return_inverse=True)
    del linear_idx
    n_voxels = len(unique_idx)
    del unique_idx

    # Accumulate point coordinates per voxel
    coord_sums_x = cp.zeros(n_voxels, dtype=cp.float32)
    coord_sums_y = cp.zeros(n_voxels, dtype=cp.float32)
    coord_sums_z = cp.zeros(n_voxels, dtype=cp.float32)
    cupyx.scatter_add(coord_sums_x, inverse, points[:, 0])
    cupyx.scatter_add(coord_sums_y, inverse, points[:, 1])
    cupyx.scatter_add(coord_sums_z, inverse, points[:, 2])

    # Accumulate colors per voxel (float32 to prevent uint8 overflow)
    colors_f = colors.astype(cp.float32)
    color_sums_r = cp.zeros(n_voxels, dtype=cp.float32)
    color_sums_g = cp.zeros(n_voxels, dtype=cp.float32)
    color_sums_b = cp.zeros(n_voxels, dtype=cp.float32)
    cupyx.scatter_add(color_sums_r, inverse, colors_f[:, 0])
    cupyx.scatter_add(color_sums_g, inverse, colors_f[:, 1])
    cupyx.scatter_add(color_sums_b, inverse, colors_f[:, 2])
    del colors_f

    # Count points per voxel
    counts = cp.zeros(n_voxels, dtype=cp.float32)
    cupyx.scatter_add(counts, inverse, cp.ones(len(points), dtype=cp.float32))
    del inverse

    # Average
    counts_expanded = counts[:, cp.newaxis]
    avg_points = cp.stack([coord_sums_x, coord_sums_y, coord_sums_z], axis=1) / counts_expanded
    del coord_sums_x, coord_sums_y, coord_sums_z
    avg_colors = (cp.stack([color_sums_r, color_sums_g, color_sums_b], axis=1) / counts_expanded).clip(0, 255).astype(cp.uint8)
    del color_sums_r, color_sums_g, color_sums_b, counts, counts_expanded

    return avg_points, avg_colors


def downsample_voxel_grid_gpu(
    points: cp.ndarray,
    colors: cp.ndarray,
    voxel_size: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Downsample point cloud using voxel grid averaging on GPU.

    For large point clouds, processes in chunks to avoid GPU OOM:
    1. Split into chunks of CHUNK_SIZE points
    2. Voxel-average each chunk independently
    3. Concatenate chunk results
    4. Final voxel-average pass to merge duplicates across chunks

    Args:
        points: CuPy array of shape (N, 3), float32
        colors: CuPy array of shape (N, 3), uint8
        voxel_size: Voxel edge length in meters

    Returns:
        Tuple of (downsampled_points, downsampled_colors) as CuPy arrays
    """
    if len(points) == 0:
        return points, colors

    n_points = len(points)

    # Small enough to process in one pass
    if n_points <= GPU_CHUNK_SIZE:
        return _average_voxel_chunk(points, colors, voxel_size)

    # Chunked processing for large point clouds
    n_chunks = (n_points + GPU_CHUNK_SIZE - 1) // GPU_CHUNK_SIZE
    print(f"    Processing in {n_chunks} chunks ({GPU_CHUNK_SIZE:,} points each)")

    chunk_results_points = []
    chunk_results_colors = []

    for i in range(n_chunks):
        start = i * GPU_CHUNK_SIZE
        end = min(start + GPU_CHUNK_SIZE, n_points)
        print(f"    Chunk {i + 1}/{n_chunks}: points {start:,}-{end:,}")

        chunk_pts, chunk_cols = _average_voxel_chunk(
            points[start:end], colors[start:end], voxel_size
        )
        chunk_results_points.append(chunk_pts)
        chunk_results_colors.append(chunk_cols)

    # Free original arrays (no longer needed after chunking)
    del points, colors
    cp.get_default_memory_pool().free_all_blocks()

    # Concatenate all chunk results
    merged_points = cp.concatenate(chunk_results_points, axis=0)
    merged_colors = cp.concatenate(chunk_results_colors, axis=0)
    del chunk_results_points, chunk_results_colors

    print(f"    Merged chunks: {len(merged_points):,} intermediate points")

    # Final merge pass: re-run voxel average to combine duplicates across chunks
    final_points, final_colors = _average_voxel_chunk(
        merged_points, merged_colors, voxel_size
    )

    return final_points, final_colors


def downsample_gpu(
    points: cp.ndarray,
    colors: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Main downsampling entry point. Applies voxel grid downsampling on GPU.

    Args:
        points: CuPy array of shape (N, 3), float32
        colors: CuPy array of shape (N, 3), uint8

    Returns:
        Tuple of (processed_points, processed_colors) as CuPy arrays
    """
    print(f"  Downsampling: ENABLED - Method: voxel (GPU)")
    print(f"    Original points: {len(points):,}")

    start_time = time.time()

    downsampled_points, downsampled_colors = downsample_voxel_grid_gpu(
        points, colors, DOWNSAMPLING_VOXEL_SIZE
    )

    # Statistics
    reduction_ratio = len(downsampled_points) / len(points) if len(points) > 0 else 0
    points_removed = len(points) - len(downsampled_points)
    processing_time = time.time() - start_time

    print(f"    Voxel size: {DOWNSAMPLING_VOXEL_SIZE}m")
    print(f"    Downsampled points: {len(downsampled_points):,}")
    print(f"    Points removed: {points_removed:,}")
    print(f"    Reduction ratio: {reduction_ratio:.2%}")
    print(f"    Processing time: {processing_time:.2f} seconds")

    return downsampled_points, downsampled_colors
