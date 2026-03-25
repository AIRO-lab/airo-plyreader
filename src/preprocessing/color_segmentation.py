"""
GPU-accelerated color segmentation using CuPy.

Converts RGB to HSV on GPU and filters points by configurable color ranges.
HSV convention: H: 0-360, S: 0-1, V: 0-1 (matches config values directly).
"""

import cupy as cp
import time
from ..config import GPU_CHUNK_SIZE


def rgb_to_hsv_gpu(rgb_colors: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """
    Convert RGB colors to HSV on GPU.

    Args:
        rgb_colors: CuPy array of shape (N, 3), uint8 [0-255]

    Returns:
        Tuple of (h, s, v) as 1D CuPy arrays:
        - h: 0-360 (float32)
        - s: 0-1 (float32)
        - v: 0-1 (float32)
    """
    # Normalize to 0-1
    rgb = rgb_colors.astype(cp.float32) / 255.0

    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    del rgb

    cmax = cp.maximum(cp.maximum(r, g), b)
    cmin = cp.minimum(cp.minimum(r, g), b)
    delta = cmax - cmin
    del cmin

    # Hue calculation
    h = cp.zeros_like(cmax)
    mask_delta = delta > 0

    # Red is max
    mask_r = mask_delta & (cmax == r)
    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)

    # Green is max
    mask_g = mask_delta & (cmax == g)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)

    # Blue is max
    mask_b = mask_delta & (cmax == b)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    del mask_delta, mask_r, mask_g, mask_b, r, g, b

    # Ensure H is in [0, 360)
    h = h % 360.0

    # Saturation
    s = cp.where(cmax > 0, delta / cmax, cp.zeros_like(cmax))
    del delta

    # Value (reuse cmax)
    v = cmax

    return h, s, v


def _filter_chunk(
    points_chunk: cp.ndarray,
    colors_chunk: cp.ndarray,
    h_ranges: list,
    s_min: float,
    v_min: float,
    index_offset: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, dict]:
    """
    Filter a single chunk of points by HSV color criteria.

    Args:
        points_chunk: CuPy array of shape (N, 3), float32
        colors_chunk: CuPy array of shape (N, 3), uint8
        h_ranges: List of (h_min, h_max) tuples
        s_min: Minimum saturation threshold
        v_min: Minimum value/brightness threshold
        index_offset: Global index offset for this chunk

    Returns:
        Tuple of (filtered_points, filtered_colors, global_indices, hsv_stats)
        hsv_stats is a dict with min/max of H, S, V for filtered points, or None if empty
    """
    h, s, v = rgb_to_hsv_gpu(colors_chunk)

    # Create color mask
    color_mask = cp.zeros(len(h), dtype=cp.bool_)
    for h_min, h_max in h_ranges:
        color_mask |= (h >= h_min) & (h <= h_max)
    color_mask &= (s >= s_min) & (v >= v_min)

    # Collect filtered results
    filtered_points = points_chunk[color_mask]
    filtered_colors = colors_chunk[color_mask]
    local_indices = cp.where(color_mask)[0]
    global_indices = local_indices + index_offset

    # HSV stats for diagnostic output
    hsv_stats = None
    if len(filtered_points) > 0:
        hsv_stats = {
            'h_min': float(h[color_mask].min()),
            'h_max': float(h[color_mask].max()),
            's_min': float(s[color_mask].min()),
            's_max': float(s[color_mask].max()),
            'v_min': float(v[color_mask].min()),
            'v_max': float(v[color_mask].max()),
        }

    del h, s, v, color_mask, local_indices
    return filtered_points, filtered_colors, global_indices, hsv_stats


def segment_by_color(
    points: cp.ndarray,
    colors: cp.ndarray,
    h_ranges: list[tuple[float, float]],
    s_min: float,
    v_min: float,
):
    """
    Filter points by HSV color criteria on GPU. Uses chunked processing
    for large point clouds to avoid GPU OOM.

    Args:
        points: CuPy array of shape (N, 3), float32
        colors: CuPy array of shape (N, 3), uint8
        h_ranges: List of (h_min, h_max) hue range tuples
        s_min: Minimum saturation threshold
        v_min: Minimum value/brightness threshold

    Returns:
        Tuple of (colored_points, colored_colors, colored_indices)
        All returned arrays are CuPy arrays.
    """
    h_str = ", ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in h_ranges)
    print(f"Filtering points by HSV (H[{h_str}], S>={s_min:.2f}, V>={v_min:.2f}) on GPU...")
    start_time = time.time()

    n_points = len(points)

    if n_points <= GPU_CHUNK_SIZE:
        # Single pass — fits in GPU memory
        filtered_points, filtered_colors, filtered_indices, hsv_stats = _filter_chunk(
            points, colors, h_ranges, s_min, v_min, index_offset=0
        )
    else:
        # Chunked processing for large point clouds
        n_chunks = (n_points + GPU_CHUNK_SIZE - 1) // GPU_CHUNK_SIZE
        print(f"  Processing color filter in {n_chunks} chunks ({GPU_CHUNK_SIZE:,} points each)")

        chunk_points = []
        chunk_colors = []
        chunk_indices = []
        chunk_stats = []

        for i in range(n_chunks):
            start = i * GPU_CHUNK_SIZE
            end = min(start + GPU_CHUNK_SIZE, n_points)

            fp, fc, fi, stats = _filter_chunk(
                points[start:end], colors[start:end],
                h_ranges, s_min, v_min, index_offset=start,
            )

            if len(fp) > 0:
                chunk_points.append(fp)
                chunk_colors.append(fc)
                chunk_indices.append(fi)

            if stats is not None:
                chunk_stats.append(stats)

        cp.get_default_memory_pool().free_all_blocks()

        if chunk_points:
            filtered_points = cp.concatenate(chunk_points, axis=0)
            filtered_colors = cp.concatenate(chunk_colors, axis=0)
            filtered_indices = cp.concatenate(chunk_indices, axis=0)
            hsv_stats = {
                'h_min': min(s['h_min'] for s in chunk_stats),
                'h_max': max(s['h_max'] for s in chunk_stats),
                's_min': min(s['s_min'] for s in chunk_stats),
                's_max': max(s['s_max'] for s in chunk_stats),
                'v_min': min(s['v_min'] for s in chunk_stats),
                'v_max': max(s['v_max'] for s in chunk_stats),
            }
        else:
            filtered_points = cp.empty((0, 3), dtype=cp.float32)
            filtered_colors = cp.empty((0, 3), dtype=cp.uint8)
            filtered_indices = cp.empty((0,), dtype=cp.int64)
            hsv_stats = None

        del chunk_points, chunk_colors, chunk_indices

    # Statistics
    filter_time = time.time() - start_time
    n_colored = len(filtered_points)
    print(
        f"Found {n_colored:,} matched points "
        f"({n_colored / n_points * 100:.2f}%) in {filter_time:.2f} seconds"
    )

    if hsv_stats is not None:
        print(
            f"  Matched points HSV ranges: "
            f"H[{hsv_stats['h_min']:.1f}-{hsv_stats['h_max']:.1f}], "
            f"S[{hsv_stats['s_min']:.3f}-{hsv_stats['s_max']:.3f}], "
            f"V[{hsv_stats['v_min']:.3f}-{hsv_stats['v_max']:.3f}]"
        )

    return filtered_points, filtered_colors, filtered_indices
