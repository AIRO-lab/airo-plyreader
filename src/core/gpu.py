"""
GPU point cloud wrapper with CPU transfer caching.

PointCloudGPU is immutable after construction — pipeline stages that
transform data return new instances rather than mutating in place.
This guarantees the CPU cache is never stale.
"""

import cupy as cp
import numpy as np


class PointCloudGPU:
    """Immutable wrapper for point cloud data on GPU."""

    __slots__ = ('_points', '_colors', '_cpu_cache')

    def __init__(self, points: cp.ndarray, colors: cp.ndarray):
        self._points = points
        self._colors = colors
        self._cpu_cache = None

    @property
    def points(self) -> cp.ndarray:
        return self._points

    @property
    def colors(self) -> cp.ndarray:
        return self._colors

    def to_cpu(self) -> tuple[np.ndarray, np.ndarray]:
        """GPU → CPU transfer with caching (repeated calls are free)."""
        if self._cpu_cache is None:
            self._cpu_cache = (cp.asnumpy(self._points), cp.asnumpy(self._colors))
        return self._cpu_cache

    @classmethod
    def from_numpy(cls, points: np.ndarray, colors: np.ndarray) -> 'PointCloudGPU':
        """CPU → GPU transfer. Points are cast to float32 (from float64 in PLY loader)."""
        return cls(cp.asarray(points, dtype=cp.float32), cp.asarray(colors, dtype=cp.uint8))

    def __len__(self) -> int:
        return len(self._points)
