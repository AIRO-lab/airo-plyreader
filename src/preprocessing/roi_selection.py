"""
GUI-based ROI (Region of Interest) selection and GPU cropping.

Provides a Matplotlib top-down scatter plot for interactive rectangular
region selection, and GPU-accelerated cropping via CuPy boolean masking.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector, Button
import numpy as np
import cupy as cp

from ..core.gpu import PointCloudGPU

_MAX_DISPLAY_POINTS = 500_000


def crop_to_roi(
    cloud: PointCloudGPU,
    roi: tuple[float, float, float, float],
) -> PointCloudGPU:
    """Crop point cloud to a 2D XY bounding box on GPU.

    Args:
        cloud: Input point cloud on GPU.
        roi: (x_min, x_max, y_min, y_max) bounds.

    Returns:
        New PointCloudGPU containing only points within the ROI.
    """
    x_min, x_max, y_min, y_max = roi
    x = cloud.points[:, 0]
    y = cloud.points[:, 1]

    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)

    cropped = PointCloudGPU(cloud.points[mask], cloud.colors[mask])

    cp.get_default_memory_pool().free_all_blocks()

    print(f"ROI crop: {len(cloud):,} -> {len(cropped):,} points "
          f"(X[{x_min:.2f}, {x_max:.2f}], Y[{y_min:.2f}, {y_max:.2f}])")

    return cropped


def crop_to_z_roi(
    cloud: PointCloudGPU,
    h_min: float,
    h_max: float,
    z_min: float,
    z_max: float,
    axis: str,
) -> PointCloudGPU:
    """Crop point cloud using a horizontal axis + Z bounding box on GPU.

    Args:
        cloud: Input point cloud on GPU.
        h_min: Minimum value on the horizontal axis (X or Y).
        h_max: Maximum value on the horizontal axis (X or Y).
        z_min: Minimum Z value.
        z_max: Maximum Z value.
        axis: Which horizontal axis — 'x' or 'y'.

    Returns:
        New PointCloudGPU containing only points within the Z ROI.
    """
    if axis == 'x':
        h = cloud.points[:, 0]
        axis_label = 'X'
    else:
        h = cloud.points[:, 1]
        axis_label = 'Y'
    z = cloud.points[:, 2]

    mask = (h >= h_min) & (h <= h_max) & (z >= z_min) & (z <= z_max)

    cropped = PointCloudGPU(cloud.points[mask], cloud.colors[mask])

    cp.get_default_memory_pool().free_all_blocks()

    print(f"Z ROI crop: {len(cloud):,} -> {len(cropped):,} points "
          f"({axis_label}[{h_min:.2f}, {h_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}])")

    return cropped


def select_z_roi_gui(
    cloud: PointCloudGPU,
) -> tuple[float, float, float, float, str] | None:
    """Open a side-view scatter plot for interactive Z ROI selection.

    Automatically chooses XZ or YZ view based on which horizontal axis
    (X or Y) has the larger range in the current point cloud.

    Args:
        cloud: Point cloud on GPU (already XY-cropped).

    Returns:
        (h_min, h_max, z_min, z_max, axis_name) tuple, or None if skipped.
        axis_name is 'x' or 'y'.
    """
    points_cpu, colors_cpu = cloud.to_cpu()

    # Determine which axis has the larger range
    x_range = float(points_cpu[:, 0].max() - points_cpu[:, 0].min())
    y_range = float(points_cpu[:, 1].max() - points_cpu[:, 1].min())

    if x_range >= y_range:
        h_idx = 0
        axis_name = 'x'
        h_label = 'X'
    else:
        h_idx = 1
        axis_name = 'y'
        h_label = 'Y'

    print(f"Z ROI: using {h_label}Z view (X range={x_range:.2f}, Y range={y_range:.2f})")

    # Subsample for display if needed
    n_points = len(points_cpu)
    if n_points > _MAX_DISPLAY_POINTS:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(n_points, _MAX_DISPLAY_POINTS, replace=False)
        display_points = points_cpu[indices]
        display_colors = colors_cpu[indices]
        print(f"Z ROI GUI: subsampled {n_points:,} -> {_MAX_DISPLAY_POINTS:,} points for display")
    else:
        display_points = points_cpu
        display_colors = colors_cpu

    # Normalize colors for matplotlib (uint8 -> float64 [0,1])
    plot_colors = display_colors.astype(np.float64) / 255.0

    # State shared with callbacks
    state = {'roi': None, 'confirmed': False}

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        display_points[:, h_idx], display_points[:, 2],
        c=plot_colors, s=0.1, marker='.', edgecolors='none',
    )
    ax.set_xlabel(h_label)
    ax.set_ylabel('Z')
    ax.set_title(f'{h_label}Z View — Drag to select Z ROI, then click Confirm (or Skip)')
    ax.set_aspect('equal')

    def on_select(eclick, erelease):
        h1, h2 = sorted([eclick.xdata, erelease.xdata])
        z1, z2 = sorted([eclick.ydata, erelease.ydata])
        state['roi'] = (h1, h2, z1, z2)

    selector = RectangleSelector(
        ax, on_select, useblit=True,
        button=[1], interactive=True,
        props=dict(facecolor='blue', edgecolor='blue', alpha=0.15, fill=True),
    )

    # Buttons
    ax_confirm = fig.add_axes([0.7, 0.01, 0.12, 0.04])
    ax_skip = fig.add_axes([0.83, 0.01, 0.12, 0.04])
    btn_confirm = Button(ax_confirm, 'Confirm')
    btn_skip = Button(ax_skip, 'Skip')

    def on_confirm(event):
        state['confirmed'] = True
        plt.close(fig)

    def on_skip(event):
        state['roi'] = None
        plt.close(fig)

    btn_confirm.on_clicked(on_confirm)
    btn_skip.on_clicked(on_skip)

    plt.show()

    if state['confirmed'] and state['roi'] is not None:
        h_min, h_max, z_min, z_max = state['roi']
        print(f"Z ROI selected: {h_label}[{h_min:.2f}, {h_max:.2f}], Z[{z_min:.2f}, {z_max:.2f}]")
        return (h_min, h_max, z_min, z_max, axis_name)

    print("Z ROI selection skipped")
    return None


def select_roi_gui(
    cloud: PointCloudGPU,
) -> tuple[float, float, float, float] | None:
    """Open a top-down XY scatter plot for interactive ROI selection.

    Displays the point cloud projected onto the XY plane. The user drags
    a rectangle to define the ROI, then clicks Confirm. Clicking Skip
    or closing the window without selecting returns None.

    Args:
        cloud: Downsampled point cloud on GPU.

    Returns:
        (x_min, x_max, y_min, y_max) tuple, or None if skipped.
    """
    points_cpu, colors_cpu = cloud.to_cpu()

    # Subsample for display if needed
    n_points = len(points_cpu)
    if n_points > _MAX_DISPLAY_POINTS:
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(n_points, _MAX_DISPLAY_POINTS, replace=False)
        display_points = points_cpu[indices]
        display_colors = colors_cpu[indices]
        print(f"ROI GUI: subsampled {n_points:,} -> {_MAX_DISPLAY_POINTS:,} points for display")
    else:
        display_points = points_cpu
        display_colors = colors_cpu

    # Normalize colors for matplotlib (uint8 -> float64 [0,1])
    plot_colors = display_colors.astype(np.float64) / 255.0

    # State shared with callbacks
    state = {'roi': None, 'confirmed': False}

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(
        display_points[:, 0], display_points[:, 1],
        c=plot_colors, s=0.1, marker='.', edgecolors='none',
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Drag to select ROI, then click Confirm (or Skip)')
    ax.set_aspect('equal')

    def on_select(eclick, erelease):
        x1, x2 = sorted([eclick.xdata, erelease.xdata])
        y1, y2 = sorted([eclick.ydata, erelease.ydata])
        state['roi'] = (x1, x2, y1, y2)

    # RectangleSelector with interactive=True provides built-in visual
    # feedback (resizable/movable rectangle). No manual highlight needed.
    # Note: selector must stay in scope until plt.show() returns.
    selector = RectangleSelector(
        ax, on_select, useblit=True,
        button=[1], interactive=True,
        props=dict(facecolor='blue', edgecolor='blue', alpha=0.15, fill=True),
    )

    # Buttons
    ax_confirm = fig.add_axes([0.7, 0.01, 0.12, 0.04])
    ax_skip = fig.add_axes([0.83, 0.01, 0.12, 0.04])
    btn_confirm = Button(ax_confirm, 'Confirm')
    btn_skip = Button(ax_skip, 'Skip')

    def on_confirm(event):
        state['confirmed'] = True
        plt.close(fig)

    def on_skip(event):
        state['roi'] = None
        plt.close(fig)

    btn_confirm.on_clicked(on_confirm)
    btn_skip.on_clicked(on_skip)

    plt.show()

    if state['confirmed'] and state['roi'] is not None:
        roi = state['roi']
        print(f"ROI selected: X[{roi[0]:.2f}, {roi[1]:.2f}], Y[{roi[2]:.2f}, {roi[3]:.2f}]")
        return roi

    print("ROI selection skipped — using full point cloud")
    return None
