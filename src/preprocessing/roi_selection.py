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
