"""
Interactive HSV distribution analysis with filter tuning.

Displays H/S/V histograms of a point cloud with interactive sliders
for setting color filter parameters.
"""

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import Slider, Button
import numpy as np
import cupy as cp

from ..core.gpu import PointCloudGPU
from .color_segmentation import rgb_to_hsv_gpu


def analyze_hsv_gui(
    cloud: PointCloudGPU,
) -> tuple[list[tuple[float, float]], float, float] | None:
    """Open HSV histogram GUI for interactive filter parameter tuning.

    Displays H, S, V histograms of the point cloud with sliders to set
    filter ranges. Shows real-time matched point count.

    H-min > H-max enables wrap-around (e.g., 350 to 10 for red hues).

    Args:
        cloud: Point cloud on GPU (typically ROI-cropped).

    Returns:
        (h_ranges, s_min, v_min) tuple, or None if skipped.
    """
    # HSV conversion on GPU, then transfer to CPU
    h_gpu, s_gpu, v_gpu = rgb_to_hsv_gpu(cloud.colors)
    h_cpu = cp.asnumpy(h_gpu)
    s_cpu = cp.asnumpy(s_gpu)
    v_cpu = cp.asnumpy(v_gpu)
    del h_gpu, s_gpu, v_gpu
    cp.get_default_memory_pool().free_all_blocks()

    n_points = len(h_cpu)

    # State
    state = {
        'h_min': 0.0,
        'h_max': 360.0,
        's_min': 0.3,
        'v_min': 0.3,
        'confirmed': False,
    }

    # Track highlight patches for removal/re-creation
    highlights = {'h': None, 'h2': None, 's': None, 'v': None}

    # --- Layout ---
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle('HSV Distribution Analysis — Set Filter Parameters')

    # Histogram axes (top row)
    ax_h = fig.add_axes([0.05, 0.50, 0.28, 0.40])
    ax_s = fig.add_axes([0.38, 0.50, 0.28, 0.40])
    ax_v = fig.add_axes([0.71, 0.50, 0.28, 0.40])

    # Slider axes (two rows for H-min/H-max, one each for S/V)
    ax_hmin_slider = fig.add_axes([0.05, 0.36, 0.28, 0.04])
    ax_hmax_slider = fig.add_axes([0.05, 0.28, 0.28, 0.04])
    ax_s_slider = fig.add_axes([0.38, 0.36, 0.28, 0.04])
    ax_v_slider = fig.add_axes([0.71, 0.36, 0.28, 0.04])

    # Info text axis
    ax_info = fig.add_axes([0.05, 0.16, 0.60, 0.06])
    ax_info.set_axis_off()

    # Button axes
    ax_confirm = fig.add_axes([0.70, 0.05, 0.12, 0.06])
    ax_skip = fig.add_axes([0.84, 0.05, 0.12, 0.06])

    # --- Histograms ---
    # H histogram with per-bar hue coloring
    h_counts, h_edges = np.histogram(h_cpu, bins=360, range=(0, 360))
    h_centers = (h_edges[:-1] + h_edges[1:]) / 2
    h_bar_colors = [hsv_to_rgb((h_val / 360, 1, 1)) for h_val in h_centers]
    ax_h.bar(h_centers, h_counts, width=1.0, color=h_bar_colors, alpha=0.85)
    ax_h.set_title('Hue (H)')
    ax_h.set_xlabel('Degrees')
    ax_h.set_xlim(0, 360)

    ax_s.hist(s_cpu, bins=100, range=(0, 1), color='gray', alpha=0.7)
    ax_s.set_title('Saturation (S)')
    ax_s.set_xlim(0, 1)

    ax_v.hist(v_cpu, bins=100, range=(0, 1), color='gray', alpha=0.7)
    ax_v.set_title('Value (V)')
    ax_v.set_xlim(0, 1)

    # --- Info text ---
    info_text = ax_info.text(
        0.0, 0.5, '', transform=ax_info.transAxes,
        fontsize=12, verticalalignment='center',
    )

    # --- Sliders ---
    # Two separate sliders for H to support wrap-around
    slider_hmin = Slider(ax_hmin_slider, 'H min', 0, 360, valinit=0, valstep=1)
    slider_hmax = Slider(ax_hmax_slider, 'H max', 0, 360, valinit=360, valstep=1)
    slider_s = Slider(ax_s_slider, 'S min', 0, 1, valinit=0.3, valstep=0.01)
    slider_v = Slider(ax_v_slider, 'V min', 0, 1, valinit=0.3, valstep=0.01)

    def compute_match_count():
        h_lo = state['h_min']
        h_hi = state['h_max']
        s_lo = state['s_min']
        v_lo = state['v_min']

        if h_lo <= h_hi:
            h_mask = (h_cpu >= h_lo) & (h_cpu <= h_hi)
        else:
            # Wrap-around: e.g., 350 to 10 means H>=350 OR H<=10
            h_mask = (h_cpu >= h_lo) | (h_cpu <= h_hi)

        mask = h_mask & (s_cpu >= s_lo) & (v_cpu >= v_lo)
        return int(mask.sum())

    def update_display():
        matched = compute_match_count()
        pct = matched / n_points * 100 if n_points > 0 else 0

        h_lo = state['h_min']
        h_hi = state['h_max']
        wrap = h_lo > h_hi
        wrap_label = " [wrap-around]" if wrap else ""
        info_text.set_text(
            f"Matched: {matched:,} / {n_points:,} points ({pct:.2f}%)"
            f"  |  H[{h_lo:.0f}-{h_hi:.0f}]{wrap_label}, S>={state['s_min']:.2f}, V>={state['v_min']:.2f}"
        )

        # Remove old highlights
        for key in highlights:
            if highlights[key] is not None:
                highlights[key].remove()
                highlights[key] = None

        # Re-create H highlights
        if h_lo <= h_hi:
            highlights['h'] = ax_h.axvspan(h_lo, h_hi, alpha=0.2, color='blue')
        else:
            highlights['h'] = ax_h.axvspan(h_lo, 360, alpha=0.2, color='blue')
            highlights['h2'] = ax_h.axvspan(0, h_hi, alpha=0.2, color='blue')

        # S and V highlights
        highlights['s'] = ax_s.axvspan(state['s_min'], 1.0, alpha=0.2, color='green')
        highlights['v'] = ax_v.axvspan(state['v_min'], 1.0, alpha=0.2, color='orange')

        fig.canvas.draw_idle()

    def on_hmin_change(val):
        state['h_min'] = val
        update_display()

    def on_hmax_change(val):
        state['h_max'] = val
        update_display()

    def on_s_change(val):
        state['s_min'] = val
        update_display()

    def on_v_change(val):
        state['v_min'] = val
        update_display()

    slider_hmin.on_changed(on_hmin_change)
    slider_hmax.on_changed(on_hmax_change)
    slider_s.on_changed(on_s_change)
    slider_v.on_changed(on_v_change)

    # --- Buttons ---
    # Note: btn_confirm/btn_skip must stay in scope for callbacks to work
    btn_confirm = Button(ax_confirm, 'Confirm')
    btn_skip = Button(ax_skip, 'Skip')

    def on_confirm(event):
        state['confirmed'] = True
        plt.close(fig)

    def on_skip(event):
        plt.close(fig)

    btn_confirm.on_clicked(on_confirm)
    btn_skip.on_clicked(on_skip)

    # Initial display
    update_display()

    plt.show()

    if not state['confirmed']:
        print("HSV filter skipped")
        return None

    h_lo = state['h_min']
    h_hi = state['h_max']
    s_min_val = state['s_min']
    v_min_val = state['v_min']

    # Build h_ranges with wrap-around support
    if h_lo <= h_hi:
        h_ranges = [(h_lo, h_hi)]
    else:
        h_ranges = [(h_lo, 360.0), (0.0, h_hi)]

    h_str = ", ".join(f"{lo:.0f}-{hi:.0f}" for lo, hi in h_ranges)
    print(f"HSV filter set: H[{h_str}], S>={s_min_val:.2f}, V>={v_min_val:.2f}")

    return h_ranges, s_min_val, v_min_val
