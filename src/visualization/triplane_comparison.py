"""
Academic-style triplane comparison figures for PCA analysis results.

Generates two Matplotlib figures comparing the 3 zone PC1 axes
and optionally the rectangular PCA principal axis:
1. 3D vector diagram with angle arcs between all pairs
2. Unit sphere with direction points and great circle arcs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .axis_comparison import _slerp_arc


_ZONE_COLORS = ["#0000FF", "#FF00FF", "#00FF00"]
_ZONE_LABELS = ["Zone 0 (Blue)", "Zone 1 (Magenta)", "Zone 2 (Green)"]


def plot_triplane_vector_diagram(
    pc1_axes: list[np.ndarray],
    angles: dict[str, float],
    rect_axis: np.ndarray | None = None,
    save_path: str | None = None,
) -> None:
    """Generate 3D vector diagram comparing three zone PC1 axes."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    origin = [0, 0, 0]

    if rect_axis is not None:
        ax.quiver(*origin, *rect_axis, color="#FF0000", arrow_length_ratio=0.08,
                  linewidth=2.5, label="Rectangular PC1 (Red)")

    for i, (axis, color, label) in enumerate(zip(pc1_axes, _ZONE_COLORS, _ZONE_LABELS)):
        ax.quiver(*origin, *axis, color=color, arrow_length_ratio=0.08,
                  linewidth=2.0, label=label)

    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        key = f"zone_{i}_{j}"
        angle_deg = angles.get(key, 0.0)
        if angle_deg < 0.1:
            continue

        a_i, a_j = pc1_axes[i], pc1_axes[j]
        if np.dot(a_i, a_j) < 0:
            arc_target = -a_j
        else:
            arc_target = a_j

        arc_radius = 0.3 + i * 0.05
        arc_points = _slerp_arc(a_i, arc_target) * arc_radius
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                "--", color="#333333", linewidth=0.8)

        mid = arc_points[len(arc_points) // 2]
        ax.text(mid[0] * 1.5, mid[1] * 1.5, mid[2] * 1.5,
                f"{angle_deg:.1f}\u00b0", fontsize=8, ha="center")

    for axis_vec in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        ax.plot([0, axis_vec[0]], [0, axis_vec[1]], [0, axis_vec[2]],
                color="#CCCCCC", linewidth=0.5, linestyle=":")

    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved triplane vector diagram: {save_path}")
    plt.show()


def plot_triplane_unit_sphere(
    pc1_axes: list[np.ndarray],
    angles: dict[str, float],
    rect_axis: np.ndarray | None = None,
    save_path: str | None = None,
) -> None:
    """Generate unit sphere visualization comparing three zone PC1 axes."""
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#AAAAAA", alpha=0.4, linewidth=0.5)

    if rect_axis is not None:
        ax.scatter(*rect_axis, color="#FF0000", s=120, zorder=6,
                   edgecolors="black", linewidths=0.5, marker="D",
                   label="Rectangular PC1 (Red)")

    for axis, color, label in zip(pc1_axes, _ZONE_COLORS, _ZONE_LABELS):
        ax.scatter(*axis, color=color, s=100, zorder=5,
                   edgecolors="black", linewidths=0.5, label=label)

    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        key = f"zone_{i}_{j}"
        angle_deg = angles.get(key, 0.0)
        if angle_deg < 0.1:
            continue
        a_i, a_j = pc1_axes[i], pc1_axes[j]
        if np.dot(a_i, a_j) < 0:
            arc_target = -a_j
        else:
            arc_target = a_j
        arc_points = _slerp_arc(a_i, arc_target)
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                "--", color="#333333", linewidth=0.8)
        mid = arc_points[len(arc_points) // 2]
        ax.text(mid[0] * 1.3, mid[1] * 1.3, mid[2] * 1.3,
                f"{angle_deg:.1f}\u00b0", fontsize=8, ha="center")

    ax.set_xlim([-1.2, 1.2]); ax.set_ylim([-1.2, 1.2]); ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved triplane unit sphere: {save_path}")
    plt.show()


def generate_triplane_comparison(
    triplane_data: dict,
    output_dir: str | None = None,
    rect_data: dict | None = None,
) -> None:
    """Generate triplane comparison figures from JSON result data."""
    zones = triplane_data.get("zones", [])
    if len(zones) < 3:
        print("Triplane comparison: need 3 zones, skipping.")
        return

    # Select the principal axis closest to Z for each zone
    z_axes = []
    for zone in zones[:3]:
        axes = zone.get("axes")
        if axes is None or len(axes) < 3:
            print("Triplane comparison: missing axes data, skipping.")
            return
        z_abs = [abs(float(axes[k][2])) for k in range(3)]
        idx = z_abs.index(max(z_abs))
        ax = np.array(axes[idx], dtype=np.float64)
        norm = np.linalg.norm(ax)
        if norm > 1e-9:
            ax = ax / norm
        z_axes.append(ax)
    pc1_axes = z_axes  # renamed for compatibility with plot functions

    angles = triplane_data.get("angles_between_zones", {})

    rect_axis = None
    if rect_data is not None:
        axes = rect_data.get("axes")
        if axes is not None and len(axes) > 0:
            rect_axis = np.array(axes[0], dtype=np.float64)
            r_norm = np.linalg.norm(rect_axis)
            if r_norm > 1e-9:
                rect_axis = rect_axis / r_norm

    print(f"\nTriplane Z-axis comparison:")
    color_names = ["Blue", "Magenta", "Green"]
    for i, (pc1, name) in enumerate(zip(pc1_axes, color_names)):
        print(f"  {name}: [{pc1[0]:.4f}, {pc1[1]:.4f}, {pc1[2]:.4f}]")
    for key, val in angles.items():
        print(f"  {key}: {val:.3f}\u00b0")
    if rect_axis is not None:
        print(f"  Rectangular PC1: [{rect_axis[0]:.4f}, {rect_axis[1]:.4f}, {rect_axis[2]:.4f}]")

    vec_save = os.path.join(output_dir, "triplane_vector_diagram.png") if output_dir else None
    sphere_save = os.path.join(output_dir, "triplane_unit_sphere.png") if output_dir else None

    plot_triplane_vector_diagram(pc1_axes, angles, rect_axis=rect_axis, save_path=vec_save)
    plot_triplane_unit_sphere(pc1_axes, angles, rect_axis=rect_axis, save_path=sphere_save)
