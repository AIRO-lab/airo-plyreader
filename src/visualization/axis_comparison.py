"""
Academic-style axis comparison figures for PCA analysis results.

Generates two Matplotlib figures comparing Pillar PCA axis direction
against Rectangular PCA axis direction:
1. 3D vector diagram with angle arc
2. Unit sphere with direction points and great circle arc
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_angle_between_axes(axis1: np.ndarray, axis2: np.ndarray) -> float:
    """Compute angle between two axis directions in degrees.

    Uses abs(dot product) to handle axis direction ambiguity —
    a PCA axis can point in either direction. Result range: [0, 90].

    Args:
        axis1: Unit vector (3,).
        axis2: Unit vector (3,).

    Returns:
        Angle in degrees, range [0, 90].
    """
    dot = np.dot(axis1, axis2)
    angle_rad = np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
    return float(np.degrees(angle_rad))


def _slerp_arc(v1: np.ndarray, v2: np.ndarray, n_points: int = 50) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors.

    Args:
        v1: Start unit vector (3,).
        v2: End unit vector (3,).
        n_points: Number of interpolation points.

    Returns:
        Array of shape (n_points, 3) along the great circle arc.
    """
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    omega = np.arccos(dot)
    if omega < 1e-8:
        return np.tile(v1, (n_points, 1))
    t = np.linspace(0, 1, n_points)
    sin_omega = np.sin(omega)
    arc = (np.sin((1 - t)[:, None] * omega) * v1 + np.sin(t[:, None] * omega) * v2) / sin_omega
    return arc


def plot_vector_diagram(
    pillar_axis: np.ndarray,
    rect_axis: np.ndarray,
    angle_deg: float,
    save_path: str | None = None,
) -> None:
    """Generate 3D vector diagram comparing two PCA axes.

    Draws two unit vectors as arrows from the origin with an angle
    arc between them. Academic paper figure style.

    Args:
        pillar_axis: Pillar PCA axis unit vector (3,).
        rect_axis: Rectangular PCA axis unit vector (3,).
        angle_deg: Pre-computed angle in degrees.
        save_path: If provided, save PNG at this path (300 DPI).
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Draw axis arrows
    origin = [0, 0, 0]
    ax.quiver(*origin, *pillar_axis, color="#00FFFF", arrow_length_ratio=0.08,
              linewidth=2.0, label="Pillar PC1")
    ax.quiver(*origin, *rect_axis, color="#FF0000", arrow_length_ratio=0.08,
              linewidth=2.0, label="Rectangular PC1")

    # Angle arc
    if angle_deg >= 0.1:
        # Ensure vectors point in same hemisphere for arc rendering
        if np.dot(pillar_axis, rect_axis) < 0:
            arc_target = -rect_axis
        else:
            arc_target = rect_axis
        arc_radius = 0.35
        arc_points = _slerp_arc(pillar_axis, arc_target) * arc_radius
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                "k--", linewidth=1.0)
        # Angle label at arc midpoint
        mid = arc_points[len(arc_points) // 2]
        ax.text(mid[0] * 1.4, mid[1] * 1.4, mid[2] * 1.4,
                f"$\\theta$ = {angle_deg:.1f}°", fontsize=10, ha="center")
    else:
        ax.text(0.5, 0.5, 0.5, "axes are coincident", fontsize=10,
                ha="center", style="italic")

    # Reference axes (light gray)
    for axis_vec, label in [([1, 0, 0], "X"), ([0, 1, 0], "Y"), ([0, 0, 1], "Z")]:
        ax.plot([0, axis_vec[0]], [0, axis_vec[1]], [0, axis_vec[2]],
                color="#CCCCCC", linewidth=0.5, linestyle=":")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved vector diagram: {save_path}")

    plt.show()


def plot_unit_sphere(
    pillar_axis: np.ndarray,
    rect_axis: np.ndarray,
    angle_deg: float,
    save_path: str | None = None,
) -> None:
    """Generate unit sphere visualization comparing two PCA axes.

    Plots direction points on a wireframe unit sphere with a great
    circle arc connecting them. Academic paper figure style.

    Args:
        pillar_axis: Pillar PCA axis unit vector (3,).
        rect_axis: Rectangular PCA axis unit vector (3,).
        angle_deg: Pre-computed angle in degrees.
        save_path: If provided, save PNG at this path (300 DPI).
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Wireframe sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#CCCCCC", alpha=0.15, linewidth=0.3)

    # Direction points
    ax.scatter(*pillar_axis, color="#00FFFF", s=100, zorder=5,
               edgecolors="black", linewidths=0.5, label="Pillar PC1")
    ax.scatter(*rect_axis, color="#FF0000", s=100, zorder=5,
               edgecolors="black", linewidths=0.5, label="Rectangular PC1")

    # Great circle arc
    if angle_deg >= 0.1:
        if np.dot(pillar_axis, rect_axis) < 0:
            arc_target = -rect_axis
        else:
            arc_target = rect_axis
        arc_points = _slerp_arc(pillar_axis, arc_target)
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                "k--", linewidth=1.0)
        mid = arc_points[len(arc_points) // 2]
        ax.text(mid[0] * 1.3, mid[1] * 1.3, mid[2] * 1.3,
                f"$\\theta$ = {angle_deg:.1f}°", fontsize=10, ha="center")
    else:
        ax.text(pillar_axis[0] * 1.3, pillar_axis[1] * 1.3, pillar_axis[2] * 1.3,
                "axes are coincident", fontsize=10, ha="center", style="italic")

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved unit sphere: {save_path}")

    plt.show()


def generate_axis_comparison(
    pillars_data: list[dict],
    rect_data: dict,
    output_dir: str | None = None,
) -> None:
    """Generate axis comparison figures from JSON result data.

    Entry point called by viewer.py when both pillar and rectangular
    results exist. Selects the largest pillar by inlier point count.

    Args:
        pillars_data: List of pillar dicts from pillar_results.json.
        rect_data: Rectangular result dict from rectangular_pca_results.json.
        output_dir: If provided, save PNGs to this directory.
    """
    # Extract axes
    largest_pillar = max(pillars_data, key=lambda p: p.get("num_inlier_points", 0))
    pillar_axis = np.array(largest_pillar["axis"], dtype=np.float64)
    pillar_axis = pillar_axis / np.linalg.norm(pillar_axis)

    rect_axis = np.array(rect_data["axes"][0], dtype=np.float64)
    rect_axis = rect_axis / np.linalg.norm(rect_axis)

    # Compute angle
    angle_deg = compute_angle_between_axes(pillar_axis, rect_axis)
    print(f"\nAxis comparison: Pillar PC1 vs Rectangular PC1")
    print(f"  Pillar axis:      [{pillar_axis[0]:.4f}, {pillar_axis[1]:.4f}, {pillar_axis[2]:.4f}]")
    print(f"  Rectangular axis: [{rect_axis[0]:.4f}, {rect_axis[1]:.4f}, {rect_axis[2]:.4f}]")
    print(f"  Angle: {angle_deg:.2f}°")

    # Build save paths
    vec_save = os.path.join(output_dir, "axis_vector_diagram.png") if output_dir else None
    sphere_save = os.path.join(output_dir, "axis_unit_sphere.png") if output_dir else None

    plot_vector_diagram(pillar_axis, rect_axis, angle_deg, save_path=vec_save)
    plot_unit_sphere(pillar_axis, rect_axis, angle_deg, save_path=sphere_save)
