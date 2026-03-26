"""
Academic-style triplane comparison figures for PCA analysis results.

Generates two Matplotlib figures comparing the 3 detected plane normals:
1. 3D vector diagram with angle arcs between all pairs
2. Unit sphere with direction points and great circle arcs
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .axis_comparison import compute_angle_between_axes, _slerp_arc


# Colors matching the triplane visualization
_PLANE_COLORS = ["#0000FF", "#FF00FF", "#00FF00"]
_PLANE_LABELS = ["Plane 0 (Blue)", "Plane 1 (Magenta)", "Plane 2 (Green)"]


def plot_triplane_vector_diagram(
    normals: list[np.ndarray],
    angles: dict[str, float],
    save_path: str | None = None,
) -> None:
    """Generate 3D vector diagram comparing three plane normals.

    Draws three unit vectors as arrows from the origin with angle
    arcs between each pair. Academic paper figure style.

    Args:
        normals: List of 3 unit normal vectors (3,).
        angles: Dict of pairwise angles (e.g. {'plane_0_1': 45.2, ...}).
        save_path: If provided, save PNG at this path (300 DPI).
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    origin = [0, 0, 0]

    # Draw normal arrows
    for i, (normal, color, label) in enumerate(zip(normals, _PLANE_COLORS, _PLANE_LABELS)):
        ax.quiver(*origin, *normal, color=color, arrow_length_ratio=0.08,
                  linewidth=2.0, label=label)

    # Draw angle arcs between each pair
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        key = f"plane_{i}_{j}"
        angle_deg = angles.get(key, 0.0)
        if angle_deg < 0.1:
            continue

        n_i = normals[i]
        n_j = normals[j]

        # Ensure same hemisphere for arc rendering
        if np.dot(n_i, n_j) < 0:
            arc_target = -n_j
        else:
            arc_target = n_j

        arc_radius = 0.3 + i * 0.05  # Slightly different radii to avoid overlap
        arc_points = _slerp_arc(n_i, arc_target) * arc_radius
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                "--", color="#333333", linewidth=0.8)

        # Angle label at arc midpoint
        mid = arc_points[len(arc_points) // 2]
        ax.text(mid[0] * 1.5, mid[1] * 1.5, mid[2] * 1.5,
                f"{angle_deg:.1f}°", fontsize=8, ha="center")

    # Reference axes (light gray)
    for axis_vec in [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        ax.plot([0, axis_vec[0]], [0, axis_vec[1]], [0, axis_vec[2]],
                color="#CCCCCC", linewidth=0.5, linestyle=":")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved triplane vector diagram: {save_path}")

    plt.show()


def plot_triplane_unit_sphere(
    normals: list[np.ndarray],
    angles: dict[str, float],
    save_path: str | None = None,
) -> None:
    """Generate unit sphere visualization comparing three plane normals.

    Plots direction points on a wireframe unit sphere with great
    circle arcs connecting each pair.

    Args:
        normals: List of 3 unit normal vectors (3,).
        angles: Dict of pairwise angles.
        save_path: If provided, save PNG at this path (300 DPI).
    """
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Wireframe sphere
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(x, y, z, color="#AAAAAA", alpha=0.4, linewidth=0.5)

    # Direction points
    for normal, color, label in zip(normals, _PLANE_COLORS, _PLANE_LABELS):
        ax.scatter(*normal, color=color, s=100, zorder=5,
                   edgecolors="black", linewidths=0.5, label=label)

    # Great circle arcs
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        key = f"plane_{i}_{j}"
        angle_deg = angles.get(key, 0.0)
        if angle_deg < 0.1:
            continue

        n_i = normals[i]
        n_j = normals[j]

        if np.dot(n_i, n_j) < 0:
            arc_target = -n_j
        else:
            arc_target = n_j

        arc_points = _slerp_arc(n_i, arc_target)
        ax.plot(arc_points[:, 0], arc_points[:, 1], arc_points[:, 2],
                "--", color="#333333", linewidth=0.8)

        mid = arc_points[len(arc_points) // 2]
        ax.text(mid[0] * 1.3, mid[1] * 1.3, mid[2] * 1.3,
                f"{angle_deg:.1f}°", fontsize=8, ha="center")

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
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
) -> None:
    """Generate triplane comparison figures from JSON result data.

    Entry point called by viewer.py when triplane results exist.

    Args:
        triplane_data: Triplane result dict from triplane_results.json.
        output_dir: If provided, save PNGs to this directory.
    """
    planes = triplane_data.get("planes", [])
    if len(planes) < 3:
        print("Triplane comparison: need 3 planes, skipping.")
        return

    # Extract and normalize normals
    normals = []
    for plane in planes[:3]:
        normal = np.array(plane["normal"], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
        normals.append(normal)

    # Get angles from JSON
    angles = triplane_data.get("angles_between_planes", {})

    # Print summary
    print(f"\nTriplane normal comparison:")
    color_names = ["Blue", "Magenta", "Green"]
    for i, (normal, name) in enumerate(zip(normals, color_names)):
        print(f"  {name}: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
    for key, val in angles.items():
        print(f"  {key}: {val:.3f}°")

    # Build save paths
    vec_save = os.path.join(output_dir, "triplane_vector_diagram.png") if output_dir else None
    sphere_save = os.path.join(output_dir, "triplane_unit_sphere.png") if output_dir else None

    plot_triplane_vector_diagram(normals, angles, save_path=vec_save)
    plot_triplane_unit_sphere(normals, angles, save_path=sphere_save)
