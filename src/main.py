#!/usr/bin/env python3
"""
Multi-Color Pillar Detection from PLY Point Cloud Files

Pipeline stages:
1. Load PLY → 2. Downsample → 3. Color segment → 4. Cluster →
5. PCA analysis → 6. Visualize
"""

import json
import os
import time
import numpy as np
import cupy as cp
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

from . import config
from .config import (
    PLY_DIR, DOWNSAMPLE_DIR, ENABLE_INTERMEDIATE_SAVES,
    GPU_DEVICE_ID, DOWNSAMPLING_ENABLED, DOWNSAMPLING_VOXEL_SIZE, ENABLE_VISUALIZATION,
    PILLAR_JSON_FILENAME, RECTANGULAR_JSON_FILENAME, TRIPLANE_JSON_FILENAME,
    create_run_output_dir,
)
from .core.gpu import PointCloudGPU
from .core.utils import save_intermediate, format_voxel_size
from .file_io.ply_io import load_ply_file_open3d, save_ply_file_open3d
from .preprocessing.downsampling import downsample_gpu
from .preprocessing.color_segmentation import segment_by_color
from .preprocessing.roi_selection import select_roi_gui, crop_to_roi, select_z_roi_gui, crop_to_z_roi
from .preprocessing.hsv_analysis import analyze_hsv_gui
from .analysis.clustering import cluster_colored_points
from .analysis.pca_analysis import detect_pillars_with_pca, analyze_rectangular_pca
from .visualization.visualization import (
    create_visualization_output, create_clustering_visualization,
    launch_all_viewers, create_rectangular_visualization,
    show_rectangular_overlay_viewer, create_triplane_visualization,
    show_triplane_overlay_viewer,
)


# =============================================================================
# FILE SELECTION
# =============================================================================

def prompt_source_selection() -> str:
    """Let the user choose between original and downsampled PLY files.

    Returns:
        'original' or 'downsampled'
    """
    downsample_dir = Path(DOWNSAMPLE_DIR)
    has_downsampled = downsample_dir.is_dir() and any(downsample_dir.glob("*.ply"))

    if not has_downsampled:
        return "original"

    print("\nSelect source type:")
    print("  1) Original PLY")
    print("  2) Downsampled PLY")
    print()

    while True:
        try:
            choice = int(input("Select source number (1-2): "))
            if choice == 1:
                print("Selected: Original PLY\n")
                return "original"
            elif choice == 2:
                print("Selected: Downsampled PLY\n")
                return "downsampled"
        except (ValueError, EOFError):
            pass
        print("Please enter 1 or 2.")


def list_ply_files(ply_dir: str = PLY_DIR) -> list[Path]:
    """Scan directory and return sorted list of .ply file paths."""
    ply_path = Path(ply_dir)
    if not ply_path.is_dir():
        raise FileNotFoundError(f"PLY directory not found: {ply_dir}")

    ply_files = sorted(ply_path.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No .ply files found in {ply_dir}")

    return ply_files


def prompt_file_selection(ply_files: list[Path]) -> str:
    """Let the user pick a PLY file interactively."""
    if len(ply_files) == 1:
        selected = ply_files[0]
        print(f"Auto-selected (only file): {selected.name}")
        return str(selected)

    print("\nAvailable PLY files:")
    for i, f in enumerate(ply_files, 1):
        print(f"  {i}) {f.name}")
    print()

    while True:
        try:
            choice = int(input(f"Select file number (1-{len(ply_files)}): "))
            if 1 <= choice <= len(ply_files):
                selected = ply_files[choice - 1]
                print(f"Selected: {selected.name}\n")
                return str(selected)
        except (ValueError, EOFError):
            pass
        print(f"Please enter a number between 1 and {len(ply_files)}.")


def prompt_pca_method() -> str:
    """Let the user choose the PCA analysis method.

    Returns:
        'cylinder', 'traditional', 'rectangular', or 'triplane'
    """
    print("\nSelect PCA method:")
    print("  [1] cylinder    - Cylindrical shape detection")
    print("  [2] traditional - Traditional PCA (PC1 = reference axis)")
    print("  [3] rectangular - Rectangular box detection (ROI-based)")
    print("  [4] triplane    - 3-zone rectangular PCA (Z-region based)")
    print()

    while True:
        try:
            choice = int(input("Select (1-4) [default: 1]: ") or "1")
            if choice == 1:
                print("Selected: cylinder PCA\n")
                return "cylinder"
            elif choice == 2:
                print("Selected: traditional PCA\n")
                return "traditional"
            elif choice == 3:
                print("Selected: rectangular PCA\n")
                return "rectangular"
            elif choice == 4:
                print("Selected: triplane PCA\n")
                return "triplane"
        except (ValueError, EOFError):
            pass
        print("Please enter 1, 2, 3, or 4.")


# =============================================================================
# GPU INITIALIZATION
# =============================================================================

def init_gpu() -> None:
    """Initialize CUDA device."""
    cp.cuda.Device(GPU_DEVICE_ID).use()
    print(f"Using GPU device: {GPU_DEVICE_ID}")


# =============================================================================
# PIPELINE STAGES
# =============================================================================

def load_and_downsample(ply_path: str) -> PointCloudGPU:
    """Load PLY file, transfer to GPU, and optionally downsample."""
    # Load from disk (CPU)
    points_np, colors_np = load_ply_file_open3d(ply_path)

    # CPU → GPU
    cloud = PointCloudGPU.from_numpy(points_np, colors_np)
    print(f"Transferred {len(cloud):,} points to GPU")

    # Downsample
    if DOWNSAMPLING_ENABLED:
        ds_points, ds_colors = downsample_gpu(cloud.points, cloud.colors)
        cloud = PointCloudGPU(ds_points, ds_colors)

        # Always save downsampled result to cache
        cache_path = downsample_cache_path(ply_path)
        pts_cpu, cols_cpu = cloud.to_cpu()
        try:
            print(f"Saving downsampled cache to: {cache_path}")
            save_ply_file_open3d(cache_path, pts_cpu, cols_cpu)
            print(f"Downsampled cache saved successfully")
        except Exception as e:
            print(f"Warning: Failed to save downsampled cache: {str(e)}")
    else:
        print("  Downsampling: DISABLED - using original point cloud")

    return cloud


def downsample_cache_path(ply_path: str) -> str:
    """Build the cache file path for a downsampled PLY."""
    ply_name = Path(ply_path).stem
    voxel_str = format_voxel_size(DOWNSAMPLING_VOXEL_SIZE)
    return str(Path(DOWNSAMPLE_DIR) / f"{ply_name}-{voxel_str}.ply")


def load_only(ply_path: str) -> PointCloudGPU:
    """Load PLY file and transfer to GPU without downsampling."""
    points_np, colors_np = load_ply_file_open3d(ply_path)
    cloud = PointCloudGPU.from_numpy(points_np, colors_np)
    print(f"Transferred {len(cloud):,} points to GPU (downsampling skipped)")
    return cloud


def detect_pillars(cloud: PointCloudGPU, h_ranges: list, s_min: float, v_min: float, method: str = 'cylinder') -> list[Dict[str, Any]]:
    """Run color segmentation → clustering → PCA analysis."""
    print("=" * 60)
    print("[3/6] Color Filtering")
    print("=" * 60)
    # Color segmentation
    colored_points, colored_colors, colored_indices = segment_by_color(
        cloud.points, cloud.colors, h_ranges, s_min, v_min
    )

    # Save color-filtered intermediate
    if ENABLE_INTERMEDIATE_SAVES and len(colored_points) > 0:
        save_intermediate(
            cp.asnumpy(colored_points), cp.asnumpy(colored_colors),
            config.get_colored_points_path(),
            f"Filtered points ({len(colored_points):,} points)",
        )

    if len(colored_points) == 0:
        print("No matched points found. Exiting.")
        return []

    print("=" * 60)
    print("[4/6] Clustering")
    print("=" * 60)
    clusters, cluster_ids, cluster_indices = cluster_colored_points(
        colored_points, colored_indices
    )

    if len(clusters) == 0:
        print("No valid clusters found. Exiting.")
        return []

    # Save clustering visualization intermediate
    if ENABLE_INTERMEDIATE_SAVES:
        pts_cpu, cols_cpu = cloud.to_cpu()
        clusters_np = [cp.asnumpy(c) for c in clusters]
        indices_np = [cp.asnumpy(idx) for idx in cluster_indices]
        cluster_viz_points, cluster_viz_colors = create_clustering_visualization(
            pts_cpu, cols_cpu, clusters_np, cluster_ids, indices_np
        )
        save_intermediate(
            cluster_viz_points, cluster_viz_colors,
            config.CLUSTERED_PLY_PATH, "clustering visualization",
        )

    print("=" * 60)
    print("[5/6] Pillar Detection")
    print("=" * 60)
    detected_pillars = detect_pillars_with_pca(clusters, cluster_ids, method=method)
    return detected_pillars


# =============================================================================
# RESULTS
# =============================================================================

def calculate_pillar_metrics(detected_pillars: List[Dict[str, Any]]) -> list[dict]:
    """Calculate height and other metrics for each detected pillar."""
    metrics = []
    for pillar in detected_pillars:
        center = pillar['center']
        axis = pillar['axis']
        height = 0.0
        if len(pillar['inlier_points']) > 0:
            axis_norm = axis / np.linalg.norm(axis)
            projections = np.dot(pillar['inlier_points'] - center, axis_norm)
            height = float(np.max(projections) - np.min(projections))

        metric = {
            'center': center,
            'height': height,
            'num_inlier_points': len(pillar['inlier_points']),
            'analysis_method': pillar.get('analysis_method', 'PCA'),
        }
        if 'radius' in pillar:
            metric['radius'] = pillar['radius']
        if 'confidence' in pillar:
            metric['confidence'] = pillar['confidence']
        if 'eigenvalue_ratios' in pillar:
            metric['eigenvalue_ratios'] = pillar['eigenvalue_ratios']
        metrics.append(metric)
    return metrics


def save_pillar_results_json(
    detected_pillars: List[Dict[str, Any]],
    metrics: list[dict],
    source_file: str,
) -> str:
    """Save pillar detection results to JSON in the run output directory.

    Args:
        detected_pillars: Raw pillar dicts from detect_pillars_with_pca().
        metrics: Metric dicts from calculate_pillar_metrics().
        source_file: Original PLY filename.

    Returns:
        Path to the written JSON file.
    """
    pillars_json = []
    for pillar, m in zip(detected_pillars, metrics):
        entry = {
            "cluster_id": int(pillar["cluster_id"]),
            "center": pillar["center"].tolist(),
            "axis": (pillar["axis"] / np.linalg.norm(pillar["axis"])).tolist(),
            "height": m["height"],
            "radius": m.get("radius"),
            "confidence": m.get("confidence"),
            "num_inlier_points": m["num_inlier_points"],
        }
        pillars_json.append(entry)

    result = {
        "version": "1.0",
        "source_file": Path(source_file).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "pillars": pillars_json,
    }

    json_path = os.path.join(config._run_dir, PILLAR_JSON_FILENAME)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Pillar results saved to: {json_path}")
    return json_path


def save_rectangular_results_json(
    rectangular_result: Dict[str, Any],
    source_file: str,
) -> str:
    """Save rectangular PCA results to JSON in the run output directory.

    Args:
        rectangular_result: Dict from analyze_rectangular_pca().
        source_file: Original PLY filename.

    Returns:
        Path to the written JSON file.
    """
    result = {
        "version": "1.0",
        "source_file": Path(source_file).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "center": rectangular_result["center"].tolist(),
        "axes": rectangular_result["axes"].tolist(),
        "dimensions": rectangular_result["dimensions"].tolist(),
        "vertices": rectangular_result["vertices"].tolist(),
        "eigenvalue_ratios": rectangular_result["eigenvalue_ratios"].tolist(),
        "point_count": rectangular_result["point_count"],
        "analysis_method": rectangular_result["analysis_method"],
    }

    json_path = os.path.join(config._run_dir, RECTANGULAR_JSON_FILENAME)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Rectangular PCA results saved to: {json_path}")
    return json_path


def save_triplane_results_json(
    zone_results: list[Dict[str, Any]],
    z_rois: list[tuple],
    source_file: str,
) -> str:
    """Save triplane PCA v2 results to JSON in the run output directory."""
    color_names = ["blue", "magenta", "green"]
    zones_json = []
    for i, (result, z_roi) in enumerate(zip(zone_results, z_rois)):
        h_min, h_max, z_min, z_max, axis_name = z_roi
        zones_json.append({
            "zone_id": i,
            "color": color_names[i],
            "z_roi": {
                "h_min": float(h_min), "h_max": float(h_max),
                "z_min": float(z_min), "z_max": float(z_max),
                "axis": axis_name,
            },
            "center": result["center"].tolist(),
            "axes": result["axes"].tolist(),
            "dimensions": result["dimensions"].tolist(),
            "vertices": result["vertices"].tolist(),
            "eigenvalue_ratios": result["eigenvalue_ratios"].tolist(),
            "point_count": result["point_count"],
        })

    # Select the principal axis closest to Z for each zone
    def _pick_z_axis(axes):
        """Pick the principal component with the largest |Z| component."""
        z_abs = [abs(float(axes[k][2])) for k in range(3)]
        idx = z_abs.index(max(z_abs))
        ax = np.array(axes[idx])
        return ax / np.linalg.norm(ax)

    angles = {}
    z_axes = [_pick_z_axis(r["axes"]) for r in zone_results]
    for i in range(3):
        for j in range(i + 1, 3):
            dot = abs(float(np.dot(z_axes[i], z_axes[j])))
            angle = float(np.degrees(np.arccos(np.clip(dot, 0.0, 1.0))))
            angles[f"zone_{i}_{j}"] = round(angle, 3)

    output = {
        "version": "2.0",
        "source_file": Path(source_file).name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_method": "triplane_PCA",
        "zones": zones_json,
        "angles_between_zones": angles,
        "point_count_total": sum(r["point_count"] for r in zone_results),
    }

    json_path = os.path.join(config._run_dir, TRIPLANE_JSON_FILENAME)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Triplane PCA results saved to: {json_path}")
    return json_path


def print_detection_summary(
    detected_pillars: List[Dict[str, Any]],
    metrics: list[dict],
) -> None:
    """Print formatted summary of detected pillars."""
    print("\n" + "=" * 60)
    print("PILLAR DETECTION SUMMARY")
    print("=" * 60)

    if not detected_pillars:
        print("No pillars detected.")
        return

    print(f"Total pillars detected: {len(detected_pillars)}")
    print()

    for i, (pillar, m) in enumerate(zip(detected_pillars, metrics)):
        center = m['center']
        print(f"Pillar {i + 1}:")
        print(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        if 'radius' in m:
            print(f"  Radius: {m['radius']:.3f} m")
        print(f"  Height: {m['height']:.3f} m")
        if 'confidence' in m:
            print(f"  Confidence: {m['confidence']:.3f}")
        if 'eigenvalue_ratios' in m:
            ev = m['eigenvalue_ratios']
            print(f"  Eigenvalues: [{ev[0]:.3f}, {ev[1]:.3f}, {ev[2]:.3f}]")
        print(f"  Inlier points: {m['num_inlier_points']:,}")
        print(f"  Method: {m['analysis_method']}")
        print()


def visualize_results(
    detected_pillars: List[Dict[str, Any]],
    cloud: PointCloudGPU,
) -> None:
    """Create and save final visualization."""
    pts_cpu, cols_cpu = cloud.to_cpu()
    output_points, output_colors = create_visualization_output(
        pts_cpu, cols_cpu, detected_pillars
    )
    save_ply_file_open3d(config.OUTPUT_PLY_PATH, output_points, output_colors)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """Main pipeline execution."""
    source = prompt_source_selection()

    if source == "downsampled":
        input_ply_path = prompt_file_selection(list_ply_files(DOWNSAMPLE_DIR))
        is_downsampled = True
    else:
        input_ply_path = prompt_file_selection(list_ply_files())
        is_downsampled = False

    run_dir = create_run_output_dir(input_ply_path)
    print(f"Output directory: {run_dir}")

    pca_method = prompt_pca_method()

    overall_start_time = time.time()

    try:
        print("=" * 60)
        print("[1/6] Loading Point Cloud")
        print("=" * 60)
        init_gpu()

        if is_downsampled:
            cloud = load_only(input_ply_path)
        else:
            cloud = load_and_downsample(input_ply_path)

        print("=" * 60)
        print("[2/6] ROI Selection")
        print("=" * 60)
        roi = select_roi_gui(cloud)
        if roi is not None:
            cloud = crop_to_roi(cloud, roi)

        # Rectangular PCA: separate pipeline branch
        if pca_method == 'rectangular':
            # Z ROI selection (side view)
            z_roi = select_z_roi_gui(cloud)
            if z_roi is not None:
                h_min, h_max, z_min, z_max, axis_name = z_roi
                cloud = crop_to_z_roi(cloud, h_min, h_max, z_min, z_max, axis_name)

            print("=" * 60)
            print("Rectangular PCA Analysis")
            print("=" * 60)
            rect_result = analyze_rectangular_pca(cloud.points)

            if rect_result is None:
                print("Rectangular PCA failed — no result.")
                total_time = time.time() - overall_start_time
                print(f"Total processing time: {total_time:.2f} seconds")
                return

            # Print summary
            print(f"\nRectangular PCA Result:")
            center = rect_result['center']
            dims = rect_result['dimensions']
            ev = rect_result['eigenvalue_ratios']
            print(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
            print(f"  Dimensions: {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f} m")
            print(f"  Eigenvalues: [{ev[0]:.3f}, {ev[1]:.3f}, {ev[2]:.3f}]")
            print(f"  Points: {rect_result['point_count']:,}")

            # Save JSON
            save_rectangular_results_json(rect_result, input_ply_path)

            # Create visualization and save PLY
            pts_cpu, cols_cpu = cloud.to_cpu()
            output_points, output_colors = create_rectangular_visualization(
                pts_cpu, cols_cpu, rect_result
            )
            save_ply_file_open3d(config.OUTPUT_PLY_PATH, output_points, output_colors)

            total_time = time.time() - overall_start_time
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Output saved to: {config.OUTPUT_PLY_PATH}")

            # Launch viewers
            if ENABLE_VISUALIZATION:
                rect_json_path = os.path.join(config._run_dir, RECTANGULAR_JSON_FILENAME)
                with open(rect_json_path, "r", encoding="utf-8") as f:
                    rect_json = json.load(f)

                targets = [("Rectangular PCA (Final)", config.OUTPUT_PLY_PATH)]
                launch_all_viewers(
                    targets,
                    overlay=("Wireframe Overlay", config.OUTPUT_PLY_PATH, rect_json),
                    overlay_viewer_fn=show_rectangular_overlay_viewer,
                )
            return

        # Triplane PCA: 3-zone rectangular PCA branch
        if pca_method == 'triplane':
            color_names = ["Blue", "Magenta", "Green"]
            zone_results = []
            z_rois = []

            for zone_idx in range(3):
                print("=" * 60)
                print(f"Z Region {zone_idx + 1}/3 ({color_names[zone_idx]})")
                print("=" * 60)

                z_roi = select_z_roi_gui(cloud)
                if z_roi is None:
                    print(f"Z region {zone_idx + 1} cancelled.")
                    total_time = time.time() - overall_start_time
                    print(f"Total processing time: {total_time:.2f} seconds")
                    return

                h_min, h_max, z_min, z_max, axis_name = z_roi
                zone_cloud = crop_to_z_roi(cloud, h_min, h_max, z_min, z_max, axis_name)

                print(f"  Analyzing zone {zone_idx + 1} ({len(zone_cloud):,} points)...")
                rect_result = analyze_rectangular_pca(zone_cloud.points)

                if rect_result is None:
                    print(f"Rectangular PCA failed for zone {zone_idx + 1}.")
                    total_time = time.time() - overall_start_time
                    print(f"Total processing time: {total_time:.2f} seconds")
                    return

                zone_results.append(rect_result)
                z_rois.append(z_roi)

                center = rect_result['center']
                dims = rect_result['dimensions']
                ev = rect_result['eigenvalue_ratios']
                print(f"  Zone {zone_idx + 1} ({color_names[zone_idx]}):")
                print(f"    Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
                print(f"    Dimensions: {dims[0]:.3f} x {dims[1]:.3f} x {dims[2]:.3f} m")
                print(f"    Eigenvalues: [{ev[0]:.3f}, {ev[1]:.3f}, {ev[2]:.3f}]")
                print(f"    Points: {rect_result['point_count']:,}")

            # Save JSON
            save_triplane_results_json(zone_results, z_rois, input_ply_path)

            # Build triplane_result dict for visualization
            triplane_result = {"zones": zone_results}

            # Create visualization and save PLY
            pts_cpu, cols_cpu = cloud.to_cpu()
            output_points, output_colors = create_triplane_visualization(
                pts_cpu, cols_cpu, triplane_result
            )
            save_ply_file_open3d(config.OUTPUT_PLY_PATH, output_points, output_colors)

            # Free inlier_points (wireframe-only visualization)
            for result in zone_results:
                result.pop('inlier_points', None)

            total_time = time.time() - overall_start_time
            print(f"Total processing time: {total_time:.2f} seconds")
            print(f"Output saved to: {config.OUTPUT_PLY_PATH}")

            # Launch viewers
            if ENABLE_VISUALIZATION:
                triplane_json_path = os.path.join(config._run_dir, TRIPLANE_JSON_FILENAME)
                with open(triplane_json_path, "r", encoding="utf-8") as f:
                    triplane_json = json.load(f)

                targets = [("Triplane PCA (Final)", config.OUTPUT_PLY_PATH)]
                launch_all_viewers(
                    targets,
                    overlay=("Triplane Overlay", config.OUTPUT_PLY_PATH, triplane_json),
                    overlay_viewer_fn=show_triplane_overlay_viewer,
                )
            return

        # HSV analysis and filter setting
        hsv_result = analyze_hsv_gui(cloud)
        if hsv_result is None:
            print("HSV filter skipped. Exiting.")
            return
        h_ranges, s_min, v_min = hsv_result

        detected_pillars = detect_pillars(cloud, h_ranges, s_min, v_min, method=pca_method)

        # Calculate metrics and save JSON (even if empty)
        metrics = calculate_pillar_metrics(detected_pillars)
        save_pillar_results_json(detected_pillars, metrics, input_ply_path)

        if not detected_pillars:
            print("No pillars detected after PCA analysis.")
            total_time = time.time() - overall_start_time
            print(f"Total processing time: {total_time:.2f} seconds")
            return

        print("=" * 60)
        print("[6/6] Output")
        print("=" * 60)
        print_detection_summary(detected_pillars, metrics)
        visualize_results(detected_pillars, cloud)

        total_time = time.time() - overall_start_time
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Output saved to: {config.OUTPUT_PLY_PATH}")

        # Launch interactive viewers if enabled
        if ENABLE_VISUALIZATION:
            targets = []
            if not is_downsampled and DOWNSAMPLING_ENABLED:
                targets.append(("Downsampled", downsample_cache_path(input_ply_path)))
            if ENABLE_INTERMEDIATE_SAVES:
                targets.append(("Filtered Points", config.get_colored_points_path()))
                targets.append(("Clusters", config.CLUSTERED_PLY_PATH))
            targets.append(("Pillars (Final)", config.OUTPUT_PLY_PATH))
            launch_all_viewers(targets)

    except Exception as e:
        print(f"Error in pipeline execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
