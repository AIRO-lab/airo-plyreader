"""
Configuration parameters for the multi-color pillar detection pipeline.

This module contains all global configuration constants used throughout
the pillar detection pipeline, including file paths, color segmentation
parameters, clustering settings, geometric constraints, and visualization
parameters.
"""

import os
import re
from datetime import datetime

# =============================================================================
# INPUT/OUTPUT CONFIGURATION
# =============================================================================

# Input/Output Configuration
PLY_DIR = "ply"
DOWNSAMPLE_DIR = "ply/downsample"
OUTPUT_DIR = "output"
PILLAR_JSON_FILENAME = "pillar_results.json"
RECTANGULAR_JSON_FILENAME = "rectangular_pca_results.json"
TRIPLANE_JSON_FILENAME = "triplane_results.json"

# Runtime paths — set by create_run_output_dir()
_run_dir = None
OUTPUT_PLY_PATH = None
CLUSTERED_PLY_PATH = None

# Intermediate Results Configuration
ENABLE_INTERMEDIATE_SAVES = True


def create_run_output_dir(ply_path: str) -> str:
    """Create a timestamped output directory for this pipeline run.

    Args:
        ply_path: Full path to the selected PLY file.

    Returns:
        The created directory path.
    """
    global _run_dir, OUTPUT_PLY_PATH, CLUSTERED_PLY_PATH

    # Extract and sanitize filename
    ply_name = os.path.splitext(os.path.basename(ply_path))[0]
    ply_name = ply_name.replace(" ", "_")
    ply_name = re.sub(r"[^\w.\-]", "", ply_name)

    # Generate compact timestamp
    timestamp = datetime.now().strftime("%y%m%d%H%M%S")

    # Create directory
    _run_dir = os.path.join(OUTPUT_DIR, f"{ply_name}-{timestamp}")
    os.makedirs(_run_dir, exist_ok=True)

    # Set output paths
    OUTPUT_PLY_PATH = os.path.join(_run_dir, "output_pillars.ply")
    CLUSTERED_PLY_PATH = os.path.join(_run_dir, "clustered_points.ply")

    return _run_dir


# =============================================================================
# GPU CONFIGURATION
# =============================================================================
GPU_DEVICE_ID = 0                      # GPU device to use
GPU_CHUNK_SIZE = 20_000_000            # Max points per GPU processing chunk

# =============================================================================
# COLOR DETECTION MODE AND PARAMETERS
# =============================================================================
# Color Detection Mode: 'red', 'blue', 'green'
COLOR_DETECTION_MODE = None


def set_color_mode(mode: str) -> None:
    """Set the color detection mode at runtime.

    Args:
        mode: Color name (must be a key in COLOR_PARAMS).
    """
    global COLOR_DETECTION_MODE
    mode = mode.lower()
    if mode not in COLOR_PARAMS:
        raise ValueError(
            f"Unknown color mode '{mode}'. "
            f"Available: {', '.join(COLOR_PARAMS.keys())}"
        )
    COLOR_DETECTION_MODE = mode


def get_colored_points_path():
    return os.path.join(_run_dir, "filtered_points_only.ply")


# HSV Color Segmentation Parameters
# H: 0-360 degrees (pipeline uses custom GPU RGB→HSV, not OpenCV's 0-180 range)
# S: 0-1, V: 0-1
COLOR_PARAMS = {
    'red':   {'h_ranges': [(0, 10), (350, 360)], 's_min': 0.55, 'v_min': 0.45},
    'blue':  {'h_ranges': [(220, 250)],           's_min': 0.80, 'v_min': 0.25},
    'green': {'h_ranges': [(80, 160)],             's_min': 0.40, 'v_min': 0.35},
}

# =============================================================================
# DBSCAN CLUSTERING PARAMETERS
# =============================================================================

# DBSCAN Clustering Parameters (TIGHTER for more precise clusters)
# Maximum distance between points in cluster (meters)
DBSCAN_EPS = 0.02
DBSCAN_MIN_SAMPLES = 50                    # Minimum points to form a cluster

# =============================================================================
# CYLINDER GEOMETRIC CONSTRAINTS
# =============================================================================

# Cylinder Geometric Constraints (tightened to match real-world pillar dimensions)
# Minimum pillar radius (meters)
PILLAR_RADIUS_MIN = 0.01
# Maximum pillar radius (meters)
PILLAR_RADIUS_MAX = 0.30
# Minimum pillar height (meters)
PILLAR_HEIGHT_MIN = 0.1
# Maximum pillar height (meters)
PILLAR_HEIGHT_MAX = 2.0
# Maximum allowed angle between pillar axis and Z-axis (degrees)
# Set to None to disable angle constraint
PILLAR_AXIS_MAX_ANGLE_DEG = None

# =============================================================================
# VISUALIZATION PARAMETERS
# =============================================================================

# Visualization Parameters
GRAY_COLOR = (128, 128, 128)              # Gray color for original points
RED_COLOR = (255, 0, 0)                   # Red color for detected pillars
CYAN_COLOR = (0, 255, 255)
# Points per unit length for cylinder sampling
CYLINDER_SAMPLE_DENSITY = 50

# Interactive Open3D viewer after pipeline completion
ENABLE_VISUALIZATION = True

# =============================================================================
# DOWNSAMPLING PARAMETERS
# =============================================================================

# Downsampling Parameters (GPU Voxel Grid)
DOWNSAMPLING_ENABLED = True                # Enable/disable downsampling step
# Voxel size for voxel grid method (meters)
DOWNSAMPLING_VOXEL_SIZE = 0.0001

# =============================================================================
# PCA ANALYSIS PARAMETERS
# =============================================================================

# PCA Analysis Parameters
# Minimum eigenvalue ratio for primary axis to be considered cylindrical
PCA_CYLINDER_THRESHOLD = 0.6
# Maximum difference between 2nd and 3rd eigenvalue ratios for cylindrical detection
PCA_CROSS_SECTION_RATIO_THRESHOLD = 0.3
# Minimum variance required for secondary axes (to avoid degenerate cases)
PCA_MIN_SECONDARY_VARIANCE = 0.05
# Maximum allowed variance for tertiary axis in cylindrical structures
PCA_MAX_TERTIARY_VARIANCE = 0.4

# =============================================================================
# PROCESSING PARAMETERS
# =============================================================================

# Processing Parameters
# Limit for cylinder fitting to prevent memory issues
MAX_POINTS_PER_CLUSTER = 50000
# Number of top clusters (by point count) to analyze with PCA
TOP_CLUSTERS_TO_ANALYZE = 5

# =============================================================================
# TRIPLANE PCA PARAMETERS
# =============================================================================

# KNN neighbor count for local surface normal estimation
TRIPLANE_K_NEIGHBORS = 50
# Visualization square side length (meters)
TRIPLANE_SQUARE_SIZE = 0.5
# Minimum angle (degrees) between any two plane normals to accept as distinct
TRIPLANE_ANGLE_THRESHOLD = 5.0
# Maximum random point sampling attempts before giving up
TRIPLANE_MAX_ATTEMPTS = 100
# Flatness warning threshold (λ3/sum(λ) above this triggers warning)
TRIPLANE_FLATNESS_WARN = 0.1

# Triplane visualization colors (RGB 0-1 for Open3D, converted to 0-255 for PLY)
TRIPLANE_COLOR_BLUE = [0.0, 0.0, 1.0]
TRIPLANE_COLOR_MAGENTA = [1.0, 0.0, 1.0]
TRIPLANE_COLOR_GREEN = [0.0, 1.0, 0.0]
