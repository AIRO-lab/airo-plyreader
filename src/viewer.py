"""
Standalone viewer for previous pipeline results.

Scans the output directory for previous runs and opens the selected
run's PLY files in Open3D viewer windows. If pillar_results.json
exists, overlays reference axes on a user-selected downsampled PLY.
"""

import json
from pathlib import Path


def list_run_dirs() -> list[Path]:
    """Scan output directory for previous run subdirectories."""
    from .config import OUTPUT_DIR
    output_dir = Path(OUTPUT_DIR)
    if not output_dir.is_dir():
        return []
    return sorted(
        [d for d in output_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )


def title_from_filename(filename: str) -> str:
    """Derive a display title from a PLY filename.

    Example: 'downsampled_points.ply' -> 'Downsampled Points'
    """
    return filename.removesuffix(".ply").replace("_", " ").title()


def prompt_run_selection(run_dirs: list[Path]) -> Path | None:
    """Let the user pick a run directory interactively."""
    print("\nAvailable runs:")
    for i, d in enumerate(run_dirs, 1):
        ply_count = len(list(d.glob("*.ply")))
        print(f"  {i}) {d.name}  ({ply_count} PLY files)")
    print()

    while True:
        try:
            choice = int(input(f"Select run number (1-{len(run_dirs)}): "))
            if 1 <= choice <= len(run_dirs):
                selected = run_dirs[choice - 1]
                print(f"Selected: {selected.name}\n")
                return selected
        except ValueError:
            pass
        except EOFError:
            print("\nNo input available.")
            return None
        print(f"Please enter a number between 1 and {len(run_dirs)}.")


def load_pillar_results(run_dir: Path) -> list[dict] | None:
    """Load pillar_results.json from a run directory if it exists.

    Returns:
        List of pillar dicts, or None if JSON not found.
    """
    from .config import PILLAR_JSON_FILENAME
    json_path = run_dir / PILLAR_JSON_FILENAME
    if not json_path.is_file():
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pillars = data.get("pillars", [])
    if not pillars:
        print(f"JSON found but no pillars recorded.")
        return None

    print(f"Loaded {len(pillars)} pillar(s) from {PILLAR_JSON_FILENAME}")
    return pillars


def prompt_downsample_selection() -> str | None:
    """Let the user pick a downsampled PLY file from ply/downsample/.

    Returns:
        Path string, or None if no files or user cancels.
    """
    from .config import DOWNSAMPLE_DIR
    ds_dir = Path(DOWNSAMPLE_DIR)
    if not ds_dir.is_dir():
        print(f"Downsample directory not found: {DOWNSAMPLE_DIR}")
        return None

    ply_files = sorted(ds_dir.glob("*.ply"))
    if not ply_files:
        print(f"No PLY files found in {DOWNSAMPLE_DIR}")
        return None

    if len(ply_files) == 1:
        selected = ply_files[0]
        print(f"Auto-selected downsample file: {selected.name}")
        return str(selected)

    print("\nAvailable downsampled PLY files:")
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
        except ValueError:
            pass
        except EOFError:
            print("\nNo input available.")
            return None
        print(f"Please enter a number between 1 and {len(ply_files)}.")


def main() -> None:
    """Main entry point for the standalone viewer."""
    run_dirs = list_run_dirs()

    if not run_dirs:
        print("No previous runs found in output/ directory.")
        return

    selected_dir = prompt_run_selection(run_dirs)
    if selected_dir is None:
        return

    # Build targets from all PLY files in the selected directory
    ply_files = sorted(selected_dir.glob("*.ply"))

    if not ply_files:
        print(f"No PLY files found in {selected_dir.name}/")
        return

    targets = [
        (title_from_filename(f.name), str(f))
        for f in ply_files
    ]

    print(f"Opening {len(targets)} file(s):")
    for title, path in targets:
        print(f"  - {title}")
    print()

    # Check for pillar results JSON and set up overlay
    overlay = None
    pillars = load_pillar_results(selected_dir)
    if pillars is not None:
        ds_path = prompt_downsample_selection()
        if ds_path is not None:
            overlay = ("Downsampled + Axes", ds_path, pillars)

    from .visualization.visualization import launch_all_viewers
    launch_all_viewers(targets, overlay=overlay)


if __name__ == "__main__":
    main()
