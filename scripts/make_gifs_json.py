#!/usr/bin/env python3
"""
<<<<<<< HEAD
GIF Frame Extractor and Mapper

This script identifies GIF files in a specified directory, maps each GIF to its
corresponding PNG frame files, and saves this mapping to a JSON file.
"""

import json
import glob
import os
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any


def setup_logging(log_level: str) -> None:
    """Configure logging with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    logging.basicConfig(
        level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s"
=======
GIF Scanner

This script scans directories for GIF files and their corresponding PNG frame images,
then creates a JSON file mapping GIFs to their frames.

Usage:
    python gif_scanner.py [--base-dir BASE_DIR] [--output OUTPUT_PATH]

Arguments:
    --base-dir: Base directory to scan (default: 'static')
    --output: Output JSON file path (default: '<base_dir>/gifs.json')
"""

import os
import glob
import json
import argparse
import logging
from typing import Dict, List, Optional


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
>>>>>>> a6a9c8ba01222f402c21a2af607107e63693c729
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
<<<<<<< HEAD
        description="Map GIF files to their corresponding PNG frames."
    )
    parser.add_argument(
        "--gif-dir",
        type=str,
        default="static/gifs/gif",
        help="Directory containing GIF files (default: static/gifs/gif)",
    )
    parser.add_argument(
        "--png-dir",
        type=str,
        default="static/gifs/gif/png",
        help="Base directory containing PNG frames (default: static/gifs/gif/png)",
=======
        description="Scan directories for GIFs and generate a JSON mapping file."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="static",
        help="Base directory to scan (default: static)",
>>>>>>> a6a9c8ba01222f402c21a2af607107e63693c729
    )
    parser.add_argument(
        "--output",
        type=str,
<<<<<<< HEAD
        default="static/gifs.json",
        help="Output JSON file path (default: static/gifs.json)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set logging level (default: info)",
=======
        default=None,
        help="Output JSON file path (default: <base_dir>/gifs.json)",
>>>>>>> a6a9c8ba01222f402c21a2af607107e63693c729
    )
    return parser.parse_args()


<<<<<<< HEAD
def get_gif_files(gif_dir: str) -> List[Path]:
    """
    Find all GIF files in the specified directory.

    Args:
        gif_dir: Directory to search for GIF files

    Returns:
        List of Path objects for each GIF file
    """
    gif_pattern = os.path.join(gif_dir, "*.gif")
    gifs = [Path(gif) for gif in glob.glob(gif_pattern)]
    logging.info(f"Found {len(gifs)} GIF files in {gif_dir}")
    return gifs


def get_png_frames(png_dir: str, gif_name: str) -> List[str]:
    """
    Find all PNG frames for a specified GIF.

    Args:
        png_dir: Base directory containing PNG frames
        gif_name: Name of the GIF file (without extension)

    Returns:
        List of PNG file paths sorted by frame index
    """
    png_pattern = os.path.join(png_dir, gif_name, "*.png")
    pngs = sorted(
        glob.glob(png_pattern), key=lambda x: int(Path(x).stem.split("_")[-1])
    )
    logging.debug(f"Found {len(pngs)} PNG frames for {gif_name}")
    return pngs


def map_gifs_to_pngs(gif_files: List[Path], png_dir: str) -> List[Dict[str, Any]]:
    """
    Create a mapping of GIF files to their PNG frames.

    Args:
        gif_files: List of GIF file paths
        png_dir: Base directory containing PNG frames

    Returns:
        List of dictionaries mapping each GIF to its PNG frames
    """
    data = []
    for gif in gif_files:
        gif_name = gif.stem
        png_frames = get_png_frames(png_dir, gif_name)

        if not png_frames:
            logging.warning(f"No PNG frames found for {gif_name}")
            continue

        data.append({"gif": str(gif), "png": png_frames})
=======
def find_gifs(base_dir: str, gif_pattern: str = "**/gif/*.gif") -> List[str]:
    """
    Find all GIF files in the specified directory using the pattern.

    Args:
        base_dir: Base directory to start the search
        gif_pattern: Glob pattern for finding GIF files

    Returns:
        List of paths to GIF files
    """
    gif_paths = glob.glob(os.path.join(base_dir, gif_pattern), recursive=True)
    logging.info(f"Found {len(gif_paths)} GIF files")
    return gif_paths


def find_png_frames(base_dir: str, gif_path: str) -> List[str]:
    """
    Find all PNG frames associated with a GIF file.

    Args:
        base_dir: Base directory where files are located
        gif_path: Path to the GIF file

    Returns:
        List of paths to PNG frames, sorted by frame number
    """
    # Extract the GIF name without extension
    gif_name = os.path.basename(gif_path).split(".")[0]

    # Find the PNG directory by looking for patterns
    # Try several common patterns for organizing PNG frames
    patterns = [
        os.path.join(
            os.path.dirname(gif_path), "png", gif_name, "*.png"
        ),  # static/gifs/gif/png/name/*.png
        os.path.join(
            os.path.dirname(gif_path), "..", "png", gif_name, "*.png"
        ),  # static/gifs/gif/../png/name/*.png
        os.path.join(
            base_dir, "gifs", "png", gif_name, "*.png"
        ),  # static/gifs/png/name/*.png
        os.path.join(
            os.path.dirname(os.path.dirname(gif_path)), "png", gif_name, "*.png"
        ),  # static/gif/../png/name/*.png
    ]

    png_paths = []
    for pattern in patterns:
        png_paths = glob.glob(pattern)
        if png_paths:
            break

    if not png_paths:
        logging.warning(f"No PNG frames found for GIF: {gif_path}")
        return []

    # Sort PNG frames by index number in filename
    try:
        sorted_paths = sorted(
            png_paths,
            key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0]),
        )
        return sorted_paths
    except (ValueError, IndexError) as e:
        # If sorting fails, try a different sorting strategy or return unsorted
        logging.warning(f"Error sorting PNG frames for {gif_path}: {e}")
        logging.warning("Returning frames in default order")
        return sorted(png_paths)


def build_gif_data(base_dir: str, gif_paths: List[str]) -> List[Dict]:
    """
    Build a data structure mapping GIFs to their PNG frames.

    Args:
        base_dir: Base directory where files are located
        gif_paths: List of paths to GIF files

    Returns:
        List of dictionaries mapping GIFs to PNG frames
    """
    data = []

    for gif_path in gif_paths:
        # Find PNG frames for this GIF
        png_frames = find_png_frames(base_dir, gif_path)

        # Use relative paths from the base directory
        if os.path.isabs(gif_path):
            rel_gif_path = os.path.relpath(gif_path, start=base_dir)
        else:
            rel_gif_path = gif_path

        rel_png_paths = [
            (
                os.path.relpath(png_path, start=base_dir)
                if os.path.isabs(png_path)
                else png_path
            )
            for png_path in png_frames
        ]

        # Add data for this GIF
        gif_data = {
            "gif": rel_gif_path,
            "name": os.path.basename(gif_path).split(".")[0],
            "png": rel_png_paths,
            "frame_count": len(rel_png_paths),
        }

        data.append(gif_data)
        logging.info(f"Processed {rel_gif_path} with {len(rel_png_paths)} frames")
>>>>>>> a6a9c8ba01222f402c21a2af607107e63693c729

    return data


<<<<<<< HEAD
def save_json(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save the mapping data to a JSON file.

    Args:
        data: Mapping data to save
        output_path: Path to save the JSON file
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        logging.info(f"Successfully saved mapping to {output_path}")
    except Exception as e:
        logging.error(f"Failed to save JSON file: {e}")
        raise


def main() -> None:
    """Main function to execute the script."""
    args = parse_arguments()
    setup_logging(args.log_level)

    try:
        gif_files = get_gif_files(args.gif_dir)
        data = map_gifs_to_pngs(gif_files, args.png_dir)
        save_json(data, args.output)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise
=======
def save_json(data: List[Dict], output_path: str) -> None:
    """
    Save the data to a JSON file.

    Args:
        data: Data to save
        output_path: Path to the output JSON file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logging.info(f"JSON data saved to {output_path}")


def main() -> None:
    """Main function to run the GIF scanner."""
    # Setup
    setup_logging()
    args = parse_arguments()

    base_dir = args.base_dir
    output_path = args.output or os.path.join(base_dir, "gifs.json")

    logging.info(f"Scanning directory: {base_dir}")
    logging.info(f"Output will be saved to: {output_path}")

    # Find GIFs
    gif_paths = find_gifs(base_dir)

    if not gif_paths:
        logging.warning("No GIF files found. Check your base directory.")
        return

    # Build data structure
    data = build_gif_data(base_dir, gif_paths)

    # Save to JSON
    save_json(data, output_path)

    logging.info(f"Successfully processed {len(data)} GIFs")
>>>>>>> a6a9c8ba01222f402c21a2af607107e63693c729


if __name__ == "__main__":
    main()
