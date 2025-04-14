#!/usr/bin/env python3
"""
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
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Scan directories for GIFs and generate a JSON mapping file."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="static",
        help="Base directory to scan (default: static)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: <base_dir>/gifs.json)",
    )
    return parser.parse_args()


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

    return data


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


if __name__ == "__main__":
    main()
