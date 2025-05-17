#!/usr/bin/env python3
"""
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
    )


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
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
    )
    parser.add_argument(
        "--output",
        type=str,
        default="static/gifs.json",
        help="Output JSON file path (default: static/gifs.json)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set logging level (default: info)",
    )
    return parser.parse_args()


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

    return data


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


if __name__ == "__main__":
    main()
