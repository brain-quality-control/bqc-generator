#!/usr/bin/env python3
"""
Script to generate transparent overlay visualizations and GIFs from neuroimaging data.
"""

import argparse
import glob
import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Union, Callable, Any
from functools import wraps
from contextlib import contextmanager

import imageio.v3 as imageio
import joblib
import nibabel as nib
import numpy as np
import plotly.express as px
import colorlog

# Remove dependency on nilearn.plotting.displays
from tqdm import tqdm

_visu_generator_test = False  # Flag for test mode
_visu_generator_debug = False

# Define paths relative to script location for better portability
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = SCRIPT_DIR.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "pages" / "static" / "gifs"
DEFAULT_LUT = CONFIG_DIR / "FreeSurferColorLUT.txt"
DEFAULT_JSON = ROOT_DIR / "json" / "json_data.json"
DEFAULT_SEGMENTATION = "aparc.DKTatlas+aseg.mgz"

# Type aliases for better code documentation
ColorMap = Dict[int, List[int]]
Coordinates = Dict[str, List[int]]

# Set up logging
logger = logging.getLogger("neuroimaging_visualizer")


def setup_logging(debug_mode: bool) -> None:
    """
    Set up logging configuration based on debug mode.

    Args:
        debug_mode: Whether to enable debug logging
    """
    # Set up root logger to WARNING to prevent debug logs from dependencies
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure our application logger
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger.setLevel(log_level)

    # Create console handler with formatting
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    colorformatter = colorlog.ColoredFormatter(
        "%(name)s| %(log_color)s%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler.setFormatter(colorformatter)

    # Add handler to our logger
    logger.addHandler(console_handler)

    # Remove propagation to root logger to avoid duplicate logs
    logger.propagate = False

    logger.debug("Debug mode enabled")


@contextmanager
def timer(operation_name: str, debug_only: bool = True) -> None:
    """
    Context manager for timing operations.

    Args:
        operation_name: Name of the operation being timed
        debug_only: Whether to log only in debug mode
    """
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        elapsed = end_time - start_time

        if not debug_only or logger.level == logging.DEBUG:
            logger.info(
                f"Operation '{operation_name}' completed in {elapsed:.4f} seconds"
            )


def timed_function(func: Callable) -> Callable:
    """
    Decorator for timing function execution.

    Args:
        func: Function to time

    Returns:
        Wrapped function with timing
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        operation_name = func.__name__
        with timer(operation_name):
            return func(*args, **kwargs)

    return wrapper


def natural_sort_key(s: str, regexp: str = r"(\d+)") -> List[Union[int, str]]:
    """
    Generate a key for natural sorting of strings containing numbers.

    Args:
        s: String to generate a sort key for
        regexp: Regular expression pattern to extract numbers

    Returns:
        List of integers and strings for sorting
    """
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(regexp, s)
    ]


@timed_function
def get_colormap(filename: str) -> ColorMap:
    """
    Parse a FreeSurfer-style color lookup table.

    Args:
        filename: Path to the color lookup table file

    Returns:
        Dictionary mapping label values to RGB colors
    """
    logger.debug(f"Loading color map from {filename}")
    colors = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:  # Ensure we have enough elements
                label_id = int(parts[0])
                r, g, b = int(parts[2]), int(parts[3]), int(parts[4])
                colors[label_id] = [r, g, b]

    logger.debug(f"Loaded {len(colors)} color definitions")
    return colors


@timed_function
def get_slice(
    img: np.ndarray,
    axis: str,
    coord: int,
    colors: ColorMap,
    is_segmentation: bool = False,
) -> np.ndarray:
    """
    Extract a 2D slice from a 3D volume along the specified axis.

    Args:
        img: 3D volume data
        axis: Axis to slice along ('x', 'y', or 'z')
        coord: Coordinate to extract the slice at
        colors: Color lookup table for segmentation data
        is_segmentation: Whether the image is a segmentation mask

    Returns:
        RGB slice as a numpy array
    """
    coord = int(coord)
    logger.debug(f"Extracting slice along {axis}-axis at coordinate {coord}")

    # Extract the appropriate slice
    if axis == "x":
        data = img[coord, :, :]
    elif axis == "y":
        data = img[:, coord, :].T
    else:  # axis == "z"
        data = img[:, :, coord].T

    if is_segmentation:
        # Apply color mapping for segmentation
        shape = (data.shape[0], data.shape[1], 3)
        return np.asanyarray(
            [colors.get(int(val), [0, 0, 0]) for val in data.ravel()], dtype=np.uint8
        ).reshape(shape)
    else:
        # Convert to 3-channel grayscale
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8) * 255
        return np.stack([data_norm] * 3, axis=-1).astype(np.uint8)


@timed_function
def transparent_overlay(
    orig: np.ndarray,
    segmentation: np.ndarray,
    coord: Coordinates,
    colorscale: ColorMap,
    alpha: float = 0.3,
) -> List[np.ndarray]:
    """
    Create transparent overlays of segmentation masks on original images.

    Args:
        orig: Original image volume
        segmentation: Segmentation mask volume
        coord: Dictionary of slice coordinates for each axis
        colorscale: Color mapping for segmentation labels
        alpha: Transparency level (0-1)

    Returns:
        List of combined image slices with transparent overlays
    """
    logger.debug(f"Creating transparent overlays with alpha={alpha}")
    _slices = []

    for axis in coord:
        for c in coord[axis]:
            # Get slices
            img_slice = get_slice(orig, axis, c, colorscale)
            seg_slice = get_slice(
                segmentation, axis, c, colorscale, is_segmentation=True
            )

            # Create a binary mask for the segmentation
            mask = seg_slice.sum(axis=-1) > 0

            # Create the blended image
            combined = img_slice.copy()

            # Apply the segmentation with transparency
            combined[mask] = (
                (1 - alpha) * img_slice[mask] + alpha * seg_slice[mask]
            ).astype(np.uint8)

            _slices.append(combined)

    logger.debug(f"Created {len(_slices)} overlay slices")
    return _slices


@timed_function
def get_coords(image_path: str) -> Coordinates:
    """
    Find optimal slice coordinates for visualization.

    Args:
        image_path: Path to the 3D volume

    Returns:
        Dictionary of coordinates for each axis
    """
    logger.debug(f"Finding optimal slice coordinates for {image_path}")

    # Load the image data
    with timer("load_image_data"):
        img_data = nib.load(image_path).get_fdata().astype(np.uint16)

    # Calculate center slices and some slices around them
    shape = img_data.shape

    # Find indices where there's actual data (non-zero values)
    with timer("find_non_zero_indices"):
        x_indices, y_indices, z_indices = np.where(img_data > 0)

    # If no non-zero values are found, use center slices
    if len(x_indices) == 0:
        center_x = shape[0] // 2
        center_y = shape[1] // 2
        center_z = shape[2] // 2
        logger.debug("No non-zero values found, using center slices")
    else:
        # Use the middle of the data-containing region
        center_x = (np.min(x_indices) + np.max(x_indices)) // 2
        center_y = (np.min(y_indices) + np.max(y_indices)) // 2
        center_z = (np.min(z_indices) + np.max(z_indices)) // 2
        logger.debug(
            f"Found data-containing region centers: x={center_x}, y={center_y}, z={center_z}"
        )

    resolution_level = 8

    # Create some slices around the center (one before, center, one after)
    x_cuts = [
        max(0, center_x - shape[0] // resolution_level),
        center_x,
        min(shape[0] - 1, center_x + shape[0] // resolution_level),
    ]
    y_cuts = [
        max(0, center_y - shape[1] // resolution_level),
        center_y,
        min(shape[1] - 1, center_y + shape[1] // resolution_level),
    ]
    z_cuts = [
        max(0, center_z - shape[2] // resolution_level),
        center_z,
        min(shape[2] - 1, center_z + shape[2] // resolution_level),
    ]

    logger.debug(f"Selected coordinates - x: {x_cuts}, y: {y_cuts}, z: {z_cuts}")

    # Return dictionary with coordinates for each axis
    return {"x": x_cuts, "y": y_cuts, "z": z_cuts}


def get_repetition(filepath: str) -> int:
    """
    Extract repetition number from a filepath.

    Args:
        filepath: Path containing repetition information

    Returns:
        Repetition number (defaults to 0 if not found)
    """
    if match := re.search(r"rep(\d+)", filepath):
        return int(match.group(1))
    return 0


@timed_function
def generate_frame(
    image_path: str,
    subject: str,
    index: int,
    coord: Coordinates,
    colors: ColorMap,
    output_dir: str,
    transparency: float,
    segmentation_type: str,
) -> None:
    """
    Generate a single frame visualization.

    Args:
        image_path: Path to segmentation image
        subject: Subject identifier
        index: Repetition index
        coord: Slice coordinates
        colors: Color lookup table
        output_dir: Output directory
        transparency: Transparency level for overlay
    """
    logger.debug(f"Generating frame for {subject}, repetition {index}")

    # Load segmentation and original images
    segmentation = nib.load(image_path).get_fdata().astype(np.uint16)
    orig_path = image_path.replace(segmentation_type, "orig.mgz")

    if not os.path.exists(orig_path):
        logger.error(f"Original image not found: {orig_path}")
        return

    logger.debug(f"Loading original image from {orig_path}")
    with timer("load_original_image"):
        orig = nib.load(orig_path).get_fdata().astype(np.uint16)

    # Create transparent overlays
    slices = transparent_overlay(orig, segmentation, coord, colors, alpha=transparency)
    slices_array = np.array(slices)

    # Create visualization using plotly
    logger.debug("Creating plotly visualization")
    with timer("plotly_visualization"):
        fig = px.imshow(
            slices_array,
            facet_col=0,
            facet_col_wrap=3,
            facet_col_spacing=0,
            facet_row_spacing=0,
        )

        # Configure layout
        fig.layout.annotations = ()
        fig.update_layout(
            autosize=False,
            width=1000,
            height=1000,
            plot_bgcolor="black",
            paper_bgcolor="black",
            title=f"{subject} - Repetition {index}",
        )

        # Remove axes
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    # Save image
    output_path = os.path.join(
        output_dir, "gif", "png", subject, f"aparc.DKTatlas+aseg_{index}.png"
    )
    logger.debug(f"Saving frame to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with timer("save_image"):
        fig.write_image(output_path)


@timed_function
def generate_frames(
    input_dir: str,
    output_dir: str,
    subject: str,
    colormap_file: str,
    n_jobs: int,
    transparency: float,
    segmentation_type: str,
) -> bool:
    """
    Generate frames for all repetitions of a subject.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        subject: Subject identifier
        colormap_file: Path to color lookup table
        n_jobs: Number of parallel jobs
        transparency: Transparency level

    Returns:
        True if frames were generated, False otherwise
    """
    logger.info(f"Generating frames for subject {subject}")

    # Find all segmentation files for the subject
    pattern = os.path.join(input_dir, "**", subject, "mri/aparc.DKTatlas+aseg.mgz")
    segmentations = glob.glob(pattern)

    if not segmentations:
        logger.warning(f"Images for {subject} not found.")
        return False

    if _visu_generator_debug or _visu_generator_test:
        # Limit to 2 segmentations for testing/debugging
        segmentations = segmentations[:1]
        logger.debug(f"Debug/Test mode: limiting to {len(segmentations)} segmentations")

    logger.debug(f"Found {len(segmentations)} segmentation files for {subject}")

    # Create output directory
    os.makedirs(os.path.join(output_dir, "gif", "png", subject), exist_ok=True)

    # Get slice coordinates and color lookup table
    coord = get_coords(segmentations[0])
    colors = get_colormap(colormap_file)

    # Generate frames for each repetition
    segmentations = [(get_repetition(s), s) for s in segmentations]
    logger.debug(f"Processing {len(segmentations)} repetitions")

    for i, segmentation in segmentations:
        with timer(f"generate_frame_{subject}_rep{i}", debug_only=False):
            generate_frame(
                segmentation,
                subject,
                i,
                coord,
                colors,
                output_dir,
                transparency,
                segmentation_type,
            )

    return True


@timed_function
def make_gif(
    directory: str, input_pattern: str, output_path: str, duration: float
) -> None:
    """
    Create a GIF from a sequence of images.

    Args:
        directory: Directory containing input images
        input_pattern: Glob pattern for input images
        output_path: Path for output GIF
        duration: Duration between frames in seconds
    """
    logger.info(f"Creating GIF from {directory}/{input_pattern}")

    # Find and sort input files
    pattern = os.path.join(directory, input_pattern)
    filenames = sorted(
        glob.glob(pattern), key=lambda s: natural_sort_key(s, regexp=r"_(\d+).png")
    )

    if not filenames:
        logger.warning(f"No images found matching pattern: {pattern}")
        return

    logger.debug(f"Found {len(filenames)} images for GIF")

    # Ensure output path has .gif extension
    output_gif = (
        f"{output_path}.gif" if not output_path.endswith(".gif") else output_path
    )

    # Create GIF
    logger.debug(f"Creating GIF at {output_gif}")
    with timer("load_images"):
        images = [imageio.imread(f) for f in filenames]

    with timer("write_gif"):
        imageio.imwrite(output_gif, images, duration=duration, loop=0)

    logger.info(f"GIF created successfully: {output_gif} ({len(images)} frames)")


@timed_function
def generate_gif(
    subject: str,
    input_dir: str,
    output_dir: str,
    colormap_file: str,
    n_jobs: int,
    transparency: float,
    segmentation_type: str,
    duration: float = 0.1,
) -> None:
    """
    Generate a GIF for a subject.

    Args:
        subject: Subject identifier
        input_dir: Input directory
        output_dir: Output directory
        colormap_file: Path to color lookup table
        n_jobs: Number of parallel jobs
        transparency: Transparency level
        duration: Duration between frames in seconds
    """
    logger.info(f"Starting GIF generation for subject {subject}")

    # First generate all frames
    if not generate_frames(
        input_dir,
        output_dir,
        subject,
        colormap_file,
        n_jobs,
        transparency,
        segmentation_type,
    ):
        logger.warning(f"Failed to generate frames for {subject}")
        return

    # Then create GIF from frames
    input_path = os.path.join(output_dir, "gif", "png", subject)
    output_path = os.path.join(output_dir, "gif", subject)
    make_gif(input_path, "*.png", output_path, duration)

    logger.info(f"Completed GIF generation for subject {subject}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate transparent overlay visualizations and GIFs from neuroimaging data."
    )
    parser.add_argument(
        "--input", type=str, help="Input JSON file with subject information"
    )
    parser.add_argument(
        "--input-dir",
        default=str(DATA_DIR),
        type=str,
        help=f"Input directory containing imaging data (default: {DATA_DIR})",
    )
    parser.add_argument(
        "--segmentation",
        default=str(DEFAULT_SEGMENTATION),
        type=str,
        help="Segmentation file name (default: aparc.DKTatlas+aseg.mgz)",
    )
    parser.add_argument(
        "--colormap",
        default=str(DEFAULT_LUT),
        type=str,
        help=f"FreeSurfer color lookup table (default: {DEFAULT_LUT})",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        type=str,
        help=f"Output directory for visualizations (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--transparency",
        default=0.3,
        type=float,
        help="Transparency level for overlays (default: 0.3)",
    )
    parser.add_argument(
        "--duration",
        default=0.1,
        type=float,
        help="GIF frame duration in seconds (default: 0.1)",
    )
    parser.add_argument(
        "--n-jobs",
        default=-1,
        type=int,
        help="Number of parallel jobs (-1 for all cores) (default: -1)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode with limited subjects"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode with detailed logging"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (if not specified, logs to console only)",
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.

    Args:
        args: Parsed command line arguments

    Raises:
        FileNotFoundError: If specified files don't exist
        NotADirectoryError: If specified directories don't exist
    """
    logger.debug("Validating command line arguments")
    if args.input and not os.path.isfile(args.input):
        raise FileNotFoundError(f"File {args.input} not found.")
    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"Directory {args.input_dir} not found.")
    if not os.path.isfile(args.colormap):
        raise FileNotFoundError(f"File {args.colormap} not found.")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    logger.debug("Arguments validated successfully")


@timed_function
def get_subjects(args: argparse.Namespace) -> Set[str]:
    """
    Get the set of subjects to process.

    Args:
        args: Parsed command line arguments

    Returns:
        Set of subject identifiers
    """
    logger.info("Finding subjects to process")
    subjects = set()

    if args.input:
        # Get subjects from JSON file
        logger.debug(f"Loading subjects from JSON file: {args.input}")
        with open(args.input, "r") as f:
            dataset = json.load(f)
            subjects = set(dataset.values())
            logger.debug(f"Found {len(subjects)} subjects in JSON file")
    else:
        # Scan directory for subjects
        logger.debug(f"Scanning {args.input_dir} for subjects...")
        with timer("scan_directory", debug_only=False):
            orig_files = glob.glob(
                os.path.join(args.input_dir, "**", "orig.mgz"), recursive=True
            )

        logger.debug(f"Found {len(orig_files)} original image files")

        # Extract subject identifiers
        for path in tqdm(orig_files, desc="Finding subjects"):
            if match := re.search(r"sub-[^/]+", path):
                subjects.add(match.group(0))

    # Limit number of subjects in test mode
    if args.test:
        logger.info("Test mode: limiting to 2 subjects.")
        subjects_list = list(subjects)
        if len(subjects_list) > 2:
            subjects = set(subjects_list[:2])

    logger.info(f"Found {len(subjects)} subjects to process")
    if len(subjects) <= 5:
        logger.info(f"Subjects: {', '.join(sorted(subjects))}")

    return subjects


def main():
    """Main entry point."""
    start_time = time.time()

    # Parse arguments
    args = parse_args()

    # Set up logging
    setup_logging(args.debug)

    if args.debug:
        logger.debug("Debug mode enabled")
        global _visu_generator_debug
        _visu_generator_debug = True

    if args.test:
        logger.debug("Test mode enabled")
        global _visu_generator_test
        _visu_generator_test = True

    if args.log_file:
        # Add file handler if log file is specified
        file_handler = logging.FileHandler(args.log_file)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
        logger.addHandler(file_handler)

    logger.info("Starting neuroimaging visualization script")
    logger.debug(f"Arguments: {vars(args)}")

    try:
        # Validate arguments
        validate_args(args)

        # Find subjects to process
        subjects = get_subjects(args)

        if not subjects:
            logger.warning("No subjects found to process. Exiting.")
            return

        # Process subjects
        with timer("process_all_subjects", debug_only=False):
            # Process subjects in parallel
            with joblib.Parallel(n_jobs=args.n_jobs, verbose=10) as parallel:
                parallel(
                    joblib.delayed(generate_gif)(
                        subject,
                        args.input_dir,
                        args.output_dir,
                        args.colormap,
                        args.n_jobs,
                        args.transparency,
                        args.segmentation,
                        args.duration,
                    )
                    for subject in tqdm(subjects, desc="Processing subjects")
                )

        # Calculate total execution time
        end_time = time.time()
        total_time = end_time - start_time

        logger.info(
            f"Processing complete. Total execution time: {total_time:.2f} seconds"
        )

    except Exception as e:
        logger.exception(f"Error during execution: {e}")
        raise


if __name__ == "__main__":
    main()
