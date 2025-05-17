#!/usr/bin/env python3
"""
Script to generate transparent overlay visualizations and GIFs from neuroimaging data.
Optimized with Dask for distributed task parallelism.
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
import nibabel as nib
import numpy as np
import plotly.express as px
import colorlog
from scandir_rs import Walk
from tqdm import tqdm

# Dask imports for distributed computing
import dask
from dask.distributed import Client, LocalCluster, progress
import pandas as pd

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
    """
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_level = logging.DEBUG if debug_mode else logging.INFO
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    colorformatter = colorlog.ColoredFormatter(
        "%(name)s| %(log_color)s%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(colorformatter)
    logger.addHandler(console_handler)
    logger.propagate = False

    logger.debug("Debug mode enabled")


@contextmanager
def timer(operation_name: str, debug_only: bool = True) -> None:
    """
    Context manager for timing operations.
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        if not debug_only or logger.level == logging.DEBUG:
            logger.info(
                f"Operation '{operation_name}' completed in {elapsed:.4f} seconds"
            )


def timed_function(func: Callable) -> Callable:
    """
    Decorator for timing function execution.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        with timer(func.__name__):
            return func(*args, **kwargs)

    return wrapper


def natural_sort_key(s: str, regexp: str = r"(\d+)") -> List[Union[int, str]]:
    """
    Generate a key for natural sorting of strings containing numbers.
    """
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(regexp, s)
    ]


def fast_glob(dir: str, pattern: str, desc: str) -> List[str]:
    """
    Fast globbing function to find files matching a pattern.
    """

    logger.debug(f"Fast globbing for pattern: {pattern}")
    scanned = []
    for root, _, files in tqdm(
        Walk(dir, file_include=[pattern]), desc=desc, unit="file"
    ):
        scanned.extend([os.path.join(dir, root, file) for file in files])

    return scanned


@timed_function
def get_colormap(filename: str) -> ColorMap:
    """
    Parse a FreeSurfer-style color lookup table.
    """
    logger.debug(f"Loading color map from {filename}")
    colors: ColorMap = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                label_id = int(parts[0])
                r, g, b = int(parts[2]), int(parts[3]), int(parts[4])
                colors[label_id] = [r, g, b]
    logger.debug(f"Loaded {len(colors)} color definitions")
    return colors


def get_slice(
    img: np.ndarray,
    axis: str,
    coord: int,
    colors: ColorMap,
    is_segmentation: bool = False,
) -> np.ndarray:
    coord = int(coord)
    if axis == "x":
        data = img[coord, :, :]
    elif axis == "y":
        data = img[:, coord, :].T
    else:  # axis == "z"
        data = img[:, :, coord].T

    if is_segmentation:
        shape = (data.shape[0], data.shape[1], 3)
        return np.asanyarray(
            [colors.get(int(val), [0, 0, 0]) for val in data.ravel()],
            dtype=np.uint8,
        ).reshape(shape)
    else:
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8) * 255
        return np.stack([data_norm] * 3, axis=-1).astype(np.uint8)


def transparent_overlay_slice(
    orig_slice: np.ndarray,
    seg_slice: np.ndarray,
    alpha: float = 0.3,
) -> np.ndarray:
    mask = seg_slice.sum(axis=-1) > 0
    combined = orig_slice.copy()
    combined[mask] = ((1 - alpha) * orig_slice[mask] + alpha * seg_slice[mask]).astype(
        np.uint8
    )
    return combined


def transparent_overlay(
    orig: np.ndarray,
    segmentation: np.ndarray,
    coord: Dict[str, List[int]],
    colorscale: ColorMap,
    alpha: float = 0.3,
) -> List[np.ndarray]:
    slice_tasks: List[np.ndarray] = []
    for axis in coord:
        for c in coord[axis]:
            img_slice = get_slice(orig, axis, c, colorscale)
            seg_slice = get_slice(
                segmentation, axis, c, colorscale, is_segmentation=True
            )
            slice_tasks.append(transparent_overlay_slice(img_slice, seg_slice, alpha))
    return slice_tasks


def get_coords(image_path: str) -> Coordinates:
    img_data = nib.load(image_path).get_fdata().astype(np.uint16)
    shape = img_data.shape
    x_idx, y_idx, z_idx = np.where(img_data > 0)
    if len(x_idx) == 0:
        cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    else:
        cx = (x_idx.min() + x_idx.max()) // 2
        cy = (y_idx.min() + y_idx.max()) // 2
        cz = (z_idx.min() + z_idx.max()) // 2

    lvl = 8
    x_cuts = [max(0, cx - shape[0] // lvl), cx, min(shape[0] - 1, cx + shape[0] // lvl)]
    y_cuts = [max(0, cy - shape[1] // lvl), cy, min(shape[1] - 1, cy + shape[1] // lvl)]
    z_cuts = [max(0, cz - shape[2] // lvl), cz, min(shape[2] - 1, cz + shape[2] // lvl)]

    return {"x": x_cuts, "y": y_cuts, "z": z_cuts}


def get_repetition(filepath: str) -> int:
    if match := re.search(r"rep(\d+)", filepath):
        return int(match.group(1))
    return 0


def generate_frame(
    image_path: str,
    subject: str,
    index: int,
    coord: Dict[str, List[int]],
    colors: ColorMap,
    output_dir: str,
    transparency: float,
    segmentation_type: str,
) -> Optional[str]:
    segmentation = nib.load(image_path).get_fdata().astype(np.uint16)
    orig_path = image_path.replace(segmentation_type, "orig.mgz")
    if not os.path.exists(orig_path):
        logger.warning(f"Original image not found for {subject} at {orig_path}")
        return None
    orig = nib.load(orig_path).get_fdata().astype(np.uint16)

    slices_list = transparent_overlay(
        orig, segmentation, coord, colors, alpha=transparency
    )
    slices_array = np.array(slices_list)

    fig = px.imshow(
        slices_array,
        facet_col=0,
        facet_col_wrap=3,
        facet_col_spacing=0,
        facet_row_spacing=0,
    )
    fig.layout.annotations = ()
    fig.update_layout(
        autosize=False,
        width=1000,
        height=1000,
        plot_bgcolor="black",
        paper_bgcolor="black",
        title=f"{subject} - Repetition {index}",
    )
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    output_path = os.path.join(
        output_dir, "gif", "png", subject, f"{segmentation_type}_{index}.png"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_image(output_path)
    return output_path


def find_segmentations(
    input_dir: str, subject: str, segmentation_type: str
) -> List[Tuple[int, str]]:
    pattern = os.path.join(input_dir, "**", subject, "mri", segmentation_type)
    segs = glob.glob(pattern, recursive=True)
    return [(get_repetition(s), s) for s in segs]


def generate_frames(
    input_dir: str,
    output_dir: str,
    subject: str,
    colormap_file: str,
    transparency: float,
    segmentation_type: str,
) -> List[str]:
    segmentations = find_segmentations(input_dir, subject, segmentation_type)
    if not segmentations:
        logger.warning(f"No segmentations found for {subject}. Skipping.")
        return []

    os.makedirs(os.path.join(output_dir, "gif", "png", subject), exist_ok=True)
    colors = get_colormap(colormap_file)

    first_seg = segmentations[0][1]
    coord = get_coords(first_seg)

    frames: List[str] = []
    for i, seg_path in segmentations:
        frame = generate_frame(
            seg_path,
            subject,
            i,
            coord,
            colors,
            output_dir,
            transparency,
            segmentation_type,
        )
        if frame:
            frames.append(frame)
    return frames


def make_gif(
    directory: str, input_pattern: str, output_path: str, duration: float
) -> Optional[str]:
    pattern = os.path.join(directory, input_pattern)
    filenames = sorted(
        glob.glob(pattern), key=lambda s: natural_sort_key(s, regexp=r"_(\d+).png")
    )
    if not filenames:
        logger.warning(f"No files found for pattern: {pattern}")
        return None

    gif_out = f"{output_path}.gif" if not output_path.endswith(".gif") else output_path
    images = [imageio.imread(f) for f in filenames]
    imageio.imwrite(gif_out, images, duration=duration, loop=0)
    return gif_out


def generate_gif(
    subject: str,
    input_dir: str,
    output_dir: str,
    colormap_file: str,
    transparency: float,
    segmentation_type: str,
    duration: float = 0.1,
) -> Optional[str]:
    frame_paths = generate_frames(
        input_dir, output_dir, subject, colormap_file, transparency, segmentation_type
    )
    if not frame_paths:
        logger.warning(f"No frames for {subject}, skipping GIF.")
        return None

    input_path = os.path.join(output_dir, "gif", "png", subject)
    output_path = os.path.join(output_dir, "gif", subject)
    return make_gif(input_path, "*.png", output_path, duration)


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--scheduler-address",
        type=str,
        help="Address of existing Dask scheduler (if not specified, creates a local cluster)",
    )
    parser.add_argument(
        "--memory-limit",
        type=str,
        default="auto",
        help="Memory limit per worker (e.g., '4GB')",
    )
    parser.add_argument(
        "--threads-per-worker",
        type=int,
        default=1,
        help="Number of threads per worker (default: 1)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    logger.debug("Validating command line arguments")
    if args.input and not os.path.isfile(args.input):
        raise FileNotFoundError(f"File {args.input} not found.")
    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"Directory {args.input_dir} not found.")
    if not os.path.isfile(args.colormap):
        raise FileNotFoundError(f"File {args.colormap} not found.")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.debug("Arguments validated successfully")


def get_subjects(args: argparse.Namespace) -> Set[str]:
    logger.info("Finding subjects to process")
    subjects: Set[str] = set()
    if args.input:
        logger.debug(f"Loading subjects from JSON file: {args.input}")
        with open(args.input, "r") as f:
            dataset = json.load(f)
        subjects = set(dataset.values())
    else:
        logger.debug(f"Scanning {args.input_dir} for subjects...")
        orig_files = fast_glob(args.input_dir, "orig.mgz", desc="Scanning for subjects")
        for path in orig_files:
            if match := re.search(r"sub-[^/]+", path):
                subjects.add(match.group(0))

    if args.test:
        logger.info("Test mode: limiting to 2 subjects.")
        subjects = set(list(subjects)[:2])

    logger.info(f"Found {len(subjects)} subjects: {sorted(subjects)[:5]}")
    return subjects


def setup_dask_client(args: argparse.Namespace) -> Client:
    if args.scheduler_address:
        logger.info(f"Connecting to Dask scheduler at {args.scheduler_address}")
        client = Client(args.scheduler_address)
    else:
        n_workers = (
            os.cpu_count() if args.n_jobs <= 0 else min(args.n_jobs, os.cpu_count())
        )
        logger.info(f"Setting up local Dask cluster with {n_workers} workers")
        cluster = LocalCluster(
            n_workers=n_workers,
            threads_per_worker=args.threads_per_worker,
            memory_limit=args.memory_limit,
            processes=True,
            silence_logs=logging.WARNING,
        )
        client = Client(cluster)

    logger.info(f"Dask dashboard available at: {client.dashboard_link}")
    return client


def main():
    start_time = time.time()
    args = parse_args()
    setup_logging(args.debug)
    if args.debug:
        global _visu_generator_debug
        _visu_generator_debug = True
    if args.test:
        global _visu_generator_test
        _visu_generator_test = True
    if args.log_file:
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        fh.setLevel(logging.DEBUG if args.debug else logging.INFO)
        logger.addHandler(fh)

    logger.info("Starting Dask-optimized neuroimaging visualization script")
    try:
        validate_args(args)
        subjects = get_subjects(args)
        if not subjects:
            logger.warning("No subjects found to process. Exiting.")
            return

        client = setup_dask_client(args)
        with timer("process_subjects", debug_only=False):
            # Prepare argument tuples
            subject_args = [
                (
                    subject,
                    args.input_dir,
                    args.output_dir,
                    args.colormap,
                    args.transparency,
                    args.segmentation,
                    args.duration,
                )
                for subject in subjects
            ]

            # Submit one task per subject
            futures = [client.submit(generate_gif, *params) for params in subject_args]

            # Track progress and gather results
            progress(futures)
            results = client.gather(futures)

            successful = sum(1 for r in results if r)
            logger.info(
                f"Successfully generated {successful} out of {len(subjects)} GIFs"
            )

        total_time = time.time() - start_time
        logger.info(f"Processing complete in {total_time:.2f}s")
        client.close()

    except Exception:
        logger.exception("Error during execution")
        raise


if __name__ == "__main__":
    main()
