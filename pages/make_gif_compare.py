import argparse
import glob
import json
import os
import re

import imageio
import joblib
import nibabel as nib
import numpy as np
import plotly.express as px
import tqdm
from nilearn.plotting.displays import MosaicSlicer
import numpy as np

rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_json = os.path.realpath(os.path.join(rootdir, "json/json_data.json"))
freesurfer_LUT = os.path.realpath(os.path.join(rootdir, "config/FreeSurferColorLUT.txt"))
output_dir_default = os.path.realpath(os.path.join(rootdir, "pages/static/gifs"))
input_dir = os.path.realpath(os.path.join(rootdir, "data"))


def natural_sort_key(s, regexp=r"(\d+)"):
    """
    Generate a key for natural sorting. It splits the input string into a list
    of strings and integers, which is suitable for correct numeric sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(regexp, s)]


def get_colormap(filename):
    colors = {}
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.rstrip().startswith("#") or line.strip() == "":
                continue
            (i, name, r, g, b, a) = line.strip().split()
            colors[int(i)] = [int(r), int(g), int(b)]
    return colors

def get_slice(img, axis, coord, colors=None, is_segmentation=False):
    coord = int(coord)

    # Extract the appropriate slice
    if axis == "x":
        data = img[coord, :, :]
    elif axis == "y":
        data = img[:, coord, :]
    else:
        data = img[:, :, coord]

    if is_segmentation:
        # Apply color mapping for segmentation
        shape = (data.shape[0], data.shape[1], 3)
        return np.asanyarray([colors[val] for val in data.ravel()], dtype=np.uint8).reshape(shape)
    else:
        # Keep original grayscale image but convert to 3-channel grayscale
        return np.stack([data] * 3, axis=-1).astype(np.uint8)


def generate_frame_plotly(image, subject, index, coord, colorscale, output_dir):
    segmentation = nib.load(image).get_fdata().astype(np.uint16)
    orig = nib.load(image.replace("aparc.DKTatlas+aseg.mgz", "orig.mgz")).get_fdata().astype(np.uint16)

    _slices = []
    for axis in coord:
        for c in coord[axis]:
            seg_slice = get_slice(segmentation, axis, c, colorscale, is_segmentation=True)
            img_slice = get_slice(orig, axis, c, colorscale)

            # Ensure img_slice is RGB
            if img_slice.ndim == 2:
                img_rgb = np.stack([img_slice] * 3, axis=-1)  # Convert grayscale to RGB
            else:
                img_rgb = img_slice

            # Create an alpha channel
            alpha_channel = (seg_slice.sum(axis=-1) > 0) * 255  # Fully opaque where segmentation exists
            img_rgba = np.concatenate([img_rgb, np.full(img_rgb.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)

            # Blend segmentation with original grayscale image
            combined = img_rgba.copy()
            mask = alpha_channel > 0
            combined[mask] = np.concatenate([seg_slice[mask], alpha_channel[mask, None]], axis=-1)

            _slices.append(combined)

    # Convert to NumPy array
    _slices = np.array(_slices)

    fig = px.imshow(
        np.stack(_slices),
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

    # Remove x and y ticks
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    output = os.path.join(output_dir, "gif", "png", subject, f"aparc.DKTatlas+aseg_{index}.png")
    fig.write_image(output)


def get_coords(image):
    img = nib.Nifti1Image(
        nib.load(image).get_fdata(),
        affine=np.eye(4),
        dtype=np.uint16,
    )
    coord = MosaicSlicer.find_cut_coords(img, cut_coords=3)
    return coord


def get_repetition(segmentation):
    # return int(re.search(r"rep(\d+)", segmentation).group(1))
    return 0


def generate_frames(input_dir, output_dir, subject, colormap_file, n_jobs):
    regexp = os.path.join(input_dir, subject, "mri/aparc.DKTatlas+aseg.mgz")
    segmentations = glob.glob(regexp)
    
    if len(segmentations) == 0:
        print(f"Images for {subject} not found.")
        return False
    os.makedirs(os.path.join(output_dir, "gif", "png", subject), exist_ok=True)
    coord = get_coords(segmentations[0])
    colors = get_colormap(colormap_file)

    segmentations = [(get_repetition(s), s) for s in segmentations]
    for i, segmentation in segmentations:
        generate_frame_plotly(segmentation, subject, i, coord, colors, output_dir)
    return True


def make_gif(directory, input, output, duration, n_jobs):
    regex = os.path.join(directory, input)
    filenames = sorted(glob.glob(regex), key=lambda s: natural_sort_key(s, regexp=r"_(\d+).png"))
    output_gif = f"{output}.gif" if not output.endswith(".gif") else output
    images = [imageio.v3.imread(f) for f in filenames]
    imageio.v3.imwrite(output_gif, images, duration=duration, loop=0)


def generate_gif(subject, input_dir, output_dir, colormap_file, n_jobs, duration=0.1):
    print(f"Generating gif for {subject}")
    if not generate_frames(input_dir, output_dir, subject, colormap_file, n_jobs):
        return
    input_dir = os.path.join(output_dir, "gif", "png", subject)
    output = os.path.join(output_dir, "gif", subject)
    make_gif(input_dir, "*.png", output, duration, n_jobs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=input_json, type=str, help="Input json file")
    parser.add_argument("--input-dir", default=input_dir, type=str, help="Input directory")
    parser.add_argument("--colormap", default=freesurfer_LUT, type=str, help="Colormap file")
    parser.add_argument("--output-dir", default=output_dir_default, type=str, help="Output directory")
    parser.add_argument("--duration", default=0.1, type=float, help="GIF duration")
    parser.add_argument("--n-jobs", default=40, type=int, help="Number of jobs to run in parallel")

    return parser.parse_args()


def check_args(args):
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"File {args.input} not found.")
    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"Directory {args.input_dir} not found.")
    if not os.path.isfile(args.colormap):
        raise FileNotFoundError(f"File {args.colormap} not found.")
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)


def main():
    args = parse_args()

    check_args(args)

    with open(args.input, "r") as f:
        dataset = json.load(f)

    for subject in dataset.values():
        generate_gif(
            subject,
            args.input_dir,
            args.output_dir,
            args.colormap,
            args.n_jobs,
            args.duration,
        )


if __name__ == "__main__":
    main()
