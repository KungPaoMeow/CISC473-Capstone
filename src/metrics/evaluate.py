import argparse
from pathlib import Path

import cv2

from .metrics import psnr, rgb_to_grayscale

"""
command line program to compute PSNR from an image and its degraded version.
Images are assumed to be RGB. A grayscale version is computed using Rec. 601 luma weights.

This can be run from the root directory using `python -m src.metrics.evaluate "clean_filename"`.
By default, the degraded image filename will be assumed to be the clean image filename with "_out" appended to the end.
"""

parser = argparse.ArgumentParser(description="Parses input images for measurement")

# filename of the clean image (RGB)
parser.add_argument("clean_filename", help="filename of clean (unprocessed) image")

# suffix to the clean image filename that corresponds to the degraded image
# e.g. Real-ESRGAN appends "_out" to the output's filename by default. The degraded suffix will match this behavior.
# the degraded image will be found using the degraded suffix if the '-d' or '--degraded-image' argument is not provided
parser.add_argument(
    "-e",
    "--degraded-suffix",
    default="_out",
    help="Appended onto the end of the clean image filename to find the degraded image.",
)
parser.add_argument(
    "-d",
    "--degraded-image",
    help="Degraded image filename. If provided, will be used over the degraded-extension.",
)

args = parser.parse_args()

# find the degraded image
if args.degraded_image:
    degraded_image_filename = args.degraded_image
else:
    clean_filepath = Path(args.clean_filename)
    suffix = args.degraded_suffix

    degraded_image_filename = (
        clean_filepath.parent / f"{clean_filepath.stem}{suffix}{clean_filepath.suffix}"
    )

original = rgb_to_grayscale(cv2.imread(args.clean_filename))
compressed = rgb_to_grayscale(cv2.imread(degraded_image_filename))

print(f"psnr is {psnr(original, compressed)} dB.")
