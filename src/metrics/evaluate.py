import argparse
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import cv2
import lpips

from .metrics import evaluate_lpips, evaluate_psnr
from .utils import rgb_to_grayscale

"""
Command line program to evaluate PSNR and LPIPS from an image and its degraded version.
Images are assumed to be RGB. A grayscale version is computed using Rec. 601 luma weights.

This can be run from the root directory using `python -m src.metrics.evaluate "clean_filename"`.
By default, the degraded image filename will be assumed to be the clean image filename with "_out" appended to the end.
"""


@contextmanager
def lpips_gag():
    """
    lpips silencer
    """
    with open(os.devnull, "w") as devnull:
        warnings.simplefilter("ignore")
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def parse_arguments():
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
    parser.add_argument(
        "-s",
        "--spatial",
        action="store_true",
        help="Whether to return a map of perceptual distance (default False)",
    )
    parser.add_argument(
        "-g",
        "--use-gpu",
        action="store_true",
        help="Whether to use GPU to compute LPIPS distance",
    )

    args = parser.parse_args()
    return args


def main():
    # find the degraded image
    args = parse_arguments()
    if args.degraded_image:
        degraded_image_filename = args.degraded_image
    else:
        clean_filepath = Path(args.clean_filename)
        suffix = args.degraded_suffix

        degraded_image_filename = str(
            clean_filepath.parent
            / f"{clean_filepath.stem}{suffix}{clean_filepath.suffix}"
        )

    # get images
    clean_image = cv2.imread(args.clean_filename)
    degraded_image = cv2.imread(degraded_image_filename)

    # compute PSNR
    clean_grayscale = rgb_to_grayscale(clean_image)
    degraded_grayscale = rgb_to_grayscale(degraded_image)

    psnr = evaluate_psnr(clean_grayscale, degraded_grayscale)

    # compute LPIPS
    clean_lpips = lpips.im2tensor(lpips.load_image(args.clean_filename))
    degraded_lpips = lpips.im2tensor(lpips.load_image(degraded_image_filename))

    if args.use_gpu:
        clean_lpips.cuda()
        degraded_lpips.cuda()

    with lpips_gag():
        lpips_distance = evaluate_lpips(clean_lpips, degraded_lpips, args.spatial)

    # output PSNR and LPIPS
    print(f"PSNR: {round(psnr, 3)} dB")
    if not args.spatial:
        print(f"LPIPS distance: {round(lpips_distance.item(), 3)}")
    else:
        print(f"LPIPS distance: {round(lpips_distance.mean(), 3)}")


if __name__ == "__main__":
    main()
