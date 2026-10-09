import argparse
import os
import statistics
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import cv2
import lpips
from loguru import logger

from .metrics import evaluate_lpips, evaluate_psnr
from .utils import rgb_to_grayscale

"""
Command line program to evaluate PSNR and LPIPS on datasets of images, one clean copy and one degraded copy per image.
Images are assumed to be RGB. A grayscale version is computed using Rec. 601 luma weights for PSNR.

This can be run from the root directory using `python -m src.metrics.evaluate "input_directory"`.
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
    parser = argparse.ArgumentParser(
        description="Parses input images for PSNR and LPIPS evaluation"
    )

    # filename of the clean image (RGB)
    parser.add_argument(
        "input_dir", help="directory containing datasets for evaluation"
    )

    parser.add_argument(
        "-c",
        "--clean-dir",
        default="clean",
        help="name of subdirectory containing clean images. Defaults to 'clean'.",
    )

    parser.add_argument(
        "-d",
        "--degraded-dir",
        default="upscaled",
        help="name of subdirectory containing degraded/upscaled images to compare against the clean images. Defaults to 'upscaled'.",
    )

    # suffix to the clean image filename that corresponds to the degraded image
    # e.g. Real-ESRGAN appends "_out" to the output's filename by default. The degraded suffix will match this behavior.
    # the degraded image will be found using the degraded suffix if the '-d' or '--degraded-image' argument is not provided
    parser.add_argument(
        "--suffix",
        default="_out",
        help="Appended onto the end of the clean image filename to find the degraded image.",
    )
    parser.add_argument(
        "-s",
        "--spatial",
        action="store_true",
        help="Whether to return a map of perceptual distance for LPIPS (default False)",
    )
    parser.add_argument(
        "-g",
        "--use-gpu",
        action="store_true",
        help="Whether to use GPU to compute LPIPS distance",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Adds additional output."
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # iterate through datasets (folders in the input folder)
    rootdir = Path(args.input_dir)
    dataset_list = [d for d in rootdir.iterdir() if d.is_dir()]

    psnr_all = []
    lpips_all = []
    psnr_max_all = 0
    psnr_min_all = float("inf")
    lpips_max_all = 0
    lpips_min_all = float("inf")

    for dataset in dataset_list:
        clean_dir = Path(f"{dataset}/{args.clean_dir}")
        degraded_dir = Path(f"{dataset}/{args.degraded_dir}")

        psnr_dataset = []
        lpips_dataset = []
        psnr_max_dataset = 0
        psnr_min_dataset = float("inf")
        lpips_max_dataset = 0
        lpips_min_dataset = float("inf")

        # iterate through images in the "clean" directory in the dataset, match to degraded image, calculate PSNR and LPIPS
        clean_filenames = [f for f in clean_dir.iterdir() if f.is_file()]
        for clean_filename in clean_filenames:
            degraded_filename = (
                f"{clean_filename.stem}{args.suffix}{clean_filename.suffix}"
            )
            if not Path(f"{degraded_dir}/{degraded_filename}").is_file():
                logger.warning(
                    f"Degraded file {degraded_filename}, which correspondings to clean file {clean_filename} does not exist.\nSkipping to next clean file."
                )
                continue
            else:
                degraded_filepath = Path(f"{degraded_dir}/{degraded_filename}")

            # get images
            clean_image = cv2.imread(clean_filename)
            degraded_image = cv2.imread(degraded_filepath)

            # If the clean image was downscaled by a non-divisor, then it's possible that the degraded image is smaller than the clean image.
            # slice excess off of the clean image
            clean_image = clean_image[
                : degraded_image.shape[0], : degraded_image.shape[1]
            ]

            # compute PSNR
            clean_grayscale = rgb_to_grayscale(clean_image)
            degraded_grayscale = rgb_to_grayscale(degraded_image)

            psnr = evaluate_psnr(clean_grayscale, degraded_grayscale)

            psnr_max_all = max(psnr, psnr_max_all)
            psnr_min_all = min(psnr, psnr_min_all)
            psnr_max_dataset = max(psnr, psnr_max_dataset)
            psnr_min_dataset = min(psnr, psnr_min_dataset)
            psnr_all.append(psnr)
            psnr_dataset.append(psnr)

            # compute LPIPS
            clean_lpips = lpips.im2tensor(clean_image)
            degraded_lpips = lpips.im2tensor(degraded_image)

            if args.use_gpu:
                clean_lpips.cuda()
                degraded_lpips.cuda()

            with lpips_gag():
                lpips_distance = evaluate_lpips(
                    clean_lpips, degraded_lpips, args.spatial
                )

            if not args.spatial:
                lpips_distance = lpips_distance.item()
            else:
                lpips_distance = lpips_distance.mean()

            lpips_max_all = max(lpips_distance, lpips_max_all)
            lpips_min_all = min(lpips_distance, lpips_min_all)
            lpips_max_dataset = max(lpips_distance, lpips_max_dataset)
            lpips_min_dataset = min(lpips_distance, lpips_min_dataset)
            lpips_all.append(lpips_distance)
            lpips_dataset.append(lpips_distance)

            # output PSNR and LPIPS
            if args.verbose:
                print(f"{clean_filename.stem} PSNR: {round(psnr, 3)} dB")
                print(
                    f"{clean_filename.stem} LPIPS distance: {round(lpips_distance, 3)}\n"
                )

        print(f"---{str(dataset.stem).upper()} RESULTS---")

        print(
            f"{dataset.stem} PSNR average: {round(statistics.fmean(psnr_dataset), 3)}"
        )
        print(
            f"{dataset.stem} PSNR median: {round(statistics.median(psnr_dataset), 3)}"
        )
        print(f"{dataset.stem} PSNR max (best): {round(psnr_max_dataset, 3)}")
        print(f"{dataset.stem} PSNR min (worst): {round(psnr_min_dataset, 3)}\n")

        print(
            f"{dataset.stem} LPIPS average: {round(statistics.fmean(lpips_dataset), 3)}"
        )
        print(
            f"{dataset.stem} LPIPS median: {round(statistics.median(lpips_dataset), 3)}"
        )
        print(f"{dataset.stem} LPIPS min (best): {round(lpips_min_dataset, 3)}")
        print(f"{dataset.stem} LPIPS max (worst): {round(lpips_max_dataset, 3)}\n")

    print("---COMPLETE RESULTS---")

    print(f"PSNR average: {round(statistics.fmean(psnr_all), 3)}")
    print(f"PSNR median: {round(statistics.median(psnr_all), 3)}")
    print(f"PSNR max (best): {round(psnr_max_all, 3)}")
    print(f"PSNR min (worst): {round(psnr_min_all, 3)}\n")

    print(f"LPIPS average: {round(statistics.fmean(lpips_all), 3)}")
    print(f"LPIPS median: {round(statistics.median(lpips_all), 3)}")
    print(f"LPIPS min (best): {round(lpips_min_all, 3)}")
    print(f"LPIPS max (worst): {round(lpips_max_all, 3)}\n")


if __name__ == "__main__":
    main()
