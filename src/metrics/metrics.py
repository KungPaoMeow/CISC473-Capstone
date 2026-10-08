from math import log10, sqrt

import numpy as np


def psnr(clean: np.ndarray, degraded: np.ndarray, max_pixel: float = 255.0) -> float:
    """
    Calculates PSNR between two images.

    Parameters:
        clean (np.ndarray):
            Clean image (no degradations).
        degraded (np.ndarray):
            Degraded version of the above image.
        max_pixel (float):
            maximum value of a pixel. Defaults to 255, which is correct for 8-bit.

    Returns:
        psnr (float): PSNR in dB.
    """
    mse = np.mean((clean - degraded) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def rgb_to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    """
    Converts a RGB image to grayscale using Rec. 601 luma weights.

    Parameters:
        rgb_image (np.ndarray):
            RGB image to convert.

    Returns:
        grayscale_image (np.ndarray):
            Converted image.
    """
    weights = np.array([0.299, 0.587, 0.114])

    # remove alpha channel from image
    grayscale_image = np.dot(rgb_image[..., :3], weights)
    return grayscale_image
