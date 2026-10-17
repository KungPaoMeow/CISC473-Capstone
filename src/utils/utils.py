import numpy as np
import torch


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


def rgb_to_grayscale_torch(rgb_image: torch.Tensor) -> torch.Tensor:
    """
    Converts a RGB image to grayscale using Rec. 601 luma weights.

    Parameters:
        rgb_image (np.ndarray):
            RGB image to convert.

    Returns:
        grayscale_image (np.ndarray):
            Converted image.
    """
    print("calculating weights")
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=rgb_image.dtype, device=rgb_image.device)
    print("done")

    # remove alpha channel from image
    grayscale_image = torch.matmul(rgb_image[..., :3], weights)
    return grayscale_image
