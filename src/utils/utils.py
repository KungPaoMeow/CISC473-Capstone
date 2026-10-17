import numpy as np
import torch

rgb_to_grayscale_weights_torch = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


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
    Converts RGB prediction and label tensors of shape [B, 3, X, Y] to grayscale
    using Rec. 601 luma weights.

    Parameters:
        predictions (torch.Tensor): RGB tensor of shape [B, 3, X, Y]
        labels (torch.Tensor): RGB tensor of shape [B, 3, X, Y]

    Returns:
        gray_preds (torch.Tensor): Grayscale tensor of shape [B, 1, X, Y]
        gray_labels (torch.Tensor): Grayscale tensor of shape [B, 1, X, Y]
    """
    # Enforce shape requirements
    if rgb_image.shape[1] != 3:
        raise ValueError("Inputs must have exactly 3 channels at dimension 1 (RGB).")
    global rgb_to_grayscale_weights_torch
    rgb_to_grayscale_weights_torch = rgb_to_grayscale_weights_torch.to(
        rgb_image.device, dtype=rgb_image.dtype
    )

    grayscale_image = (rgb_image * rgb_to_grayscale_weights_torch).sum(
        dim=1, keepdim=True
    )

    return grayscale_image
