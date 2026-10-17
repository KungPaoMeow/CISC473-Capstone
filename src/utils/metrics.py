from math import log10, sqrt

import lpips
import numpy as np
import torch


def evaluate_psnr(
    clean: np.ndarray, degraded: np.ndarray, max_pixel: float = 255.0
) -> float:
    """
    Evaluates PSNR between two images.

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

def evaluate_psnr_torch(predictions: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
    if predictions.shape[1] != 3 or labels.shape[1] != 3:
        raise ValueError("Inputs must have exactly 3 channels at dimension 1 (RGB).")

    # 1. Define Rec. 601 weights matching input device and dtype
    # Reshape weights to [1, 3, 1, 1] so they broadcast seamlessly across [B, 3, X, Y]
    weights = torch.tensor([0.299, 0.587, 0.114], dtype=predictions.dtype, device=predictions.device)
    weights = weights.view(1, 3, 1, 1)
    
    # 2. Match labels device and precision to predictions
    labels = labels.to(dtype=predictions.dtype, device=predictions.device)

    # 3. Compute weighted sum across the channel dimension (dim=1)
    # keepdim=True preserves the channel dimension, resulting in shape [B, 1, X, Y]
    gray_preds = (predictions * weights).sum(dim=1, keepdim=True)
    gray_labels = (labels * weights).sum(dim=1, keepdim=True)

    return gray_preds, gray_labels


def evaluate_lpips(
    image0: torch.Tensor, image1: torch.Tensor, spatial: bool = False
) -> float:
    """
    Evaluates LPIPS between two images.

    Parameters:
        image0 (torch.Tensor):
            Image to compare in torch tensor format
        image1 (torch.Tensor):
            Image to compare in torch tensor format

    Returns:
        d (float): LPIPS distance between `image0` and `image1`.
    """
    loss_fn = lpips.LPIPS(net="vgg", spatial=spatial)
    d = loss_fn.forward(image0, image1)
    return d
