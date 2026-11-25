from math import log10, sqrt

import lpips
import numpy as np
import torch


def evaluate_psnr(
    clean: np.ndarray, degraded: np.ndarray, max_pixel: float = 1
) -> float:
    """
    Evaluates PSNR between two images.

    Parameters:
        clean (np.ndarray):
            Clean image (no degradations).
        degraded (np.ndarray):
            Degraded version of the above image.
        max_pixel (float):
            maximum value of a pixel. Defaults to 1.

    Returns:
        psnr (float): PSNR in dB.
    """
    mse = np.mean((clean - degraded) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def evaluate_psnr_torch(
    clean: torch.Tensor, degraded: torch.Tensor, max_pixel: float = 1
):
    mse = torch.mean((clean - degraded) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def evaluate_lpips(
    image0: torch.Tensor, image1: torch.Tensor, loss_fn: lpips.LPIPS
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
    d = loss_fn.forward(image0, image1)
    return d


def loss_fn_fft(pred, target):
    _, _, H, W = pred.shape    # pred and target: [B, C, H, W]

    # ortho normalizes forward and inverse transforms by square root of number of transformed elements
    # rfft shape is [B, C, N, M/2 + 1]
    #   -> in Fourier, i (matrix col) < 0 is discarded since F(j,i) = conj(F(-j,-i)) if input all real
    pred_fft = torch.fft.rfft2(pred, norm="ortho")
    target_fft = torch.fft.rfft2(target, norm="ortho")

    # Generate frequency coordinates
    u = torch.fft.rfftfreq(W, device=pred.device)   # rfft since only positive values left
    v = torch.fft.fftfreq(H, device=pred.device)    # fft for whole range

    # 2D radial wavenumber
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    wavenumber_squared = uu**2 + vv**2

    # Complex coefficient difference
    complex_diff = pred_fft - target_fft
    error_mag = torch.abs(complex_diff)
    
    return (wavenumber_squared * error_mag).mean()