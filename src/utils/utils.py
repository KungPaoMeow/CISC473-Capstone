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
    Converts an RGB tensor of shape [B, 3, X, Y] to grayscale using Rec. 601 luma weights.

    Parameters:
        rgb_image (torch.Tensor): RGB tensor of shape [B, 3, X, Y]

    Returns:
        grayscale_image (torch.Tensor): Grayscale tensor of shape [B, 1, X, Y]
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


def rgb_to_ycbcr(rgb_image: np.ndarray) -> np.ndarray:
    weights = np.array([
        [ 0.299,     0.587,     0.114   ],
        [-0.168736, -0.331264,  0.5     ],
        [ 0.5,      -0.418688, -0.081312]
    ])

    ycbcr_image = rgb_image @ weights.T
    return ycbcr_image

def ycbcr_to_rgb(ycbcr_image: np.ndarray) -> np.ndarray:
    weights = np.array([
        [ 0.299,     0.587,     0.114   ],
        [-0.168736, -0.331264,  0.5     ],
        [ 0.5,      -0.418688, -0.081312]
    ])
    inverse_weights = np.linalg.inv(weights)

    rgb_image = ycbcr_image @ inverse_weights.T
    return rgb_image


def ycbcr_to_rgb_torch(ycbcr_image: torch.Tensor) -> torch.Tensor:
    """
        Converts a YCbCr tensor of shape [B, 3, X, Y] to RGB
    
        Parameters:
            ycbcr_image (torch.Tensor): YCrCb tensor of shape [B, 3, X, Y]
    
        Returns:
            rgb_image (torch.Tensor): RGB tensor of shape [B, 3, X, Y]
    """
    weights = torch.tensor([
        [ 0.299,      0.587,      0.114    ],
        [-0.168736,  -0.331264,   0.5      ],
        [ 0.5,       -0.418688,  -0.081312 ]
    ], dtype=ycbcr_image.dtype, device=ycbcr_image.device)

    inverse_weights = torch.linalg.inv(weights)

    # [B, 3, X, Y] -> [B, X, Y, 3]
    ycbcr_image = ycbcr_image.permute(0, 2, 3, 1)

    rgb_image = ycbcr_image @ inverse_weights.T

    # [B, X, Y, 3] -> [B, 3, X, Y]
    return rgb_image.permute(0, 3, 1, 2)

def rgb_to_ycrcb_torch(ycbcr_image: torch.Tensor) -> torch.Tensor:
    """
        Converts a YCbCr tensor of shape [3, X, Y] to RGB
    
        Parameters:
            ycbcr_image (torch.Tensor): YCrCb tensor of shape [B, 3, X, Y]
    
        Returns:
            rgb_image (torch.Tensor): RGB tensor of shape [B, 3, X, Y]
    """
    weights = torch.tensor([
        [ 0.299,      0.587,      0.114    ],
        [-0.168736,  -0.331264,   0.5      ],
        [ 0.5,       -0.418688,  -0.081312 ]
    ], dtype=ycbcr_image.dtype, device=ycbcr_image.device)


    # [3, X, Y] -> [X, Y, 3]
    ycbcr_image = ycbcr_image.permute(1, 2, 0)

    rgb_image = ycbcr_image @ weights.T

    # [X, Y, 3] -> [3, X, Y]
    return rgb_image.permute(2, 0, 1)