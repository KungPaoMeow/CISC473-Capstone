import torch
from torch import nn


class UpscaleModel(nn.Module):
    """
    FSRCNN that takes in all 3 channels with a pixel shuffle layer at the end to upscale.
    """

    def __init__(self, num_channels=3, d=56, s=16, m=4, scale_factor=2):
        super().__init__()

        # feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(
                in_channels=num_channels, out_channels=d, kernel_size=5, padding=5 // 2
            ),
            nn.PReLU(d),
        )

        # shrinking
        self.shrinking = nn.Sequential(
            nn.Conv2d(d, s, kernel_size=1),
            nn.PReLU(s),
        )

        # mapping
        mapping_layers = []
        for _ in range(m):
            mapping_layers.append(nn.Conv2d(s, s, kernel_size=3, padding=3 // 2))
        self.mapping = nn.Sequential(*mapping_layers)

        # expanding
        self.expanding = nn.Sequential(nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d))

        # upscale with deconvolution + pixel shuffle
        out_channels = num_channels * scale_factor**2
        self.upscale = nn.Sequential(
            nn.Conv2d(d, out_channels, kernel_size=1),
            nn.PReLU(out_channels),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, x):
        print("CHYME1")
        extracted = self.feature_extraction(x)
        print("CHYME2")
        shrunk = self.shrinking(extracted)
        print("CHYME3")
        mapped = self.mapping(shrunk)
        print("CHYME4")
        expanded = self.expanding(mapped)
        print("CHYME5")
        upscaled = self.upscale(expanded)
        print("CHYME6")
        return upscaled
