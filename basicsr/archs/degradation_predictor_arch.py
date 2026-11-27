import torch
from torch import nn

from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class DegradationPredictor(nn.Module):
    """
    Degradation predictor for conditioned SR models (BSRNCond, RRDBCond)


    """

    def __init__(self, degradation_dim=3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 48, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, degradation_dim),
        )

    def forward(self, x):
        features = self.features(x)
        degradation = self.regressor(features)
        return degradation

