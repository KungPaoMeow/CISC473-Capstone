import torch
import torch.nn as nn

class FSRCNN(nn.Module):
    def __init__(self, scale_factor=3, num_channels=1, d=56, s=12, m=4):
        super().__init__()
        self.layer1 = nn.Sequential