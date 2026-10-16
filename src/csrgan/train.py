import lpips
import numpy as np
import torch
from torch import nn

from .model import UpscaleModel

# load data
data_dir = ""
x_train = np.load(data_dir)
print(f"shape of training data: {x_train.shape}")
print(f"data type: {type(x_train)}")

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = UpscaleModel().to(device)
print(model)

x = torch.zeros(1, 3, 40, 40).to(device)
y = torch.ones(1, 3, 80, 80).to(device)
output = model(x)

loss_fn_vgg = lpips.LPIPS(net="vgg").cuda()
loss_perceptual = loss_fn_vgg.forward(output, y)
print(loss_perceptual)
