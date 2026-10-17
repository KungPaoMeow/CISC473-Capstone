import os

import lpips
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..utils.metrics import evaluate_psnr_torch
from ..utils.utils import rgb_to_grayscale_torch
from .model import UpscaleModel

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class Div2kDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, patch_size=None, train=True):
        """
        Args:
            lr_dir (str): Path to Low-Resolution folder
            hr_dir (str): Path to High-Resolution folder
            patch_size (int): Size of the LR patch (if cropping during training)
            train (bool): Toggle for training vs validation behaviors
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.is_train = train

        self.lr_filenames = sorted(os.listdir(lr_dir))
        self.hr_filenames = sorted(os.listdir(hr_dir))
        if not train:
            self.lr_filenames = self.lr_filenames[:10]
            self.hr_filenames = self.hr_filenames[:10]

        assert len(self.lr_filenames) == len(self.hr_filenames), (
            f"Mismatch: found {len(self.lr_filenames)} LR images but {len(self.hr_filenames)} HR images."
        )

    def __len__(self):
        return len(self.lr_filenames)

    def __getitem__(self, idx):
        lr_path = os.path.join(self.lr_dir, self.lr_filenames[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_filenames[idx])
        lr_image = Image.open(lr_path).convert("RGB")
        hr_image = Image.open(hr_path).convert("RGB")

        if self.is_train and self.patch_size is not None:
            # crop images to patch size
            # calculate upscale factor
            scale = hr_image.size[0] // lr_image.size[0]

            lr_width, lr_height = lr_image.size
            # select a top-left starting coordinate on the LR image
            x = torch.randint(0, lr_width - self.patch_size + 1, (1,)).item()
            y = torch.randint(0, lr_height - self.patch_size + 1, (1,)).item()

            # crop LR image
            lr_image = F.crop(lr_image, y, x, self.patch_size, self.patch_size)
            # crop HR image to match the cropped LR image
            hr_image = F.crop(
                hr_image,
                y * scale,
                x * scale,
                self.patch_size * scale,
                self.patch_size * scale,
            )

        # transform to Pytorch Tensors
        lr_tensor = F.to_tensor(lr_image)
        hr_tensor = F.to_tensor(hr_image)

        return lr_tensor, hr_tensor


def get_datasets(patch_size, batch_size, scale_factor=2):
    training_data = Div2kDataset(
        f"datasets/DIV2K_train_LR_bicubic/X{scale_factor}",
        "datasets/DIV2K_train_HR",
        patch_size,
        train=True,
    )
    testing_data = Div2kDataset(
        f"datasets/DIV2K_valid_LR_bicubic/X{scale_factor}",
        "datasets/DIV2K_valid_HR",
        train=False,
    )

    train_dataloader = DataLoader(
        dataset=training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        dataset=testing_data,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    return train_dataloader, test_dataloader


# train loop
def train_loop(
    dataloader: DataLoader, model, loss_fn_lpips: lpips.LPIPS, loss_fn_pixel, optimizer
):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        # loss = loss_fn_lpips.forward(pred * 2 - 1, y * 2 - 1).mean()
        loss = loss_fn_pixel(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            print(
                f"loss: {loss.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]"
            )


# test loop
def test_loop(dataloader, model, loss_fn_lpips: lpips.LPIPS):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    lpips_loss = 0
    psnr = 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            pred = torch.clamp(pred, min=0.0, max=1.0)
            lpips_loss += loss_fn_lpips.forward(pred * 2 - 1, y * 2 - 1).mean().item()

            pred_gray = rgb_to_grayscale_torch(pred)
            y_gray = rgb_to_grayscale_torch(y)
            # evaluate psnr
            psnr += evaluate_psnr_torch(pred_gray, y_gray)
    psnr /= num_batches
    lpips_loss /= num_batches
    print(f"average LPIPS loss: {lpips_loss}")
    print(f"average PSNR: {psnr}")


def __main__():
    scale_factor = 2
    learning_rate = 1e-3
    batch_size = 64
    epochs = 5

    patch_size = 64

    train_dataloader, test_dataloader = get_datasets(
        patch_size=patch_size, batch_size=batch_size, scale_factor=scale_factor
    )

    print(f"Using {device} device")

    model = UpscaleModel(scale_factor=scale_factor).to(device)

    loss_fn_vgg = lpips.LPIPS(net="vgg").cuda()
    loss_fn_pixel = nn.L1Loss().to(device)

    # hyperparameters

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        print("TRAINING")
        train_loop(train_dataloader, model, loss_fn_vgg, loss_fn_pixel, optimizer)
        print("\nTESTING")
        test_loop(test_dataloader, model, loss_fn_vgg)


if __name__ == "__main__":
    __main__()
