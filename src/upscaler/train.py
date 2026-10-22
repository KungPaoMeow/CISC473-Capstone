import os

import lpips
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..utils.metrics import evaluate_psnr_torch
from .model import UpscaleModel

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


class Div2kDataset(Dataset):
    def __init__(
        self, lr_dir, hr_dir, patch_size=None, train=True, max_size: int | None = None
    ):
        """
        Args:
            lr_dir (str): Path to Low-Resolution folder
            hr_dir (str): Path to High-Resolution folder
            patch_size (int): Size of the LR patch (if cropping during training)
            train (bool): Toggle for training vs validation behaviors
            max_size (int | None): If not `None`, takes the first `max_size` images from the dataset
        """
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size
        self.is_train = train

        self.lr_filenames = np.array(sorted(os.listdir(lr_dir)))
        self.hr_filenames = np.array(sorted(os.listdir(hr_dir)))
        if max_size is not None:
            self.lr_filenames = self.lr_filenames[:max_size]
            self.hr_filenames = self.hr_filenames[:max_size]

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


def get_train_and_test_datasets(patch_size, batch_size, scale_factor=2):
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
        max_size=5,
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
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn_lpips: lpips.LPIPS,
    loss_fn_pixel,
    optimizer,
):
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        loss_lpips = loss_fn_lpips.forward(pred * 2 - 1, y * 2 - 1).mean()
        loss_pixel = loss_fn_pixel(pred, y)

        loss_combined = loss_lpips + loss_pixel
        # backpropagation
        loss_combined.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            print(
                f"pixel loss: {loss_pixel.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]"
            )
            print(
                f"LPIPS distance: {loss_lpips.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]"
            )
            print(
                f"loss: {loss_combined.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]"
            )


# test loop
def test_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn_lpips: lpips.LPIPS,
    display: bool = False,
):
    model.eval()
    num_batches = len(dataloader)
    lpips_loss = 0
    psnr = 0

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)

            # compute LPIPS loss
            lpips_single = loss_fn_lpips.forward(pred * 2 - 1, y * 2 - 1).mean().item()

            # compute PSNR
            pred_gray = pred[:, 0, :, :]
            y_gray = y[:, 0, :, :]
            psnr_single = evaluate_psnr_torch(pred_gray, y_gray)

            lpips_loss += lpips_single
            psnr += psnr_single

            if display:
                import matplotlib.pyplot as plt

                print(f"IMAGE {i}")
                print(f"LPIPS distance: {lpips_single:>4f}")
                print(f"PSNR: {psnr_single:>2f}")
                plt.imshow(pred.permute(2, 3, 1, 0).cpu().numpy().squeeze())
                plt.show()
                print()
    psnr /= num_batches
    lpips_loss /= num_batches
    print(f"average LPIPS distance: {lpips_loss:>4f}")
    print(f"average PSNR: {psnr:>2f}")


def save_model(model, file_path, file_name):
    os.makedirs(file_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(file_path, file_name))


def __main__():
    scale_factor = 4
    learning_rate = 1e-3
    batch_size = 64
    epochs = 30

    patch_size = 32

    train_dataloader, test_dataloader = get_train_and_test_datasets(
        patch_size=patch_size, batch_size=batch_size, scale_factor=scale_factor
    )

    print(f"Using {device} device")

    model = UpscaleModel(scale_factor=scale_factor).to(device)

    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
    loss_fn_pixel = nn.L1Loss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        print("TRAINING")
        train_loop(train_dataloader, model, loss_fn_vgg, loss_fn_pixel, optimizer)
        if t == epochs - 1:
            print("\nTESTING")
            test_loop(test_dataloader, model, loss_fn_vgg)
        print()

    save_model(model, "models", "model.pth")

    # evaluate and output images
    eval_data = Div2kDataset(
        f"datasets/DIV2K_valid_LR_bicubic/X{scale_factor}",
        "datasets/DIV2K_valid_HR",
        train=False,
        max_size=5,
    )
    eval_dataloader = DataLoader(
        dataset=eval_data,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loop(eval_dataloader, model, loss_fn_vgg, display=True)


if __name__ == "__main__":
    __main__()
