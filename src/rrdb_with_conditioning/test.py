import argparse
import os
import random

import lpips
from PIL import Image
import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset

from ..utils.metrics import evaluate_psnr_torch
from .model import RRDBNet


device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"

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

        lr_by_id = {
            os.path.splitext(filename)[0].split("x")[0]: filename
            for filename in sorted(os.listdir(lr_dir))
        }
        self.hr_filenames = np.array(
            [
                filename
                for filename in sorted(os.listdir(hr_dir))
                if os.path.splitext(filename)[0] in lr_by_id
            ]
        )
        self.lr_filenames = np.array(
            [lr_by_id[os.path.splitext(filename)[0]] for filename in self.hr_filenames]
        )
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

        # transform to Pytorch Tensors in range [0, 1]
        lr_tensor = F.to_tensor(lr_image)
        hr_tensor = F.to_tensor(hr_image)

        return lr_tensor, hr_tensor, self.lr_dir, self.lr_filenames[idx]


def test_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn_lpips: lpips.LPIPS,
    display: bool = False,
    patch_size: int = 256,
):
    model.eval()
    num_batches = len(dataloader)
    lpips_scores = []
    psnr_scores = []

    with torch.no_grad():
        for i, (X, y, dir, filename) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)     # shape torch.Size([batch_size, channels, height, width])
            _, _, height, width = pred.shape

            y = y[:, :, :height, :width]    # Standard SR practice: crop ground truth 'y' to match 'pred'

            # compute LPIPS loss in batches
            loss = patches = 0
            for x_pixel in range(0, width, patch_size):
                for y_pixel in range(0, height, patch_size):
                    pred_patch = pred[:, :, y_pixel:y_pixel + patch_size, x_pixel:x_pixel + patch_size]
                    target_patch = y[:, :, y_pixel:y_pixel + patch_size, x_pixel:x_pixel + patch_size]

                    loss += loss_fn_lpips.forward(pred_patch * 2 - 1, target_patch * 2 - 1).mean().item()
                    patches += 1

            lpips_single = loss / patches

            # compute PSNR
            pred_gray = pred[:, 0, :, :]
            y_gray = y[:, 0, :, :]
            psnr_single = evaluate_psnr_torch(pred_gray, y_gray)

            lpips_scores.append(lpips_single)
            psnr_scores.append(psnr_single)

            if display:
                import matplotlib.pyplot as plt

                print(
                    f"IMAGE {i + 1}/{num_batches} | LPIPS: {lpips_single:.4f} "
                    f"| PSNR: {psnr_single:.2f} dB"
                )
                # plt.imshow(pred.permute(2, 3, 1, 0).cpu().numpy().squeeze())
                # plt.show()
                img = pred.permute(2, 3, 1, 0).cpu().numpy().squeeze()
                img = np.clip(img, 0, 1)  # Assuming values are normalized to [0, 1]
                dir, filename = str(dir[0]), str(filename[0])
                os.makedirs(f"outputs/{dir}_SR", exist_ok=True)
                Image.fromarray((img * 255).astype(np.uint8)).save(f"outputs/{dir}_SR/{filename}")

    print(f"Validation summary ({num_batches} images)")
    print(f"LPIPS: {np.mean(lpips_scores):.4f} +/- {np.std(lpips_scores):.4f}")
    print(f"PSNR: {np.mean(psnr_scores):.2f} +/- {np.std(psnr_scores):.2f} dB")


def __main__():
    scale_factor = 4

    print(f"Using {device} device")

    model = RRDBNet(3, 3, scale_factor=scale_factor).to(device)
    model.load_state_dict(torch.load("models/model.pth"))
    # model.load_state_dict(torch.load("models/rrdb-200-4x-24feat-4block-10grow-3xpixel-1xlpips-50xfft.pth"))

    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device).eval()

    # evaluate and output images
    test_datasets = [
        (f"Set5/LRbicx{scale_factor}", f"Set5/original"), 
        (f"Set14/LRbicx{scale_factor}", f"Set14/original")
    ]

    for test_images, test_truth in test_datasets:
        eval_data = Div2kDataset(
            f"datasets/{test_images}",
            f"datasets/{test_truth}",
            train=False,
            max_size=100,
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
