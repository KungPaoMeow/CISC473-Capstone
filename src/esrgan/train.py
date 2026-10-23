import argparse
import os
import random

import lpips
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torch import nn
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from ..utils.metrics import evaluate_psnr_torch
from .rrdbnet_arch import RRDBNet

IS_DISTRIBUTED = "LOCAL_RANK" in os.environ and torch.accelerator.is_available()

if IS_DISTRIBUTED:
    LOCAL_RANK = int(os.environ["LOCAL_RANK"])              # device ID of GPU executing this script
    RANK = int(os.environ["RANK"])
    IS_MASTER = RANK == 0

    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device(f"cuda:{LOCAL_RANK}")

    torch.distributed.init_process_group(
        backend="nccl",      # GPU communication backend, nccl not supported on Windows
    )    
else:
    IS_MASTER = True
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


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

        return lr_tensor, hr_tensor


def get_train_and_test_datasets(
    patch_size, batch_size, scale_factor=2, num_validations=5
):
    training_data = Div2kDataset(
        f"datasets/DIV2K_train_LR_bicubic/X{scale_factor}",
        "datasets/DIV2K_train_HR",
        patch_size,
        train=True,
        max_size=400,
    )
    testing_data = Div2kDataset(
        f"datasets/DIV2K_valid_LR_bicubic/X{scale_factor}",
        "datasets/DIV2K_valid_HR",
        train=False,
        max_size=num_validations,
    )

    train_sampler = (
        DistributedSampler(training_data, shuffle=True)
        if IS_DISTRIBUTED
        else None
    )

    train_dataloader = DataLoader(
        dataset=training_data,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
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
    epoch
):
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
    
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
                f"pixel loss: {loss_pixel.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]\n"
                f"LPIPS distance: {loss_lpips.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]\n"
                f"loss: {loss_combined.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]\n"
            )


# test loop
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
        for i, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            pred = model(X)     # shape torch.Size([batch_size, channels, height, width])
            _, _, height, width = pred.shape

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
                plt.imshow(pred.permute(2, 3, 1, 0).cpu().numpy().squeeze())
                plt.show()

    print(f"Validation summary ({num_batches} images)")
    print(
        f"LPIPS: {np.mean(lpips_scores):.4f} +/- {np.std(lpips_scores):.4f}"
    )
    print(f"PSNR: {np.mean(psnr_scores):.2f} +/- {np.std(psnr_scores):.2f} dB")


def save_model(model, file_path, file_name):
    os.makedirs(file_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(file_path, file_name))


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train and validate the upscaler")
    parser.add_argument(
        "-s", "--skip", action="store_true", help="Skip displaying validation images"
    )
    parser.add_argument(
        "-nv",
        "--num-validations",
        type=int,
        default=5,
        help="Number of validation images to evaluate (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducible runs"
    )
    args = parser.parse_args()
    if args.num_validations < 1:
        parser.error("--num-validations must be at least 1")
    return args


def __main__():
    args = parse_arguments()
    set_random_seed(args.seed)
    scale_factor = 4
    learning_rate = 1e-3
    batch_size = 64
    epochs = 50

    patch_size = 32

    train_dataloader, test_dataloader = get_train_and_test_datasets(
        patch_size=patch_size,
        batch_size=batch_size,
        scale_factor=scale_factor,
        num_validations=args.num_validations,
    )

    if IS_DISTRIBUTED and IS_MASTER:
        print("Using distributed multi-GPU environment")
    print(f"Using {device} device")

    model = RRDBNet(3, 3, scale=scale_factor).to(device)
    if IS_DISTRIBUTED:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)


    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
    loss_fn_pixel = nn.L1Loss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs):
        if IS_MASTER:
            print(f"Epoch {t + 1}\n-------------------------------")
            print("TRAINING")
        train_loop(train_dataloader, model, loss_fn_vgg, loss_fn_pixel, optimizer, epoch=t)
        if t == epochs - 1:
            if IS_MASTER:
                print("\nTESTING")
                # test_loop(test_dataloader, model, loss_fn_vgg, display=not args.skip)
        print()

    if IS_MASTER:
        model_to_save = model.module if IS_DISTRIBUTED else model   # DDP model weights in .module
        save_model(model_to_save, "models", "model.pth")

        # evaluate and output images
        eval_data = Div2kDataset(
            f"datasets/DIV2K_valid_LR_bicubic/X{scale_factor}",
            "datasets/DIV2K_valid_HR",
            train=False,
            max_size=10,
        )
        eval_dataloader = DataLoader(
            dataset=eval_data,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        test_loop(eval_dataloader, model, loss_fn_vgg, display=True)

    if IS_DISTRIBUTED:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    __main__()
