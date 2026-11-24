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
from torch.utils.data.distributed import DistributedSampler

from ..utils.metrics import evaluate_psnr_torch
from .model import RRDBNet

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

        return lr_tensor, hr_tensor


def get_train_and_test_datasets(
    patch_size, batch_size, scale_factor=2, num_validations=100
):
    training_data = Div2kDataset(
        f"datasets/DIV2K_train_LR_bicubic/X{scale_factor}",
        "datasets/DIV2K_train_HR",
        patch_size,
        train=True,
        max_size=800,
    )
    validation_data = Div2kDataset(
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

    validation_sampler = (
        DistributedSampler(validation_data, shuffle=False)
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
    validation_dataloader = DataLoader(
        dataset=validation_data,
        batch_size=1,
        shuffle=False,
        sampler=validation_sampler,
        num_workers=2,
        pin_memory=True,
    )
    return train_dataloader, validation_dataloader


# train loop
def train_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn_lpips: lpips.LPIPS,
    loss_fn_pixel,
    optimizer,
    scheduler,
    epoch
):
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)
    
    size = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    model.train()

    # Accumulate weighted loss totals
    loss_l1_pixel_sum = torch.zeros((), device=device)
    loss_lpips_sum = torch.zeros((), device=device)
    num_samples = 0

    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)

        loss_lpips = loss_fn_lpips.forward(pred * 2 - 1, y * 2 - 1).mean()
        loss_pixel = loss_fn_pixel(pred, y)

        current_batch_size = X.size(0)
        loss_l1_pixel_sum += loss_pixel.detach() * current_batch_size
        loss_lpips_sum += loss_lpips.detach() * current_batch_size
        num_samples += current_batch_size

        loss_combined = loss_lpips + 3*loss_pixel

        # backpropagation
        loss_combined.backward()
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if batch % 2 == 0:
            print(
                f"{'rank ' + str(RANK) + '\n' if IS_DISTRIBUTED else ''}"
                f"pixel loss: {loss_pixel.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]\n"
                f"LPIPS distance: {loss_lpips.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]\n"
                f"loss: {loss_combined.item():>7f}  [{batch * batch_size + len(X):>5d}/{size:>5d}]\n"
            )

    # Combine totals from all GPUs
    totals = torch.stack([
        loss_l1_pixel_sum,
        loss_lpips_sum,
        torch.tensor(num_samples, device=device, dtype=torch.float32),
    ])

    if IS_DISTRIBUTED:
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)

    return totals[0] / totals[2], totals[1] / totals[2]


# test loop
def validation_loop(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn_lpips: lpips.LPIPS,
    patch_size: int = 128,
):
    model.eval()
    eval_model = model.module if IS_DISTRIBUTED else model
    
    lpips_scores = []
    psnr_scores = []

    with torch.no_grad():
        for i, (X, y) in enumerate(dataloader):
            X = X.to(device)
            y = y.to(device)

            pred = eval_model(X)     # shape torch.Size([batch_size, channels, height, width])
            _, _, height, width = pred.shape

            # compute LPIPS loss in batches
            loss = patches = 0
            for x_pixel in range(0, width, patch_size):
                for y_pixel in range(0, height, patch_size):
                    # Shift start coordinate back if the patch goes past the image border
                    x_start = max(0, min(x_pixel, width - patch_size))
                    y_start = max(0, min(y_pixel, height - patch_size))

                    pred_patch = pred[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size]
                    target_patch = y[:, :, y_start:y_start + patch_size, x_start:x_start + patch_size]

                    loss += loss_fn_lpips.forward(pred_patch * 2 - 1, target_patch * 2 - 1).mean().item()
                    patches += 1

            lpips_single = loss / patches

            # compute PSNR
            pred_gray = pred[:, 0, :, :]
            y_gray = y[:, 0, :, :]
            psnr_single = evaluate_psnr_torch(pred_gray, y_gray)

            lpips_scores.append(lpips_single)
            psnr_scores.append(psnr_single)

    # Combine metrics across GPUs using a single tensor all_reduce
    totals = torch.tensor(
        [sum(psnr_scores), sum(lpips_scores), len(lpips_scores)],
        device=device,
        dtype=torch.float64,
    )

    if IS_DISTRIBUTED:
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)

    mean_psnr = (totals[0] / totals[2]).item()
    mean_lpips = (totals[1] / totals[2]).item()

    return mean_psnr, mean_lpips


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
    print(f"LPIPS: {np.mean(lpips_scores):.4f} +/- {np.std(lpips_scores):.4f}")
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
        default=100,
        help="Number of validation images to evaluate (default: 100)",
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
    batch_size = 50
    resume_epoch = 0
    epochs = 200

    patch_size = 32

    train_dataloader, validation_dataloader = get_train_and_test_datasets(
        patch_size=patch_size,
        batch_size=batch_size,
        scale_factor=scale_factor,
        num_validations=args.num_validations,
    )

    if IS_DISTRIBUTED and IS_MASTER:
        print("Using distributed multi-GPU environment")
    print(f"Using {device} device")

    model = RRDBNet(3, 3, scale_factor=scale_factor).to(device)
    if IS_DISTRIBUTED:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device).eval()
    loss_fn_pixel = nn.L1Loss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_l1_pixel_loss = []
    train_lpips_loss = []
    test_psnr_scores = []
    test_lpips_scores = []

    CHECKPOINT_REBOOT = True
    if CHECKPOINT_REBOOT:
        chkpt = torch.load("models/checkpoint_epoch_100-rrdb-100-4x-24feat-4block-10grow-2xpixel-1xlpips.pt", map_location=device)
        if IS_DISTRIBUTED:
            model.module.load_state_dict(chkpt["model"])
        else:
            model.load_state_dict(chkpt["model"])
        resume_epoch = chkpt["epoch"] + 1
        optimizer.load_state_dict(chkpt["optimizer"])
        # scheduler.load_state_dict(chkpt["scheduler"])
        train_l1_pixel_loss = chkpt["train_l1_pixel_loss"]
        train_lpips_loss = chkpt["train_lpips_loss"]
        test_psnr_scores = chkpt["test_psnr_scores"]
        test_lpips_scores = chkpt["test_lpips_scores"]

        # Override the learning rate inside the loaded optimizer
        new_lr = learning_rate * 0.5
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
            param_group["initial_lr"] = new_lr  # Required by PyTorch schedulers

        # Create a fresh scheduler for the next x epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs-resume_epoch, eta_min=1e-6
        )

    for t in range(resume_epoch, epochs):
        if IS_MASTER:
            print(f"Epoch {t + 1}\n-------------------------------")
            print("TRAINING")
        l1_loss, lpips_loss = train_loop(train_dataloader, model, loss_fn_vgg, loss_fn_pixel, optimizer, scheduler, epoch=t)
        train_l1_pixel_loss.append(l1_loss.item())
        train_lpips_loss.append(lpips_loss.item())

        psnr_score, lpips_score = validation_loop(validation_dataloader, model, loss_fn_vgg)
        test_psnr_scores.append(psnr_score)
        test_lpips_scores.append(lpips_score)
        print()

        if IS_MASTER and (t + 1) % 10 == 0:
            checkpoint = {
                "epoch": t,
                "model": model.module.state_dict() if IS_DISTRIBUTED else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "train_l1_pixel_loss": train_l1_pixel_loss,
                "train_lpips_loss": train_lpips_loss,
                "test_psnr_scores": test_psnr_scores,
                "test_lpips_scores": test_lpips_scores,
            }
            torch.save(checkpoint, f"models/checkpoint_epoch_{t + 1}.pt")

    if IS_MASTER:
        model_to_save = model.module if IS_DISTRIBUTED else model   # DDP model weights in .module
        save_model(model_to_save, "models", "model.pth")

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
        import matplotlib.pyplot as plt
   
        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        epoch_axis = range(1, epochs + 1)

        # 1. Perceptual Loss (Train vs. Val on the same scale)
        axes[0].plot(epoch_axis, train_lpips_loss, label='Train LPIPS', marker='o')
        axes[0].plot(epoch_axis, test_lpips_scores, label='Val LPIPS', marker='s')
        axes[0].set_ylabel('LPIPS (Lower is better)')
        axes[0].set_title('Training & Validation Metrics Over Epochs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Pixel Loss (Train L1 on its own scale)
        axes[1].plot(epoch_axis, train_l1_pixel_loss, label='Train L1 Loss', color='tab:green', marker='^')
        axes[1].set_ylabel('L1 Loss (Lower is better)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # 3. Image Quality Score (Val PSNR on its own scale)
        axes[2].plot(epoch_axis, test_psnr_scores, label='Val PSNR', color='tab:purple', marker='d')
        axes[2].set_ylabel('PSNR in dB (Higher is better)')
        axes[2].set_xlabel('Epoch')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
        
        test_loop(eval_dataloader, model, loss_fn_vgg, display=not args.skip)

    if IS_DISTRIBUTED:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    __main__()
