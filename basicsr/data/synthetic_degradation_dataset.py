import os
import random

import cv2
import numpy as np
import torch
from basicsr.utils import FileClient, imfrombytes, img2tensor, scandir
from basicsr.utils.matlab_functions import rgb2ycbcr
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data


@DATASET_REGISTRY.register()
class SyntheticDegradationDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the template excludes the file extension.
                Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_hflip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.gt_folder = opt["dataroot_gt"]

        self.io_backend_opt = opt["io_backend"].copy()
        self.file_client = None

        self.gt_size = opt.get("gt_size", 256)
        self.skip = opt.get("skip", 0.10)

        self.paths = [
            os.path.join(self.gt_folder, v)
            for v in scandir(self.gt_folder)
            if v.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        ]

        # degradation parameter ranges
        self.blur_range = opt.get("blur_range", [0.2, 1.5])  # in HR pixels
        self.noise_range = opt.get("noise_range", [0, 10])
        self.jpeg_range = opt.get("jpeg_range", [60, 100])
        self.scale = opt.get("scale", 4)
        assert self.gt_size % self.scale == 0

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # image range: [0, 1], float32.
        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path, "gt")
        img_gt = imfrombytes(img_bytes, float32=True)

        h, w = img_gt.shape[:2]
        if h < self.gt_size or w < self.gt_size:
            raise ValueError(f"image {gt_path} is smaller than gt_size {self.gt_size}")

        # crop the image randomly
        top = random.randint(0, h - self.gt_size)
        left = random.randint(0, w - self.gt_size)
        img_gt = img_gt[top : top + self.gt_size, left : left + self.gt_size]

        # degradation pipeline
        # 1. isotropic gaussian blur
        # 2. bicubic or bilinear downsampling
        # 3. noise injection
        # 4. JPEG compression

        # skip everything except downsampling 5% of the time so some data is very simple
        skip = random.random() < self.skip

        # blur
        # don't always blur, leave image unblurred 20% of the time
        if not skip and random.random() > 0.2:
            blur_factor = random.uniform(*self.blur_range)
            img_lq = cv2.GaussianBlur(
                img_gt, (0, 0), sigmaX=blur_factor, sigmaY=blur_factor
            )
        else:
            blur_factor = 0
            img_lq = img_gt.copy()

        # downsampling
        h, w = img_lq.shape[:2]
        interpolation = random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR])

        img_lq = cv2.resize(
            img_lq, (w // self.scale, h // self.scale), interpolation=interpolation
        )

        # noise injection
        if not skip:
            noise_factor = random.uniform(*self.noise_range)

            noise = np.random.normal(0, noise_factor / 255, img_lq.shape).astype(
                np.float32
            )

            img_lq = np.clip(img_lq + noise, 0, 1)
        else:
            noise_factor = 0

        # jpeg compression
        if not skip:
            jpeg_quality = random.randint(
                int(self.jpeg_range[0]), int(self.jpeg_range[1])
            )

            img_255 = np.round(img_lq * 255).astype(np.uint8)

            success, encoded = cv2.imencode(
                ".jpg", img_255, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )

            if not success:
                raise RuntimeError("vacuous jpeg compression failed")

            img_lq = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32) / 255
        else:
            jpeg_quality = 100

        # end of degradation pipeline
        # convert to pytorch tensors

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=True, float32=True)

        degradation_parameters = torch.tensor(
            [
                blur_factor,
                noise_factor,
                jpeg_quality,
                self.scale,
            ],
            dtype=torch.float32,
        )

        # print(f"\nDEGRADATION PARAMS: {degradation_parameters}")

        # color space transform
        if "color" in self.opt and self.opt["color"] == "y":
            img_gt = rgb2ycbcr(img_gt, y_only=True)[..., None]
            img_lq = rgb2ycbcr(img_lq, y_only=True)[..., None]

        return {
            "lq": img_lq,
            "gt": img_gt,
            "lq_path": gt_path,
            "gt_path": gt_path,
            "degradation": degradation_parameters,
        }

    def __len__(self):
        return len(self.paths)
