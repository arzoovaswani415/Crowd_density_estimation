import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ShanghaiTechDataset(Dataset):
    def __init__(self, img_dir, gt_dir, resize=None):

        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.resize = resize

        all_imgs = sorted(os.listdir(img_dir))
        self.img_names = []

        # Keep only images with valid GT
        for img_name in all_imgs:
            gt_name = img_name.replace(".jpg", ".h5")
            gt_path = os.path.join(gt_dir, gt_name)

            if os.path.exists(gt_path):
                self.img_names.append(img_name)

        print("Total valid samples:", len(self.img_names))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):

        img_name = self.img_names[idx]

        # ----- Load image -----
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0

        # ----- Load GT -----
        gt_name = img_name.replace(".jpg", ".h5")
        gt_path = os.path.join(self.gt_dir, gt_name)

        with h5py.File(gt_path, "r") as f:
            density = np.array(f["density"], dtype=np.float32)

        # ----- Resize if needed -----
        if self.resize is not None:

            h, w = img.shape[:2]
            new_h, new_w = self.resize

            # Resize image
            img = np.array(
                Image.fromarray((img * 255).astype(np.uint8))
                .resize((new_w, new_h), Image.BILINEAR)
            ).astype(np.float32) / 255.0

            # Resize density to SAME resolution
            density = np.array(
                Image.fromarray(density)
                .resize((new_w, new_h), Image.BILINEAR)
            )

            # Preserve total count
            density = density * (h * w) / (new_h * new_w)

        # ----- Convert to tensor -----
        img = torch.from_numpy(img).permute(2, 0, 1)
        density = torch.from_numpy(density).unsqueeze(0)

        return img, density