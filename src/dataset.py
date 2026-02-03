import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ShanghaiTechDataset(Dataset):
    def __init__(self, img_dir, gt_dir, resize=None):
        """
        img_dir : path to images folder
        gt_dir  : path to ground-truth-h5 folder
        resize  : (H, W) tuple or None
        """
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.resize = resize

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # ---------- Load image ----------
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0

             # ---------- Load density map ----------
        gt_name = img_name.replace(".jpg", ".h5")
        gt_path = os.path.join(self.gt_dir, gt_name)

        try:
            with h5py.File(gt_path, "r") as f:
                density = np.array(f["density"], dtype=np.float32)
        except OSError:
            # Skip corrupted file by moving to next sample
            return self.__getitem__((idx + 1) % len(self))


        # ---------- Resize (if required) ----------
        if self.resize is not None:
            h, w = img.shape[:2]
            new_h, new_w = self.resize

            # resize image
            img = np.array(
                Image.fromarray((img * 255).astype(np.uint8))
                .resize((new_w, new_h), Image.BILINEAR)
            ).astype(np.float32) / 255.0

            # resize density map
            density = np.array(
                Image.fromarray(density)
                .resize((new_w, new_h), Image.BILINEAR)
            )

            # VERY IMPORTANT: preserve crowd count
            density = density * (h * w) / (new_h * new_w)

        # ---------- Convert to tensors ----------
        img = torch.from_numpy(img).permute(2, 0, 1)     # [3, H, W]
        density = torch.from_numpy(density).unsqueeze(0)  # [1, H, W]

        return img, density
