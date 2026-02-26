import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ShanghaiTechDataset(Dataset):
    def __init__(self, img_dir, gt_dir, resize=None, downsample=4):
        """
        resize: (H, W)
        downsample: model output downsampling factor
        """
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.resize = resize
        self.downsample = downsample

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]

        # ---- Load image ----
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = np.array(img).astype(np.float32) / 255.0

        # ---- Load density safely ----
        gt_name = img_name.replace(".jpg", ".h5")
        gt_path = os.path.join(self.gt_dir, gt_name)

        # If GT file does not exist → skip
        if not os.path.exists(gt_path):
            return self.__getitem__((idx + 1) % len(self))

        try:
            with h5py.File(gt_path, "r") as f:
                density = np.array(f["density"], dtype=np.float32)
        except Exception:
            # Skip corrupted file
            return self.__getitem__((idx + 1) % len(self))
        # ---- Resize image ----
        if self.resize is not None:
          h, w = img.shape[:2]
          new_h, new_w = self.resize

          # Resize image
          img = np.array(
              Image.fromarray((img * 255).astype(np.uint8))
              .resize((new_w, new_h), Image.BILINEAR)
          ).astype(np.float32) / 255.0

          # Resize density to MODEL OUTPUT SIZE
          d_h = new_h // self.downsample
          d_w = new_w // self.downsample

          density = np.array(
              Image.fromarray(density)
              .resize((d_w, d_h), Image.BILINEAR)
          )

          # Preserve total count correctly
          density = density * (h * w) / (d_h * d_w)
        # ---- Convert to tensor ----
        img = torch.from_numpy(img).permute(2, 0, 1)
        density = torch.from_numpy(density).unsqueeze(0)

        return img, density