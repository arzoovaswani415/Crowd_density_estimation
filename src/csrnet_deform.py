# src/csrnet_deform.py

import torch
import torch.nn as nn
from torchvision import models
from .deform_block import DeformBlock   # IMPORTANT: relative import
from .deform_attention import DeformAttention


class CSRNetDeform(nn.Module):
    """
    CSRNet + Deformable Convolution in Backend
    """

    def __init__(self):
        super().__init__()

        # -----------------------------
        # Frontend (Pretrained VGG16)
        # -----------------------------
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        # Remove 3rd maxpool (like CSRNet paper)
        self.frontend = nn.Sequential(
            *features[:16],
            *features[17:23]
        )

        # -----------------------------
        # Backend (Dilated + Deform)
        # -----------------------------
        self.backend = nn.Sequential(

            # Replace first backend conv with deformable conv
            DeformBlock(512, 512),

            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
        )
        self.backend = nn.Sequential(

    # 1️⃣ Deformable Conv
    DeformBlock(512, 512),

    # 2️⃣ 🔥 Deformable Attention (NEW)
    DeformAttention(512),

    nn.Conv2d(512, 512, 3, padding=2, dilation=2),
    nn.ReLU(inplace=True),

    nn.Conv2d(512, 512, 3, padding=2, dilation=2),
    nn.ReLU(inplace=True),

    nn.Conv2d(512, 256, 3, padding=2, dilation=2),
    nn.ReLU(inplace=True),

    nn.Conv2d(256, 128, 3, padding=2, dilation=2),
    nn.ReLU(inplace=True),

    nn.Conv2d(128, 64, 3, padding=2, dilation=2),
    nn.ReLU(inplace=True),
)

        # Output density map
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x