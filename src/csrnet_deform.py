import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .deform_block import DeformBlock
from .deform_attention import DeformableAttention



class CSRNetDeform(nn.Module):

    def __init__(self):
        super().__init__()

        # -----------------------------
        # Frontend (Pretrained VGG16)
        # -----------------------------
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        self.frontend = nn.Sequential(
            *features[:16],
            *features[17:23]
        )

        # -----------------------------
        # Backend
        # -----------------------------
        self.backend = nn.Sequential(

            # 1️⃣ Deformable Conv
            DeformBlock(512, 512),

            # 2️⃣ Deformable Attention
            DeformableAttention(512, 512),

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

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)

        # 🔥 VERY IMPORTANT — force positive density
        x = F.softplus(x)

        return x
