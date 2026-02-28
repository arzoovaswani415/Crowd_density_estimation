
import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import DeformConv2d


# -------------------------
# Spatial Attention Module
# -------------------------
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = self.sigmoid(self.conv(x))
        return x * attn


# -------------------------
# Deformable + Attention Block
# -------------------------
class DeformAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Learn offsets
        self.offset = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)

        # Deformable convolution
        self.deform = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        # Spatial Attention
        self.attn = SpatialAttention(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset(x)
        x = self.deform(x, offset)
        x = self.attn(x)
        return self.relu(x)


# -------------------------
# Full Model
# -------------------------
class CSRNetDeformAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained VGG frontend
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        self.frontend = nn.Sequential(*features[:23])

        self.backend = nn.Sequential(
            DeformAttentionBlock(512, 512),
            DeformAttentionBlock(512, 512),
            DeformAttentionBlock(512, 256),
            DeformAttentionBlock(256, 128),
            DeformAttentionBlock(128, 64),
        )

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
