import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import DeformConv2d


class DeformableBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Learn offsets (18 = 2 * 3 * 3)
        self.offset = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)

        self.deform = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset(x)
        x = self.deform(x, offset)
        return self.relu(x)


class CSRNetDeform(nn.Module):
    def __init__(self):
        super().__init__()

        # Pretrained VGG frontend
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        features = list(vgg.features.children())

        # Keep until conv4_3 (output stride = 8)
        self.frontend = nn.Sequential(*features[:23])

        # Replace dilated backend with deformable blocks
        self.backend = nn.Sequential(
            DeformableBlock(512, 512),
            DeformableBlock(512, 512),
            DeformableBlock(512, 256),
            DeformableBlock(256, 128),
            DeformableBlock(128, 64),
        )

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
