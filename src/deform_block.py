# src/deform_block.py

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class DeformBlock(nn.Module):
    """
    Deformable Convolution Block
    Learns spatial offsets dynamically.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Offset generator
        # 3x3 kernel → 2 * 3 * 3 = 18 offset channels
        self.offset_conv = nn.Conv2d(
            in_channels,
            18,
            kernel_size=3,
            padding=1
        )

        # Deformable convolution
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        x = self.relu(x)
        return x