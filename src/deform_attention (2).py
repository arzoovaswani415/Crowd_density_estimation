import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class DeformableAttention(nn.Module):
    """
    Deformable Attention Module
    Combines DCN + Spatial Attention
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Offset generator
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)

        # Deformable convolution
        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        # Spatial attention (lightweight)
        self.attention_conv = nn.Conv2d(
            out_channels,
            1,
            kernel_size=1
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # Geometric adaptation
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)

        # Attention weighting
        attn = torch.sigmoid(self.attention_conv(x))
        x = x * attn

        x = self.relu(x)
        return x
