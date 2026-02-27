import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class DeformableAttention(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)

        self.deform_conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1
        )

        self.attention_conv = nn.Conv2d(out_channels, 1, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)

        attn = torch.sigmoid(self.attention_conv(x))

        # 🔥 Stable residual attention
        x = x * (1 + attn)

        x = self.relu(x)

        return x
