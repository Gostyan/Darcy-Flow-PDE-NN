"""
UNet2d — 2D U-Net for image-to-image regression.
Adapted from PDEBench (https://github.com/pdebench/PDEBench), MIT License.
Only the 2D variant is kept.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


class UNet2d(nn.Module):
    """
    2D U-Net with 4 encoder/decoder levels and skip connections.

    For 2D Darcy Flow:
        in_channels  = 1  (permeability field a)
        out_channels = 1  (pressure field u)
        forward(x): x shape (B, 1, H, W) → output (B, 1, H, W)

    Note: H and W must be divisible by 16 (4 pooling layers of stride 2).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, init_features: int = 32):
        super().__init__()
        f = init_features

        # Encoder
        self.enc1 = UNet2d._block(in_channels, f,      name="enc1")
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = UNet2d._block(f,      f * 2,  name="enc2")
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = UNet2d._block(f * 2,  f * 4,  name="enc3")
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = UNet2d._block(f * 4,  f * 8,  name="enc4")
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = UNet2d._block(f * 8, f * 16, name="bottleneck")

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(f * 16, f * 8,  2, stride=2)
        self.dec4    = UNet2d._block(f * 16, f * 8,  name="dec4")
        self.upconv3 = nn.ConvTranspose2d(f * 8,  f * 4,  2, stride=2)
        self.dec3    = UNet2d._block(f * 8,  f * 4,  name="dec3")
        self.upconv2 = nn.ConvTranspose2d(f * 4,  f * 2,  2, stride=2)
        self.dec2    = UNet2d._block(f * 4,  f * 2,  name="dec2")
        self.upconv1 = nn.ConvTranspose2d(f * 2,  f,      2, stride=2)
        self.dec1    = UNet2d._block(f * 2,  f,      name="dec1")

        self.out_conv = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        bn = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.upconv4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], dim=1))

        return self.out_conv(d1)

    @staticmethod
    def _block(in_ch: int, features: int, name: str) -> nn.Sequential:
        return nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv2d(in_ch,    features, 3, padding=1, bias=False)),
            (name + "norm1", nn.BatchNorm2d(features)),
            (name + "act1",  nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv2d(features, features, 3, padding=1, bias=False)),
            (name + "norm2", nn.BatchNorm2d(features)),
            (name + "act2",  nn.ReLU(inplace=True)),
        ]))
