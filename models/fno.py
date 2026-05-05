"""
FNO2d — Fourier Neural Operator for 2D problems.
Adapted from PDEBench (https://github.com/pdebench/PDEBench), MIT License.
Only the 2D variant is kept; unused 1D/3D classes are removed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class SpectralConv2d(nn.Module):
    """2D Fourier layer: FFT → linear transform → IFFT."""

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def _compl_mul2d(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_ft = torch.fft.rfft2(x)

        out_ft = torch.zeros(B, self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self._compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class FNO2d(nn.Module):
    """
    Fourier Neural Operator for 2D steady/time-dependent problems.

    For 2D Darcy Flow (static):
        initial_step = 1, num_channels = 1
        forward(x, grid):
            x    : (B, H, W, initial_step * num_channels)  — input field(s)
            grid : (H, W, 2)                               — spatial coordinates
        returns: (B, H, W, num_channels, 1)
    """

    def __init__(
        self,
        num_channels: int = 1,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        initial_step: int = 1,
    ):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad domain if non-periodic

        # input: initial_step * num_channels + 2 (grid coords)
        self.fc0 = nn.Linear(initial_step * num_channels + 2, width)

        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """
        x    : (B, H, W, C)  where C = initial_step * num_channels
        grid : (H, W, 2)
        """
        # expand grid to batch dimension if needed
        if grid.dim() == 3:
            grid = grid.unsqueeze(0).expand(x.size(0), -1, -1, -1)

        x = torch.cat((x, grid), dim=-1)   # (B, H, W, C+2)
        x = self.fc0(x)                    # (B, H, W, width)
        x = x.permute(0, 3, 1, 2)          # (B, width, H, W)

        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x);  x2 = self.w0(x);  x = F.gelu(x1 + x2)
        x1 = self.conv1(x);  x2 = self.w1(x);  x = F.gelu(x1 + x2)
        x1 = self.conv2(x);  x2 = self.w2(x);  x = F.gelu(x1 + x2)
        x1 = self.conv3(x);  x2 = self.w3(x);  x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]  # unpad
        x = x.permute(0, 2, 3, 1)   # (B, H, W, width)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)              # (B, H, W, num_channels)
        return x.unsqueeze(-2)       # (B, H, W, num_channels, 1)
