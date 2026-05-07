"""
DarcyDataset — HDF5 dataloader for 2D steady-state Darcy Flow.

Data keys (PDEBench format):
    "nu"     : permeability field a(x,y),  shape (N, H, W)
    "tensor" : pressure field u(x,y),      shape (N, 1, H, W)

Each sample returns:
    a    : (H, W)   — permeability
    u    : (H, W)   — pressure (label)
    grid : (H, W, 2) — spatial coordinates in [0,1]^2  (shared across samples)
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DarcyDataset(Dataset):
    """2D Darcy Flow dataset.

    Parameters
    ----------
    filepath : str | Path
        Path to the HDF5 file (e.g. ``dataset/2D_DarcyFlow_beta1.0_Train.hdf5``).
    train : bool
        If True load the training split, else the test split.
    test_ratio : float
        Fraction of samples held out for testing.
    reduced_resolution : int
        Spatial sub-sampling factor (1 = full resolution).
    num_samples_max : int
        Cap on the total number of samples used (-1 = all).
    num_train_samples : int
        Cap on training samples only (-1 = all). Test set is always full-sized.
        When set, num_samples_max is ignored for the test split.
    """

    def __init__(
        self,
        filepath: str | Path,
        train: bool = True,
        test_ratio: float = 0.1,
        reduced_resolution: int = 1,
        num_samples_max: int = -1,
        num_train_samples: int = -1,
    ):
        filepath = Path(filepath)
        assert filepath.exists(), f"Dataset not found: {filepath}"

        with h5py.File(filepath, "r") as f:
            # permeability: (N, H, W)
            a = np.array(f["nu"], dtype=np.float32)
            # pressure: (N, 1, H, W) → (N, H, W)
            u = np.array(f["tensor"], dtype=np.float32)[:, 0, :, :]

        # spatial sub-sampling
        a = a[:, ::reduced_resolution, ::reduced_resolution]
        u = u[:, ::reduced_resolution, ::reduced_resolution]

        N = a.shape[0]
        if num_samples_max > 0:
            N = min(N, num_samples_max)

        # Full-data split (test set always uses this boundary)
        split_full = int(N * (1 - test_ratio))

        if train:
            a_train, u_train = a[:split_full], u[:split_full]
            # Optionally restrict training samples further
            if num_train_samples > 0:
                a_train = a_train[:num_train_samples]
                u_train = u_train[:num_train_samples]
            self.a = torch.from_numpy(a_train)
            self.u = torch.from_numpy(u_train)
        else:
            self.a = torch.from_numpy(a[split_full:])
            self.u = torch.from_numpy(u[split_full:])

        # spatial grid [0, 1]^2
        H, W = self.a.shape[-2], self.a.shape[-1]
        xs = torch.linspace(0.0, 1.0, H)
        ys = torch.linspace(0.0, 1.0, W)
        X, Y = torch.meshgrid(xs, ys, indexing="ij")
        self.grid = torch.stack([X, Y], dim=-1)  # (H, W, 2)

        self.H, self.W = H, W

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.a)

    def __getitem__(self, idx: int):
        return self.a[idx], self.u[idx], self.grid
