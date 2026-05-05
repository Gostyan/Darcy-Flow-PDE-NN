"""
Physics utilities for 2D steady-state Darcy Flow.

PDE:   -∇·(a ∇u) = f    on Ω = [0,1]²,   f = 1
BC:     u = 0            on ∂Ω  (Dirichlet)

All tensors are expected on the same device as the model output.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def darcy_pde_residual(
    a: torch.Tensor,
    u: torch.Tensor,
    f: float = 1.0,
) -> torch.Tensor:
    """Finite-difference PDE residual at interior grid points.

    Uses a cell-face discretisation:
        -∇·(a ∇u) ≈  -(div_x + div_y)   where
        div_x[i,j] = (a_{i+1/2,j}*(u_{i+1,j}-u_{i,j}) - a_{i-1/2,j}*(u_{i,j}-u_{i-1,j})) / h²
        div_y[i,j] = similarly in the y-direction

    Parameters
    ----------
    a : (B, H, W)  permeability field
    u : (B, H, W)  predicted pressure field
    f : float      source term (constant = 1 for PDEBench Darcy)

    Returns
    -------
    residual : (B, H-2, W-2)  PDE residual at interior points (should be ~0)
    """
    H, W = a.shape[-2], a.shape[-1]
    dx = 1.0 / (H - 1)
    dy = 1.0 / (W - 1)

    # Face-centred permeability (arithmetic mean)
    a_e = 0.5 * (a[:, :-1, :] + a[:, 1:, :])   # east faces  (B, H-1, W)
    a_n = 0.5 * (a[:, :, :-1] + a[:, :, 1:])   # north faces (B, H,   W-1)

    # Diffusive fluxes
    F_x = a_e * (u[:, 1:, :] - u[:, :-1, :]) / dx   # (B, H-1, W)
    F_y = a_n * (u[:, :, 1:] - u[:, :, :-1]) / dy   # (B, H,   W-1)

    # Divergence (finite difference of fluxes)
    div_x = (F_x[:, 1:, :] - F_x[:, :-1, :]) / dx   # (B, H-2, W)
    div_y = (F_y[:, :, 1:] - F_y[:, :, :-1]) / dy   # (B, H,   W-2)

    # Extract interior points
    div_x_int = div_x[:, :, 1:-1]   # (B, H-2, W-2)
    div_y_int = div_y[:, 1:-1, :]   # (B, H-2, W-2)

    # Residual: -∇·(a∇u) - f  (ideally zero)
    return -(div_x_int + div_y_int) - f


def darcy_boundary_loss(u: torch.Tensor) -> torch.Tensor:
    """L2 loss enforcing u = 0 on all four boundaries.

    Parameters
    ----------
    u : (B, H, W)  predicted pressure field

    Returns
    -------
    scalar loss
    """
    bc = torch.cat([
        u[:, 0, :].reshape(-1),    # top
        u[:, -1, :].reshape(-1),   # bottom
        u[:, :, 0].reshape(-1),    # left
        u[:, :, -1].reshape(-1),   # right
    ])
    return (bc ** 2).mean()
