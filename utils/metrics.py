"""
Evaluation metrics for 2D Darcy Flow prediction.

Reported metrics
----------------
mse          : mean squared error
rel_l2       : relative L2 error  (‖u_pred - u_true‖₂ / ‖u_true‖₂)
pde_residual : mean absolute PDE residual  |−∇·(a∇û) − 1|
boundary_err : mean absolute boundary value |û|_∂Ω
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .physics import darcy_boundary_loss, darcy_pde_residual


@torch.no_grad()
def compute_metrics(
    a: torch.Tensor,
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
) -> dict[str, float]:
    """Compute all metrics for a batch of Darcy Flow predictions.

    Parameters
    ----------
    a      : (B, H, W)  permeability
    u_pred : (B, H, W)  predicted pressure
    u_true : (B, H, W)  ground-truth pressure

    Returns
    -------
    dict with keys: mse, rel_l2, pde_residual, boundary_err
    """
    # MSE
    mse = F.mse_loss(u_pred, u_true).item()

    # Relative L2
    diff_norm = torch.norm(u_pred - u_true, p=2, dim=(-2, -1))
    true_norm = torch.norm(u_true,          p=2, dim=(-2, -1)) + 1e-8
    rel_l2 = (diff_norm / true_norm).mean().item()

    # PDE residual (MAE at interior points)
    res = darcy_pde_residual(a, u_pred)
    pde_res = res.abs().mean().item()

    # Boundary error
    bc_err = darcy_boundary_loss(u_pred).item() ** 0.5   # RMSE on boundary

    return {
        "mse":          mse,
        "rel_l2":       rel_l2,
        "pde_residual": pde_res,
        "boundary_err": bc_err,
    }
