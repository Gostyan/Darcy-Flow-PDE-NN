"""
Evaluation metrics for 2D Darcy Flow prediction.

Reported metrics
----------------
mse              : Mean Squared Error
rmse             : Root Mean Squared Error  = sqrt(MSE)
mae              : Mean Absolute Error
rel_l2           : Relative L2 error  (||u_pred - u_true||_2 / ||u_true||_2)
max_err          : Maximum absolute error (L-inf norm)
pde_residual     : Mean absolute PDE residual  |-div(a*grad(u)) - 1|  (MAE, interior)
pde_residual_rmse: Root-mean-square PDE residual (RMSE, interior)
boundary_err     : Boundary RMSE  ||u||_dOmega
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
    dict with keys: mse, rmse, mae, rel_l2, max_err,
                    pde_residual, pde_residual_rmse, boundary_err
    """
    diff = u_pred - u_true

    # MSE / RMSE / MAE / Max-Error
    mse     = F.mse_loss(u_pred, u_true).item()
    rmse    = mse ** 0.5
    mae     = diff.abs().mean().item()
    max_err = diff.abs().max().item()

    # Relative L2  (||delta_u||_2 / ||u_true||_2, averaged over batch)
    diff_norm = torch.norm(diff,   p=2, dim=(-2, -1))
    true_norm = torch.norm(u_true, p=2, dim=(-2, -1)) + 1e-8
    rel_l2    = (diff_norm / true_norm).mean().item()

    # PDE residual at interior points
    res              = darcy_pde_residual(a, u_pred)
    pde_res_mae      = res.abs().mean().item()
    pde_res_rmse     = res.pow(2).mean().sqrt().item()

    # Boundary RMSE  (u_pred should be 0 on boundary)
    bc_err = darcy_boundary_loss(u_pred).item() ** 0.5   # sqrt(MSE on boundary)

    return {
        "mse":               mse,
        "rmse":              rmse,
        "mae":               mae,
        "rel_l2":            rel_l2,
        "max_err":           max_err,
        "pde_residual":      pde_res_mae,
        "pde_residual_rmse": pde_res_rmse,
        "boundary_err":      bc_err,
    }
