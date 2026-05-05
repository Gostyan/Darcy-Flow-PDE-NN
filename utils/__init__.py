from .dataset import DarcyDataset
from .physics import darcy_pde_residual, darcy_boundary_loss
from .metrics import compute_metrics

__all__ = [
    "DarcyDataset",
    "darcy_pde_residual",
    "darcy_boundary_loss",
    "compute_metrics",
]
