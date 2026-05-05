"""
evaluate.py — Compare U-Net, FNO, and PINN-U-Net on the Darcy Flow test set.

Loads the best checkpoint for each model, runs evaluation, prints a summary
table, and saves prediction visualisations to results/.

Usage:
    python evaluate.py
    python evaluate.py --reduced_res 2 --num_samples 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from models import FNO2d, UNet2d
from utils import DarcyDataset, compute_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Darcy Flow models")
    p.add_argument("--data_path",   type=str, default="dataset/2D_DarcyFlow_beta1.0_Train.hdf5")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--reduced_res", type=int, default=1)
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--results_dir", type=str, default="results")
    # checkpoint paths
    p.add_argument("--ckpt_unet",      type=str, default="checkpoints/unet/best_model.pt")
    p.add_argument("--ckpt_fno",       type=str, default="checkpoints/fno/best_model.pt")
    p.add_argument("--ckpt_pinn_unet", type=str, default="checkpoints/pinn_unet/best_model.pt")
    # model hyper-parameters (must match training)
    p.add_argument("--init_features",  type=int, default=32)
    p.add_argument("--fno_modes",      type=int, default=12)
    p.add_argument("--fno_width",      type=int, default=32)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(model, loader, device, model_type: str = "unet") -> dict[str, float]:
    """Accumulate metrics over the full test set."""
    accum = {"mse": 0., "rel_l2": 0., "pde_residual": 0., "boundary_err": 0.}
    n_total = 0

    for a, u, grid in loader:
        a    = a.to(device)
        u    = u.to(device)
        grid = grid.to(device)

        if model_type == "unet":
            u_pred = model(a.unsqueeze(1)).squeeze(1)
        elif model_type == "fno":
            grid0  = grid[0] if grid.dim() == 4 else grid
            u_pred = model(a.unsqueeze(-1), grid0).squeeze(-1).squeeze(-1)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        m = compute_metrics(a, u_pred, u)
        B = a.size(0)
        for k in accum:
            accum[k] += m[k] * B
        n_total += B

    return {k: v / n_total for k, v in accum.items()}


def load_unet(ckpt_path: str, init_features: int, device) -> UNet2d | None:
    if not Path(ckpt_path).exists():
        print(f"[skip] UNet checkpoint not found: {ckpt_path}")
        return None
    model = UNet2d(in_channels=1, out_channels=1, init_features=init_features).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def load_fno(ckpt_path: str, modes: int, width: int, device) -> FNO2d | None:
    if not Path(ckpt_path).exists():
        print(f"[skip] FNO checkpoint not found: {ckpt_path}")
        return None
    model = FNO2d(num_channels=1, modes1=modes, modes2=modes,
                  width=width, initial_step=1).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def save_comparison_figure(
    models: dict,  # name -> (model, model_type)
    sample_a: torch.Tensor,   # (H, W)
    sample_u: torch.Tensor,   # (H, W)
    sample_grid: torch.Tensor,
    device,
    save_path: Path,
):
    """Save a side-by-side comparison of ground truth vs each model's prediction."""
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models + 2, figsize=(4 * (n_models + 2), 4))

    a_  = sample_a.to(device).unsqueeze(0)   # (1, H, W)
    u_  = sample_u.numpy()

    vmin, vmax = float(sample_u.min()), float(sample_u.max())

    # permeability
    im = axes[0].imshow(sample_a.numpy(), origin="lower", cmap="viridis")
    axes[0].set_title("Permeability $a$")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    # ground truth
    im = axes[1].imshow(u_, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("Ground Truth $u$")
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    for ax, (name, (model, mtype)) in zip(axes[2:], models.items()):
        if model is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(name)
            continue
        grid0 = sample_grid.to(device)
        if mtype == "unet":
            u_pred = model(a_).squeeze().cpu().numpy()
        else:
            u_pred = model(a_.unsqueeze(-1), grid0).squeeze().cpu().numpy()
        err = np.abs(u_pred - u_)
        im = ax.imshow(u_pred, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(f"{name}\nMAE={err.mean():.4f}")
        plt.colorbar(im, ax=ax, fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- data ----
    test_set = DarcyDataset(args.data_path, train=False,
                            reduced_resolution=args.reduced_res,
                            num_samples_max=args.num_samples)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    print(f"Test set: {len(test_set)} samples  ({test_set.H}×{test_set.W})\n")

    # ---- load models ----
    unet      = load_unet(args.ckpt_unet,      args.init_features, device)
    fno       = load_fno(args.ckpt_fno,        args.fno_modes, args.fno_width, device)
    pinn_unet = load_unet(args.ckpt_pinn_unet, args.init_features, device)

    model_registry = {
        "U-Net":      (unet,      "unet"),
        "FNO":        (fno,       "fno"),
        "PINN-U-Net": (pinn_unet, "unet"),
    }

    # ---- evaluate ----
    header = f"{'Model':<14} {'MSE':>12} {'Rel-L2':>12} {'PDE-Res':>12} {'BC-Err':>12}"
    print(header)
    print("-" * len(header))

    for name, (model, mtype) in model_registry.items():
        if model is None:
            print(f"{name:<14}   (checkpoint not found)")
            continue
        metrics = evaluate_model(model, test_loader, device, mtype)
        print(f"{name:<14} "
              f"{metrics['mse']:>12.4e} "
              f"{metrics['rel_l2']:>12.4e} "
              f"{metrics['pde_residual']:>12.4e} "
              f"{metrics['boundary_err']:>12.4e}")

    # ---- visualisation ----
    sample_a, sample_u, sample_grid = test_set[0]
    active_models = {k: v for k, v in model_registry.items() if v[0] is not None}
    if active_models:
        save_comparison_figure(
            active_models, sample_a, sample_u, sample_grid, device,
            results_dir / "comparison.png",
        )


if __name__ == "__main__":
    main(get_args())
