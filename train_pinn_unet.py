"""
train_pinn_unet.py — PINN-U-Net: U-Net with Darcy PDE physics constraints.

Loss function:
    L = L_data + λ_pde * L_pde + λ_bc * L_bc

where
    L_data  = MSE(û, u_true)
    L_pde   = mean(|−∇·(a ∇û) − 1|²)   at interior grid points
    L_bc    = mean(û² on ∂Ω)

Usage:
    python train_pinn_unet.py
    python train_pinn_unet.py --lambda_pde 0.1 --lambda_bc 10.0 --epochs 200
    python train_pinn_unet.py --pretrain_path pretrain_model/unet/DarcyFlow_Unet.tar
"""

from __future__ import annotations

import json

import argparse
import io
import tarfile
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import UNet2d
from utils import DarcyDataset, compute_metrics, darcy_pde_residual, darcy_boundary_loss


# ---------------------------------------------------------------------------
# Pretrained weight loader
# ---------------------------------------------------------------------------

def _remap_unet_keys(state_dict: dict) -> dict:
    """Remap PDEBench pretrained UNet key names to our UNet2d attribute names."""
    prefix_map = {
        "encoder1.": "enc1.", "encoder2.": "enc2.",
        "encoder3.": "enc3.", "encoder4.": "enc4.",
        "decoder4.": "dec4.", "decoder3.": "dec3.",
        "decoder2.": "dec2.", "decoder1.": "dec1.",
    }
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in prefix_map.items():
            if new_k.startswith(old):
                new_k = new + new_k[len(old):]
                break
        if new_k in ("conv.weight", "conv.bias"):
            new_k = "out_conv." + new_k[5:]
        new_sd[new_k] = v
    return new_sd


def load_pretrain(tar_path: str, model: nn.Module, beta: str = "1.0") -> None:
    p = Path(tar_path)
    if not p.exists():
        print(f"[INFO] Pretrain not found at {tar_path}, training from scratch.")
        return

    # --- plain .pt checkpoint (e.g. output of train_unet.py) ---
    if p.suffix == ".pt":
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)  # bare state_dict or wrapped
        try:
            model.load_state_dict(state_dict)
            print(f"[INFO] Loaded pretrained weights: {p}")
        except RuntimeError as e:
            print(f"[WARN] Pretrained weight mismatch, training from scratch.\n      ({e})")
        return

    # --- PDEBench .tar archive ---
    with tarfile.open(p) as t:
        names = t.getnames()
        match = [n for n in names if f"beta{beta}" in n]
        fname = match[0] if match else names[0]
        raw = t.extractfile(fname).read()
    ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = _remap_unet_keys(state_dict)
    try:
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded pretrained weights: {fname}")
    except RuntimeError as e:
        print(f"[WARN] Pretrained weight mismatch, training from scratch.\n      ({e})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PINN-U-Net on 2D Darcy Flow")
    p.add_argument("--data_path",     type=str,   default="dataset/2D_DarcyFlow_beta1.0_Train.hdf5")
    p.add_argument("--pretrain_path", type=str,   default="pretrain_model/unet/DarcyFlow_Unet.tar",
                   help="Path to pretrained UNet .tar; set to '' to train from scratch")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--init_features", type=int,   default=32)
    # Physics loss weights
    p.add_argument("--lambda_pde",    type=float, default=1e-4,
                   help="Weight for PDE residual loss (raw residual^2 ~400, needs small scale)")
    p.add_argument("--lambda_bc",     type=float, default=10.0,
                   help="Weight for boundary condition loss")
    p.add_argument("--reduced_res",   type=int,   default=1)
    p.add_argument("--num_samples",   type=int,   default=-1)
    p.add_argument("--num_train_samples", type=int, default=-1,
                   help="Limit training samples only; test set stays full (-1=all)")
    p.add_argument("--checkpoint_dir",type=str,   default="checkpoints/pinn_unet")
    p.add_argument("--log_interval",  type=int,   default=10)
    p.add_argument("--patience",      type=int,   default=20,
                   help="Early stopping patience (in log_interval units; 0 = disabled)")
    p.add_argument("--num_workers",   type=int,   default=4,
                   help="DataLoader worker processes (0 = main process only)")
    p.add_argument("--resume",        action="store_true",
                   help="Resume training from checkpoint_dir/last_state.pt")
    # ---- Improvement flags ----
    p.add_argument("--dynamic_lambda",         action="store_true",
                   help="GradNorm: adjust lambda_pde so PDE/data grad ratio -> grad_ratio")
    p.add_argument("--grad_ratio",             type=float, default=0.1)
    p.add_argument("--lambda_update_interval", type=int,   default=5)
    p.add_argument("--freeze_encoder",         action="store_true",
                   help="Freeze enc1-enc4+bottleneck, physics updates decoder only")
    p.add_argument("--normalize_pde",          action="store_true",
                   help="Normalize PDE residual by (H-1)^2 to remove 1/dx^2 amplification")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    print(f"λ_pde={args.lambda_pde}  λ_bc={args.lambda_bc}")

    # ---- data ----
    train_set = DarcyDataset(args.data_path, train=True,
                             reduced_resolution=args.reduced_res,
                             num_samples_max=args.num_samples,
                             num_train_samples=args.num_train_samples)
    test_set  = DarcyDataset(args.data_path, train=False,
                             reduced_resolution=args.reduced_res,
                             num_samples_max=args.num_samples)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False,
                              pin_memory=True, num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0)

    print(f"Train: {len(train_set)} samples | Test: {len(test_set)} samples")
    print(f"Grid : {train_set.H} x {train_set.W}")

    # ---- model ----
    model = UNet2d(in_channels=1, out_channels=1,
                   init_features=args.init_features).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PINN-UNet params: {n_params:,}")

    if args.pretrain_path:
        load_pretrain(args.pretrain_path, model)

    # ---- two-stage: freeze encoder + bottleneck ----
    if args.freeze_encoder:
        for mname in ["enc1", "enc2", "enc3", "enc4", "bottleneck"]:
            for param in getattr(model, mname).parameters():
                param.requires_grad = False
        n_frozen    = sum(p.numel() for mname in ["enc1","enc2","enc3","enc4","bottleneck"]
                          for p in getattr(model, mname).parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[FreezeEncoder] Frozen {n_frozen:,} | Trainable {n_trainable:,}")
    # ---- optimiser (only trainable params) ----
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()

    # ---- checkpoint dir ----
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_rel_l2 = float("inf")
    no_improve  = 0
    start_epoch = 1
    history: dict = {"epoch": [], "train_mse": [], "val_rel_l2": [], "val_pde_residual": [], "val_bc_err": []}

    if args.resume:
        resume_path = ckpt_dir / "last_state.pt"
        if resume_path.exists():
            state = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            best_rel_l2 = state["best_rel_l2"]
            no_improve  = state["no_improve"]
            start_epoch = state["epoch"] + 1
            if "lambda_pde" in state:
                args.lambda_pde = state["lambda_pde"]
                print(f"[Resume] Restored lambda_pde={args.lambda_pde:.3e}")
            hist_path = ckpt_dir / "history.json"
            if hist_path.exists():
                history = json.loads(hist_path.read_text())
            print(f"[Resume] Epoch {start_epoch}, best_rel_L2={best_rel_l2:.4e}")
        else:
            print(f"[Resume] No checkpoint found at {resume_path}, starting fresh.")

    for epoch in range(start_epoch, args.epochs + 1):
        # -- train with per-batch progress bar --
        model.train()
        total_loss = total_ldata = total_lpde = total_lbc = 0.0

        # -- GradNorm: directly compute target lambda_pde in one shot --
        if args.dynamic_lambda and (epoch - 1) % args.lambda_update_interval == 0:
            model.eval()
            a_v, u_v, _ = next(iter(train_loader))
            a_v = a_v.to(device); u_v = u_v.to(device)
            u_p = model(a_v.unsqueeze(1)).squeeze(1)
            l_d = criterion(u_p, u_v)
            res_v = darcy_pde_residual(a_v, u_p.float())
            if args.normalize_pde:
                res_v = res_v / (a_v.shape[-2] - 1) ** 2
            l_p   = (res_v ** 2).mean()
            trainable = [p for p in model.parameters() if p.requires_grad]
            # g_p uses unnormalized L_pde (no lambda), so lambda_target = grad_ratio * n_d / n_p
            g_d = torch.autograd.grad(l_d, trainable, retain_graph=True, allow_unused=True)
            g_p = torch.autograd.grad(l_p,             trainable, allow_unused=True)
            n_d = sum(g.pow(2).sum() for g in g_d if g is not None).sqrt().item()
            n_p = sum(g.pow(2).sum() for g in g_p if g is not None).sqrt().item()
            if n_p > 1e-10 and n_d > 1e-10:
                # Direct target: lambda s.t. |grad(lambda*Lpde)| = grad_ratio * |grad(Ldata)|
                lam_target = args.grad_ratio * n_d / n_p
                # EMA blend to avoid sudden jumps; range [1e-8, 1e-1]
                args.lambda_pde = float(max(1e-5, min(1e-1,
                    0.7 * args.lambda_pde + 0.3 * lam_target)))
            model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:4d}/{args.epochs}", leave=False,
                    unit="batch", dynamic_ncols=True)
        for a, u, _ in pbar:
            a = a.to(device, non_blocking=True)
            u = u.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                u_pred = model(a.unsqueeze(1)).squeeze(1)
                l_data = criterion(u_pred, u)
                residual = darcy_pde_residual(a, u_pred.float())
                if args.normalize_pde:
                    residual = residual / (a.shape[-2] - 1) ** 2
                l_pde = (residual ** 2).mean()
                l_bc  = darcy_boundary_loss(u_pred.float())
            loss  = l_data + args.lambda_pde * l_pde + args.lambda_bc * l_bc
            loss.backward()
            optimizer.step()

            B = a.size(0)
            total_loss  += loss.item()   * B
            total_ldata += l_data.item() * B
            total_lpde  += l_pde.item()  * B
            total_lbc   += l_bc.item()   * B
            pbar.set_postfix(loss=f"{loss.item():.4e}",
                             pde=f"{l_pde.item():.3e}",
                             bc=f"{l_bc.item():.3e}",
                             lam=f"{args.lambda_pde:.2e}")
        pbar.close()

        scheduler.step()
        N_tr = len(train_set)
        train_loss = total_ldata / N_tr   # data-loss per sample (for history)

        # -- evaluate every log_interval epochs --
        if epoch % args.log_interval == 0 or epoch == args.epochs:
            model.eval()
            metrics_accum = {"mse": 0., "rel_l2": 0., "pde_residual": 0., "boundary_err": 0.}
            with torch.no_grad():
                for a, u, _ in test_loader:
                    a = a.to(device, non_blocking=True)
                    u = u.to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        u_pred = model(a.unsqueeze(1)).squeeze(1)
                    m = compute_metrics(a, u_pred.float(), u)
                    for k in metrics_accum:
                        metrics_accum[k] += m[k] * a.size(0)
            n = len(test_set)
            metrics_accum = {k: v / n for k, v in metrics_accum.items()}

            rel_l2 = metrics_accum["rel_l2"]
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"L_total={total_loss/N_tr:.3e}  "
                  f"L_data={total_ldata/N_tr:.3e}  "
                  f"L_pde={total_lpde/N_tr:.3e}  "
                  f"L_bc={total_lbc/N_tr:.3e}  | "
                  f"rel_L2={rel_l2:.4e}  "
                  f"pde_res={metrics_accum['pde_residual']:.4e}  "
                  f"bc_err={metrics_accum['boundary_err']:.4e}  "
                f"lambda_pde={args.lambda_pde:.2e}")

            # ---- history ----
            history["epoch"].append(epoch)
            history["train_mse"].append(train_loss)
            history["val_rel_l2"].append(rel_l2)
            history["val_pde_residual"].append(metrics_accum["pde_residual"])
            history["val_bc_err"].append(metrics_accum["boundary_err"])
            (ckpt_dir / "history.json").write_text(json.dumps(history))
            if rel_l2 < best_rel_l2:
                best_rel_l2 = rel_l2
                no_improve  = 0
                torch.save(model.state_dict(), ckpt_dir / "best_model.pt")
                torch.save({
                    "epoch": epoch, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_rel_l2": best_rel_l2, "no_improve": no_improve,
                    "lambda_pde": args.lambda_pde,
                }, ckpt_dir / "last_state.pt")
                print(f"  --> Best model saved (rel_L2={best_rel_l2:.4e})")
            else:
                no_improve += 1
                if args.patience > 0 and no_improve >= args.patience:
                    print(f"\n[Early Stop] No improvement for {no_improve} evaluations. "
                          f"Best rel_L2={best_rel_l2:.4e}")
                    break

    print(f"\nBest Relative L2: {best_rel_l2:.4e}")
    print(f"Checkpoint saved to: {ckpt_dir / 'best_model.pt'}")


if __name__ == "__main__":
    train(get_args())
