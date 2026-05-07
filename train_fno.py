"""
train_fno.py — FNO baseline for 2D Darcy Flow prediction.

Usage:
    python train_fno.py
    python train_fno.py --epochs 200 --batch_size 8 --modes 12 --width 32
    python train_fno.py --pretrain_path pretrain_model/fno/DarcyFlow_FNO.tar
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

from models import FNO2d
from utils import DarcyDataset, compute_metrics


# ---------------------------------------------------------------------------
# Pretrained weight loader
# ---------------------------------------------------------------------------

def detect_fno_width(tar_path: str, beta: str = "1.0") -> int | None:
    """Infer FNO width from a pretrained checkpoint (.pt or .tar) before building the model."""
    p = Path(tar_path)
    if not p.exists():
        return None
    try:
        if p.suffix == ".pt":
            ckpt = torch.load(p, map_location="cpu", weights_only=True)
            sd = ckpt.get("model_state_dict", ckpt)
            return int(sd["fc0.weight"].shape[0])
        with tarfile.open(p) as t:
            names = t.getnames()
            match = [n for n in names if f"beta{beta}" in n]
            fname = match[0] if match else names[0]
            raw = t.extractfile(fname).read()
        ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        return int(state_dict["fc0.weight"].shape[0])
    except Exception:
        return None


def load_pretrain(tar_path: str, model: nn.Module, beta: str = "1.0") -> None:
    p = Path(tar_path)
    if not p.exists():
        print(f"[INFO] Pretrain not found at {tar_path}, training from scratch.")
        return
    if p.suffix == ".pt":
        ckpt = torch.load(p, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(state_dict)
            print(f"[INFO] Resumed from checkpoint: {p}")
        except RuntimeError as e:
            print(f"[WARN] Resume failed, training from scratch.\n      ({e})")
        return
    with tarfile.open(p) as t:
        names = t.getnames()
        match = [n for n in names if f"beta{beta}" in n]
        fname = match[0] if match else names[0]
        raw = t.extractfile(fname).read()
    ckpt = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    try:
        model.load_state_dict(state_dict)
        print(f"[INFO] Loaded pretrained weights: {fname}")
    except RuntimeError as e:
        print(f"[WARN] Pretrained weight mismatch, training from scratch.\n      ({e})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train FNO on 2D Darcy Flow")
    p.add_argument("--data_path",     type=str,   default="dataset/2D_DarcyFlow_beta1.0_Train.hdf5")
    p.add_argument("--pretrain_path", type=str,   default="pretrain_model/fno/DarcyFlow_FNO.tar",
                   help="Path to pretrained .tar; set to '' to train from scratch")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch_size",    type=int,   default=8)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--modes",         type=int,   default=12,
                   help="Number of Fourier modes")
    p.add_argument("--width",         type=int,   default=32,
                   help="FNO channel width")
    p.add_argument("--reduced_res",   type=int,   default=1)
    p.add_argument("--num_samples",   type=int,   default=-1)
    p.add_argument("--num_train_samples", type=int, default=-1,
                   help="Limit training samples only; test set stays full (-1=all)")
    p.add_argument("--checkpoint_dir",type=str,   default="checkpoints/fno")
    p.add_argument("--log_interval",  type=int,   default=10)
    p.add_argument("--patience",      type=int,   default=20,
                   help="Early stopping patience (in log_interval units; 0 = disabled)")
    p.add_argument("--num_workers",   type=int,   default=4,
                   help="DataLoader worker processes (0 = main process only)")
    p.add_argument("--resume",        action="store_true",
                   help="Resume training from checkpoint_dir/last_state.pt")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = device.type == "cuda"

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
    width = args.width
    if args.pretrain_path:
        detected = detect_fno_width(args.pretrain_path)
        if detected is not None and detected != width:
            print(f"[INFO] Pretrained width={detected}, overriding --width {width} → {detected}")
            width = detected
    model = FNO2d(num_channels=1, modes1=args.modes, modes2=args.modes,
                  width=width, initial_step=1).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"FNO2d params: {n_params:,}")

    if args.pretrain_path:
        load_pretrain(args.pretrain_path, model)

    # ---- optimiser ----
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
            hist_path = ckpt_dir / "history.json"
            if hist_path.exists():
                history = json.loads(hist_path.read_text())
            print(f"[Resume] Epoch {start_epoch}, best_rel_L2={best_rel_l2:.4e}")
        else:
            print(f"[Resume] No checkpoint found at {resume_path}, starting fresh.")

    for epoch in range(start_epoch, args.epochs + 1):
        # -- train with per-batch progress bar --
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch:4d}/{args.epochs}", leave=False,
                    unit="batch", dynamic_ncols=True)
        for a, u, grid in pbar:
            a    = a.to(device, non_blocking=True)
            u    = u.to(device, non_blocking=True)
            grid = grid.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                a_in  = a.unsqueeze(-1)
                grid0 = grid[0] if grid.dim() == 4 else grid
                u_pred = model(a_in, grid0).squeeze(-1).squeeze(-1)
                loss = criterion(u_pred, u)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * a.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4e}")
        pbar.close()

        scheduler.step()
        train_loss = total_loss / len(train_set)

        # -- evaluate every log_interval epochs --
        if epoch % args.log_interval == 0 or epoch == args.epochs:
            model.eval()
            metrics_accum = {"mse": 0., "rel_l2": 0., "pde_residual": 0., "boundary_err": 0.}
            with torch.no_grad():
                for a, u, grid in test_loader:
                    a    = a.to(device, non_blocking=True)
                    u    = u.to(device, non_blocking=True)
                    grid = grid.to(device, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        grid0 = grid[0] if grid.dim() == 4 else grid
                        u_pred = model(a.unsqueeze(-1), grid0).squeeze(-1).squeeze(-1)
                    m = compute_metrics(a, u_pred.float(), u)
                    for k in metrics_accum:
                        metrics_accum[k] += m[k] * a.size(0)
            n = len(test_set)
            metrics_accum = {k: v / n for k, v in metrics_accum.items()}

            rel_l2 = metrics_accum["rel_l2"]
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train_mse={train_loss:.4e}  "
                  f"rel_L2={rel_l2:.4e}  "
                  f"pde_res={metrics_accum['pde_residual']:.4e}  "
                  f"bc_err={metrics_accum['boundary_err']:.4e}")

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
