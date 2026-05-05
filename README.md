# Darcy Flow PDE — Neural Operator Benchmark

用神经网络求解二维稳态 Darcy 流方程：

$$-\nabla \cdot (a \nabla u) = 1 \quad \text{on } \Omega=[0,1]^2, \quad u=0 \text{ on } \partial\Omega$$

实现了三种方法：**U-Net**、**FNO**（Fourier Neural Operator）、**PINN-UNet**（物理约束 U-Net）。

---

## 环境依赖

```bash
# 创建并激活 conda 环境（Python 3.10+）
conda create -n darcy python=3.10 -y
conda activate darcy

# 安装 PyTorch（示例：CUDA 12.6）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 安装其余依赖
pip install h5py numpy matplotlib tqdm
```

---

## 数据集与预训练权重

数据集和预训练权重来自 [PDEBench](https://github.com/pdebench/PDEBench)，不包含在本仓库中，需手动下载后放置：

```
dataset/
    2D_DarcyFlow_beta1.0_Train.hdf5   ← PDEBench Darcy Flow 数据集
pretrain_model/
    unet/DarcyFlow_Unet.tar           ← PDEBench UNet 预训练权重
    fno/DarcyFlow_FNO.tar             ← PDEBench FNO 预训练权重
```

---

## 训练

所有脚本默认自动加载对应预训练权重（找不到则从头训练），并带早停机制。

```bash
# U-Net 基线
python train_unet.py

# FNO
python train_fno.py

# PINN-UNet（物理约束）
python train_pinn_unet.py

# 两阶段训练：先用 U-Net 收敛，再用 PINN 微调
python train_unet.py --epochs 100
python train_pinn_unet.py --pretrain_path checkpoints/unet/best_model.pt --lr 1e-4
```

常用参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 100 | 训练轮数 |
| `--batch_size` | 8 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--patience` | 20 | 早停耐心值（评估次数）|
| `--pretrain_path` | 自动 | 预训练权重路径（`.tar` 或 `.pt`）|
| `--lambda_pde` | 0.1 | PDE 残差损失权重（仅 PINN）|
| `--lambda_bc` | 10.0 | 边界条件损失权重（仅 PINN）|

训练完成后检查点保存在 `checkpoints/<model>/best_model.pt`。

---

## 评估

```bash
python evaluate.py
```

对三个模型统一评估，输出以下指标并保存对比图 `results/comparison.png`：

- **rel_L2**：相对 L2 误差
- **MSE**：均方误差
- **pde_residual**：PDE 残差 MAE（衡量物理一致性）
- **boundary_err**：边界条件误差

---

## 项目结构

```
├── models/
│   ├── fno.py          # Fourier Neural Operator (FNO2d)
│   └── unet.py         # U-Net (UNet2d)
├── utils/
│   ├── dataset.py      # HDF5 数据加载
│   ├── physics.py      # Darcy PDE 残差 / 边界损失
│   └── metrics.py      # 评估指标
├── train_unet.py       # U-Net 训练
├── train_fno.py        # FNO 训练
├── train_pinn_unet.py  # PINN-UNet 训练
├── evaluate.py         # 统一评估 + 可视化
└── requirements.txt
```
