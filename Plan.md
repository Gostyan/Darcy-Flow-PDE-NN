## 1. 问题定义

本课题研究的是 **二维 Darcy 渗流场预测问题**。在多孔介质中，流体流动可以由 Darcy 方程描述。给定二维区域内的渗透率分布，需要预测对应的压力场。

设二维计算区域为：

$$
\Omega=[0,1]\times[0,1]
$$

输入为渗透率场：

$$
a(x,y)
$$

输出为压力场：

$$
u(x,y)
$$

二维稳态 Darcy 控制方程为：

$$
-\nabla \cdot \left(a(x,y)\nabla u(x,y)\right)=f(x,y), \quad (x,y)\in \Omega
$$

其中，$a(x,y)$ 表示介质渗透率，$u(x,y)$ 表示压力场，$f(x,y)$ 表示源项。课程设计中的学习任务可以表述为：

$$
a(x,y) \longrightarrow u(x,y)
$$

也就是让神经网络学习从 **渗透率场图像** 到 **压力场图像** 的映射关系。

本课题的核心问题是：

> 纯数据驱动的 U-Net 虽然可以预测压力场，但预测结果可能不满足 Darcy 方程；因此需要引入 PINN 物理信息约束，使模型同时具有较好的预测精度和物理一致性。

---

## 2. PINN 物理信息约束设计

PINN 的核心思想是：模型训练时不仅最小化预测值和真实值之间的误差，还要让预测结果满足物理控制方程。

对于 U-Net 输出的预测压力场：

$$
\hat{u}(x,y)=U_{\theta}(a(x,y))
$$

根据 Darcy 方程，可以构造 PDE 残差：

$$
r(x,y)=-\nabla \cdot \left(a(x,y)\nabla \hat{u}(x,y)\right)-f(x,y)
$$

如果预测压力场完全满足 Darcy 方程，那么残差 $r(x,y)$ 应该接近 0。

因此，物理残差损失定义为：

$$
L_{pde}=\frac{1}{M}\sum_{j=1}^{M}|r(x_j,y_j)|^2
$$

同时，为了保证预测结果满足边界条件，还可以加入边界损失：

$$
L_{bc}=\frac{1}{B}\sum_{k=1}^{B}|\hat{u}(x_k,y_k)-g(x_k,y_k)|^2
$$

数据拟合损失为：

$$
L_{data}=\frac{1}{N}\sum_{i=1}^{N}|\hat{u}_i-u_i|_2^2
$$

最终联合损失函数为：

$$
L=L_{data}+\lambda_{pde}L_{pde}+\lambda_{bc}L_{bc}
$$

其中：

| 损失项             | 作用                  |
| --------------- | ------------------- |
| $L_{data}$      | 保证预测压力场接近真实压力场      |
| $L_{pde}$       | 保证预测结果满足 Darcy 控制方程 |
| $L_{bc}$        | 保证预测结果满足边界条件        |
| $\lambda_{pde}$ | 控制 PDE 约束强度         |
| $\lambda_{bc}$  | 控制边界约束强度            |

这个部分是课题的关键创新点：
**不是单纯训练 U-Net，而是在 U-Net 的输出结果上继续施加 Darcy 方程约束，使预测场更符合物理规律。**

---

## 3. U-Net 结构设计

U-Net 是本课题的主要预测网络。它适合处理二维场预测任务，因为 Darcy 渗流问题可以看作一个“图像到图像”的回归问题：

$$
\text{渗透率场图像} \rightarrow \text{压力场图像}
$$

U-Net 主要由三部分组成：

### 3.1 编码器

编码器负责提取输入渗透率场的多尺度空间特征。

输入：

$$
a(x,y)\in \mathbb{R}^{H\times W}
$$

经过多层卷积和下采样，得到不同尺度的特征图。浅层特征保留局部空间细节，深层特征包含更大范围的全局结构信息。

---

### 3.2 解码器

解码器负责将低分辨率的高层特征逐步恢复为原始分辨率的压力场。

输出：

$$
\hat{u}(x,y)\in \mathbb{R}^{H\times W}
$$

通过上采样和卷积操作，模型最终生成与输入尺寸相同的二维压力场。

---

### 3.3 跳跃连接

U-Net 的重要结构是跳跃连接。它将编码器中的浅层特征直接传递给解码器对应层，使模型同时利用：

* 浅层的局部细节信息；
* 深层的全局结构信息。

这对于 Darcy 渗流场预测很重要，因为压力场既受局部渗透率变化影响，也受整体边界条件和全局空间结构影响。

整体结构可以表示为：

```text
输入渗透率场 a(x,y)
        ↓
编码器：卷积 + 下采样
        ↓
瓶颈层：全局特征表示
        ↓
解码器：上采样 + 卷积
        ↓
跳跃连接融合多尺度特征
        ↓
输出预测压力场 û(x,y)
        ↓
计算 L_data + L_pde + L_bc
```

因此，本课题中的模型可以表示为：

$$
\hat{u}=U_{\theta}(a)
$$

训练目标是：

$$
\min_{\theta}\left(L_{data}+\lambda_{pde}L_{pde}+\lambda_{bc}L_{bc}\right)
$$

U-Net 负责学习复杂的空间映射关系，PINN 物理约束负责保证预测结果符合 Darcy 方程。两者结合后，模型既具有深度网络的表达能力，又具有物理规律约束。

---


# 可能遇到的问题


## 问题 1：PDE residual 计算和数据中的方程不完全一致

这是非常重要的问题。

如果用公开数据，但不清楚其源项、边界条件、渗透率定义方式，那么你自己写的 PDE residual 可能和数据生成时的真实 PDE 不完全一致。

例如，你以为方程是：

$$
-\nabla \cdot(a\nabla u)=1
$$

但数据中可能使用了不同源项或边界设定。

### 解决方案

课程设计中建议采用两种处理方式之一：

### 方式 A：查清楚数据对应的 PDE 设置

从 PDEBench 或 FNO 数据说明中确认：

* 源项 $f(x,y)$；
* 边界条件；
* 渗透率场定义；
* 是否归一化。

然后严格按它计算 PDE residual。

---

## 问题 2：加入物理损失后，MSE 可能不降反升

PINN 物理约束不一定直接降低数据误差。它更直接改善的是：

* PDE residual；
* 边界一致性；
* 低数据泛化；
* 噪声鲁棒性。

### 解决方案

报告中不要只看 MSE，要同时报告：

| 指标             | 作用      |
| -------------- | ------- |
| MSE            | 预测精度    |
| Relative L2    | 场预测相对误差 |
| PDE residual   | 物理一致性   |
| Boundary error | 边界条件一致性 |
| 低数据测试误差        | 泛化能力    |

如果出现：

```text
PINN-U-Net 的 MSE 稍高，但 PDE residual 明显更低
```

这并不是失败，而是可以解释为：

> 物理约束改善了模型的物理一致性，但与数据拟合精度之间存在权衡。

---


## 问题 3：有限差分计算 PDE residual 时边界不好处理

PDE residual 需要计算：

$$
-\nabla\cdot(a\nabla \hat{u})
$$

如果使用有限差分，会涉及边界点。边界点附近没有完整邻域，容易产生 shape mismatch 或数值错误。

### 解决方案

只在内部网格点计算 PDE residual：

```text
i = 1 到 H-2
j = 1 到 W-2
```

边界点单独用边界损失：

$$
L_{bc}
$$

这样实现最简单，也最稳定。

---

## 问题 4：渗透率场归一化后，物理方程尺度被改变

如果你对输入 `a` 和输出 `u` 做了标准化：

```python
a_norm = (a - mean_a) / std_a
u_norm = (u - mean_u) / std_u
```

那么直接用归一化后的 `a_norm` 和 `u_norm` 计算 PDE residual，可能不再对应原始物理方程。

### 解决方案

有两种选择：

| 方式   | 说明                            |
| ---- | ----------------------------- |
| 简单做法 | 物理 residual 也在归一化空间中计算，只作为正则项 |
| 严谨做法 | 计算 PDE residual 前反归一化到物理量尺度   |

---

## 问题 5：U-Net 输出边界不满足 Dirichlet 条件

如果边界条件为：

$$
u=0,\quad \partial\Omega
$$

普通 U-Net 可能预测出非零边界。

### 解决方案

可以使用两种方法：

### 方法 A：加入边界损失

$$
L_{bc}
$$

直接约束四条边界。

### 方法 B：硬约束边界

网络输出后强制：

```python
pred[:, :, 0, :] = 0
pred[:, :, -1, :] = 0
pred[:, :, :, 0] = 0
pred[:, :, :, -1] = 0
```

课程设计建议用方法 A，因为它更符合 PINN 思想。

---



---

# 五、最终建议

你的课程设计最稳路线是：

```text
数据：Zenodo Darcy Flow 或 PDEBench 2D Darcy Flow
模型：CNN、U-Net、PINN-U-Net
核心创新：在 U-Net 输出上加入 Darcy PDE residual
主要实验：预测误差 + PDE residual + 低数据泛化
```

最小可完成版本：

| 模块             | 是否必须 |
| -------------- | ---- |
| Darcy 数据读取     | 必须   |
| U-Net baseline | 必须   |
| PINN baseline  | 必须   |
| **PINN-U-Net** | 必须   |
| FNO baseline   | 可选   |


[1]: https://github.com/pdebench/PDEBench?utm_source=chatgpt.com "PDEBench: An Extensive Benchmark for Scientific Machine ..."
[2]: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-2986&utm_source=chatgpt.com "PDEBench Datasets - DaRUS - Universität Stuttgart"
[3]: https://zenodo.org/records/10994262?utm_source=chatgpt.com "Darcy Flow Dataset"
[4]: https://darus.uni-stuttgart.de/dataset.xhtml?persistentId=doi%3A10.18419%2Fdarus-2987&utm_source=chatgpt.com "PDEBench Pretrained Models - DaRUS - Universität Stuttgart"
[5]: https://github.com/scaomath/fourier_neural_operator/?utm_source=chatgpt.com "Fourier Neural Operator"
[6]: https://neuraloperator.github.io/dev/auto_examples/training/plot_incremental_FNO_darcy.html?utm_source=chatgpt.com "Training an FNO with incremental meta-learning"
[7]: https://proceedings.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Supplemental-Datasets_and_Benchmarks.pdf?utm_source=chatgpt.com "An Extensive Benchmark for Scientific Machine Learning. ..."
