# Rryyz：深度学习框架

> 从张量计算、自动求导到模型训练，亲手构建一套深度学习框架。

Rryyz 是一个从零构建的深度学习框架，将深度学习训练链路拆解为清晰、可读、可扩展的核心组件。它以 NumPy 为基础，提供动态计算图、自动微分、神经网络层、模型封装、优化器、数据集与数据加载器，并支持通过 CuPy 使用 GPU 进行计算。

## 核心能力

| 模块 | 能力 |
| --- | --- |
| 自动求导 | 基于 `Variable` 与 `Function` 构建动态计算图，支持前向计算、反向传播和梯度管理 |
| 函数算子 | 加减乘除、幂运算、指数、三角函数、激活函数、矩阵乘法、变形、转置、求和等 |
| 神经网络 | `Layer`、`Linear`、`Model`、`MLP` 等基础组件，可组合出完整模型 |
| 优化器 | 提供 SGD、Momentum SGD，以及优化器 Hook 扩展机制 |
| 数据管线 | `Dataset`、`DataLoader`、`SeqDataLoader`，支持批处理、打乱和设备迁移 |
| 数据集 | 提供 Spiral、MNIST、CIFAR-10、CIFAR-100、ImageNet、SinCurve、Shakespear 等数据集接口 |
| 预处理 | 提供组合变换、缩放、裁剪、数组转换、归一化、展平和随机翻转等操作 |
| 计算设备 | 默认使用 NumPy；检测到 CuPy 时可切换到 GPU 计算 |
| 计算图可视化 | 支持将模型计算图导出为图像，帮助理解前向与反向传播结构 |

## 框架结构

```text
Rryyz/
├── core.py           # Variable、Function、Parameter、Config 与自动求导核心
├── core_simple.py    # 便于理解的简化核心实现
├── functions.py      # 函数算子、损失函数与常用神经网络运算
├── layers.py         # Layer、Linear 等网络层
├── models.py         # Model、MLP 等模型封装
├── optimizers.py     # SGD、MomentumSGD 与优化器基类
├── datasets.py       # 数据集接口与常用数据集
├── dataloaders.py    # 批处理、打乱与序列数据加载
├── transforms.py     # 数据预处理与变换
├── cuda.py           # NumPy / CuPy 设备支持
└── utils.py          # 工具函数与计算图可视化
```

## 快速开始

安装基础依赖：

```bash
pip install numpy matplotlib
```

下面的例子使用 Rryyz 的自动求导能力完成一个线性回归训练过程：

```python
import numpy as np
from Rryyz import Variable
import Rryyz.functions as F

x = Variable(np.random.rand(100, 1))
t = Variable(5 + 2 * x.data + np.random.rand(100, 1))

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))

for _ in range(100):
    y = F.matmul(x, W) + b
    loss = F.sum((y - t) ** 2) / len(t)

    W.cleargrad()
    b.cleargrad()
    loss.backward()

    W.data -= 0.1 * W.grad.data
    b.data -= 0.1 * b.grad.data

print(loss)
```

也可以直接运行仓库中的示例脚本：

```bash
python 42线性回归.py
python 43曲线回归.py
python 48训练多分类.py
```

## 从自动求导到模型训练

使用模型和优化器时，训练流程可以进一步抽象为：

```python
import Rryyz.functions as F
from Rryyz.models import MLP
from Rryyz import optimizers

model = MLP((1000, 10))
optimizer = optimizers.SGD().setup(model)

y = model(x)
loss = F.softmax_cross_entropy(y, target)
model.cleargrads()
loss.backward()
optimizer.update()
```

## GPU 计算

安装与本机 CUDA 环境匹配的 CuPy 后，Rryyz 可以将数据、模型和训练过程迁移到 GPU：

```bash
pip install cupy-cuda12x
python 52GPU训练MINIST.py
```

实际使用时请根据本机 CUDA 版本选择对应的 CuPy 安装包。

## 示例索引

根目录下的编号脚本展示了从基础数值计算到模型训练的完整使用路径：

| 示例 | 内容 |
| --- | --- |
| `42线性回归.py` | 使用自动求导完成线性回归 |
| `43曲线回归.py` | 多项式曲线拟合 |
| `48训练多分类.py` | 多分类模型训练 |
| `50训练测试多分类.py` | 训练集与测试集分离 |
| `51MINIST.py` | MNIST 手写数字训练 |
| `52GPU训练MINIST.py` | GPU 加速 MNIST 训练 |
| `53模型的保存与加载.py` | 模型权重保存与加载 |

## 项目定位

Rryyz 关注深度学习框架内部机制的清晰表达：计算图如何建立，梯度如何传播，参数如何组织，模型如何训练，以及数据如何进入整个流程。它既可以作为理解深度学习底层原理的框架，也可以作为实验网络结构、优化方法和数据管线的轻量工具。
