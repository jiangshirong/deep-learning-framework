# 深度学习框架（Rryyz）

一个从零实现的迷你深度学习框架，只依赖 NumPy（有 GPU 时走 CuPy），外加一组跟着步骤推进的学习脚本。

## 框架本体

`Rryyz/` 是框架核心，实现了自动求导与训练所需的基本构件：

| 模块 | 内容 |
| --- | --- |
| `core.py` | `Variable`、`Function`、`Parameter`、`Config`，自动求导与反向传播的骨架，含 `Add` / `Mul` / `Sub` / `Div` / `Pow` / `Neg` 等基础算子 |
| `core_simple.py` | 简化版核心，去掉内存优化后便于对照阅读 |
| `functions.py` | 各类函数：`Square`、`Exp`、`Sin`、`Cos`、`Tanh`、`Reshape`、`Transpose`、`Sum`、`MatMul`、`Linear`、`Sigmoid`、`MeanSquaredError` 等 |
| `layers.py` | `Layer` 基类与 `Linear` 等层 |
| `models.py` | `Model` 基类与 `MLP` |
| `optimizers.py` | `Optimizer` 基类、`SGD`、`MomentumSGD` |
| `datasets.py` | `Dataset` 基类，以及 `Spiral`、`MNIST`、`CIFAR10`、`CIFAR100`、`ImageNet`、`SinCurve`、`Shakespear` |
| `dataloaders.py` | `DataLoader`、`SeqDataLoader`，负责批处理与打乱 |
| `transforms.py` | 数据预处理：`Compose`、`Convert`、`Resize`、`CenterCrop`、`ToArray`、`ToPIL`、`RandomHorizontalFlip`、`Normalize`、`Flatten` 等 |
| `utils.py` | 工具函数 |
| `cuda.py` | GPU（CuPy）相关支持 |

## 学习脚本

根目录下的脚本按顺序推进，编号沿用教材的 step 编号：

| 脚本 | 内容 |
| --- | --- |
| `42线性回归.py` | 线性回归，从零搭出前向与反向 |
| `43曲线回归.py` | 多项式曲线拟合 |
| `48训练多分类.py` | 多分类模型的训练 |
| `50训练测试多分类.py` | 多分类的训练与测试分离 |
| `51MINIST.py` | MNIST 手写数字训练 |
| `52GPU训练MINIST.py` | 用 GPU 训练 MNIST |
| `53模型的保存与加载.py` | 模型权重的保存与重新加载 |
| `step.py` / `test.py` | 过程中的试验与临时验证脚本 |

`my_mlp.npz` 是 `53模型的保存与加载.py` 跑出来的权重文件。

## 运行

```bash
pip install numpy matplotlib
# 可选：要用 GPU 训练时安装 cupy
python 42线性回归.py
```

`datasets.py` 中的 MNIST / CIFAR 等数据集在首次使用时自行下载。
