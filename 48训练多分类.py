if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
import matplotlib.pyplot as plt
import Rryyz
from Rryyz import optimizers
import Rryyz.functions as F
from Rryyz.models import MLP

# 超参数
# 训练的轮数
max_epoch = 300
# 一次处理30个数据
batch_size = 30
hidden_size = 10
lr = 1.0

# Rryyz的datasets.py里内置了一些经典数据集
# 从数据库获取训练数据（如果train为False则返回测试数据）
# x是矩阵，每一行有两列，代表其数据的横纵坐标
# t是向量，代表与之索引对应的数据的类别
x, t = Rryyz.datasets.get_spiral(train=True)
# 创建有两个层的全连接层类，并指定输出的大小为3
model = MLP((hidden_size,3))
# 创建参数更新器，设置为梯度下降法，并绑定model
optimizer = optimizers.SGD(lr).setup(model)
# 获取x的数据长度
data_size = len(x)
# 算出进行一整轮的训练需要多少批次。math.ceil()是向上取整的函数
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # 创造一个包含0~299的数组，并打乱
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        # 第一次取出第1到30的数据，第二次取出第31到60的数据，依次类推
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        # np矩阵可以以这种形式x[0,1,2,3]取出数据
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        # 给model投入数据，它会自动按照x的尺寸生成w，然后进行变换
        y = model(batch_x)
        # sofmax+交叉熵二合一函数。对比得到的预测值和真实值的差异，算出损失值
        loss = F.softmax_cross_entropy(y, batch_t)
        # 清理导数
        model.cleargrads()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.update()
        # 损失值
        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    # 每一轮完成后打印当前轮数和损失值
    print('epoch %d, loss %.2f' % (epoch + 1, avg_loss))

# 画图
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with Rryyz.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40,  marker=markers[c], c=colors[c])
plt.show()