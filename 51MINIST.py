if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import Rryyz
import Rryyz.functions as F
from Rryyz import optimizers
from Rryyz import DataLoader
from Rryyz.models import MLP


max_epoch = 5
batch_size = 100
hidden_size = 1000

# 下载好的压缩包在C:\Users\用户名\.Rryyz内
# 若第一次下载时中途退出，后面再运行会报错，需要进入该目录手动删除压缩包
train_set = Rryyz.datasets.MNIST(train=True)
test_set = Rryyz.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# model = MLP((hidden_size, 10))
model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.SGD().setup(model)
# optimizer = optimizers.Adam().setup(model)

for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print('轮数: {}'.format(epoch+1))
    print('训练损失: {:.4f}, 识别精度: {:.4f}'.format(
        sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with Rryyz.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('测试损失: {:.4f}, 识别精度: {:.4f}'.format(
        sum_loss / len(test_set), sum_acc / len(test_set)))