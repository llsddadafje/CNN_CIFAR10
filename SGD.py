import tensorboard
import torchvision
from sklearn.metrics import precision_score, recall_score, f1_score
from torch import nn
import torch
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from matplotlib import font_manager
from torch.utils.tensorboard import SummaryWriter
import numpy

import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime

# 获取当前时间
now = datetime.now()
# 实例化tensorboard对象
writer = {
    'SGD_train_loss_3': SummaryWriter("./logs/SGD_train_loss_3"),  # 必须要不同的writer
    'SGD_acc_3': SummaryWriter("./logs/SGD_acc_3"),
    'SGD_recall_3': SummaryWriter("./logs/SGD_recall_3"),
    'SGD_precision_3': SummaryWriter("./logs/SGD_precision_3"),
    'SGD_f1_3': SummaryWriter("./logs/SGD_f1_3")
}

font = font_manager.FontProperties(fname=r'C:\Windows\Fonts\STSONG.TTF')
train_loss = list()
test_loss = list()
precision_list = list()
recall_list = list()


def plot(train_loss, test_loss):
    plt.figure(figsize=(5, 5))
    plt.plot(train_loss, label='train_loss', alpha=0.5)
    plt.plot(test_loss, label='test_loss', alpha=0.5)
    plt.title(f'使用非线性函数和图像数据增强手段处理CIFAR10数据集', fontproperties=font)
    plt.xlabel('训练次数', fontproperties=font)
    plt.ylabel('损失', fontproperties=font)
    plt.legend()
    plt.show()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data_set = torchvision.datasets.CIFAR10(root='dataset', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

]), download=True)

test_data_set = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

]), download=True)

train_data_load = DataLoader(dataset=train_data_set, batch_size=64, shuffle=True, drop_last=True)
test_data_load = DataLoader(dataset=test_data_set, batch_size=32, shuffle=True, drop_last=True)

train_data_size = len(train_data_set)
test_data_size = len(test_data_set)
print(f'训练集的大小为{train_data_size}')
print(f'测试集的大小为{test_data_size}')


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)  # 考虑到池化层后的尺寸
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 64 * 8 * 8)  # flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


mynet = ConvNet()
mynet = mynet.to(device)
print(mynet)
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# learning_rate = 1e-2  # 第一次实验
# learning_rate = 1e-1  # 第二次实验
learning_rate = 1e-3  # 第三次实验



# 优化器
# 优化器
# 优化器
optim = torch.optim.SGD(mynet.parameters(), lr=learning_rate)
# optim=torch.optim.Adagrad(mynet.parameters(), lr=learning_rate)


j = 0

train_step = 0
test_step = 0
epoch = 100
all_preds = []
all_targets = []
total = 0
correct = 0
if __name__ == '__main__':
    for i in range(epoch):
        print(f'----第{i + 1}轮训练开始-------')
        mynet.train()
        now = datetime.now()
        train_for_loss = 0
        for j, (imgs, targets) in enumerate(train_data_load):
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = mynet(imgs)

            loss = loss_fn(outputs, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            train_step += 1
            train_for_loss += loss.item()

            if train_step % 100 == 0:
                print(f'损失第{train_step}次,loss={loss}')
                train_loss.append(loss.item())  # loss下降曲线
        writer['SGD_train_loss_3'].add_scalar('value', float(train_for_loss / (len(train_data_load))), i)

        mynet.eval()
        accuracy = 0
        accuracy_total = 0
        with torch.no_grad():
            for j, (imgs, targets) in enumerate(test_data_load):
                imgs = imgs.to(device)
                targets = targets.to(device)

                outputs = mynet(imgs)
                loss = loss_fn(outputs, targets)

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                accuracy = (outputs.argmax(axis=1) == targets).sum()
                accuracy_total += accuracy



                # 将指标保存到 TensorBoard
                test_step += 1
                if test_step % 100 == 0:
                    j += 1
                    test_loss.append(loss.item())
        precision = precision_score(all_targets, all_preds, average='macro', zero_division=1)
        recall = recall_score(all_targets, all_preds, average='macro', zero_division=1)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=1)
        writer['SGD_acc_3'].add_scalar('value', float(accuracy_total / test_data_size), i)
        writer['SGD_precision_3'].add_scalar('value', float(precision), i)
        writer['SGD_recall_3'].add_scalar('value', float(recall), i)
        writer['SGD_f1_3'].add_scalar('value', float(f1), i)
        print(f'第{i + 1}轮训练结束，准确率{accuracy_total / test_data_size}')
        print(now.strftime("%Y-%m-%d %H:%M:%S"))
        # torch.save(mynet, f'CIAFR_10_{i + 1}_acc_{accuracy_total / test_data_size}.pth')

    # plot(train_loss, test_loss)
    writer['SGD_acc_3'].close()
    writer['SGD_precision_3'].close()
    writer['SGD_recall_3'].close()
    writer['SGD_f1_3'].close()
    writer['SGD_train_loss_3'].close()
