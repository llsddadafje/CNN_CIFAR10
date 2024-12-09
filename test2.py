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

writer = SummaryWriter(log_dir="./logs/module")


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


fake_img = torch.randn(1, 3, 32, 32)
net = ConvNet()
print(net)
writer.add_graph(net, fake_img)
writer.close()
