import os

import torch
from torch import nn
import torchvision.transforms
from PIL import Image


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128),

        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),

            nn.Linear(256, 10)

        )

    def forward(self, x):
        return self.fc(self.main(x))

targets_idx={
    0:'airplane',
    1:'car',
    2:'bird',
    3:'cat',
    4:'deer',
    5:'dog',
    7:'horse',
    8:'ship',
    9:'truck'
}


root_dir = 'test_CIFAR_10'
obj_dir = 'test3.png'

img_dir = os.path.join(root_dir, obj_dir)
img = Image.open(img_dir)
tran_pose = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(32, 32)),
    torchvision.transforms.ToTensor()

])


