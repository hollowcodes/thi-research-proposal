
import matplotlib.pyplot as plt
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


torch.manual_seed(0)


class Model(nn.Module):
    def __init__(self, class_amount: int=10):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=(1, 1))
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1))
        self.batchnorm3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.batchnorm5 = nn.BatchNorm2d(128)

        self.maxpool2x2 = nn.MaxPool2d(2)

        self.dense1 = nn.Linear(128 * 3 * 3, 256)
        self.dense2 = nn.Linear(256, class_amount)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, print_: bool=False):
        if print_: print(x.shape)

        # res 1
        x = self.conv1(x)
        x1 = self.batchnorm1(x)
        x = F.relu(x1)

        if print_: print(x.shape)

        x = self.conv2(x1)
        x = self.batchnorm2(x)
        x += x1
        x = F.relu(x)
        x = self.maxpool2x2(x)

        if print_: print(x.shape)

        # res 2
        x = self.conv3(x)
        x2 = self.batchnorm3(x)
        x = F.relu(x2)

        if print_: print(x.shape)

        x = self.conv4(x)
        x = self.batchnorm4(x)
        x += x2
        x = F.relu(x)
        x = self.maxpool2x2(x)

        if print_: print(x.shape)

        # res 3
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = F.relu(x)
        x = self.maxpool2x2(x)

        if print_: print(x.shape)

        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])

        
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        #x = F.softmax(self.dense2(x), dim=1)

        return x


"""class Model(nn.Module):
    def __init__(self, class_amount: int=10):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.batchnorm1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.batchnorm2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.batchnorm3 = nn.BatchNorm2d(128)

        # self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3))
        # self.batchnorm4 = nn.BatchNorm2d(256)

        self.maxpool2x2 = nn.MaxPool2d(2)

        self.dense1 = nn.Linear(64 * 6 * 6, 512)
        self.dense2 = nn.Linear(512, 128)
        self.dense3 = nn.Linear(128, class_amount)
        self.dropout = nn.Dropout(0.6)

    def forward(self, x, print_: bool=False):
        if print_: print(x.shape)

        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool2x2(x)

        if print_: print(x.shape)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool2x2(x)

        if print_: print(x.shape)

        # x = self.conv3(x)
        # x = self.batchnorm3(x)
        # x = F.relu(x)
        # x = self.maxpool2x2(x)

        if print_: print(x.shape)

        x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
        x = F.relu(self.dense1(x), inplace=True)
        x = self.dropout(x)
        x = F.relu(self.dense2(x), inplace=True)
        x = self.dropout(x) 
        x = F.softmax(self.dense3(x), dim=1)

        return x"""


