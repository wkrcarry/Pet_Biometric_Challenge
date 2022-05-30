import torch
import torch.nn as nn

#from nets.vgg import VGG16
import linecache
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
import os
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim, device
from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional
import torch.utils.checkpoint as checkpoint

class Bottleneck(nn.Module):  # 针对更深层次的残差结构
    # 以50层conv2_x为例，卷积层1、2的卷积核个数=64，而第三层卷积核个数=64*4=256，故expansion = 4
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 对于第一层卷积层，无论是实线残差结构还是虚线，stride都=1
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # 对于第二层卷积层，实线残差结构和虚线的stride是不同的，stride采用传入的方式
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel * self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    # 正向传播过程
    def forward(self, x):
        identity = x
        # self.downsample=none对应实线残差结构，否则为虚线残差结构
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)  # 卷积层
        out = self.bn1(out)  # BN层
        out = self.relu(out)  # 激活层

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    # 若选择浅层网络结构block=BasicBlock，否则=Bottleneck
    # blocks_num所使用的残差结构的数目（是一个列表），若选择34层网络结构，blocks_num=[3,4,6,3]
    # num_classes训练集的分类个数
    # include_top参数便于在ResNet网络基础上搭建更复杂的网络
    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64  # 输入特征矩阵的深度（经过最大下采样层之后的）

        # 第一个卷积层，对应表格中7*7的卷积层，输入特征矩阵的深度RGB图像，故第一个参数=3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 对应3*3那个maxpooling
        # conv2_x对应的残差结构，是通过_make_layer函数生成的
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        # conv3_x
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        # conv4_x
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        # conv5_x
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.fully_connect1 = torch.nn.Linear(2048, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)
        if self.include_top:
            # 平均池化下采样层，AdaptiveAvgPool2d自适应的平均池化下采样操作，所得到特征矩阵的高和宽都是（1,1）
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            # 全连接层（输出节点层）
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        # 卷积层初始化操作
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    # block为BasicBlock或Bottleneck
    # channel残差结构中所对应的第一层的卷积核的个数（值为64/128/256/512）
    # block_num对应残差结构中每一个conv*_x卷积层的个数(该层一共包含了多少个残差结构)例：34层的conv2_x：block_num取值为3
    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None  # 下采样赋值为none
        # 对于18层、34层conv2_x不满足此if语句（不执行）
        # 而50层、101层、152层网络结构的conv2_x的第一层也是虚线残差结构，需要调整特征矩阵的深度而高度和宽度不需要改变
        # 但对于conv3_x、conv4_x、conv5_x不论ResNet为多少层，特征矩阵的高度、宽度、深度都需要调整（高和宽缩减为原来的一半）
        if stride != 1 or self.in_channel != channel * block.expansion:
            # 生成下采样函数
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []  # 定义一个空列表
        # 参数依次为输入特征矩阵的深度，残差结构所对应主分支上第一个卷积层的卷积核个数
        # 18层34层的conv2_x的layer1没有经过下采样函数那个if语句downsample=none
        # conv2_x对应的残差结构，通过此函数_make_layer生成的时，没有传入stride参数，stride默认=1
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        # conv3_x、conv4_x、conv5_x的第一层都是虚线残差结构，
        # 而从第二层开始都是实线残差结构了，直接压入统一处理
        for _ in range(1, block_num):  # 由于第一层已经搭建好，从1开始
            # self.in_channel：输入特征矩阵的深度，channel：残差结构主分支第一层卷积的卷积核个数
            layers.append(block(self.in_channel, channel))

        # 通过非关键字参数的形式传入到nn.Sequential，nn.Sequential将所定义的一系列层结构组合在一起并返回
        return nn.Sequential(*layers)

    # 正向传播
    def forward_once(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        '''
        if self.include_top:
            x = self.avgpool(x)  # 平均池化下采样
            x = torch.flatten(x, 1)  # 展平处理
            x = self.fc(x)  # 全连接
        '''
        return x

    def forward(self, x):
        x1, x2 = x
        x1 = self.forward_once(x1)
        x2 = self.forward_once(x2)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        x = torch.abs(x1 - x2)
        x = self.fully_connect1(x)
        x = self.fully_connect2(x)

        return x
    
def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)