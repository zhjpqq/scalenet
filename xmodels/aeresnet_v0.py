# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F

"""
只有 双层 的 WaveResNet
"""
class AEBlock(nn.Module):
    exp1 = 1
    exp2 = 1

    def __init__(self, depth, after=True):
        super(AEBlock, self).__init__()
        self.after = after
        self.bn1 = nn.BatchNorm2d(depth)
        self.conv1 = nn.Conv2d(depth, depth*self.exp1, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth*self.exp1)
        self.conv2 = nn.Conv2d(depth*self.exp1, depth*self.exp2, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(depth*self.exp2)
        self.deconv3 = nn.ConvTranspose2d(depth*self.exp2, depth*self.exp1, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(depth*self.exp1)
        self.deconv4 = nn.ConvTranspose2d(depth*self.exp1, depth, 3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        if isinstance(x, tuple):
            x1, x2, x3 = x
        else:
            x1, x2, x3 = x, None, None
        res1 = self.conv1(F.relu(self.bn1(x1)))
        out = res1 if x2 is None else res1 + x2
        res2 = self.conv2(F.relu(self.bn2(out)))
        out = res2 if x3 is None else res2 + x3
        res3 = self.deconv3(F.relu(self.bn3(out)))
        res4 = self.deconv4(F.relu(self.bn4(res3 + res1)))
        res4 = res4 + x1

        if self.after:
            return res4, res3, res2
        else:
            return res4


class DownSample(nn.Module):
    def __init__(self, indepth, outdepth, stride=2):
        super(DownSample, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdepth)
        self.conv2 = nn.Conv2d(outdepth, outdepth, 3, stride, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        return x

    def __call__(self, x):
        return self.forward(x)


class AEResNet(nn.Module):
    def __init__(self, layers=(2, 3, 4, 5), num_classes=1000):
        super(AEResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # padding=19
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        indepth = 64
        self.layer1 = self._make_aelayer(AEBlock, 1 * indepth, layers[0])  # 112*112*64
        self.downsize1 = DownSample(1 * indepth, 2 * indepth, stride=2)  # 56*56*128
        self.layer2 = self._make_aelayer(AEBlock, 2 * indepth, layers[1])  # 56*56*128
        self.downsize2 = DownSample(2 * indepth, 4 * indepth, stride=2)  # 28*28*256
        self.layer3 = self._make_aelayer(AEBlock, 4 * indepth, layers[2])  # 28*28*256
        self.downsize3 = DownSample(4 * indepth, 8 * indepth, stride=2)  # 14*14*512
        self.layer4 = self._make_aelayer(AEBlock, 8 * indepth, layers[3])  # 14*14*512
        self.downsize4 = DownSample(8 * indepth, 16 * indepth, stride=2)  # 7*7*1024

        self.avg_pool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(16 * indepth, num_classes, bias=True)

    def _make_aelayer(self, block, indepth, block_nums):
        layers = []
        for i in range(block_nums - 1):
            layers.append(block(depth=indepth, after=True))
        layers.append(block(depth=indepth, after=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.downsize1(x)
        x = self.layer2(x)
        x = self.downsize2(x)
        x = self.layer3(x)
        x = self.downsize3(x)
        x = self.layer4(x)
        x = self.downsize4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class CifarAEResNet(nn.Module):
    def __init__(self, layers=(2, 3, 4), num_classes=10):
        super(CifarAEResNet, self).__init__()

        self.conv_3x3 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        indepth = 16
        self.layer1 = self._make_aelayer(AEBlock, 1 * indepth, layers[0])  # 32*32*16 -> 16*16*16 -> 8*8*16
        self.downsize1 = DownSample(1 * indepth, 2 * indepth, stride=2)  # 16*16*32
        self.layer2 = self._make_aelayer(AEBlock, 2 * indepth, layers[1])  # 16*16*32 -> 8*8*32 -> 4*4*32
        self.downsize2 = DownSample(2 * indepth, 4 * indepth, stride=2)  # 8*8*64
        self.layer3 = self._make_aelayer(AEBlock, 4 * indepth, layers[2])  # 8*8*64 -> 4*4*64 -> 2*2*64
        self.downsize3 = DownSample(4 * indepth, 8 * indepth, stride=2)  # 4*4*128

        self.avg_pool = nn.AvgPool2d(4)
        self.classifier = nn.Linear(8 * indepth, num_classes)

    def _make_aelayer(self, block, indepth, block_nums):
        layers = []
        for i in range(block_nums - 1):
            layers.append(block(depth=indepth, after=True))
        layers.append(block(depth=indepth, after=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_3x3(x)

        x = self.layer1(x)
        x = self.downsize1(x)
        x = self.layer2(x)
        x = self.downsize2(x)
        x = self.layer3(x)
        x = self.downsize3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':

    model = AEResNet(layers=[5, 6, 3, 2], num_classes=1000)
    print(model)
    x = torch.ones(4, 3, 256, 256)
    y = model(x)
    print(y, y.shape, y.max(1))

    model = CifarAEResNet(layers=[5, 3, 2], num_classes=10)
    print(model)
    x = torch.randn(4, 3, 32, 32)
    y = model(x)
    print(y, y.shape, y.max(1))
