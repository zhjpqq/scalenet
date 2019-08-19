# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F
from xmodules.downsample import Downsample
from xmodules.classifier import ReturnX


# Note: A/4 M/4, B/2 C/2 for ImageNet
class RockBlock(nn.Module):
    def __init__(self):
        super(RockBlock, self).__init__()

    def forward(self, *input):
        raise NotImplementedError


############################################################
"""  RockBlock with 3 branchses  for  WaveResNet 
"""


############################################################

# New Version to Suppurt BottleNeck only For  WaveResNet ########################


class RockBlockU(RockBlock):
    # 第1/2/3分支分别独立处理图像
    def __init__(self, indepth=3, outdepth=16, branch=3, expand=(0, 0), dataset='cfar'):
        super(RockBlockU, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)
            if indepth == outdepth:
                self.branch1 = ReturnX()
            if branch >= 2:
                self.branch2 = nn.Conv2d(indepth, outdepth + expand[0], kernel_size=3, stride=2, padding=1, bias=False)
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth + expand[0] + expand[1], kernel_size=3, stride=2, padding=1,
                              bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(indepth, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth + expand[0], kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(outdepth + expand[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
                )
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth + expand[0] + expand[1], kernel_size=7, stride=2, padding=3,
                              bias=False),
                    nn.BatchNorm2d(outdepth + expand[0] + expand[1]),
                    nn.ReLU(inplace=True),  # ??
                    nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                )

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            return x1, x2, None, pred
        elif self.branch == 3:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            return x1, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockV(RockBlock):
    # 第1分支处理图像，第2/3分支分别处理第1分支
    def __init__(self, indepth=3, outdepth=16, branch=3, expand=(1, 1), dataset='cfar'):
        super(RockBlockV, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.BatchNorm2d(outdepth),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(outdepth, outdepth + expand[0], kernel_size=2, stride=2, padding=0, bias=False)
                )
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.BatchNorm2d(outdepth),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(outdepth, outdepth + expand[0] + expand[1], kernel_size=2, stride=2, padding=0,
                              bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                )
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(indepth, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(outdepth, outdepth + expand[0], kernel_size=2, stride=2, padding=0, bias=False))
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(outdepth, outdepth + expand[0] + expand[1], kernel_size=2, stride=2, padding=0,
                              bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                )

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x = self.branch1(x)
            x2 = self.branch2(x)
            return x, x2, None, pred
        elif self.branch == 3:
            x = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            return x, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockM(RockBlock):
    # 第1分支处理图像，第2分支Maxpool2d第1分支，第3分支Maxpool2d第2分支
    # 特点是采用了 maxpool（ksize=2, stride=2）, 防止降采样时像素重叠
    def __init__(self, indepth=3, outdepth=16, branch=3, expand=(0, 0), dataset='cfar'):
        super(RockBlockM, self).__init__()
        assert expand == (0, 0), 'No Channel Expand in RockBlockM，Should expand==(0,0) ==> growth[0]==0'
        self.indepth = indepth
        self.outdepth = outdepth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)
            if branch >= 2:
                self.branch2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            if branch >= 3:
                self.branch3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(indepth, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            if branch >= 2:
                self.branch2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            if branch >= 3:
                self.branch3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x = self.branch1(x)
            x2 = self.branch2(x)
            return x, x2, None, pred
        elif self.branch == 3:
            x = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x2)
            return x, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockR(RockBlock):
    # 直接resize原图像为金字塔，Conv-work-on-金字塔
    def __init__(self, indepth=3, outdepth=16, branch=3, expand=(0, 0), dataset='cfar'):
        super(RockBlockR, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.branch = branch
        self.expand = expand
        self.dataset = dataset
        self.mode = ['nearest', 'linear', 'bilinear', 'trilinear'][0]
        if dataset == 'cifar':
            self.branch1 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)
            if branch >= 2:
                self.branch2 = nn.Conv2d(indepth, outdepth + expand[0], kernel_size=3, stride=1, padding=1, bias=False)
            if branch >= 3:
                self.branch3 = nn.Conv2d(indepth, outdepth + expand[0] + expand[1], kernel_size=3, stride=1, padding=1,
                                         bias=False)
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(indepth, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth + expand[0], kernel_size=5, stride=2, padding=2, bias=False),
                    nn.BatchNorm2d(outdepth + expand[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth + expand[0] + expand[1], kernel_size=3, stride=2, padding=1,
                              bias=False),
                    nn.BatchNorm2d(outdepth + expand[0] + expand[1]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x1 = self.branch1(x)
            # x2 = self.branch2(F.interpolate(x, scale_factor=1.0/2, mode=self.mode))
            x2 = self.branch2(F.max_pool2d(x, kernel_size=2, stride=2, padding=0))
            return x1, x2, None, pred
        elif self.branch == 3:
            x1 = self.branch1(x)
            # x2 = self.branch2(F.interpolate(x, scale_factor=1.0/2, mode=self.mode))
            x2 = self.branch2(F.max_pool2d(x, kernel_size=2, stride=2, padding=0))
            # x3 = self.branch3(F.interpolate(x, scale_factor=1.0/4, mode=self.mode))
            x3 = self.branch3(F.max_pool2d(x, kernel_size=4, stride=4, padding=0))
            return x1, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockG(RockBlock):
    # 与 RockBlockU 相同，但imagenet上非(7, 2, 3)而(5, 2, 2)
    # 第1/2/3分支分别独立处理图像
    def __init__(self, indepth=3, outdepth=16, branch=3, expand=(0, 0), dataset='cfar'):
        super(RockBlockG, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)
            if indepth == outdepth:
                self.branch1 = ReturnX()
            if branch >= 2:
                self.branch2 = nn.Conv2d(indepth, outdepth + expand[0], kernel_size=3, stride=2, padding=1, bias=False)
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth + expand[0] + expand[1], kernel_size=3, stride=2, padding=1,
                              bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(indepth, outdepth, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            )
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth + expand[0], kernel_size=5, stride=2, padding=2, bias=False),
                    nn.BatchNorm2d(outdepth + expand[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
                )
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth + expand[0] + expand[1], kernel_size=5, stride=2, padding=2,
                              bias=False),
                    nn.BatchNorm2d(outdepth + expand[0] + expand[1]),
                    nn.ReLU(inplace=True),  # ??
                    nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                )

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            return x1, x2, None, pred
        elif self.branch == 3:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            return x1, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockN(RockBlock):
    # 与 RockBlockU 相同，但imagenet上非(7, 2, 3)而(3, 2, 1)
    # 第1/2/3分支分别独立处理图像
    def __init__(self, indepth=3, outdepth=16, branch=3, expand=(0, 0), dataset='cfar'):
        super(RockBlockN, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)
            if indepth == outdepth:
                self.branch1 = ReturnX()
            if branch >= 2:
                self.branch2 = nn.Conv2d(indepth, outdepth + expand[0], kernel_size=3, stride=2, padding=1, bias=False)
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth + expand[0] + expand[1], kernel_size=3, stride=2, padding=1,
                              bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(indepth, outdepth, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            )
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth + expand[0], kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(outdepth + expand[0]),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
                )
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth + expand[0] + expand[1], kernel_size=3, stride=2, padding=1,
                              bias=False),
                    nn.BatchNorm2d(outdepth + expand[0] + expand[1]),
                    nn.ReLU(inplace=True),  # ??
                    nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                )

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            return x1, x2, None, pred
        elif self.branch == 3:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            return x1, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')

############################################################
"""  RockBlock with 4 branchses  for  WaveDenseNet 
"""


############################################################


class RockBlockX(nn.Module):
    """
    特征金字塔： 第1分支处理原图，第2分支处理第1分支的map.
    第3分支处理第2分支的map，第4分支处理第1分支的map.
    特征图平面尺寸 x1:大， x2:中， x3:小, x4:中
    """

    def __init__(self, indepth=3, outdepth=16, branch=4, dataset='cfar'):
        super(RockBlockX, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.branch = branch
        if dataset == 'cifar':
            self.branch1 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)
            if branch >= 2:
                self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if branch >= 3:
                self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if branch >= 4:
                self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(indepth, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            if branch >= 2:
                self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if branch >= 3:
                self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if branch >= 4:
                self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, None, pred
        elif self.branch == 2:
            x = self.branch1(x)
            x2 = self.branch2(x)
            return x, x2, None, None, pred
        elif self.branch == 3:  # 大 中 小
            x = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x2)
            return x, x2, x3, None, pred
        elif self.branch == 4:
            x1 = self.branch1(x)
            x2 = self.branch2(x1)
            x3 = self.branch3(x2)
            x4 = self.branch4(x1)  # x3
            return x1, x2, x3, x4, pred  # 大 中 小 中
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockQ(nn.Module):
    """
    图像金字塔： 第1,2,3,4分支分别独立处理图像，缩放步长不同,
    各自产生独立的featureMap，平面尺寸 x1:大， x2:中， x3:小, x4:中
    """

    def __init__(self, indepth=3, outdepth=16, branch=4, dataset='cfar'):
        super(RockBlockQ, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.branch = branch
        if dataset == 'cifar':
            self.branch1 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)
            if branch >= 2:
                self.branch2 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=2, padding=1, bias=False)
            if branch >= 3:
                self.branch3 = self.branch3 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            if branch >= 4:
                self.branch4 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(indepth, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(outdepth),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=4, stride=4, padding=0))
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                    nn.BatchNorm2d(outdepth),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            if branch >= 4:
                self.branch4 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=7, stride=4, padding=3, bias=False),
                    nn.BatchNorm2d(outdepth),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, None, pred
        elif self.branch == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            return x1, x2, None, None, pred
        elif self.branch == 3:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            return x1, x2, x3, None, pred
        elif self.branch == 4:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            x4 = self.branch4(x)
            return x1, x2, x3, x4, pred  # 大 中 小 中
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockO(nn.Module):
    """
    图像金字塔： 第1,2,3,4分支分别独立处理图像，缩放步长不同,
    各自产生独立的featureMap，平面尺寸 x1:大， x2:中， x3:小, x4:中.
    """

    def __init__(self, indepth=3, outdepth=16, branch=4, dataset='cfar'):
        super(RockBlockO, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.branch = branch
        self.mode = ['nearest', 'linear', 'bilinear', 'trilinear'][0]
        if dataset == 'cifar':
            self.branch1 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)
            if branch >= 2:
                self.branch2 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)
            if branch >= 3:
                self.branch3 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)
            if branch >= 4:
                self.branch4 = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)

        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(indepth, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=5, stride=2, padding=2, bias=False),
                    nn.BatchNorm2d(outdepth),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(outdepth),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
            if branch >= 4:
                self.branch4 = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(outdepth),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, None, pred
        elif self.branch == 2:
            x1 = self.branch1(x)
            # x2 = self.branch2(F.interpolate(x, scale_factor=1.0/2, mode=self.mode))
            x2 = self.branch2(F.max_pool2d(x, kernel_size=2, stride=1, padding=0))
            return x1, x2, None, None, pred
        elif self.branch == 3:
            x1 = self.branch1(x)
            # x2 = self.branch2(F.interpolate(x, scale_factor=1.0/2, mode=self.mode))
            x2 = self.branch2(F.max_pool2d(x, kernel_size=2, stride=2, padding=0))
            # x3 = self.branch3(F.interpolate(x, scale_factor=1.0/4, mode=self.mode))
            x3 = self.branch3(F.max_pool2d(x, kernel_size=4, stride=4, padding=0))
            return x1, x2, x3, None, pred
        elif self.branch == 4:
            x1 = self.branch1(x)
            # x2 = self.branch2(F.interpolate(x, scale_factor=1.0/2, mode=self.mode))
            x2 = self.branch2(F.max_pool2d(x, kernel_size=2, stride=2, padding=0))
            # x3 = self.branch3(F.interpolate(x, scale_factor=1.0/4, mode=self.mode))
            x3 = self.branch3(F.max_pool2d(x, kernel_size=4, stride=4, padding=0))
            # x4 = self.branch4(F.interpolate(x, scale_factor=1.0/2, mode=self.mode))
            x4 = self.branch4(F.max_pool2d(x, kernel_size=2, stride=2, padding=0))
            return x1, x2, x3, x4, pred  # 大 中 小 中
        else:
            raise ValueError('check branch must be in [1, 2, 3, 4]!')


############################################################
"""  RockBlock with 3 branchses  for  WaveResNet 
     Not Support BottleNeck... 
     
     Deprecated !!!!
"""


############################################################


class RockBlockA(RockBlock):
    def __init__(self, depth=16, branch=3, expand=(1, 1), dataset='cfar', rock='conv2x2'):
        super(RockBlockA, self).__init__()
        self.depth = depth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(depth),
                # nn.ReLU(inplace=True),
            )
            if branch >= 2:
                self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if branch >= 3:
                self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            if branch >= 2:
                if expand[0] == 1:
                    self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch2 = nn.Sequential(
                        nn.Conv2d(depth, depth * expand[0], kernel_size=7, stride=2, padding=3, bias=False))
            if branch >= 3:
                if expand[0] == 1:
                    self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch3 = nn.Sequential(
                        nn.Conv2d(depth, depth * expand[1], kernel_size=7, stride=2, padding=3, bias=False))

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x = self.branch1(x)
            x2 = self.branch2(x)
            return x, x2, None, pred
        elif self.branch == 3:
            x = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x2)
            return x, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockB(RockBlock):
    def __init__(self, depth=16, branch=3, expand=(1, 1), dataset='cfar', rock='conv2x2'):
        super(RockBlockB, self).__init__()
        self.depth = depth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(depth),
                # nn.ReLU(inplace=True),
            )
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.BatchNorm2d(depth),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(depth, depth, kernel_size=2, stride=2, padding=0, bias=False),
                )
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.BatchNorm2d(depth),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(depth, depth, kernel_size=2, stride=2, padding=0, bias=False),
                )
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # **** /2
            )
            if branch >= 2:
                if expand[0] == 1:
                    self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch2 = nn.Sequential(
                        nn.Conv2d(depth, depth * expand[0], kernel_size=7, stride=2, padding=3, bias=False))
            if branch >= 3:
                if expand[0] == 1:
                    self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch3 = nn.Sequential(
                        nn.Conv2d(depth, depth * expand[1], kernel_size=7, stride=2, padding=3, bias=False))

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x = self.branch1(x)
            x2 = self.branch2(x)
            return x, x2, None, pred
        elif self.branch == 3:
            x = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x2)
            return x, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockC(RockBlock):
    def __init__(self, depth=16, branch=3, expand=(1, 1), dataset='cfar', rock='conv2x2'):
        super(RockBlockC, self).__init__()
        self.depth = depth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(depth),
                # nn.ReLU(inplace=True),
            )
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(3, depth, kernel_size=2, stride=2, padding=0, bias=False),
                )
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(3, depth, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),     # ***/2
            )
            if branch >= 2:
                if expand[0] == 1:
                    self.branch2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                elif expand[0] > 1:
                    self.branch2 = nn.Sequential(
                        nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False))
            if branch >= 3:
                if expand[0] == 1:
                    self.branch3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                elif expand[0] > 1:
                    self.branch3 = nn.Sequential(
                        nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False))

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            return x1, x2, None, pred
        elif self.branch == 3:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            return x1, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockD(RockBlock):
    def __init__(self, depth=16, branch=3, expand=(1, 1), dataset='cfar', rock='conv2x2'):
        super(RockBlockD, self).__init__()
        self.depth = depth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(depth),
                # nn.ReLU(inplace=True),
            )
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(3, depth, kernel_size=2, stride=2, padding=0, bias=False),
                )
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(3, depth, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.Conv2d(depth, depth, kernel_size=2, stride=2, padding=0, bias=False),
                )

        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            if branch >= 2:
                if expand[0] == 1:
                    self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch2 = nn.Sequential(
                        nn.Conv2d(depth, depth * expand[0], kernel_size=7, stride=2, padding=3, bias=False))
            if branch >= 3:
                if expand[0] == 1:
                    self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch3 = nn.Sequential(
                        nn.Conv2d(depth, depth * expand[1], kernel_size=7, stride=2, padding=3, bias=False))

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            return x1, x2, None, pred
        elif self.branch == 3:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            return x1, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockE(RockBlock):
    def __init__(self, depth=16, branch=3, expand=(1, 1), dataset='cfar', rock='conv2x2'):
        super(RockBlockE, self).__init__()
        self.depth = depth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Conv2d(3, depth, kernel_size=3, stride=1, padding=1, bias=False)
            if branch >= 2:
                self.branch2 = nn.Conv2d(3, depth, kernel_size=3, stride=2, padding=1, bias=False)
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(3, depth, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                )

        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            if branch >= 2:
                if expand[0] == 1:
                    self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch2 = nn.Sequential(
                        nn.Conv2d(depth, depth * expand[0], kernel_size=7, stride=2, padding=3, bias=False))
            if branch >= 3:
                if expand[0] == 1:
                    self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch3 = nn.Sequential(
                        nn.Conv2d(depth, depth * expand[1], kernel_size=7, stride=2, padding=3, bias=False))

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            return x1, x2, None, pred
        elif self.branch == 3:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            return x1, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockH(RockBlock):
    # some like E
    def __init__(self, depth=16, branch=3, expand=(1, 1), dataset='cfar', rock='conv2x2'):
        super(RockBlockH, self).__init__()
        self.depth = depth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(depth),
                # nn.ReLU(inplace=True),
            )
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(3, depth, kernel_size=3, stride=2, padding=1, bias=False),
                )
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(3, depth, kernel_size=3, stride=3, padding=0, bias=False),  # 10
                    nn.MaxPool2d(kernel_size=3, stride=1, padding=0),  # 8
                )

        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            if branch >= 2:
                if expand[0] == 1:
                    self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch2 = nn.Sequential(
                        nn.Conv2d(depth, depth * expand[0], kernel_size=7, stride=2, padding=3, bias=False))
            if branch >= 3:
                if expand[0] == 1:
                    self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch3 = nn.Sequential(
                        nn.Conv2d(depth, depth * expand[1], kernel_size=7, stride=2, padding=3, bias=False))

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            return x1, x2, None, pred
        elif self.branch == 3:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            return x1, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class RockBlockF(RockBlock):
    def __init__(self, depth=16, branch=3, expand=(1, 1), dataset='cfar', rock='conv2x2'):
        super(RockBlockF, self).__init__()
        self.depth = depth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(depth),
                # nn.ReLU(inplace=True),
            )
            if branch >= 2:
                self.branch2 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(3, depth, kernel_size=3, stride=1, padding=1, bias=False),
                )
            if branch >= 3:
                self.branch3 = nn.Sequential(
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                    nn.Conv2d(3, depth, kernel_size=2, stride=2, padding=0, bias=False)
                )
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            if branch >= 2:
                if expand[0] == 1:
                    self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch2 = nn.Sequential(
                        nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False))
            if branch >= 3:
                if expand[0] == 1:
                    self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                elif expand[0] > 1:
                    self.branch3 = nn.Sequential(
                        nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False))

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            return x1, x2, None, pred
        elif self.branch == 3:
            x1 = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            return x1, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')
