# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F

inplace = [False, True][1]


class TransitionA(nn.Module):
    """
    used in WaveNet
    """
    exp = 2

    def __init__(self, indepth, outdepth, growth=0, pool='avg', active='relu'):
        super(TransitionA, self).__init__()
        self.indepth = indepth
        self.oudepth = outdepth
        self.growth = growth
        if pool == 'avg':
            self.pool2d = F.avg_pool2d
        elif pool == 'max':
            self.pool2d = F.max_pool2d
        self.active = getattr(nn.functional, active)
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(self.active(self.bn1(x)))
        x = self.pool2d(x, 2)
        return x


class TransitionB(nn.Module):
    """
    used in NameiNet
    """
    exp = 2

    def __init__(self, indepth, outdepth, growth=0, pool='avg', active='relu'):
        super(TransitionB, self).__init__()
        self.indepth = indepth
        self.oudepth = outdepth
        self.growth = growth
        self.pool = pool

        self.active = getattr(nn.functional, active)
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)

        if pool == 'avg':
            self.pool2d = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        elif pool == 'max':
            self.pool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1)
        elif pool == 'convk2s2':
            self.bn2 = nn.BatchNorm2d(outdepth)
            self.pool2d = nn.Conv2d(outdepth, outdepth, kernel_size=2, stride=2, padding=0, dilation=1)
        elif pool == 'convk3s2':
            self.bn2 = nn.BatchNorm2d(outdepth)
            self.pool2d = nn.Conv2d(outdepth, outdepth, kernel_size=3, stride=2, padding=1, dilation=1)

    def forward(self, x):
        x = self.conv1(self.active(self.bn1(x)))
        if 'conv' in self.pool:
            x = self.pool2d(self.active(self.bn2(x)))
        else:
            x = self.pool2d(x)
        return x


class TransitionC(nn.Module):
    """
    used in NameiNet
    """
    exp = 2

    def __init__(self, indepth, outdepth, growth=0, pool='avg', active='relu'):
        super(TransitionC, self).__init__()
        self.indepth = indepth
        self.oudepth = outdepth
        self.growth = growth

        self.active = getattr(nn.functional, active)
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(self.active(self.bn1(x)))
        return x


class TransBlock(nn.Module):
    exp = 2

    def __init__(self, indepth, outdepth, growth=None, branch=4, pool='avg', active='relu'):
        super(TransBlock, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.growth = growth
        self.branch = branch
        if pool == 'avg':
            self.pool2d = F.avg_pool2d
        elif pool == 'max':
            self.pool2d = F.max_pool2d
        self.active = getattr(nn.functional, active)
        for i in range(1, branch +1):
            trans_layer = nn.Sequential(
                nn.BatchNorm2d(indepth),
                nn.ReLU(inplace),
                nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            setattr(self, 'trans%s' % i, trans_layer)
        # if branch >= 1:
        #     self.bn1 = nn.BatchNorm2d(indepth)
        #     self.conv1 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)
        # if branch >= 2:
        #     self.bn2 = nn.BatchNorm2d(indepth)
        #     self.conv2 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)
        # if branch >= 3:
        #     self.bn3 = nn.BatchNorm2d(indepth)
        #     self.conv3 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)
        # if branch >= 4:
        #     self.bn4 = nn.BatchNorm2d(indepth)
        #     self.conv4 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x1, x2, x3, x4, pred = x
        else:
            x1, x2, x3, x4, pred = x, None, None, None, None
        x, pred = x[:-1], x[-1]
        for i, xx in enumerate(x[::-1]):
            xx = getattr(self, '')
        if self.branch >= 1:
            x1 = self.conv1(self.active(self.bn1(x1)))
            x1 = self.pool2d(x1, 2)
        if self.branch >= 2:
            x2 = self.conv2(self.active(self.bn2(x2)))
            x2 = self.pool2d(x2, 2)
        if self.branch >= 3:
            x3 = self.conv3(self.active(self.bn3(x3)))
            x3 = self.pool2d(x3, 2)
        if self.branch >= 4:
            x4 = self.conv4(self.active(self.bn4(x4)))
            x4 = self.pool2d(x4, 2)
        return x1, x2, x3, x4, pred

