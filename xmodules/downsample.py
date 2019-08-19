# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F

inplace = [False, True][1]


class Downsample(nn.Module):
    """
    forked from /torch/nn/modules/upsampling.py
    Args:
        size (tuple, optional): a tuple of ints `([optional D_out], [optional H_out], W_out)` output sizes
        scale_factor (int / tuple of ints, optional): the multiplier for the image height / width / depth
        mode (string, optional): the upsampling algorithm: one of `nearest`, `linear`, `bilinear` and `trilinear`.
                                    Default: `nearest`
        align_corners (bool, optional): if True, the corner pixels of the input
            and output tensors are aligned, and thus preserving the values at
            those pixels. This only has effect when :attr:`mode` is `linear`,
            `bilinear`, or `trilinear`. Default: False
    """
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(Downsample, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class DownSampleA(nn.Module):
    def __init__(self, indepth, outdepth):
        super(DownSampleA, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdepth)
        # will led to overlapped pixels, k=3, s=2
        self.conv2 = nn.Conv2d(outdepth, outdepth, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x), inplace))
        x = self.conv2(F.relu(self.bn2(x), inplace))
        return x


class DownSampleD(nn.Module):
    def __init__(self, indepth, outdepth, pool='conv3x2'):  # conv3x3  #conv2x3  #conv1x2
        super(DownSampleD, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdepth)
        # will not led to overlapped pixels, k=2, s=2
        self.conv2 = nn.Conv2d(outdepth, outdepth, 2, stride=2, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x), inplace))
        x = self.conv2(F.relu(self.bn2(x), inplace))
        return x


class DownSampleO(nn.Module):

    def __init__(self, indepth=0, outdepth=0, stride=2, method='avgpool'):
        super(DownSampleO, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.stride = stride
        self.method = method
        if stride == 2:
            if method == 'avgpool':
                self.downsize = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
            elif method == 'maxpool':
                self.downsize = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            elif method == 'convk2':
                self.downsize = nn.Conv2d(indepth, outdepth, kernel_size=2, stride=2, padding=0, bias=False)
            elif method == 'convk3':
                self.downsize = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=2, padding=1, bias=False)

        elif stride == 4:
            if method in ['avgpool', 'convk1a']:
                self.downsize = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
                )
            elif method in ['maxpool', 'convk1m']:
                self.downsize = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.MaxPool2d(kernel_size=4, stride=4, padding=0),
                )
            elif method in ['convk2', 'convk2a']:
                self.downsize = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                )
            elif method == 'convk2m':
                self.downsize = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )
            elif method in ['convk3', 'convk3a']:
                self.downsize = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                )
            elif method == 'convk3m':
                self.downsize = nn.Sequential(
                    nn.Conv2d(indepth, outdepth, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )

    def forward(self, x):
        return self.downsize(x)


# ******CHANNLE & PLAIN ALL Seperated !!*********
class DownSampelH(nn.Module):
    def __init__(self, indepth, outdepth, pool='conv1x2'):
        super(DownSampelH, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(outdepth)
        self.conv2 = nn.Conv2d(outdepth, outdepth, 2, stride=2, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x), inplace))
        x = self.conv2(F.relu(self.bn2(x), inplace))
        return x


class DownSampleB(nn.Module):
    def __init__(self, indepth, outdepth, pool='max'):
        super(DownSampleB, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)
        if pool == 'max':
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x), inplace))
        x = self.pool(x)
        return x


class DownSampleC(nn.Module):
    def __init__(self, indepth, outdepth):
        super(DownSampleC, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x), inplace))
        return x


class DownSampleE(nn.Module):
    def __init__(self, indepth, outdepth):
        super(DownSampleE, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 2, stride=2, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x), inplace))
        return x


class ChannelSample(nn.Module):
    def __init__(self, indepth, outdepth, double=True):
        super(ChannelSample, self).__init__()
        self.double = double
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdepth)
        self.conv2 = nn.Conv2d(outdepth, outdepth, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x), inplace))
        if self.double:
            x = self.conv2(F.relu(self.bn2(x), inplace))
        return x
