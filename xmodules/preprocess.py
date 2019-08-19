# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F


class PreProcess(nn.Module):
    """
    起始预处理
    """

    def __init__(self, indepth=3, outdepth=16, dataset='cifar'):
        super(PreProcess, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        if dataset == 'cifar':
            self.process = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)

        elif dataset == 'imagenet':
            self.process = nn.Sequential(
                nn.Conv2d(indepth, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    def forward(self, x):
        pred = []
        x = self.process(x)
        return x, pred


class PreProc(nn.Module):
    """
    起始预处理, used in XResNet
    """

    def __init__(self, indepth=3, outdepth=16, dataset='cifar', ksp=723, **kwargs):
        super(PreProc, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        ksp = [int(x) for x in list(str(ksp))]

        if dataset.startswith('cifar'):
            self.process = nn.Conv2d(indepth, outdepth, kernel_size=3, stride=1, padding=1, bias=False)

        elif dataset.startswith('imagenet'):
            k, s, p = ksp[0], ksp[1], ksp[2]
            self.process = nn.Sequential(
                nn.Conv2d(indepth, outdepth, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        else:
            raise NotImplementedError('Unknown Dataset %s' % dataset)

    def forward(self, x):
        x = self.process(x)
        return x