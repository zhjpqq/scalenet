# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F

inplace = [False, True][1]


class ViewLayer(nn.Module):
    def __init__(self, dim=-1):
        super(ViewLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        # print('view-layer -> ', x.size())
        x = x.view(x.size(0), self.dim)
        return x

    def __repr__(self):
        myname = '%s' \
                 % self.__class__.__name__ \
                 + '(dim=%s)' % self.dim
        return myname


class Squeeze(nn.Module):
    def __init__(self, dim=-1):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        # print('view-layer -> ', x.size())
        x = x.view(x.size(0), self.dim)
        return x


class AdaPoolView(nn.Module):
    def __init__(self, pool='avg', dim=-1, which=0):
        super(AdaPoolView, self).__init__()
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size=1)
        else:
            raise NotImplementedError
        self.view = ViewLayer(dim=-1)

        self.dim = dim
        self.which = which

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[self.which]
        # print(self.__class__.__name__, ' ---> ', x.size())
        x = self.pool(x)
        # x = x.view(x.size(0), self.dim)
        x = self.view(x)
        return x


class AdaAvgPool(nn.AvgPool2d):
    def __init__(self):
        super(AdaAvgPool, self).__init__(kernel_size=1)

    def forward(self, x):
        # print('AdaAvg-layer -> ', x.size())
        h, w = x.size(2), x.size(3)
        # super(AdaAvgPool, self).__init__(kernel_size=(h, w))
        return F.avg_pool2d(x, kernel_size=(h, w))


class Activate(nn.Module):
    def __init__(self, method='relu'):
        super(Activate, self).__init__()
        if method == 'relu':
            self.method = nn.ReLU(inplace)
        elif method == 'sigmoid':
            self.method = nn.Sigmoid()
        elif method == 'leaky_relu':
            self.method = nn.LeakyReLU(negative_slope=0.02, inplace=inplace)
        else:
            raise NotImplementedError('--->%s' % method)

    def forward(self, x):
        return self.method(x)


class ReturnX(nn.Module):

    def __init__(self):
        super(ReturnX, self).__init__()

    def forward(self, x):
        return x


class XClassifier(nn.Module):

    def __init__(self, indepth, nclass):
        super(XClassifier, self).__init__()
        self.batch = nn.BatchNorm2d(indepth)
        self.relu = nn.ReLU(inplace)
        self.pool = AdaAvgPool()
        self.view = ViewLayer(dim=-1)
        self.fc = nn.Linear(indepth, nclass)
        # nn.Sequential(
        #     nn.BatchNorm2d(depth * expand),
        #     nn.ReLU(inplace),
        #     AdaAvgPool(),
        #     ViewLayer(dim=-1),
        #     nn.Linear(depth * expand, nclass)
        # )

    def forward(self, x):
        x = self.batch(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.view(x)
        x = self.fc(x)
        return x


class ConvClassifier(nn.Module):
    def __init__(self, indepth, outdepth, ksize, stride=1, padding=0, pool='no'):
        # outdepth === nclass, ksize=x.size(h,w) or //2
        super(ConvClassifier, self).__init__()
        assert pool in ['avg', 'max', 'no'], 'pool must in [avg, max, no], but now: %s' % pool
        ksize = ksize if pool == 'no' else ksize // 2
        self.pool = pool
        if pool == 'avg':
            self.pool2d = nn.AvgPool2d(2, stride=2, padding=0)
        elif pool == 'max':
            self.pool2d = nn.MaxPool2d(2, stride=2, padding=0)
        elif pool == 'no':
            self.pool2d = ReturnX()
        self.bnorm = nn.BatchNorm2d(indepth)
        self.relu = nn.ReLU(inplace)
        self.conv2d = nn.Conv2d(indepth, outdepth, ksize, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        # print(x.size())
        x = self.conv2d(self.relu(self.bnorm(self.pool2d(x))))
        # assert x.size()[-2:] == (1, 1)
        # print(x.size())
        x = x.view(x.size(0), -1)
        return x


class FcClassifier(nn.Module):
    def __init__(self, indepth, outdepth, ksize):
        # outdepth === nclass, ksize=x.size(h,w)
        super(FcClassifier, self).__init__()
        self.indepth = indepth
        self.outdepth = outdepth
        self.ksize = ksize

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(indepth),
            nn.ReLU(inplace),
            # nn.AvgPool2d(self.classify),
            AdaAvgPool(),
            ViewLayer(dim=-1),
            nn.Linear(indepth, outdepth)
        )

    def forward(self, x):
        return self.classifier(x)
