# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import torch
import torch.nn as nn
from torch.nn import init
from xmodules.res_utils import DownsampleA
from xmodules.classifier import AdaAvgPool, ViewLayer, ReturnX
from xmodules.preprocess import PreProc
import torch.nn.functional as F
import math

"""
基于Taylor级数的高阶ResNet
"""


class PolyActive(nn.Module):

    def __init__(self, fname='fx', a=0, b=0, c=0):
        super(PolyActive, self).__init__()
        self.fname = fname
        self.a = a
        self.b = b
        self.c = c
        self.active = getattr(self, fname, None)
        assert self.active is not None

    def fr(self, x):
        return F.relu(x)

    def fr6(self, x):
        return F.relu6(x)

    def fs(self, x):
        return x + x.sigmoid()

    def fe(self, x):
        return -1 + 2 * x + torch.exp(-x)

    def fg(self, x):
        return 1.0 / (1. - x) - 1.

    def fx(self, x):
        return x

    def f3(self, x):
        return x - x * x + x * x * x

    def forward(self, x):
        return self.active(x)

    def __repr__(self):
        itis = self.__class__.__name__
        itis += '(' + self.fname + ')'
        return itis


class ResPlain(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, Type, bottle=1, fname='fx'):
        super(ResPlain, self).__init__()
        assert bottle == 1, 'In ResPlain block, <bottle> is Invalid, just a placeholder.'
        self.Type = Type

        self.bn_a = nn.BatchNorm2d(inplanes)
        self.relu_a = nn.ReLU(inplace=True)
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn_b = nn.BatchNorm2d(planes)
        self.relu_b = nn.ReLU(inplace=True)
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.active = PolyActive(fname=fname)
        self.downsample = downsample

    def forward(self, x):
        skip = self.active(x)

        x = self.bn_a(x)
        x = self.relu_a(x)
        if self.Type == 'both_preact':
            skip = self.active(x)
        elif self.Type != 'normal':
            assert False, 'Unknow type : {}'.format(self.Type)
        x = self.conv_a(x)

        x = self.bn_b(x)
        x = self.relu_b(x)
        x = self.conv_b(x)

        if self.downsample is not None:
            skip = self.downsample(skip)

        out = skip + x
        out = self.active(out)
        return out


class ResBottle(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, Type, bottle=0.8, fname='f1'):
        """
        bottle = 1 时，蜕化为 ResPlain block. =1:平坦, >1:膨胀，<1:收缩.
        """
        super(ResBottle, self).__init__()

        interplanes = math.ceil(planes * bottle)

        self.Type = Type

        self.bn_a = nn.BatchNorm2d(inplanes)
        self.relu_a = nn.ReLU(inplace=True)
        self.conv_a = nn.Conv2d(inplanes, interplanes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn_b = nn.BatchNorm2d(interplanes)
        self.relu_b = nn.ReLU(inplace=True)
        self.conv_b = nn.Conv2d(interplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.active = PolyActive(fname=fname)
        self.downsample = downsample

    def forward(self, x):
        skip = self.active(x)

        x = self.bn_a(x)
        x = self.relu_a(x)
        if self.Type == 'both_preact':
            skip = self.active(x)
        elif self.Type != 'normal':
            assert False, 'Unknow type : {}'.format(self.Type)
        x = self.conv_a(x)

        x = self.bn_b(x)
        x = self.relu_b(x)
        x = self.conv_b(x)

        if self.downsample is not None:
            skip = self.downsample(skip)

        out = self.active(skip + x)
        return out


class XResNet(nn.Module):
    """
    https://arxiv.org/abs/1512.03385.pdf
    """

    residual = {'P': ResPlain, 'B': ResBottle}
    datasets = {'cifar10': 10, 'cifar100': 100, 'imagenet': 1000}

    def __init__(self, block='P', nblocks=(3, 3, 3), inplanes=16, bottle=1, active='fx', dataset='cifar10'):
        """ Constructor
        Args:
            bottle: =1 a ResPlain block; >1 a dilate ResBottle block; <1, a shrink ResBottle block.
        """
        super(XResNet, self).__init__()
        assert block in self.residual.keys(), 'Unknown residual block: %s.' % block
        assert dataset in self.datasets.keys(), 'Unsupported dataset: %s.' % dataset
        assert len(nblocks) == [4, 3][dataset != 'imagenet'], 'Assure <nblocks> match with dataset.'

        block = self.residual[block]
        self.block = block
        self.nblocks = nblocks
        self.inplanes = inplanes
        self.bottle = bottle
        self.active = active
        self.dataset = dataset
        nlabels = self.datasets[dataset]

        nplanes = [inplanes, 2 * inplanes, 4 * inplanes, 8 * inplanes]
        nblocks = nblocks if len(nblocks) >= 4 else list(nblocks) + [-1]  # 添加伪数，防止越界
        nbottls = [bottle] * len(nblocks)

        self.preproc = PreProc(3, nplanes[0], dataset)

        self.stage_1 = self._make_layer(block, nplanes[0], nblocks[0], nbottls[0], stride=1)
        self.stage_2 = self._make_layer(block, nplanes[1], nblocks[1], nbottls[1], stride=2)
        self.stage_3 = self._make_layer(block, nplanes[2], nblocks[2], nbottls[2], stride=2)
        self.stage_4 = [self._make_layer(block, nplanes[3], nblocks[3], nbottls[3], stride=2), ReturnX()][
            dataset != 'imagenet']
        out_planes = [nplanes[-1], nplanes[-2]][dataset != 'imagenet']

        self.squeeze = nn.Sequential(nn.BatchNorm2d(out_planes * block.expansion),
                                     nn.ReLU(inplace=True),
                                     AdaAvgPool(),
                                     ViewLayer())

        self.classifier = nn.Linear(out_planes * block.expansion, nlabels)

        self._init_params()

    def _make_layer(self, block, planes, blocks, bottle, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, 'both_preact', bottle, self.active))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, 'normal', bottle, self.active))

        return nn.Sequential(*layers)

    def _init_params(self):
        # http://studyai.com/article/de23cbb0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.preproc(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        # print(x.shape)
        x = self.squeeze(x)
        x = self.classifier(x)
        return x


xresnet_cifar10 = {'block': 'P', 'nblocks': (3, 3, 3), 'inplanes': 16,
                   'bottle': 1, 'active': 'fr', 'dataset': 'cifar10'}

xresnet_cifar100 = {'block': 'B', 'nblocks': (3, 3, 3), 'inplanes': 20,
                    'bottle': 1.2, 'active': 'fx', 'dataset': 'cifar100'}

xresnet_imgnet = {'block': 'B', 'nblocks': (3, 3, 3, 3), 'inplanes': 64,
                  'bottle': 0.8, 'active': 'fg', 'dataset': 'imagenet'}

if __name__ == '__main__':
    import xtils, time, random

    seed = [2019, time.time()][0]
    random.seed(seed)
    torch.manual_seed(seed)

    # # # ImageNet
    model = XResNet(**xresnet_imgnet)
    print(model)
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'classifier'))
    xtils.calculate_FLOPs_scale(model, input_size=224, multiply_adds=True, use_gpu=False)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_time_cost(model, insize=224, toc=1, use_gpu=False, pritout=True)

    # CIFAR10 & CIFAR100
    #
    # model = XResNet(**xresnet_cifar100)
    # print(model)
    # utils.calculate_layers_num(model, layers=('conv2d', 'classifier'))
    # utils.calculate_FLOPs_scale(model, input_size=32, multiply_adds=True, use_gpu=False)
    # utils.calculate_params_scale(model, format='million')
    # utils.calculate_time_cost(model, insize=32, toc=3, use_gpu=False, pritout=True)
