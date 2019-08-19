# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/7/23 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# r = F.relu(0)
#
# R = nn.ReLU()
#
# b = F.batch_norm(0, 0, 0)
#
# B = nn.BatchNorm2d(0)
#
# C = nn.Conv2d(0, 0, 0)


class AdaReLU(nn.Module):
    _abc_attr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']
    """
    => ReLU(): abc_nums=0, abc_val=0, abc_bias=0, abc_relu=-1, abc_addx=False 
    """

    def __init__(self, threshold=0, value=0, momentum=0.9, inplace=False,
                 abc_nums=3, abc_val=0.001, abc_bias=0, abc_relu=1, abc_addx=True):

        super(AdaReLU, self).__init__()
        self.threshold = threshold
        self.value = value
        self.momentum = momentum
        self.inplace = inplace

        self.abc_nums = abc_nums  # abc中的指数项，次幂从1~N
        self.abc_bias = abc_bias  # abc中的常数项，次幂=0；==0时不包含常数项
        self.abc_relu = abc_relu  # relu放在abc的之前(-1)，之后(1)，还是无relu(0) ?.
        self.abc_addx = abc_addx  # abc运算完之后是否需要加回x: abc(x) + x ?

        assert len(self._abc_attr) >= abc_nums
        self.abc_attr = self._abc_attr[:abc_nums]
        if isinstance(abc_val, (tuple, list)):
            assert len(abc_val) == abc_nums
        else:
            abc_val = [abc_val] * abc_nums
        self.abc_val = abc_val
        for i, attr in enumerate(self.abc_attr):
            val = Parameter(torch.ones(1) * abc_val[i])
            setattr(self, attr, val)
        if abc_bias != 0:
            val = Parameter(torch.ones(1) * abc_bias)
            setattr(self, 'bias', val)

    def gradient_me(self, yes=True):
        for p in self.parameters():
            p.requires_grad = yes

    def forward(self, x):
        if self.abc_relu == -1:
            x = F.relu(x)
        z = 0
        for i, attr in enumerate(self.abc_attr):
            val = getattr(self, attr)
            z += val * torch.pow(x, (i + 1))
        if self.abc_bias != 0:
            z += getattr(self, 'bias')
        if self.abc_addx is True:
            z += x
        if self.abc_relu == 1:
            z = F.relu(z)
        return z

    def __repr__(self):
        strme = self.__class__.__name__ + '(' + \
                'abc_nums={0}, abc_val={1}, abc_bias={2}, abc_relu={3}, abc_addx={4}'.format(
                    self.abc_nums, self.abc_val, self.abc_bias, self.abc_relu, self.abc_addx) + ')'
        return strme


if __name__ == '__main__':
    torch.manual_seed(1)
    br = nn.ReLU()
    ar = AdaReLU(abc_nums=3, abc_val=0.01, abc_bias=0, abc_relu=1, abc_addx=True)
    ar.gradient_me(yes=True)
    x = torch.rand(2, 3)
    z = ar(x)
    e = br(x)
    print('x', x)
    print('z', z)
    print('e', e)
    print(ar)
