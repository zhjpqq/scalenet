# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils
from xmodels.scalenet import ScaleNet
from xmodels.richnet import RichNet
from xmodels.fishnet import FishNet, fishnet99, fishnet201, fishnet150
from xmodels.dxnet import DxNet
from xmodels.mobilev3x import MobileV3X
from xmodels.mobilev3y import MobileV3Y
from xmodels.mobilev3z import MobileV3Z
from xmodels.mobilev3 import MobileV3, mbvxl, mbvdl
from xmodels.hrnet import HRNets, hrw18, hrw30, hrw40, hrw44, hrw48
from xmodels.efficientnet import EFFNets, effb0, effb1, effb2, effb3
import torchvision as tv
from torch import nn


class ArchImgNet(object):
    # res18   11.69M  3.63G   prec@1-69.758  prec@5-89.076  516M   0.00276s
    # res34   21.80M  7.34G   prec@1-73.314  prec@5-91.420  575M   0.00466s
    # res50   25.56M  8.20G   prec@1-76.130  prec@5-92.862  697M   0.00778s
    # res101  44.54M  15.63   prec@1-77.374  prec@5-93.546  846M   0.01459s
    # res152  60.19M  23.07G  prec@1-78.312  prec@5-94.046  1007M  0.02109s

    # fish99: 16.63M  4.31G   693M  0.01121s
    # fish150: 24.96M 6.45G   804M  0.01696S

    # hrw18   21.30M  8.60G   751M   0.0341s
    # hrw30   37.71M  16.25G  898M   0.0341s
    # hrw40   57.56M  25.43G  1057M  0.0340s
    # hrw44   67.06M  29.82G  1120M  0.0335s
    # hrw48   77.46M  34.61G  1213M  0.0405s

    # effb0   666M  0.01082s
    # effb1   773M  0.01695s
    # effb2   795M  0.01625s
    # effb3   917M  0.02011s

    # fishnet99  16.63M  8.57G    691M   0.0156s?

    vo22 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
            'layers': (3, 3, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
            'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
            'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
            'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
            'last_expand': 1024 - 320, 'version': 3}  # 15.16M  6.10G  39L  0.31s  1024fc  572M  0.00680s  73.02%

    vo72 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
            'layers': (4, 5, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
            'growth': (-8, -20, -50), 'classify': (0, 0, 0), 'expand': (1 * 120, 2 * 120), 'afisok': False,
            'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
            'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
            'last_expand': 1700 - 440, 'version': 3}  # 30.51M  10.83G  59L  0.51s  1700fc  678M  0.01519s?  74.86%


if __name__ == '__main__':

    device = torch.device('cuda:0')

    netis = ['scalenet', 'dxnet', 'densenet', 'resnet', 'fishnet', 'mobilev3', 'hrnet', 'effnet'][-1]

    if netis == 'densenet':
        model = tv.models.densenet201()
    elif netis == 'resnet':
        model = tv.models.resnet152()
    elif netis == 'scalenet':
        model = ScaleNet(**ArchImgNet().vo72)
    elif netis == 'dxnet':
        model = DxNet(**ArchImgNet().exp1)
    elif netis == 'mobilev3':
        model = MobileV3(**mbvdl)
    elif netis == 'fishnet':
        # fish99: 16.63M  4.31G  100L  693M  0.01121s
        # fish150: 24.96M 6.45G 151L  804M  0.01696S
        model = [fishnet99(), fishnet150(), fishnet201()][0]
    elif netis == 'hrnet':
        model = HRNets(**hrw18)
    elif netis == 'effnet':
        model = EFFNets(**effb3)
    else:
        raise NotImplementedError

    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_FLOPs_scale(model, input_size=224, multiply_adds=True)
    xtils.calculate_layers_num(model, layers=['conv2d', 'deconv2d', 'linear'])

    model = model.to(device)
    model.eval()

    bn, c, h, w = 1, 3, 224, 224
    x = torch.Tensor(bn, c, h, w)
    x = x.to(device)
    x.requires_grad = False

    N = 2000
    tic = time.time()
    for i in range(N):
        out = model(x)
    toc = time.time() - tic

    print('total time cost is %s s' % toc)
    print('batch img total time cost is %s s' % (toc / N,))
    print('single img time cost is %s s' % (toc / N / bn,))
