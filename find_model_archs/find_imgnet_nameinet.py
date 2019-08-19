# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils
from xmodels.nameinet import NameiNet

nm1d = {'stages': 1, 'indepth': 16, 'growth': 16, 'poolmode': 'max', 'active': 'relu',
        'layers': (20, 0), 'blocks': ('D', '-'), 'bottle': (0, 0), 'classify': (0, 0),
        'trans': ('B', '-'), 'reduction': (0, 0),
        'last_branch': 1, 'last_down': True, 'last_expand': 0,
        'summer': 'split', 'nclass': 1000}

nm2d = {'stages': 2, 'indepth': 6, 'growth': 6, 'poolmode': 'max', 'active': 'relu',
        'layers': (3, 3), 'blocks': ('D', 'S'), 'bottle': (0, 0), 'classify': (0, 0),
        'trans': ('B', 'B'), 'reduction': (0.5, 0),
        'last_branch': 1, 'last_down': True, 'last_expand': 0,
        'summer': 'split', 'nclass': 1000}

nm3d = {'stages': 3, 'indepth': 6, 'growth': 3, 'poolmode': 'max', 'active': 'relu',
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'bottle': (0, 0, 0),
        'classify': (0, 0, 0), 'trans': ('B', 'B', 'B'), 'reduction': (0.5, 0.5, 0),
        'last_branch': 1, 'last_down': True, 'last_expand': 10,
        'summer': 'split', 'nclass': 1000}

model = NameiNet(**nm1d)
print('\n', model, '\n')


# model = tv.models.resnet18()

# utils.tensorboard_add_model(model, x)
xtils.calculate_params_scale(model, format='million')
xtils.calculate_FLOPs_scale(model, input_size=224, use_gpu=False, multiply_adds=True)
xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
x = torch.randn(4, 3, 224, 224)
tic, toc = time.time(), 1
y = [model(x) for _ in range(toc)][0]
toc = (time.time() - tic) / toc
print('处理时间: %.5f 秒\t' % toc)
