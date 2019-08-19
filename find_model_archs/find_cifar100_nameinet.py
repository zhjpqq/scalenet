# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils
from xmodels.nameinet import NameiNet

nm1s = {'stages': 1, 'indepth': 24, 'growth': 12,
        'layers': (3, 0), 'blocks': ('D', '-'), 'bottle': (0, 0), 'classify': (0, 0),
        'trans': ('B', '-'), 'reduction': (0, 0),
        'last_branch': 1, 'last_down': True, 'last_expand': 0,
        'poolmode': 'max', 'active': 'relu', 'summer': 'split', 'nclass': 100}

nm2s = {'stages': 2, 'indepth': 24, 'growth': 12,
        'layers': (3, 3), 'blocks': ('D', 'S'), 'bottle': (0, 0), 'classify': (0, 0),
        'trans': ('B', 'B'), 'reduction': (0.5, 0),
        'last_branch': 1, 'last_down': False, 'last_expand': 0,
        'poolmode': 'max', 'active': 'relu', 'summer': 'split', 'nclass': 100}

nm3s = {'stages': 3, 'indepth': 24, 'growth': 8,
        'layers': (10, 10, 10), 'blocks': ('D', 'D', 'S'), 'bottle': (0, 0, 0), 'classify': (0, 0, 0),
        'trans': ('B', 'B', 'B'), 'reduction': (0.5, 0.5, 0),
        'last_branch': 1, 'last_down': False, 'last_expand': 10,
        'poolmode': 'max', 'active': 'relu', 'summer': 'split', 'nclass': 100}

model = NameiNet(**nm3s)
print('\n', model, '\n')

use_gpu = False and torch.cuda.is_available()

# utils.tensorboard_add_model(model, x)
xtils.calculate_params_scale(model, format='million')
xtils.calculate_FLOPs_scale(model, input_size=32, use_gpu=use_gpu)
xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
x = torch.randn(4, 3, 32, 32)
if use_gpu:
    x = x.cuda()
    model = model.cuda()
tic, toc = time.time(), 3
y = [model(x) for _ in range(toc)][0]
toc = (time.time() - tic) / toc
print('处理时间: %.5f 秒' % (toc,))
