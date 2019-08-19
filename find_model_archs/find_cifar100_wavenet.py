# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils
from xmodels.wavenet import WaveNet

we1s = {'stages': 1, 'rock': 'Q', 'branch': 3, 'indepth': 24, 'growth': 12, 'multiway': 4,
        'layers': (3, 0, 0), 'blocks': ('D', '-', '-'), 'bottle': (0, 0, 0), 'classify': (0, 0, 0),
        'trans': ('A', '-', '-'), 'reduction': (0, 0, 0),
        'last_branch': 1, 'last_down': True, 'last_expand': 0,
        'poolmode': 'avg', 'active': 'relu', 'summer': 'split', 'nclass': 100,
        'afisok': False, 'afkeys': ('af1', 'af2'), 'convon': True,
        'convlayers': 1, 'convdepth': 4}

we2s = {'stages': 2, 'rock': 'Q', 'branch': 3, 'indepth': 24, 'growth': 12, 'multiway': 4,
        'layers': (3, 3, 0), 'blocks': ('D', 'D', '-'), 'bottle': (3, 3, 3), 'classify': (0, 0, 0),
        'trans': ('A', 'A', '-'), 'reduction': (0.5, 0, 0),
        'last_branch': 1, 'last_down': True, 'last_expand': 0,
        'poolmode': 'avg', 'active': 'relu', 'summer': 'split', 'nclass': 100,
        'afisok': False, 'afkeys': ('af1', 'af2'), 'convon': True,
        'convlayers': 1, 'convdepth': 4}

we3s = {'stages': 3, 'rock': 'Q', 'branch': 3, 'indepth': 24, 'growth': 12, 'multiway': 4,
        'layers': (10, 10, 10), 'blocks': ('D', 'D', 'S'), 'bottle': (0, 0, 0), 'classify': (0, 0, 0),
        'trans': ('A', 'A', 'A'), 'reduction': (0.5, 0.5, 0),
        'last_branch': 1, 'last_down': False, 'last_expand': 10,
        'poolmode': 'avg', 'active': 'relu', 'summer': 'split', 'nclass': 100,
        'afisok': False, 'afkeys': ('af1', 'af2', 'af3'), 'convon': True,
        'convlayers': 1, 'convdepth': 4}

we1 = {'stages': 1, 'rock': 'Q', 'branch': 3, 'indepth': 24, 'growth': 12, 'multiway': 4,
       'layers': (17, 0, 0), 'blocks': ('D', '-', '-'), 'bottle': (0, 0, 0),
       'trans': ('A', '-', '-'), 'reduction': (0, 0, 0), 'classify': (0, 0, 0),
       'last_branch': 1, 'last_down': True, 'last_expand': 0, 'afisok': False,
       'poolmode': 'avg', 'active': 'relu', 'summer': 'split', 'nclass': 100}  # 1.05M  0.44G  73L  0.17S

we2 = {'stages': 2, 'rock': 'Q', 'branch': 3, 'indepth': 24, 'growth': 12, 'multiway': 4,
       'layers': (13, 13, 0), 'blocks': ('D', 'D', '-'), 'bottle': (0, 0, 0), 'classify': (0, 0, 0),
       'trans': ('A', 'A', '-'), 'reduction': (0.5, 0, 0), 'afisok': False,
       'last_branch': 1, 'last_down': True, 'last_expand': 0, 'poolmode': 'avg',
       'active': 'relu', 'summer': 'split', 'nclass': 100}   # 1.74M  0.38G  113L  0.161s

we3 = {'stages': 3, 'rock': 'Q', 'branch': 3, 'indepth': 24, 'growth': 12, 'multiway': 4,
       'layers': (11, 11, 11), 'blocks': ('D', 'D', 'S'), 'bottle': (0, 0, 0), 'classify': (0, 0, 0),
       'trans': ('A', 'A', 'A'), 'reduction': (0.5, 0.5, 0), 'afisok': False,
       'last_branch': 1, 'last_down': False, 'last_expand': 10, 'poolmode': 'avg',
       'active': 'relu', 'summer': 'split', 'nclass': 100}  # 1.75M  0.30G  122L  0.16s

we4 = {'stages': 3, 'rock': 'Q', 'branch': 3, 'indepth': 24, 'growth': 18, 'multiway': 4,
       'layers': (10, 10, 10), 'blocks': ('D', 'D', 'S'), 'bottle': (0, 0, 0), 'classify': (0, 0, 0),
       'trans': ('A', 'A', 'A'), 'reduction': (0.5, 0.5, 0), 'afisok': False,
       'last_branch': 1, 'last_down': False, 'last_expand': 10, 'poolmode': 'avg',
       'active': 'relu', 'summer': 'split', 'nclass': 100}  # 3.11M  0.53G  112l  0.17s  wd=0.00001


we5 = {'stages': 3, 'rock': 'Q', 'branch': 3, 'indepth': 24, 'growth': 23, 'multiway': 4,
       'layers': (10, 10, 10), 'blocks': ('D', 'D', 'S'), 'bottle': (0, 0, 0), 'classify': (0, 0, 0),
       'trans': ('A', 'A', 'A'), 'reduction': (0.5, 0.5, 0), 'afisok': False,
       'last_branch': 1, 'last_down': False, 'last_expand': 10, 'poolmode': 'avg',
       'active': 'relu', 'summer': 'split', 'nclass': 100}  # 4.94M  0.83G  112l  0.22s


we6 = {'stages': 3, 'rock': 'Q', 'branch': 3, 'indepth': 36, 'growth': 33, 'multiway': 4,
       'layers': (10, 10, 10), 'blocks': ('D', 'D', 'S'), 'bottle': (0, 0, 0), 'classify': (0, 0, 0),
       'trans': ('A', 'A', 'A'), 'reduction': (0.5, 0.5, 0), 'afisok': False,
       'last_branch': 1, 'last_down': False, 'last_expand': 10, 'poolmode': 'avg',
       'active': 'relu', 'summer': 'split', 'nclass': 100}  # 10.10M  1.71G  112l  0.30s

model = WaveNet(**we2)
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