# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils
import torchvision as tv
import xmodels.tvm_densenet as tvmd
import xmodels.tvm_resnet as tvmr
from xmodels.dxnet import DxNet

# # imageNet
exp5 = {'stages': 5, 'branch': 3, 'rock': 'N', 'depth': 16, 'kldloss': False,
        'layers': (6, 5, 4, 3, 2), 'blocks': ('D', 'D', 'D', 'S', 'S'), 'slink': ('A', 'A', 'A', 'A', 'A'),
        'growth': (0, 0, 0, 0, 0), 'classify': (1, 1, 1, 1, 1),
        'expand': (1 * 16, 2 * 16, 4 * 16, 8 * 16), 'dfunc': ('O', 'D', 'O', 'A'),
        'fcboost': 'none', 'nclass': 1000, 'summer': 'merge', 'version': 5,
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'A', 'last_expand': 15}

exp4 = {'stages': 4, 'branch': 3, 'rock': 'N', 'depth': 16, 'kldloss': False,
        'layers': (6, 5, 4, 3), 'blocks': ('D', 'D', 'D', 'S'), 'slink': ('A', 'A', 'A', 'A'),
        'growth': (0, 0, 0, 0), 'classify': (0, 0, 1, 1), 'expand': (10, 20, 30),
        'dfunc': ('O', 'D', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
        'last_branch': 2, 'last_down': False, 'last_dfuc': 'A', 'last_expand': 15,
        'version': 5}

exp3 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
        'layers': (3, 4, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120),
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'),
        'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 256,
        'version': 5}  # 10.35M  1.63G  26L  0.27s

# ---------------------------------------------------------------------------------------------
dx1 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
       'layers': (3, 4, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120),
       'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'),
       'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 256,
       'version': 5}  # 10.35M  1.63G  26L  0.27s  656fc  523M   0.0065s  @1080TI

dx2 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
       'layers': (4, 7, 9), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120),
       'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'),
       'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 1400 - 400,
       'version': 5}  # 20.83M  2.88G  39L  0.45s  598M  0.0111s  @1080TI

dx3 = {'stages': 2, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
       'layers': (15, 15), 'blocks': ('D', 'D'), 'slink': ('A', 'A'),
       'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 80,),
       'dfunc': ('O',), 'dstyle': ('maxpool', 'convk2m', 'convk2'),
       'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 256,
       'version': 5}

dx4 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
       'layers': (20,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (),
       'dfunc': (), 'dstyle': ('maxpool', 'convk2m', 'convk2'),
       'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 1024 - 80,
       'version': 5}  # 5.06M  4.20G  67L  0.81s

model = DxNet(**dx3)

# model = tv.models.resnet50()
# resnet18-11.68M-3.62G-18L-0.39s
# resnet34-21.80M-7.34G-37L-0.70s
# resnet50-25.55M-8.20G-54L-25.19%-0.93s
# resnet101-44.55M-15.64G-105L-24.10%-1.75s

# model = tvmd.densenet169()
# model = tvmd.densenet201()
# model = tvmd.densenet161()
# dense169-14.15M-6.76G-169L-25.76%-1.27s
# dense201-20.01M-8.63G-201L-25.33%-1.57s
# dense161-28.68M-15.52G-161L-23.97%-2.19S
# dense264-33.34M-5.82G-264L

# model.eval()
print('\n', model, '\n')
# utils.tensorboard_add_model(model, x)
xtils.calculate_params_scale(model, format='million')
xtils.calculate_FLOPs_scale(model, input_size=224, multiply_adds=True)
xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
xtils.calculate_time_cost(model, insize=224, toc=1, use_gpu=False, pritout=True)
print('\nover!')

# 查看模型的权值衰减曲线
if [True, False][1]:
    from config.configure import Config

    cfg = Config()
    cfg.weight_decay = 0.0001
    cfg.decay_fly = {'flymode': ['nofly', 'stepall'][1],
                     'a': 0, 'b': 1, 'f': xtils.Curves(4).func2,
                     'wd_start': 0.0000001, 'wd_end': 0.00001, 'wd_bn': None}
    model.visual_weight_decay(cfg=cfg, visual=True)
