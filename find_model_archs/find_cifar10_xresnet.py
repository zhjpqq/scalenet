# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils
from xmodels.xresnet import XResNet

xs1 = {'block': 'B', 'nblocks': (3, 3, 3), 'inplanes': 16,
       'bottle': 0.8, 'active': 'fr', 'dataset': 'cifar10'}

xs2 = {'block': 'B', 'nblocks': (3, 3, 3), 'inplanes': 16,
       'bottle': 0.8, 'active': 'fg', 'dataset': 'cifar10'}

xs3 = {'block': 'B', 'nblocks': (3, 3, 3), 'inplanes': 16,
       'bottle': 0.8, 'active': 'fs', 'dataset': 'cifar10'}   # 0.22M  0.03G  20L  0.029s

xs4 = {'block': 'B', 'nblocks': (3, 3, 3), 'inplanes': 16,
       'bottle': 0.8, 'active': 'fs', 'dataset': 'cifar10'}

model = XResNet(**xs4)
print('\n', model, '\n')

# utils.tensorboard_add_model(model, x)
xtils.calculate_params_scale(model, format='million')
xtils.calculate_FLOPs_scale(model, input_size=32)
xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
xtils.calculate_time_cost(model, insize=32, toc=3, use_gpu=False, pritout=True)

if [True, False][1]:
    # 查看模型的权值衰减曲线
    from config.configure import Config

    cfg = Config()
    cfg.decay_fly = {'flymode': ['nofly', 'stepall'][1],
                     'a': 0, 'b': 5, 'f': xtils.Curves(6).func1, 'wd_bn': None,
                     'weight_decay': 0.0001, 'wd_start': 0.00001, 'wd_end': 0.0001}
    cfg.decay_fly = {'flymode': 'stepall', 'a': 0, 'b': 1, 'f': xtils.Curves().func1, 'weight_decay': 0.0001,
                     'wd_start': 0, 'wd_end': 1, 'wd_bn': 0}
    model.visual_weight_decay(cfg=cfg, visual=True)

    # # 查看模型的可学习参数列表
    # for n, p in model.named_parameters():
    #     print('---->', n, '\t', p.size())
