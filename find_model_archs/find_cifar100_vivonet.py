# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils
from xmodels.vivonet_v1 import VivoNet

vp1 = {'depth': 40, 'growthRate': 24, 'reduction': 0.5, 'nClasses': 100,
       'kinds': (3, 3, 3), 'dividers': ((10, 8, 6), (10, 8, 6), (10, 8, 6)),
       'branchs': ('v1', 'v1', 'v1')}  # 0.71M 0.29G  94L

model = VivoNet(**vp1)
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
