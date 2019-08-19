# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time, torchvision as tv
import xtils
from xmodels.vivonet_v1 import VivoNet

# imageNet

ve1 = {}

model = VivoNet(**ve1)
print('\n', model, '\n')

# model = tv.models.resnet50()
# resnet18-11.68M-1.82G-18L-0.37s
# resnet34-21.80M-3.67G-37L-0.70s
# resnet50-25.55M-4.11G-54-0.93s

# model = tv.models.densenet121()
# print(model)
# dense201-20.01M-4.34G-201L

# utils.tensorboard_add_model(model, x)
xtils.calculate_params_scale(model, format='million')
xtils.calculate_FLOPs_scale(model, input_size=224)
xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
x = torch.randn(4, 3, 224, 224)
tic, toc = time.time(), 1
y = [model(x) for _ in range(toc)][0]
toc = (time.time() - tic) / toc
print('处理时间: %.5f 秒\t' % toc)
