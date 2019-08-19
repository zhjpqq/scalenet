# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils
from xmodels.richnet import RichNet
import arch_params
import cfg_params

rh1 = {'stages': 1, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (7,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (-10,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.03M  32L  0.03s

rh2 = {'stages': 2, 'depth': 27, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (20, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (8, 8), 'classify': (0, 0), 'expand': (1 * 27,), 'dfunc': ('O',),
       'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 64,
       'afisok': False, 'version': 2}  # 1.27MM  0.36G 98L  0.07s  =>ci13

rh3 = {'stages': 2, 'depth': 32, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (20, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (8, 8), 'classify': (0, 0), 'expand': (1 * 32,), 'dfunc': ('O',),
       'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 32,
       'afisok': False, 'version': 2}  # 1.65MM  0.48G  98L  0.09s

model = RichNet(**rh3)
print('\n', model, '\n')

# train_which & eval_which 在组合上必须相互匹配
# model.set_train_which(part=['conv+rock', 'xfc+boost', 'xfc-only', 'boost-only'][1])
model.set_eval_which(part=['conv+rock', 'conv+rock+xfc', 'conv+rock+boost', 'conv+rock+xfc+boost'][0])
# print(model.stage1[1].conv1.training)
# print(model.stage1[1].classifier.training)
# print(model.stage2[0].classifier.training)
# print(model.summary.classifier1.training)

# utils.tensorboard_add_model(model, x)
xtils.calculate_params_scale(model, format='million')
xtils.calculate_FLOPs_scale(model, input_size=32)
xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
x = torch.randn(4, 3, 32, 32)
tic, toc = time.time(), 3
y = [model(x) for _ in range(toc)][0]
toc = (time.time() - tic) / toc
print('有效分类支路：', len(y), '\t共有blocks：', sum(model.layers), '\t处理时间: %.5f 秒' % toc)
print(len(y), sum(model.layers), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])
