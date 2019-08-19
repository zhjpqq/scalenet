# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils
from xmodels.richnet import RichNet

ch1 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (6, 7, 8), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 512 - 256,
       'version': 2}  # 9.39M  2.38G 44L  0.4s  512fc   ==> vo6

ch2 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 72, 'kldloss': False,
       'layers': (2, 2, 2), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 72, 2 * 72),
       'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'),
       'fcboost': 'none', 'nclass': 1000, 'summer': 'split', 'afisok': False,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 512 - 288,
       'version': 2}  # 6.10M  1.73G  27L  0.28s  512fc

ch3 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 72, 'kldloss': False,
       'layers': (2, 2, 2), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (-4, -4, -8), 'classify': (0, 0, 0), 'expand': (1 * 72, 2 * 72),
       'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'),
       'fcboost': 'none', 'nclass': 1000, 'summer': 'split', 'afisok': False,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 512 - 288,
       'version': 2}  # 6.16M  1.68G  32L  0.31s  512fc

model = RichNet(**ch1)
print('\n', model, '\n')

# train_which & eval_which 在组合上必须相互匹配
# model.train_which(part=['conv+rock', 'xfc+boost', 'xfc-only', 'boost-only'][1])
model.set_eval_which(part=['conv+rock', 'conv+rock+xfc', 'conv+rock+boost', 'conv+rock+xfc+boost'][1])
# print(model.stage1[1].conv1.training)
# print(model.stage1[1].classifier.training)
# print(model.stage2[0].classifier.training)
# print(model.summary.classifier1.training)

# model = tv.models.resnet18()

# utils.tensorboard_add_model(model, x)
xtils.calculate_params_scale(model, format='million')
xtils.calculate_FLOPs_scale(model, input_size=224)
xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
x = torch.randn(4, 3, 224, 224)
tic, toc = time.time(), 1
y = [model(x) for _ in range(toc)][0]
toc = (time.time() - tic) / toc
print('处理时间: %.5f 秒\t' % toc)
