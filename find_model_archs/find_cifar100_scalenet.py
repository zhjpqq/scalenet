# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils

nn = torch.nn
from xmodels.scalenet import ScaleNet
from config.configure import Config

vx22 = {'stages': 3, 'depth': 16, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (78, 28, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (-3, -3, -3), 'classify': (0, 0, 0), 'expand': (1 * 11, 1 * 16),
        'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
        'afisok': False}  # 1.74M   444L  0.18s

vx23 = {'stages': 3, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (2, 2, 2), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 32, 2 * 32),
        'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
        'afisok': False}  # 1.72M   32L  0.26s

vx8 = {'stages': 2, 'depth': 16, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (50, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (4, 4), 'classify': (0, 0), 'expand': (1 * 16,), 'dfunc': ('O',),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 32,
       'afisok': False}  # 1.23M  220L  0.17s

vx24 = {'stages': 2, 'depth': 32, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (20, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 32,), 'dfunc': ('O',),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 64,
        'afisok': False}  # 1.72M  100L  0.10s

vx25 = {'stages': 2, 'depth': 32, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (10, 10), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 32,), 'dfunc': ('O',),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 64,
        'afisok': False, 'version': 2}  # 1.71M  67L  0.08s

vx26 = {'stages': 2, 'depth': 40, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (6, 6), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 40,), 'dfunc': ('O',),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 80,
        'afisok': False, 'version': 2}  # 1.70M  43L  0.07s

vx27 = {'stages': 2, 'depth': 32, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (17, 6), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 32,), 'dfunc': ('O',),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 100,
        'afisok': False, 'version': 2}  # 1.72M  87L  0.11s

vx28 = {'stages': 2, 'depth': 32, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (18, 6), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 32,), 'dfunc': ('O',),
        'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 64,
        'afisok': False, 'version': 2}  # 1.71M  91L  0.11s

vx29 = {'stages': 2, 'depth': 32, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (20, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 32,), 'dfunc': ('O',),
        'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 64,
        'afisok': False, 'version': 2}  # 1.71M  97L  0.12s

vv29 = {'stages': 2, 'depth': 32, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (20, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 32,), 'dfunc': ('O',),
        'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 64,
        'afisok': False, 'version': 2}  # 1.71M  97L  0.12s

vx30 = {'stages': 2, 'depth': 40, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (18, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (-4, -4), 'classify': (0, 0), 'expand': (1 * 40,), 'dfunc': ('O',),
        'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 64,
        'afisok': False, 'version': 2}  # 1.74M  91L  0.10s

vx31 = {'stages': 2, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (5, 4), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 36,), 'dfunc': ('O',),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 64,
        'afisok': False, 'version': 2}  # 1.04M  35L  0.04s

yy1 = {'stages': 2, 'depth': 24, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (21, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 24,), 'dfunc': ('O',),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 82,
       'afisok': False, 'version': 2}  # 1.04M  101L  0.08s

yy2 = {'stages': 2, 'depth': 24, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10, 10), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 24,), 'dfunc': ('O',),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 82,
       'afisok': False, 'version': 2}  # 1.01M  67L  0.05s

yy3 = {'stages': 2, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (2, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (-3, 0), 'classify': (0, 0), 'expand': (1 * 36,), 'dfunc': ('O',),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 100,
       'afisok': False, 'version': 2}  # 1.04M  28L  0.03s

yy4 = {'stages': 2, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (4, 3), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (2, 4), 'classify': (0, 0), 'expand': (1 * 36,), 'dfunc': ('O',),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 128,
       'afisok': False, 'version': 2}  # 1.04M  32L  0.04s

cc1 = {'stages': 3, 'depth': 48, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (9, 8, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (2, 5, 8), 'classify': (0, 0, 0), 'expand': (1 * 48, 2 * 48),
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'afisok': False}  # 10.3M   90L  0.13s

cc2 = {'stages': 3, 'depth': 48, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (9, 8, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (2, 5, 8), 'classify': (0, 0, 0), 'expand': (1 * 48, 2 * 48),
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'afisok': False}  # 10.3M   90L  0.13s  == cc1, wd不同

cs1 = {'stages': 3, 'depth': 48, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (12, 9, 9), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (2, 5, 8), 'classify': (0, 0, 0), 'expand': (1 * 48, 2 * 48),
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'afisok': False, 'version': 2}  # 15.02M   114L  0.17s

cs2 = {'stages': 3, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (30, 30, 7), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 36, 2 * 36),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 100,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 15.10M  304L  0.26s

cs4 = {'stages': 3, 'depth': 56, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (8, 8, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 56, 2 * 56),
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'afisok': False, 'version': 2}  # 15.00M   87L  0.09s  224fc  == cs3 on titan

cs6 = {'stages': 3, 'depth': 48, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (23, 10, 7), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 48, 2 * 48),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 100,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 15.12M  157L  0.22s  192fc

cs7 = {'stages': 4, 'depth': 40, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (3, 3, 3, 3), 'blocks': ('D', 'D', 'D', 'S'), 'slink': ('A', 'A', 'A', 'A'),
       'growth': (4, 4, 4, 8), 'classify': (0, 0, 0, 0), 'expand': (1 * 42, 2 * 42, 4 * 42),
       'dfunc': ('O', 'O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 100,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 15.13M  58l  0.07s   334fc

cs8 = {'stages': 3, 'depth': 56, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (12, 9, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 56, 2 * 56),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 100,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 512 - 224,
       'summer': 'split', 'afisok': False, 'version': 2}  # 15.05M  101L  0.17s

cj3 = {'stages': 3, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (7, 7, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 36, 2 * 36),
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 100,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 5.0M  72L  0.07s

cj4 = {'stages': 3, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (7, 6, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 36, 2 * 36),
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 100,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'C', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 5.2M  73L  0.07s

cj5 = {'stages': 3, 'depth': 16, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (272, 200, 50), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (2, 2, 4), 'classify': (0, 0, 0), 'expand': (1 * 10, 2 * 10),
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'afisok': False}  # 15.23M  2000L  0.65s

ro1 = {'stages': 1, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (12,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (6,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.06M  52L  0.06s

ro2 = {'stages': 1, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (7,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (-10,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.03M  32L  0.06s

de1 = {'stages': 2, 'depth': 64, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (4, 3), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 64,), 'dfunc': ('O',),
       'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 0,
       'afisok': False, 'version': 3}  # 1.67M  28L  0.39G  0.05s

de2 = {'stages': 2, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (4, 3), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 64,), 'dfunc': ('O',),
       'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 0,
       'afisok': False, 'version': 3}  # 1.67M  28L  0.39G  0.05s

vx32 = {'stages': 3, 'depth': 16, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (78, 28, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (-3, -3, 4), 'classify': (0, 0, 0), 'expand': (1 * 11, 1 * 16),
        'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
        'afisok': False, 'version': 1}  # 1.70M  0.46G  444L  0.21s

xb2 = {'stages': 1, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (1,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 2.06M  0.96G  44L  0.10s  ==>ax18  + xfc

vx35 = {'stages': 1, 'depth': 75, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
        'growth': (0,), 'classify': (1,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 3}  # 2.04M  0.84G  44L  0.11s  ==>ax18  + xfc   74.02%

vx36 = {'stages': 1, 'depth': 75, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
        'growth': (0,), 'classify': (1,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 3}  # 2.04M  0.84G  44L  0.11s  ==>ax18  + xfc

vx39 = {'stages': 1, 'depth': 80, 'branch': 1, 'rock': 'U', 'kldloss': False,
        'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
        'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 3}  # 2.32M  0.93G  42L  0.11s  ==>ax18  + xfc   75.52%

vx41 = {'stages': 1, 'depth': 76, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
        'growth': (0,), 'classify': (1,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 3}  # 2.09M  0.84G  44L  0.11s  ==>ax18  + xfc  75.11%

vx43 = {'stages': 1, 'depth': 100, 'branch': 1, 'rock': 'U', 'kldloss': False,
        'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
        'growth': (0,), 'classify': (1,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 3}  # 3.72M  1.44G  52L  0.11s  ==>ax18  + xfc

vx40 = {'stages': 1, 'depth': 100, 'branch': 1, 'rock': 'U', 'kldloss': False,
        'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
        'growth': (0,), 'classify': (1,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False,
        'version': 3}  # 3.72M  1.44G  42L+10L=52L  0.11s  76.09%  ==>ax18  + xfc  // load vx43 to Train xfc

# NET WIDTH
bo1 = {'stages': 1, 'depth': 100, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 3.62M  1.44G  42L  0.22s

bo2 = {'stages': 1, 'depth': 90, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 2.93M  1.18G  42L  0.22s

bo3 = {'stages': 1, 'depth': 80, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 2.32M  0.93G  42L  0.22s

bo4 = {'stages': 1, 'depth': 70, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 1.78M  0.71G  42L  0.17s

bo5 = {'stages': 1, 'depth': 60, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 1.30M  0.52G  42L  0.17s

model = ScaleNet(**bo1)
print('\n', model, '\n')

# train_which & eval_which 在组合上必须相互匹配
# model.train_which(part=['conv+rock', 'xfc+boost', 'xfc-only', 'boost-only'][1])
# model.set_train_which(part=['conv+rock', 'conv+rock+xfc', 'conv+rock+boost', 'conv+rock+xfc+boost', 'rock'][0])
model.set_eval_which(part=['conv+rock', 'conv+rock+xfc', 'conv+rock+boost', 'conv+rock+xfc+boost'][1])
# print(model.stage1[1].conv1.training)
# print(model.stage1[1].classifier.training)
# print(model.stage2[0].classifier.training)
# print(model.summary.classifier1.training)

# utils.tensorboard_add_model(model, x)
xtils.calculate_params_scale(model, format='million')
xtils.calculate_FLOPs_scale(model, input_size=32, use_gpu=False)
xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
xtils.calculate_time_cost(model, insize=32, toc=3, use_gpu=False, pritout=True)

# #### 优化器  ########
#
# # parameters = model.parameters()
# # parameters = list(parameters)
# # optimizer = torch.optim.SGD(parameters[:-2], 0.1, momentum=0.9, weight_decay=0.0003)
# # optimizer.add_param_group({'params': parameters[-2:], 'lr': 0.3, 'momentum': 0.5, 'weight_decay': 0.0005})  # linear
# # print(optimizer)
#
# nparams = model.named_parameters()
# names = [n for n, p in nparams]
# print('\n', names, '\n')
#
# cfg = Config()
# cfg.linear_fly = {'isfly': True, 'lr': 0.2, 'momentum': 0.3, 'weight_decay': 0.0005}
#
# model = [model, nn.DataParallel(model, device_ids=[0, 1])][1]
#
# # optimizer
# if isinstance(model, nn.DataParallel):
#     optimizer = model.module.init_optimizer(cfg=cfg)
# elif isinstance(model, nn.Module):
#     optimizer = model.init_optimizer(cfg=cfg)
# else:
#     raise NotImplementedError
# print(optimizer)
