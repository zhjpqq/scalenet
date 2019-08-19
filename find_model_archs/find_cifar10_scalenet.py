# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils
from xmodels.scalenet import ScaleNet

tx2 = {'stages': 3, 'depth': 16, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (272, 200, 50), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (2, 2, 4), 'classify': (0, 0, 0), 'expand': (1 * 10, 2 * 10),
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'afisok': False}  # 15.23M  2000L  0.65s

tx3 = {'stages': 3, 'depth': 24, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (70, 70, 15), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (2, 3, 4), 'classify': (0, 0, 0), 'expand': (1 * 24, 2 * 24),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False}  # 15.20M  602L  0.26s

tx4 = {'stages': 3, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (30, 30, 26), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 24, 2 * 24),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False}  # 15.10M  304L  0.24s

tx5 = {'stages': 3, 'depth': 32, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (31, 32, 21), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 32, 2 * 32),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 15.08M  306L  0.24s

gd1 = {'stages': 2, 'depth': 3, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (48, 3), 'blocks': ('D', 'S'), 'slink': ('A', 'A'), 'growth': (16, 8),
       'classify': (0, 0), 'dfunc': ('O',), 'dstyle': 'maxpool', 'expand': (32,),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split', 'version': 2,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 64 - 35,
       'afisok': False}  # 1.01M  204L  0.07s  64fc

gd2 = {'stages': 3, 'depth': 3, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (12, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (24, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 32, 65),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'C', 'last_expand': 33,
       'summer': 'split', 'afisok': False, 'version': 2}  # 1.52M  75L  0.03s  100fc

gd3 = {'stages': 2, 'depth': 3, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (22, 3), 'blocks': ('D', 'S'), 'slink': ('A', 'A'), 'growth': (24, 8),
       'classify': (0, 0), 'dfunc': ('O',), 'dstyle': 'maxpool', 'expand': (40,),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split', 'version': 2,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 64 - 43,
       'afisok': False}  # 1.00M  100L  0.04s  64fc

xw1 = {'stages': 3, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (18, 18, 18), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 36, 2 * 36),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 15.25M  186L  0.18s

ci1 = {'stages': 2, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (5, 4), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 36,), 'dfunc': ('O',),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 64,
       'afisok': False, 'version': 2}  # 1.04M  35L  0.04s  ==vx31 ==yy6

ci2 = {'stages': 2, 'depth': 32, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (20, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 32,), 'dfunc': ('O',),
       'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 64,
       'afisok': False, 'version': 2}  # 1.71M  97L  0.1s

ci3 = {'stages': 3, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (7, 7, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 36, 2 * 36),
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 4.9M  72L  0.07s

ci4 = {'stages': 3, 'depth': 48, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (9, 8, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (2, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 48, 2 * 48),
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 10.2M  90L  0.12s

ci5 = {'stages': 3, 'depth': 48, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (23, 10, 7), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 48, 2 * 48),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 15.12M  157L  0.22s  192fc

ci6 = {'stages': 2, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (5, 3), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (6, 8), 'classify': (0, 0), 'expand': (1 * 36,), 'dfunc': ('O',),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.03M  35L  0.04s

ci7 = {'stages': 3, 'depth': 22, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 22, 2 * 22),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 1M  35L  0.22s  88fc

ci8 = {'stages': 3, 'depth': 16, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10, 7, 9), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 16, 2 * 16),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 1.73M  97L  0.22s  64fc

ci9 = {'stages': 3, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (11, 10, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 24, 2 * 24),
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 5.0M  108  0.07s

ci10 = {'stages': 2, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (5, 3), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (6, 8), 'classify': (0, 0), 'expand': (1 * 36,), 'dfunc': ('O',),
        'fcboost': 'none', 'nclass': 10, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 2}  # 1.03M  35L  0.04s

ci12 = {'stages': 2, 'depth': 42, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (5, 3), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 42,), 'dfunc': ('O',),
        'fcboost': 'none', 'nclass': 10, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 32,
        'afisok': False, 'version': 2}  # 1.03M  35L  0.04s

ci13 = {'stages': 2, 'depth': 27, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (20, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
        'growth': (8, 8), 'classify': (0, 0), 'expand': (1 * 27,), 'dfunc': ('O',),
        'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 64,
        'afisok': False, 'version': 2}  # 1.71M  97L  0.1s

ci14 = {'stages': 3, 'depth': 48, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (2, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 48, 2 * 48),
        'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10,
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
        'summer': 'split', 'afisok': False, 'version': 2}  # 5.02M  42L  0.12s

ci15 = {'stages': 3, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (6, 6, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (2, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 36, 2 * 36),
        'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10,
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
        'summer': 'split', 'afisok': False, 'version': 2}  # 5.02M  70L  0.12s

ci16 = {'stages': 3, 'depth': 48, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (11, 7, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 48, 2 * 48),
        'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10,
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
        'summer': 'split', 'afisok': False, 'version': 2}  # 10.12M  90L  0.22s  192fc

ci17 = {'stages': 1, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (12,), 'blocks': ('D',), 'slink': ('A',),
        'growth': (6,), 'classify': (0,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 10, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 2}  # 1.06M  52L  0.06s

ci18 = {'stages': 1, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (12,), 'blocks': ('D',), 'slink': ('A',),
        'growth': (6,), 'classify': (0,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 10, 'summer': 'convt',
        'last_branch': 3, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 2}  # 1.06M  52L  0.06s

cu13 = {'stages': 2, 'depth': 27, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (20, 5), 'blocks': ('D', 'S'), 'slink': ('C', 'C'),
        'growth': (8, 8), 'classify': (0, 0), 'expand': (1 * 27,), 'dfunc': ('O',),
        'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 64,
        'afisok': False, 'version': 2}  # 1.71M  97L  0.1s  ==ci13  but slink

bx1 = {'stages': 1, 'depth': 50, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (4,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'convf',
       'last_branch': 3, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.04M  20L  0.05s

bx2 = {'stages': 1, 'depth': 50, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (8,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.04M  36L  0.05s

ne1 = {'stages': 2, 'depth': 64, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (4, 3), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 64,), 'dfunc': ('O',),
       'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 0,
       'afisok': False, 'version': 3}  # 1.67M  28L  0.39G  0.05s

ne2 = {'stages': 2, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (4, 3), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
       'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 64,), 'dfunc': ('O',),
       'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 0,
       'afisok': False, 'version': 3}  # 1.67M  28L  0.39G  0.05s

# wave 系列， wa开头

wa1 = {'stages': 1, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (7,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (-10,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.03M  32L 0.53G  0.03s  ==>ax14  5.5%

wa2 = {'stages': 1, 'depth': 64, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (8,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.65M  36L  0.77G  0.08s

pm1 = {'stages': 3, 'depth': 16, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (100, 100, 100), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 16, 2 * 16),
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'afisok': False, 'version': 3}  # 14.5M  1.28G  1011L  0.43s

ax14a = {'stages': 1, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
         'layers': (7,), 'blocks': ('D',), 'slink': ('C',),
         'growth': (-10,), 'classify': (0,), 'expand': (), 'dfunc': (),
         'fcboost': 'none', 'nclass': 10, 'summer': 'split',
         'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
         'afisok': False, 'version': 2}  # 1.03M  32L  0.03s  slink=C

ro3a = {'stages': 1, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (7,), 'blocks': ('D',), 'slink': ('C',),
        'growth': (-10,), 'classify': (0,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 100, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 2}  # 1.03M  32L  0.06s  ==ro2, but wd, slink=C

ci13a = {'stages': 2, 'depth': 27, 'branch': 1, 'rock': 'U', 'kldloss': False,
         'layers': (20, 5), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
         'growth': (8, 8), 'classify': (0, 0), 'expand': (1 * 27,), 'dfunc': ('O',),
         'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
         'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 64,
         'afisok': False, 'version': 2}  # 1.71M  96L  0.11s  ==>ci13 branch=1

ci19 = {'stages': 1, 'depth': 64, 'branch': 1, 'rock': 'U', 'kldloss': False,
        'layers': (8,), 'blocks': ('D',), 'slink': ('A',),
        'growth': (-5,), 'classify': (0,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 10, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 3}  # 1.06M  0.94G  42L  0.08s  93.64%  ==> uv3/titan/94.15%

cj2 = {'stages': 1, 'depth': 64, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (8,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (-5,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 1.00M  0.42G  34L  0.06s  93.81%   ==> uv3/titan/94.15%  200ep

cj3 = {'stages': 1, 'depth': 64, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (6,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (5,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 1.03M  0.40G  26L  0.05s      200ep

cj4 = {'stages': 1, 'depth': 64, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (6,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (5,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 1.03M  0.40G  26L  0.05s   300ep

cj5 = {'stages': 1, 'depth': 64, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (5,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (10,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 1.00M  0.36G  22L  0.05s   run   300ep

ax16 = {'stages': 1, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
        'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 10, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 2}  # 2.06M  44L  0.05s  94.75%

ax18 = {'stages': 1, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
        'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
        'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
        'fcboost': 'none', 'nclass': 10, 'summer': 'split',
        'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
        'afisok': False, 'version': 2}  # 2.06M  44L + 10L = 54L  0.05s  ==>ax16  but xfc-only


uv7 = {'stages': 3, 'depth': 16, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (236, 236, 50), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (2, 2, 4), 'classify': (0, 0, 0), 'expand': (1 * 10, 2 * 10),
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'afisok': False, 'version': 3}  # 11.60M  3.51G  2000L  0.30s  waite

zs1 = {'stages': 3, 'depth': 16, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (100, 100, 100), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 6, 8), 'classify': (0, 0, 0), 'expand': (1 * 16, 2 * 16),
       'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
       'nclass': 10, 'summer': 'split', 'last_branch': 1, 'last_down': False,
       'last_dfuc': 'D', 'last_expand': 32, 'afisok': False, 'version': 3}  # 15.03M  1.32G  1012  0.51s

# NET Width
ao1 = {'stages': 1, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 2.06M  0.96G  44L  0.05s  94.57%   => ax16@titan but-only-lastfc

ao2 = {'stages': 1, 'depth': 56, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.57M  0.74G  44L  0.05s

ao3 = {'stages': 1, 'depth': 48, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.16M  0.54G  44L  0.05s


ao4 = {'stages': 1, 'depth': 40, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 0.81M  0.38G  44L  0.04s


ao5 = {'stages': 1, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 0.65M  0.30G  44L  0.03s


model = ScaleNet(**ao5)
print('\n', model, '\n')

# train_which & eval_which 在组合上必须相互匹配
# model.set_train_which(part=['conv+rock', 'xfc+boost', 'xfc-only', 'boost-only'][2])
model.set_eval_which(part=['conv+rock', 'conv+rock+xfc', 'conv+rock+boost', 'conv+rock+xfc+boost'][1])
# print(model.stage1[1].conv1.training)
# print(model.stage1[1].classifier.training)
# print(model.stage2[0].classifier.training)
# print(model.summary.classifier1.training)

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
