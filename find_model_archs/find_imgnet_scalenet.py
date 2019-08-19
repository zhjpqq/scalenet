# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch, time
import xtils
import torchvision as tv
import xmodels.tvm_densenet as tvmd
from xmodels.scalenet import ScaleNet

# imageNet
exp5 = {'stages': 5, 'branch': 3, 'rock': 'U', 'depth': 16, 'kldloss': False,
        'layers': (6, 5, 4, 3, 2), 'blocks': ('D', 'D', 'D', 'S', 'S'), 'slink': ('A', 'A', 'A', 'A', 'A'),
        'growth': (12, 12, 12, 12, 12), 'classify': (1, 1, 1, 1, 1),
        'expand': (1 * 16, 2 * 16, 4 * 16, 8 * 16), 'dfunc': ('O', 'D', 'O', 'A'),
        'fcboost': 'none', 'nclass': 1000, 'summer': 'merge',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'A', 'last_expand': 15,
        'afisok': False, 'afkeys': ('af1', 'af2'), 'convon': True,
        'convlayers': 1, 'convdepth': 4}

exp4 = {'stages': 4, 'branch': 3, 'rock': 'U', 'depth': 16, 'kldloss': False,
        'layers': (6, 5, 4, 3), 'blocks': ('D', 'D', 'D', 'S'), 'slink': ('A', 'A', 'A', 'A'),
        'growth': (10, 15, 20, 30), 'classify': (0, 0, 1, 1), 'expand': (10, 20, 30),
        'dfunc': ('O', 'D', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
        'last_branch': 2, 'last_down': False, 'last_dfuc': 'A', 'last_expand': 15,
        'afisok': True, 'afkeys': ('af1', 'af2'), 'convon': True,
        'convlayers': 1, 'convdepth': 4}

exp3 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 16, 'kldloss': False,
        'layers': (3, 3, 1), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (5, 5, 5), 'classify': (0, 0, 0), 'expand': (1 * 16, 2 * 16),
        'dfunc': ('D', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'merge',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 256,
        'afisok': False, 'afkeys': ('af1', 'af2'), 'convon': True,
        'convlayers': 2, 'convdepth': 4}

############################################################################################################

# 25M 模型
ox1 = {'stages': 4, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (27, 5, 3, 2), 'blocks': ('D', 'D', 'S', 'S'), 'slink': ('A', 'A', 'A', 'A'),
       'growth': (-6, -2, 0, 2), 'classify': (0, 0, 0, 0), 'expand': (64, 2 * 64, 3 * 64),
       'dfunc': ('O', 'O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split', 'afisok': False,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 256}  # 25.15M   153L  0.43s

ox4 = {'stages': 4, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (7, 5, 3, 2), 'blocks': ('D', 'D', 'S', 'S'), 'slink': ('A', 'A', 'A', 'A'),
       'growth': (-4, -2, 0, 2), 'classify': (0, 0, 0, 0), 'expand': (64, 2 * 64, 4 * 64),
       'dfunc': ('O', 'O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split', 'afisok': False,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 256}  # 25.07M   73L  0.37s

ox2 = {'stages': 4, 'branch': 3, 'rock': 'U', 'depth': 128, 'kldloss': False, 'afisok': False,
       'layers': (3, 3, 2, 2), 'blocks': ('D', 'D', 'S', 'S'), 'slink': ('A', 'A', 'A', 'A'),
       'growth': (-20, -20, -40, -40), 'classify': (0, 0, 0, 0), 'expand': (1 * 64, 2 * 64, 3 * 64),
       'dfunc': ('O', 'O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 256}  # 25.20M   47L  0.83s

ox3 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (14, 13, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (-4, 0, 2), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('D', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'merge',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 256}  # 24.97M  134L 1.24s

kx1 = {'stages': 4, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False, 'afisok': False,
       'layers': (3, 4, 4, 2), 'blocks': ('D', 'D', 'S', 'S'), 'slink': ('A', 'A', 'A', 'A'),
       'growth': (0, -2, -4, 2), 'classify': (0, 0, 0, 0), 'expand': (64, 2 * 64, 4 * 64),
       'dfunc': ('O', 'O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 256}  # 25.16M   55L  0.62s

ox5 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (13, 13, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (-4, -8, 2), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('D', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'merge',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 256}  # 25.11M  132L 1.22s   #512

oo1 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (13, 13, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (-4, 0, 2), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 356}  # 25.25M  128L 1.17s  # 612fc

ew1 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (12, 9, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (-4, -20, -64), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'merge',
       'last_branch': 2, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 256}  # 25.07M  110L 1.08s  # 1129fc

ew2 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (12, 9, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (-4, -20, -64), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D',
       'last_expand': 256}  # 25.07M  110L 1.08s  # 1129fc # no good

# 35M 模型
tf1 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (13, 15, 7), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 2, 4), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 356 + 412}  # 35.06M  140L 1.38s  # 1024fc

# 45M 模型
sw1 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (31, 24, 10), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (-4, 0, 2), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 256}  # 45.00M  254L 2.044s

sw2 = {'stages': 4, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False, 'afisok': False,
       'layers': (4, 4, 3, 3), 'blocks': ('D', 'D', 'D', 'S'), 'slink': ('A', 'A', 'A', 'A'),
       'growth': (0, -8, -8, -8), 'classify': (0, 0, 0, 0), 'expand': (64, 2 * 64, 4 * 64),
       'dfunc': ('O', 'O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 512}  # 44.02M   68L  0.75s

sw3 = {'stages': 4, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False, 'afisok': False,
       'layers': (9, 7, 4, 3), 'blocks': ('D', 'D', 'D', 'S'), 'slink': ('A', 'A', 'A', 'A'),
       'growth': (0, -16, -32, -64), 'classify': (0, 0, 0, 0), 'expand': (64, 2 * 64, 4 * 64),
       'dfunc': ('O', 'O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 512}  # 45.04M   104L  1.00s

oo5 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 70, 'kldloss': False,
       'layers': (8, 7, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 70, 2 * 70), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 512 - 280,
       'version': 2}  # 24.96M  85L  0.99s  # 512fc

oo6 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 70, 'kldloss': False,
       'layers': (8, 7, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 70, 2 * 70), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 512 - 280,
       'version': 2}  # 24.96M  85L  0.99s  # 512fc  # == cfgoo5  but weight_decay

oo7 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 70, 'kldloss': False,
       'layers': (4, 4, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 100, 2 * 100), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 512 - 370,
       'version': 2}  # 25.10M  54L  0.85s  # 512fc

# mm1 > oo8 > oo2 > uu1

mm1 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 75, 'kldloss': False,
       'layers': (4, 4, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 75, 2 * 75), 'afisok': False,
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split', 'version': 2,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 1024 - 300}  # 25.16M  50L 0.82s  # 1024fc

oo8 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 100, 'kldloss': False,
       'layers': (4, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 100, 2 * 100), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 640 - 400,
       'version': 2}  # 25.09M  42L  0.93s  640fc

oo2 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 70, 'kldloss': False,
       'layers': (7, 7, 7), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 70, 2 * 70), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 232,
       'version': 2}  # 25.4M  78L  1.05s  # 512fc

oo9 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 80, 'kldloss': False,
       'layers': (4, 5, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 1230 - 320,
       'version': 2}  # 25.60M  53L  0.81s  1230fc

bb1 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 100, 'kldloss': False,
       'layers': (3, 3, 2), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 100, 2 * 100), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 2048 - 400,
       'version': 2}  # 25.43M  35L  0.77s  2048fc

bb2 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 80, 'kldloss': False,
       'layers': (4, 4, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (2, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 1230 - 320,
       'version': 2}  # 25.36M  55L  0.80s  1230fc

bb3 = {'stages': 4, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False, 'afisok': False,
       'layers': (2, 2, 2, 2), 'blocks': ('D', 'D', 'S', 'S'), 'slink': ('A', 'A', 'A', 'A'),
       'growth': (-2, -3, -4, -8), 'classify': (0, 0, 0, 0), 'expand': (64, 2 * 64, 4 * 64),
       'dfunc': ('O', 'O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split', 'version': 2,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 2048 - 512}  # 25.06M 40L 0.54s 1024

mm2 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 80, 'kldloss': False,
       'layers': (4, 4, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split', 'version': 2,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 2048 - 320}  # 25.00M  47L 0.71s  # 2048fc

uu1 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 100, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (-10, -20, -40), 'classify': (0, 0, 0), 'expand': (1 * 100, 2 * 100), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 900 - 400,
       'version': 2}  # 25.25M  42L  0.88s  900fc

vv1 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (4, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 512 - 256,
       'version': 2}  # 10.10M  41L  0.56s  512fc

oo10 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 100, 'kldloss': False,
        'layers': (4, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 100, 2 * 100), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 1024 - 400,
        'version': 2}  # 25.22M  41L  0.78s  1024fc

mm3 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 75, 'kldloss': False,
       'layers': (5, 5, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 75, 2 * 75), 'afisok': False,
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split', 'version': 2,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 1800 - 300}  # 25.25M  57L 0.70s  # 1800fc

oo11 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 75, 'kldloss': False,
        'layers': (5, 5, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 75, 2 * 75), 'afisok': False,
        'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split', 'version': 2,
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 2014 - 300}  # 23.02M  57L 0.71s  # 2014fc

mm5 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (13, 13, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'merge', 'version': 2,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 1024 - 256}  # 25.13M  121L 1.17s  #1024fc

# == mm5
mm6 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (13, 13, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'merge', 'version': 2,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 1024 - 256}  # 25.13M  121L 1.17s  #1024fc

pp1 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 100, 'kldloss': False,
       'layers': (4, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 100, 2 * 100), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'avgpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 1024 - 400,
       'version': 2}  # 25.22M  41L  0.86s  1024fc

pp2 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 1024 - 256,
       'version': 2}  # 10.3M  37L  0.53s  1024fc

pp3 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 512 - 256,
       'version': 2}  # 10.0M  37L  0.53s  512fc

pp4 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 512 - 256,
       'version': 2}  # 10.5M  42L  0.53s  512fc

pp5 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (4, 4, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 4, 8), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 512 - 256,
       'version': 2}  # 15.0M  54L  0.63s  512fc

pp6 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 71, 'kldloss': False,
       'layers': (4, 4, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 71, 2 * 71), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 512 - 284,
       'version': 2}  # 15.2M  47L  0.70s  512fc

pp7 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 70, 'kldloss': False,
       'layers': (6, 5, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 70, 2 * 70), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 1024 - 280,
       'version': 2}  # 20.04M  61L  0.80s  1024fc

exp2 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (3, 3, 1), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'expand': (1, 2, 4), 'growth': (0, 0, 0), 'dfunc': ('D', 'D', 'D'),
        'classify': (0, 0, 0), 'fcboost': 'none', 'nclass': 1000, 'summer': 'merge',
        'last_branch': 1, 'last_down': True, 'last_expand': 256, 'afisok': False}  # 70.97%  9.581M

vo1 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 72, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 72, 2 * 72), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 512 - 288,
       'version': 3}  # 9.4M  37L  2.41G  0.45s  512fc

vo3 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (4, 5, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 512 - 256,
       'version': 3}  # 10.18M  2.64G 51L  0.4s  512fc

vo7 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 72, 'kldloss': False,
       'layers': (2, 2, 2), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (-8, -8, -8), 'classify': (0, 0, 0), 'expand': (1 * 72, 2 * 72),
       'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'),
       'fcboost': 'none', 'nclass': 1000, 'summer': 'split', 'afisok': False,
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 512 - 288,
       'version': 3}  # 5.96M  1.61G  31L  0.28s  512fc

vo10 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 72, 'kldloss': False,
        'layers': (5, 5, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (-4, -4, -4), 'classify': (0, 0, 0), 'expand': (1 * 72, 2 * 72), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 1024 - 288,
        'version': 3}  # 14.39M  3.31G  59L  0.38s  1024fc

vo5 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 72, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 72, 2 * 72), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'C', 'last_expand': 512 - 288,
       'version': 3}  # 9.39M  2.41G 37L  0.4s  512fc

vo6 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (6, 7, 8), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 512 - 256,
       'version': 3}  # 9.39M  2.38G 44L  0.4s  512fc

vo8 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (6, 7, 8), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 512 - 256,
       'version': 3}  # 9.36M  2.10G 42L 0.27s  512fc  ==>vo6  branch=1

vo9 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (6, 7, 8), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 1024 - 256,
       'version': 3}  # 8.70M  1.87G  44L  0.25s  1024fc  ==>vo8  1024fc

vo11 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (6, 7, 8), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 2048 - 320,
        'version': 3}  # 14.88M  2.89G  41L  0.35s  2048fc

vo13 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1024 - 400, 'version': 3}  # 15.32M  3.33G  40L  0.42s  1024fc

vo12 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 100, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 2048 - 420,
        'version': 3}  # 20.12M  3.50G  41L  0.35s  2048fc

vo14 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 512 - 400, 'version': 3}  # 13.99mM  3.29G  40L  0.41s  512fc

vo15 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1024 - 400, 'version': 3}  # 15.29M  3.03G  38l  0.42s  1024fc

voo = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 160, 'kldloss': False,
       'layers': (2, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (-40, -40), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'E', 'last_expand': 512 - 80,
       'version': 3}  # 19.97M  4.26G  49L  0.41s  2048fc

vo16 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1024 - 400, 'version': 3}  # 15.32M  3.33G  40L  0.42s  1024fc  ==> vo13 on 1080ti

vo17 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (4, 4, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1280 - 400, 'version': 3}  # 20.02M  4.27G  50L  0.55s  1024fc

vo18 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (4, 4, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1280 - 400, 'version': 3}  # 20.02M  4.27G  50L  0.55s  1280fc

vo19 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (4, 4, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1314 - 400, 'version': 3}  # 20.09M  3.97G  48L  0.40s  1314fc  73.99%

vo27 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (6, 6, 8), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 3000 - 256, 'version': 3}  # 20.05M  3.61G  69L  0.39s  3000fc

vo20 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 920 - 400, 'version': 3}  # 15.03M  3.02G  38L  0.32s  920fc

vo22 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (3, 3, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1024 - 320, 'version': 3}  # 15.16M  6.10G  39L  0.31s  1024fc

vo22a = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
         'layers': (3, 3, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
         'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
         'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
         'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
         'last_expand': 1024 - 320, 'version': 3}  # 15.16M  6.10G  39L  0.31s  1024fc wd=0.00007/vo22

vo22b = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
         'layers': (3, 3, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
         'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
         'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
         'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
         'last_expand': 1024 - 320, 'version': 3}  # 15.16M  6.10G  39L  0.31s  1024fc wd=0.00005/vo22

vo23 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (5, 5, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2048 - 256, 'version': 3}  # 15.02M  5.81G  57L  0.32s  2048fc

vo23a = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
         'layers': (5, 5, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
         'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
         'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
         'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
         'last_expand': 2048 - 256, 'version': 3}  # 15.02M  5.81G  57L  0.32s  2048fc wd=0.00007*vo23

vo24 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (4, 4, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 3000 - 256, 'version': 3}  # 15.03M  2.45G  47L  0.28s  3000fc

vo25 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (5, 5, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 3000 - 256, 'version': 3}  # 17.00M  2.96G  57L  0.33s  3000fc

vo26 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (3, 3, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1024 - 320, 'version': 3}  # 15.16M  6.10G  39L  0.31s  1024fc  ==vo22

ok1 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (6, 7, 8), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
       'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 512 - 256,
       'version': 3}  # 9.36M  4.20G 42L  0.28s  512fc  ==> vo6 but branch=1

ok2 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (4, 6, 8), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
       'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True,
       'last_dfuc': 'D', 'last_expand': 512 - 256, 'version': 3}  # 9.20M  4.12G 42L  0.27s  512fc  ==>

vot = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 6, 'kldloss': False,
       'layers': (1, 1, 1), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 6, 2 * 6), 'afisok': False,
       'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
       'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True,
       'last_dfuc': 'D', 'last_expand': 0, 'version': 3}

vo28 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2048 - 256, 'version': 3}  # 10.00M  3.50G  35L  0.24s  2048fc   wd=0.0000

vo29 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (6, 6, 12), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 512 - 256, 'version': 3}  # 7.56M  3.67G  40L  0.23s  512fc

vo30 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (3, 3, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (6, 6, 12), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1111 - 256, 'version': 3}  # 10.01M  4.03G  42L  0.26s  1111fc

vo31 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (6, 7, 8), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
        'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 512 - 256,
        'version': 3}  # 9.36M  4.20G  42L  0.29s  512fc  =>vo6/branch=1

vo21 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (4, 5, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1746 - 400, 'version': 3}  # 25.01M  4.64G  54L  0.46s  1746fc  74.62%

vo32 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (6, 8, 8), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (6, 6, 12), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2048 - 320, 'version': 3}  # 30.06M  12.95G  82L  0.65s  2048fc

vo33 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (6, 9, 10), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (6, 6, 12), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2101 - 320, 'version': 3}  # 35.00M  14.49G  90L  0.64s  2048fc

vo35 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (6, 6, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2022 - 400, 'version': 3}  # 30.00M  11.72G  68L  0.54s  2022fc

vo36 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (7, 7, 7), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (4, 5, 6), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2022 - 400, 'version': 3}  # 35.01M  14.14G  81L  0.64s  2022fc

vo37 = {'stages': 2, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (8, 8), 'blocks': ('D', 'D'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 80,), 'afisok': False,
        'dfunc': ('O',), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 512 - 160, 'version': 3}  # 10.11M  9.48G  68L  0.46s  512fc  73.08%

vo37S = {'stages': 2, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
         'layers': (8, 8), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
         'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 80,), 'afisok': False,
         'dfunc': ('O',), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
         'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
         'last_expand': 512 - 160, 'version': 3}  # 6.42M  8.57G  52L  0.42s  512fc

vo39 = {'stages': 2, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (8, 8), 'blocks': ('D', 'D'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 64,), 'afisok': False,
        'dfunc': ('O',), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': False, 'last_dfuc': 'E',
        'last_expand': 512 - 128, 'version': 3}  # 6.07M  6.03G  67L  0.32s  128fc  # bad-128fc

vo38 = {'stages': 2, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (8, 8), 'blocks': ('D', 'D'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 64,), 'afisok': False,
        'dfunc': ('O',), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 512 - 128, 'version': 3}  # 6.71M  6.13G  68L  0.35s  512fc  71.18%

vo38N = {'stages': 2, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
         'layers': (8, 8), 'blocks': ('D', 'D'), 'slink': ('A', 'A'),
         'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 64,), 'afisok': False,
         'dfunc': ('O',), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
         'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
         'last_expand': 512 - 128, 'version': 3}  # 6.71M  5.94G  68L  0.34s  512fc

vo38W = {'stages': 2, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
         'layers': (8, 8), 'blocks': ('D', 'D'), 'slink': ('A', 'A'),
         'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 64,), 'afisok': False,
         'dfunc': ('O',), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
         'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
         'last_expand': 512 - 128, 'version': 3}  # 6.71M  6.13G  68L  0.35s  512fc  wd=fuc2(0.00002~0.0002)

vo38S = {'stages': 2, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
         'layers': (8, 8), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
         'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 64,), 'afisok': False,
         'dfunc': ('O',), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
         'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
         'last_expand': 512 - 128, 'version': 3}  # 4.34M  5.56G  52L  0.31s  512fc

vo50 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
        'layers': (15,), 'blocks': ('D',), 'slink': ('A',), 'growth': (10,),
        'classify': (0,), 'expand': (), 'dfunc': (), 'afisok': False,
        'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 512 - 80, 'version': 3}  # 5.06M  10.32G  63L  0.52s  512fc  67.50%-run

vo51 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (15,), 'blocks': ('D',), 'slink': ('A',), 'growth': (10,),
        'classify': (0,), 'expand': (), 'dfunc': (), 'afisok': False,
        'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 512 - 64, 'version': 3}  # 3.61M  6.92G  63L  0.47s  512fc   68.67%-ep93

vo52 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 36, 'kldloss': False,
        'layers': (15,), 'blocks': ('D',), 'slink': ('A',), 'growth': (10,),
        'classify': (0,), 'expand': (), 'dfunc': (), 'afisok': False,
        'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 360 - 36, 'version': 3}  # 1.6M  2.55G  63L  0.22s  360fc

vo53 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (25,), 'blocks': ('D',), 'slink': ('A',), 'growth': (6,),
        'classify': (0,), 'expand': (), 'dfunc': (), 'afisok': False,
        'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 520 - 64, 'version': 3}  # 5.08M  10.51G  103L  0.65s  520fc  69.54%

vo55 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (15,), 'blocks': ('D',), 'slink': ('A',), 'growth': (0,),
        'classify': (0,), 'expand': (), 'dfunc': (), 'afisok': False,
        'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1100 - 64, 'version': 3}  # 3.61M  5.92G  63L  0.43s  1100fc

vo56 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (15,), 'blocks': ('D',), 'slink': ('A',), 'growth': (0,),
        'classify': (1,), 'expand': (), 'dfunc': (), 'afisok': False,
        'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1100 - 64, 'version': 3}  # 3.61M  5.92G  63L  0.43s  1100fc    69.27%

vo56a = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
         'layers': (15,), 'blocks': ('D',), 'slink': ('A',), 'growth': (0,),
         'classify': (1,), 'expand': (), 'dfunc': (), 'afisok': False,
         'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
         'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
         'last_expand': 1100 - 64, 'version': 3}  # 3.61M  5.92G  63L  0.43s  1100fc

vo57 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 52, 'kldloss': False,
        'layers': (20,), 'blocks': ('D',), 'slink': ('A',), 'growth': (6,),
        'classify': (0,), 'expand': (), 'dfunc': (), 'afisok': False,
        'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1100 - 52, 'version': 3}  # 3.18M  4.55G  123L  0.44s  1100fc

vo58 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (3, 5, 15), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2000 - 256, 'version': 3}  # 25.25M  7.69G  67L  0.43s  2000fc

vo59 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (3, 5, 20), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 1 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2000 - 192, 'version': 3}  # 20.45M  6.38G  80L  0.36s  2000fc

vo60 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 36, 'kldloss': False,
        'layers': (40,), 'blocks': ('D',), 'slink': ('A',), 'growth': (0,),
        'classify': (1,), 'expand': (), 'dfunc': (), 'afisok': False,
        'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1516 - 36, 'version': 3}  # 3.62M  4.95G  163L  0.51s  1516fc  wd~0.0001, +xfc

vo61 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (3, 5, 27), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 1 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1992 - 192, 'version': 3}  # 25.09M  7.52G  94L  0.41s  1992fc

vo62 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (3, 5, 19), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2056 - 256, 'version': 3}  # 30.08M  8.65G  75L  0.44s  2056fc  @mic118  73.87%

vo63 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (4, 6, 10), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2000 - 256, 'version': 3}  # 20.08M  6.77G  65L  0.37s  2000fc  @1080ti

vo64 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (3, 5, 16), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1300 - 256, 'version': 3}  # 25.00M  7.71G  69L  0.38s  1300fc  @mic119  73.65%

vo65 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (37,), 'blocks': ('D',), 'slink': ('A',), 'growth': (0,),
        'classify': (0,), 'expand': (), 'dfunc': (), 'afisok': False,
        'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1200 - 64, 'version': 3}  # 6.98M  13.91G  151L  0.82s  1200fc

vo66 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
        'layers': (3, 3, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1600 - 400, 'version': 3}  # 25.43M  8.02G  44L  0.38s  1600

vo67 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
        'layers': (3, 3, 8), 'blocks': ('D', 'S', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 100), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1600 - 360, 'version': 3}  # 25.02M  8.00G  41L  0.38s  1600

vo68 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (2, 2, 21), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1800 - 256, 'version': 3}  # 30.00M  7.76G  63L  0.38S  1800fc  @casia

vo69 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (25,), 'blocks': ('D',), 'slink': ('A',), 'growth': (0,),
        'classify': (0,), 'expand': (), 'dfunc': (), 'afisok': False,
        'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1100 - 64, 'version': 3}  # 5.08M  9.53G  103L  0.65s  520fc  71.63%

vo56x = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
         'layers': (15,), 'blocks': ('D',), 'slink': ('A',), 'growth': (0,),
         'classify': (1,), 'expand': (), 'dfunc': (), 'afisok': False,
         'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
         'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
         'last_expand': 1100 - 64, 'version': 3}  # 3.61M  5.92G  63L  0.43s  1100fc

vo70 = {'stages': 2, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
        'layers': (6, 14), 'blocks': ('D', 'D'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 80,), 'afisok': False,
        'dfunc': ('O',), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 512 - 160, 'version': 3}  # 15.18M  11.49G  84L  0.55s  512fc

vo19 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (4, 4, 4), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1314 - 400, 'version': 3}  # 20.09M  7.93G  48L  0.40s  1314fc  74.03%

vo21 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (4, 5, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1746 - 400, 'version': 3}  # 25.01M  4.64G  54L  0.46s  1746fc  74.62%

vo37 = {'stages': 2, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (8, 8), 'blocks': ('D', 'D'), 'slink': ('A', 'A'),
        'growth': (0, 0), 'classify': (0, 0), 'expand': (1 * 80,), 'afisok': False,
        'dfunc': ('O',), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 512 - 160, 'version': 3}  # 10.11M  9.48G  68L  0.46s  512fc  73.08%

vo58 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 64, 'kldloss': False,
        'layers': (3, 5, 15), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2000 - 256, 'version': 3}  # 25.25M  7.69G  67L  0.43s  2000fc  73.87%

vo62 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (3, 5, 19), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 2056 - 256, 'version': 3}  # 30.08M  8.65G  75L  0.44s  2056fc  @mic118  73.87%

vo64 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (3, 5, 16), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 64, 2 * 64), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1300 - 256, 'version': 3}  # 25.00M  7.71G  69L  0.38s  1300fc  @mic119  73.65%

vo71 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (4, 5, 7), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1746 - 400, 'version': 3}  # 30.78M  10.69G  58L  0.50s  1746fc

vo72 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
        'layers': (4, 5, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (-8, -20, -50), 'classify': (0, 0, 0), 'expand': (1 * 120, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1700 - 440, 'version': 3}  # 30.51M  10.83G  59L  0.51s  1700fc  74.68%

vo73 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 76, 'kldloss': False,
        'layers': (4, 4, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (-8, -40, -100), 'classify': (0, 0, 0), 'expand': (1 * 114, 310), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1600 - 500, 'version': 3}  # 30.83M  9.87G  54L  0.47s  1600fc

vo75 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (4, 7, 21), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 5, 10), 'classify': (0, 0, 0), 'expand': (1 * 72, 1 * 96), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1600 - 232, 'version': 3}  # 30.29M  9.93G  97L  0.51s  1600fc  @mic118

vo76 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
        'layers': (4, 5, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (-8, -8, -8), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1730 - 320, 'version': 3}  # 20.02M  7.62G  57L  0.25s  1730fc

model = ScaleNet(**vo76)

# train_which & eval_which 在组合上必须相互匹配
# model.train_which(part=['conv+rock', 'xfc+boost', 'xfc-only', 'boost-only'][1])
model.set_eval_which(part=['conv+rock', 'conv+rock+xfc', 'conv+rock+boost', 'conv+rock+xfc+boost'][1])
# print(model.stage1[1].conv1.training)
# print(model.stage1[1].classifier.training)
# print(model.stage2[0].classifier.training)
# print(model.summary.classifier1.training)

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


# 查看模型的权值衰减曲线
if [True, False][1]:
    from config.configure import Config

    cfg = Config()
    cfg.weight_decay = 0.0001
    cfg.decay_fly = {'flymode': ['nofly', 'stepall'][1],
                     'a': 0, 'b': 1, 'f': xtils.Curves(4).func2,
                     'wd_start': 0.0000001, 'wd_end': 0.00001, 'wd_bn': None}
    model.visual_weight_decay(cfg=cfg, visual=True)
