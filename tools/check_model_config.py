#!/usr/bin/env python
# -*- coding: utf-8 -*-
__date__ = '2019/3/6 23:10'
__author__ = 'ooo'

import os
import time
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
from config.configure import Config
import xtils

exp1 = 'imagenet-scalenet44-ep37-it185185-acc62.55-best62.55-train54.2969-par10.04M-best-exp1.ckpt'  # bh=256
exp2 = 'imagenet-scalenet44-ep39-it200199-acc70.91-best70.97-train80.4196-par9.58M-norm-exp2.ckpt'   # bh=256
exp4 = 'imagenet-scalenet76-ep59-it300299-acc73.04-best73.04-train90.2098-par27.83M-norm-exp4.ckpt'  # bh=256
exp5 = 'imagenet-scalenet44-ep38-it190190-acc68.04-best68.47-train70.3125-par9.58M-norm-exp5.ckpt'   # bh=256
mix1 = 'imgnet-resmix96-ep100-acc69.70-best69.83-kval69.79-train68.95-ktrain68.92-par25.55M-exp1.ckpt'  # bh=256
mix2 = 'imgnet-resmix88-ep100-acc74.08-best74.16-kval74.14-train91.00-ktrain90.91-par18.99M-exp2.ckpt'  # bh=256
mix3 = 'imgnet-resmix72-ep100-acc73.63-best73.85-kval73.79-train89.78-ktrain89.73-par15.56M-exp3.ckpt'  # bh=256
mix4 = 'imgnet-resmix52-ep100-acc68.33-best68.33-kval68.27-train66.76-ktrain66.69-par14.50M-exp4.ckpt'  # bh=256

vx13 = 'cifar100-scalenet444-ep10-it3910-acc35.73-best36.18-train46.8750-par1.72M-norm-uxp.vx13.ckpt'
vx14= 'cifar100-scalenet444-ep10-it3910-acc29.09-best37.14-train50.7812-par1.67M-norm-uxp.vx14.ckpt'

tt1 = 'cifar10-scalenet2672-ep117-it44574-acc89.92-best91.61-train99.2188-par15.03M-norm-uxp.tt1.ckpt'

vx24 = 'cifar100-scalenet100-ep299-it117299-acc75.48-best75.75-train100.0000-par1.72M-norm-uxp.vx24.ckpt'

gx1 = 'cifar100-scalenet1432-ep109-it86019-acc63.39-best66.01-train76.5625-par10.21M-norm-uxp.vx6.ckpt'

vo6 = ['imagenet-scalenet44-ep40-it205204-acc64.59-best65.13-train60.8392-par9.39M-norm-uxp.vo6.ckpt',
       'imagenet-scalenet44-ep40-it205204-acc64.59-best65.13-train60.8392-par9.39M-norm-uxp.vo6.ckpt',
       'imagenet-scalenet44-ep99-it500499-acc71.42-best71.50-train72.0280-par9.39M-norm-uxp.vo6.ckpt'][1]

oo1 = 'imagenet-scalenet128-ep99-it500499-acc74.13-best74.65-train88.1119-par25.25M-norm-uxp.oo1.ckpt'

file = os.path.join('/data1/zhangjp/classify/checkpoints/imagenet/mobilev3/mobilev3-uxp.mb1/',
                    'imagenet-mobilev356-ep47-it240239-acc61.29-best61.29-topv82.67-par2.51M-best-uxp.mb1.ckpt')
ckpt = torch.load(f=file, map_location='cpu')
cfg_dict = ckpt['config']
for k, v in cfg_dict.items():
    print(k, '', v)
