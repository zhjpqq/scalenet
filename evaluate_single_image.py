# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2018/12/15 12:17'
#  适用所有数据, 所有模型, 带冻结的测试评估过程

import time
import argparse
import os
from datetime import datetime

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
# from tensorboardX import SummaryWriter
from xtils import AverageMeter, accuracy
from factory.data_factory_bp import data_factory
from factory.model_factory import model_factory
from xmodels.scalenet import ScaleNet
from config.configure import Config

# cfg -flow
cfg = Config()

# ckpt_dir = 'E:\Checkpoints\scalenet-uxp.gd2'
ckpt_dir = 'E:\Checkpoints'

# ckpt_name = 'cifar10-scalenet75-ep0-it390-acc43.51-best43.51-train53.7500-par1.52M-norm-uxp.gd2.ckpt'
# ckpt_name = 'cifar10-scalenet75-ep299-it117299-acc92.78-best92.79-train100.0000-par1.52M-norm-uxp.gd2.ckpt'
# ckpt_name = 'imagenet-scalenet128-ep99-it500499-acc74.13-best74.65-train88.1119-par25.25M-norm-uxp.oo1.ckpt'
ckpt_name = 'cifar10-scalenet54-ep300-it117690-acc94.75-best94.75-train100.0000-par2.07M-best-uxp.ax16.ckpt'

cfg.resume = os.path.join(ckpt_dir, ckpt_name)
cfg.gpu_ids = [0, 1, 2, 3][0:1] and []
cfg.config_gpus()
cfg.exclude_keys = ('exclude_keys', 'resume', 'device', 'gpu_ids')

checkpoint = torch.load(f=cfg.resume, map_location=cfg.device)
cfg.dict_to_class(checkpoint['config'], exclude=cfg.exclude_keys)

# model-flow
model, params, layers = model_factory(cfg.arch_name, cfg.arch_kwargs, cfg.dataset, with_info=True)
model = model.to(cfg.device)
try:
    model.load_state_dict(checkpoint['model'])
except:
    # multi-GPU-Training --> single-GPU-evaluate
    model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
    model.load_state_dict(checkpoint['model'])

# transform-flow
# cifar
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# # imagenet
# insize = 224
# xscale = [480, 256]
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]
#
# train_transform = transforms.Compose([
#     transforms.RandomChoice([transforms.Resize(x) for x in xscale]),
#     transforms.RandomAffine(degrees=(-8, 8), translate=None, scale=(1, 1.2), shear=(-8, 8)),
#     transforms.RandomCrop(insize),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean, std)]
# )
#
# val_transform = transforms.Compose([
#     transforms.Resize(xscale[-1]),
#     transforms.TenCrop(insize),
#     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#     transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(mean, std)(crop) for crop in crops])),
# ])

# data-flow
image = 'C://Users//xo//Pictures//cat1.jpg'
image = Image.open(image)
# image.show()
image = image.resize((64, 64))
image.show()
image.save('C://Users//xo//Pictures//58b3eb3e7db0e-x.jpg')

image = val_transform(image)
image = image.unsqueeze(0)
# image = torch.cat([image, image], dim=0)
image.to(cfg.device)

if image.dim() == 4:
    pred = model(image)
    print([(p.size(), p.max(1)) for p in pred])
elif image.dim() == 5:
    bh, ncrops, c, h, w = image.size()
    print(bh, ncrops, c, h, w)
    pred = model(image.view(-1, c, h, w))
    pred = [p.view(bh, ncrops, -1).mean(1) for p in pred]
    print([(p.size(), p.max(1)) for p in pred])
else:
    print(image.size())
    raise NotImplementedError
