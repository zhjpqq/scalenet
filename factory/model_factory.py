# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2018/12/15 12:17'

# pakages
import time
import argparse
import shutil
import os
from datetime import datetime
from collections import namedtuple

import torch
from torchvision import models as tvmodels

import xmodels.resnet_org as resnet_cifar
from xmodels.resnXt_org import CifarResNeXt
from xmodels.densenet import DenseNet as CifarDenseNet
from xmodels.preresnet import PreResNet
from xmodels import tvm_resnet, tvm_densenet, tvm_vggs
from xmodels.tvm_resnet import RESNets
from xmodels.tvm_densenet import DENSENets
from xmodels.mobilev3 import MobileV3
from xmodels.hrnet import HRNets
from xmodels.fishnet import FISHNets
from xmodels import _init_model as init_model
import xtils


def model_factory(arch_name, arch_kwargs, dataset, with_info=False):
    model = None

    if dataset.startswith('cifar'):
        if arch_name.startswith('resnet'):
            model = getattr(resnet_cifar, arch_name)(**arch_kwargs)
        elif arch_name.startswith('densenet'):
            model = CifarDenseNet(**arch_kwargs)
        elif arch_name.startswith('preresnet'):
            model = PreResNet(**arch_kwargs)
        else:
            try:
                model = getattr(init_model, arch_name)
            except AttributeError:
                raise AttributeError('未找到模型: <%s>, 请检查是否已将该模型注册到 <xmodel._init_model> .' % arch_name)
            model = model(**arch_kwargs)

    elif dataset.startswith('imagenet'):
        #  导入ImageNet模型，只需在 xmodel._init_model 中添加自己的模型即可，无需在此处添加循环分支
        try:
            model = getattr(init_model, arch_name)
        except AttributeError:
            raise AttributeError('未找到模型: <%s>, 请检查是否已将该模型注册到 <xmodel._init_model> .' % arch_name)
        model = model(**arch_kwargs)

    if model is None:
        raise NotImplementedError('Unkown <arch_name:%s> for <dataset:%s>, '
                                  'check mdoel_factory.py' % (arch_name, dataset))

    if with_info:
        params = xtils.calculate_params_scale(model, format='million')
        insize = 224 if dataset == 'imagenet' else 32
        gflops = xtils.calculate_FLOPs_scale(model, input_size=insize, multiply_adds=True, use_gpu=False)
        depth = xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
        if with_info == 'return':
            return model, params, gflops, depth
    return model
