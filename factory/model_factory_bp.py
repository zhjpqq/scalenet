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

from xmodels import tvm_resnet, tvm_densenet, tvm_vggs, fishnet
import xmodels.resnet_org as resnet_cifar
from xmodels.resnXt_org import CifarResNeXt
from xmodels.densenet import DenseNet as CifarDenseNet
from xmodels.preresnet import PreResNet
from xmodels.mobilev3 import MobileV3
from xmodels import _init_model as init_model
import xtils


def model_factory(arch_name, arch_kwargs, dataset, with_info=False):
    model = None

    if arch_name.startswith('resnet'):
        if dataset == 'cifar':
            model = getattr(resnet_cifar, arch_name)(**arch_kwargs)
        elif dataset == 'imagenet':
            # model = getattr(tvmodels, arch_name)(**arch_kwargs)    # pretrained  model_url
            model = getattr(tvm_resnet, arch_name)(**arch_kwargs)  # pretrained  model_path

    elif arch_name.startswith('densenet'):
        if dataset == 'cifar':
            model = CifarDenseNet(**arch_kwargs)
        elif dataset == 'imagenet':
            # model = getattr(tvmodels, arch_name)(**arch_kwargs)      # pretrained  model_url
            model = getattr(tvm_densenet, arch_name)(**arch_kwargs)  # pretrained  model_path

    elif arch_name.startswith('vgg'):
        if dataset == 'imagenet':
            # model = getattr(tvmodels, arch_name)(**arch_kwargs)      # pretrained  model_url
            model = getattr(tvm_vggs, arch_name)(**arch_kwargs)

    elif arch_name.startswith('preresnet'):
        if dataset == 'cifar10':
            model = PreResNet(**arch_kwargs)

    elif arch_name.startswith('fishnet'):
        if dataset == 'imagenet':
            model = getattr(fishnet, arch_name)(**arch_kwargs)      # pretrained  model_path

    elif arch_name.startswith('mobilev3'):
        if dataset == 'imagenet':
            model = MobileV3(**arch_kwargs)     # pretrained  model_path

    #  导入自定义模型，只需在 xmodel._init_model 中添加自己的模型即可，无需在此处添加循环分支
    else:
        try:
            model = getattr(init_model, arch_name)
        except AttributeError:
            raise AttributeError('未找到模型: <%s>, 请检查是否已将该模型注册到 <xmodel._init_model> .' % arch_name)
        model = model(**arch_kwargs)

    if model is None:
        raise NotImplementedError('Unkown <arch_name:%s> for <dataset:%s>, '
                                  'check mdoel_factory.py' % (arch_name, dataset))
    if with_info:
        # print(model)
        parameters = xtils.calculate_params_scale(model, format='million')
        if dataset == 'imagenet' and True:
            gflops = xtils.calculate_FLOPs_scale(model, input_size=224, multiply_adds=True, use_gpu=False)
        model_depth = xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
        return model, parameters, model_depth

    return model
