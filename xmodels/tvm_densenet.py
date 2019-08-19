# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2018/12/15 12:17'

import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from torchvision import models
from torchvision.models import DenseNet
import xtils

"""
与torchvision.models 相同，但原码只能从 <url加载预训练模型>。

此处增加了 model_path 参数，可以直接从 <本地加载预训练模型>。

当上述两个参数都不指定时，加载一个空白的初始化模型。
"""

"""
ResNet for ImageNet - 1000  classes
"""

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def densenet121(pretrained=False, model_path=None, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        if model_path is not None:
            state_dict = torch.load(model_path)
        else:
            state_dict = model_zoo.load_url(model_urls['densenet121'])

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet169(pretrained=False, model_path=None, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        if model_path is not None:
            state_dict = torch.load(model_path)
        else:
            state_dict = model_zoo.load_url(model_urls['densenet169'])

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet201(pretrained=False, model_path=None, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        if model_path is not None:
            state_dict = torch.load(model_path)
        else:
            state_dict = model_zoo.load_url(model_urls['densenet201'])

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, model_path=None, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        if model_path is not None:
            state_dict = torch.load(model_path)
        else:
            state_dict = model_zoo.load_url(model_urls['densenet161'])

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet264(pretrained=False, model_path=None, **kwargs):
    r"""Densenet-264 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 64, 48),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

        if model_path is not None:
            state_dict = torch.load(model_path)
        else:
            state_dict = model_zoo.load_url(model_urls['densenet264'])

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


# 统一接口 interface for model_factory ##########################################

model_dir = xtils.get_pretrained_models()

dense121 = {'arch': 'densenet121',
            'model_path': [os.path.join(model_dir, 'densenet121-a639ec97.pth'), 'download', ''][0],
            'cfg': {'num_init_features': 64, 'growth_rate': 32, 'block_config': (6, 12, 24, 16)}}

dense169 = {'arch': 'densenet169',
            'model_path': [os.path.join(model_dir, 'densenet169-b2777c0a.pth'), 'download', ''][2],
            'cfg': {'num_init_features': 64, 'growth_rate': 32, 'block_config': (6, 12, 32, 32)}}

dense201 = {'arch': 'densenet201',
            'model_path': [os.path.join(model_dir, 'densenet201-c1103571.pth'), 'download', ''][2],
            'cfg': {'num_init_features': 64, 'growth_rate': 32, 'block_config': (6, 12, 48, 32)}}

dense161 = {'arch': 'densenet161',
            'model_path': [os.path.join(model_dir, 'densenet161-8d451a50.pth'), 'download', ''][2],
            'cfg': {'num_init_features': 96, 'growth_rate': 48, 'block_config': (6, 12, 36, 24)}}

dense264 = {'arch': 'densenet264',
            'model_path': [os.path.join(model_dir, ''), 'download', ''][2],
            'cfg': {'num_init_features': 64, 'growth_rate': 32, 'block_config': (6, 12, 64, 48)}}


def DENSENets(arch, cfg, model_path):
    """
    自定义接口 for model_factory
    :param arch: dense121, dense201, dense161, dense264
    :param cfg:  arch configs
    :param model_path: state_dict.pth
    :return: a blank model or pre-trained model
    """
    pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.'
                         r'((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    model = DenseNet(**cfg)

    state_dict = {}
    if os.path.isfile(model_path):
        print('\n=> loading model.pth from %s.' % model_path)
        state_dict = torch.load(model_path)
    elif model_path == 'download':
        print('\n=> downloading model.pth from %s.' % model_urls[arch])
        state_dict = model_zoo.load_url(model_urls[arch])
    else:
        assert model_path == '', '<model_path> must refer to valid-model.ckpt || ''download'' || "".'

    if state_dict:
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
        print('\nSuccess: loaded model.pth from %s.\n' % model_path)

    return model


if __name__ == '__main__':
    import xtils

    # model = models.densenet121()
    model = DENSENets(**dense121)
    print(model)
    xtils.calculate_layers_num(model)
    xtils.calculate_FLOPs_scale(model, input_size=224, use_gpu=False, multiply_adds=True)
    xtils.calculate_params_scale(model, 'million')
