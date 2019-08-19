# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2018/12/15 12:17'

import os
import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls
from torchvision import models
import xtils

"""
与torchvision.models 相同，但原码只能从 <url加载预训练模型>。

此处增加了 model_path 参数，可以直接从 <本地加载预训练模型>。

当上述两个参数都不指定时，加载一个空白的初始化模型。
"""

"""
ResNet for ImageNet - 1000  classes
https://github.com/KaimingHe/deep-residual-networks
https://github.com/facebookresearch/ResNeXt
"""


def resnet18(pretrained=False, model_path=None, **kwargs):
    """Constructs a ResNet-18 model. top1-acc-69.758%  parameter-11.69M

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, model_path=None, **kwargs):
    """Constructs a ResNet-34 model. top1-acc-73.314%  parameter-21.80M

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, model_path=None, **kwargs):
    """Constructs a ResNet-50 model. top1-acc-76.130%  parameter-25.56M

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, model_path=None, **kwargs):
    """Constructs a ResNet-101 model. top1-acc-77.340%  parameter-44.55M

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, model_path=None, **kwargs):
    """Constructs a ResNet-152 model. top1-acc-%   parameter-60.20M

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


# 统一接口 interface for model_factory ##########################################

res18 = {'arch': 'resnet18', 'cfg': {'block': 'BasicBlock', 'layers': [2, 2, 2, 2], 'num_classes': 1000},
         'model_path': ['local', 'download', '', 'xxx.pth'][2]}

res34 = {'arch': 'resnet34', 'cfg': {'block': 'BasicBlock', 'layers': [3, 4, 6, 3], 'num_classes': 1000},
         'model_path': ['local', 'download', '', 'xxx.pth'][2]}

res50 = {'arch': 'resnet50', 'cfg': {'block': 'Bottleneck', 'layers': [3, 4, 6, 3], 'num_classes': 1000},
         'model_path': ['local', 'download', '', 'xxx.pth'][0]}

res101 = {'arch': 'resnet101', 'cfg': {'block': 'Bottleneck', 'layers': [3, 4, 23, 3], 'num_classes': 1000},
          'model_path': ['local', 'download', '', 'xxx.pth'][2]}

res152 = {'arch': 'resnet152', 'cfg': {'block': 'Bottleneck', 'layers': [3, 8, 36, 3], 'num_classes': 1000},
          'model_path': ['local', 'download', '', 'xxx.pth'][2]}


def RESNets(arch, cfg, model_path=''):
    """
    自定义接口 for model_factory
    :param arch: res18, res34, res50, res101, res152
    :param cfg:  arch configs
    :param model_path: state_dict.pth
    :return: a blank model or pre-trained model
    """
    _block = {'BasicBlock': BasicBlock, 'Bottleneck': Bottleneck}

    model_name_map = {
        'res18': 'resnet18',
        'res34': 'resnet34',
        'res50': 'resnet50',
        'res101': 'resnet101',
        'res152': 'resnet152',
    }
    model_cfg_map = {
        'resnet18': res18['cfg'],
        'resnet34': res34['cfg'],
        'resnet50': res50['cfg'],
        'resnet101': res101['cfg'],
        'resnet152': res152['cfg'],
    }
    model_ckpt_map = {
        'resnet18': 'resnet18-5c106cde.pth',
        'resnet34': 'resnet34-333f7ec4.pth',
        'resnet50': 'resnet50-19c8e357.pth',
        'resnet101': 'resnet101-5d3b4d8f.pth',
        'resnet152': 'resnet152-b121ed2d.pth',
    }

    try:
        # 调用官方模型
        name = model_name_map[arch]
    except:
        # 使用自定义模型，如fish33, fish55
        name = arch

    if cfg == '':
        # 调用官方配置
        cfg = model_cfg_map[name]
    else:
        # 使用自定义配置
        assert isinstance(cfg, dict)
        cfg = cfg

    if cfg['block'] in _block.keys():
        cfg['block'] = _block[cfg['block']]

    model = ResNet(**cfg)

    model_dir = xtils.get_pretrained_models()
    if os.path.isfile(model_path):
        print('\n=> loading model.pth from %s.' % model_path)
        model.load_state_dict(torch.load(model_path))
        print('\nSuccess: loaded model.pth from %s.\n' % model_path)
    elif model_path == 'local':
        model_path = os.path.join(model_dir, model_ckpt_map[name])
        print('\n=> loading model.pth from %s.' % model_path)
        model.load_state_dict(torch.load(model_path))
        print('\nSuccess: loaded model.pth from %s.\n' % model_path)
    elif model_path == 'download':
        print('\n=> downloading model.pth from %s.' % model_urls[name])
        model.load_state_dict(model_zoo.load_url(model_urls[name], model_dir))
        print('\nSuccess: downloaded model.pth from %s.\n' % model_urls[name])
    else:
        assert model_path == '', '<model_path> must refer to valid-model.ckpt || ''download'' || "".'

    return model


if __name__ == '__main__':
    import xtils

    res1 = {'arch': 'resnet50', 'cfg': '', 'model_path': ''}

    resx = {'arch': 'resx', 'cfg': {'block': 'Bottleneck', 'layers': [2, 2, 2, 2], 'num_classes': 1000},
            'model_path': ''}

    # model = models.resnet50()
    model = RESNets(**res50)
    model = RESNets(**resx)
    print(model)
    xtils.calculate_layers_num(model)
    xtils.calculate_FLOPs_scale(model, input_size=224, use_gpu=False, multiply_adds=False)
    xtils.calculate_params_scale(model, 'million')
    xtils.calculate_time_cost(model, insize=224, use_gpu=False, toc=1, pritout=True)
