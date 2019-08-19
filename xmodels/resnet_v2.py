# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2018/12/15 12:17'

import os
import torch
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls
from torchvision import models

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


def pretrained(model_dir, arch_name='resnet50'):
    """
    :model_dir:
    :arch_name:
    :return the path of the pretrained model for torch.load(model_path)
    """
    arch_name_list = [
        'resnet18-5c106cde.pth',
        'resnet34-333f7ec4.pth',
        'resnet50-19c8e357.pth',
        'resnet101-5d3b4d8f.pth',
        'resnet152-b121ed2d.pth',
        'densenet121-a639ec97.pth',
        'densenet169-b2777c0a.pth',
        'densenet201-c1103571.pth',
        'densenet161-8d451a50.pth',
    ]
    arch_name = [name for name in arch_name_list if name.startswith(arch_name)]
    if len(arch_name) == 1:
        arch_name = arch_name[0]
    elif len(arch_name) > 1:
        raise Warning('too much choices for %s ... !' % arch_name)
    else:
        raise Warning('no checkpoint exist ... !')
    model_path = os.path.join(model_dir, arch_name)
    return model_path


if __name__ == '__main__':
    import xtils
    # model = models.resnet50()
    model = models.densenet169()
    print(model)
    xtils.calculate_layers_num(model)
    xtils.calculate_params_scale(model, 'million')
