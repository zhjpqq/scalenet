# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/7/23 12:17'

"""
   使用改进的ReLU()优化ResNet.  

   AdaReLU() = {
        'case1': ReLU(x + ax + bx^2 + cx^3 + dx^4 + ... + bias), 
        'case2': ReLU(x) + ax + bx^2 + cx^3 + dx^4 + ... + bias,
        'case3': ax + bx^2 + cx^3 + dx^4 + ... + bias,
        }
"""

import os
import math
import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
import xtils
from xmodules.relu_func import AdaReLU

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = AdaReLU(abc_nums=3, abc_val=0.0001, abc_bias=0, abc_relu=1, abc_addx=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = AdaReLU(abc_nums=3, abc_val=0.0001, abc_bias=0, abc_relu=1, abc_addx=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = AdaReLU(abc_nums=3, abc_val=0.0001, abc_bias=0, abc_relu=1, abc_addx=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = AdaReLU(abc_nums=3, abc_val=0.0001, abc_bias=0, abc_relu=1, abc_addx=True)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu3 = AdaReLU(abc_nums=3, abc_val=0.0001, abc_bias=0, abc_relu=1, abc_addx=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class ActiveRes(nn.Module):
    _block = {'basic': BasicBlock, 'bottle': Bottleneck}

    def __init__(self, stage=4, block='bottle', layers=(3, 4, 6, 3), channels=(64, 128, 256, 512),
                 num_classes=1000, ckpt_path=None, ckpt_who='all'):
        self.inplanes = channels[0]
        super(ActiveRes, self).__init__()
        assert len(channels) == stage
        assert len(layers) == stage
        assert ckpt_who in ['resnet', 'backbone', 'all']

        block = self._block[block]
        self.stage = stage

        if num_classes == 1000:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif num_classes in [10, 100]:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.Sequential()

        self.layer1 = self._make_layer(block, channels[0], layers[0])
        if stage >= 2:
            self.layer2 = self._make_layer(block, channels[1], layers[1], stride=2)
        if stage >= 3:
            self.layer3 = self._make_layer(block, channels[2], layers[2], stride=2)
        if stage >= 4:
            self.layer4 = self._make_layer(block, channels[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(channels[-1] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, AdaReLU):
            #     m.abc_nums = 3
            #     m.abc_val = 0

        if ckpt_path is not None:
            if ckpt_who in ['resnet', 'backbone']:
                strict = False
            else:
                strict = True
            self.load_trained_weights(ckpt_path, strict)

        self.train_which_now = {'bone+active': False, 'active': False}
        self.eval_which_now = {'bone+active': False}

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def load_trained_weights(self, ckpt_path, strict=True):
        """
        加载backbone, 不包括AdaReLU || load ResNet
        """
        state_dict = torch.load(f=ckpt_path)
        self.load_state_dict(state_dict, strict=strict)

    def train_mode(self, ite, cfg):
        # 当迭代次数 ite 超过设定值，开启对应的训练阶段
        which = None
        for key in sorted(cfg.train_which.keys())[::-1]:
            if ite >= key:
                which = cfg.train_which[key]
                break
        self.set_train_which(part=which, name_which=cfg.name_which)

    def eval_mode(self, ite, cfg):
        # 当迭代次数 ite 超过设定值，开启对应的测试阶段
        which = None
        for key in sorted(cfg.eval_which.keys())[::-1]:
            if ite >= key:
                which = cfg.eval_which[key]
                break
        self.set_eval_which(part=which, name_which=cfg.name_which)

    def reset_mode(self, mode='train'):
        if mode == 'train':
            for k, v in self.train_which_now.items():
                self.train_which_now[k] = False
        elif mode == 'val':
            for k, v in self.eval_which_now.items():
                self.eval_which_now[k] = False

    def set_train_which(self, part, name_which='none'):
        assert part in self.train_which_now, '设定超出可选项范围--> %s' % part
        self.reset_mode(mode='val')
        if self.train_which_now[part]:
            return
        else:
            self.reset_mode(mode='train')
            self.train_which_now[part] = True

        if part == 'bone+active':
            for n, m in self.named_modules():
                m.train()
                for p in m.parameters():
                    p.requires_grad = True
        elif part == 'active':
            for n, m in self.named_modules():
                if isinstance(m, AdaReLU):
                    m.train()
                    for p in m.parameters():
                        p.requires_grad = True
                else:
                    m.eval()
                    for p in m.parameters():
                        p.requires_grad = False

    def set_eval_which(self, part, name_which='none'):
        assert part in self.eval_which_now, '设定超出可选项范围--> %s' % part
        self.reset_mode(mode='train')
        if self.eval_which_now[part]:
            return
        else:
            self.reset_mode(mode='val')
            self.eval_which_now[part] = True

        for n, m in self.named_modules():
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(1, self.stage + 1):
            x = getattr(self, 'layer%s' % i)(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


active50 = ActiveRes(stage=4, block='bottle', channels=(64, 128, 256, 512),
                     layers=[3, 4, 6, 3], num_classes=1000, ckpt_path=None)

actcifar10 = ActiveRes(stage=3, block='bottle', channels=(16, 32, 48),
                       layers=[3, 4, 6], num_classes=10, ckpt_path=None)

if __name__ == '__main__':
    # # imagenet
    model_path = os.path.join('E:\PreTrainedModels', 'resnet50-19c8e357.pth')
    model = ActiveRes(stage=4, block='bottle', channels=(64, 128, 256, 512),
                      layers=[3, 4, 6, 3], num_classes=1000, ckpt_path=model_path, ckpt_who='backbone')
    x = torch.rand(2, 3, 224, 224)
    z = model(x)
    print(model)
    print('z=>', z)

    # # cifar
    # model = actcifar10
    # print(model)
    #
    # xtils.calculate_layers_num(model)
    # xtils.calculate_FLOPs_scale(model, input_size=32)
    # xtils.calculate_params_scale(model, format='million')
    # xtils.calculate_time_cost(model, insize=32, toc=3)
    #
    # x = torch.rand(2, 3, 32, 32)
    # z = model(x)
    # print('\nz=>', z)
