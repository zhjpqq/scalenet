# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/15 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F


class ViewLayer(nn.Module):
    def __init__(self, dim=-1):
        super(ViewLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        # print('view-layer -> ', x.size())
        x = x.view(x.size(0), self.dim)
        return x


class AdaAvgPool(nn.Module):
    def __init__(self, size=0):
        self.size = size
        super(AdaAvgPool, self).__init__()

    def forward(self, x):
        # print('avg-layer -> ', x.size())
        if self.size == -1:
            return x
        if self.size == 0:
            h, w = x.size(2), x.size(3)
            assert h == w
        elif self.size >= 1:
            h, w = self.size, self.size
        else:
            raise NotImplementedError('check the avg kernel size !')
        return F.avg_pool2d(x, kernel_size=(h, w))


class Activate(nn.Module):
    def __init__(self, method='relu'):
        super(Activate, self).__init__()
        if method == 'relu':
            self.method = nn.ReLU(inplace=True)
        elif method == 'sigmoid':
            self.method = nn.Sigmoid()
        elif method == 'leaky_relu':
            self.method = nn.LeakyReLU(negative_slope=0.02)
        else:
            raise NotImplementedError('--->%s' % method)

    def forward(self, x):
        return self.method(x)


class SweetBlock(nn.Module):
    def __init__(self, depth, inter=1, downexp=2, downsize=False):
        super(SweetBlock, self).__init__()
        self.downsize = downsize
        self.bn1 = nn.BatchNorm2d(depth)
        self.conv1 = nn.Conv2d(depth, depth * inter, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth * inter)
        self.deconv2 = nn.ConvTranspose2d(depth * inter, depth, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if downsize:
            self.down1 = nn.Sequential(
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
                nn.Conv2d(depth, depth * downexp, 3, stride=1, padding=1, bias=False),
                nn.AvgPool2d(2)
            )
            self.down2 = nn.Sequential(
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
                nn.Conv2d(depth, depth * downexp, 3, stride=1, padding=1, bias=False),
                nn.AvgPool2d(2),
                # nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            assert len(x) == 3, 'len of x is: %s ...' % len(x)
            x1, x2, pred = x  # (big, small, pred)
        else:
            x1, x2, pred = x, None, None
        res1 = self.conv1(self.relu(self.bn1(x1)))
        res2 = self.deconv2(self.relu(self.bn2(res1)))
        res1 = res1 + x2
        res2 = res2 + x1
        if self.downsize:
            res2 = self.down2(res2)
            res1 = self.down1(res1)
        # utils.print_size([res2, res1])
        return res2, res1, pred


class TransBlock(nn.Module):
    def __init__(self, indepth, outdepth):
        super(TransBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(indepth)
        self.conv2 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x1, x2, pred = x
        else:
            x1, x2, pred = x, None, None
        x1 = self.conv1(F.relu(self.bn1(x1)))
        x1 = F.avg_pool2d(x1, 2)
        x2 = self.conv2(F.relu(self.bn2(x2)))
        x2 = F.avg_pool2d(x2, 2)
        return x1, x2


class SumaryBlock(nn.Module):
    def __init__(self, depth, classify=1, avgpool=True, active='relu', nclass=1000):
        super(SumaryBlock, self).__init__()
        self.classify = classify
        if self.classify >= 1:
            self.classifier1 = nn.Sequential(
                nn.BatchNorm2d(depth),
                Activate(active),
                AdaAvgPool(),
                ViewLayer(),
                nn.Linear(depth, nclass)
            )
        if self.classify >= 2:
            self.classifier2 = nn.Sequential(
                nn.BatchNorm2d(depth),
                Activate(active),
                AdaAvgPool(),
                ViewLayer(),
                nn.Linear(depth, nclass)
            )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x1, x2, pred = x
        else:
            x1, x2, pred = x, None, None
        if self.classify == 1:
            x1 = self.classifier1(x1)
            pred.extend([x1])
        elif self.classify == 2:
            x1 = self.classifier1(x1)
            x2 = self.classifier2(x2)
            pred.extend([x2, x1])
        else:
            raise NotImplementedError
        return pred


class RockBlock(nn.Module):
    def __init__(self, outdepth, branch=2, dataset='cifar'):
        super(RockBlock, self).__init__()
        self.branch = branch
        if dataset == 'cifar':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, outdepth, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(depth),
                # nn.ReLU(inplace=True),
            )
            if branch >= 2:
                self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, pred
        elif self.branch == 2:
            x = self.branch1(x)
            x2 = self.branch2(x)
            return x, x2, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class SweetNet(nn.Module):
    def __init__(self, branch=2, depth=64, layers=(2, 3, 3, 3), expand=(1, 2, 4, 8), downexp=2, downlast=False,
                 inter=(1, 1, 1, 1), classify=1, active='relu', nclass=1000):
        super(SweetNet, self).__init__()
        self.layers = layers
        self.layer0 = RockBlock(depth, branch, dataset='imagenet')
        self.layer1 = self._make_sweet_layer(SweetBlock, layers[0], depth * expand[0], inter[0], downexp, down=True)
        self.layer2 = self._make_sweet_layer(SweetBlock, layers[1], depth * expand[1], inter[1], downexp, down=True)
        self.layer3 = self._make_sweet_layer(SweetBlock, layers[2], depth * expand[2], inter[2], downexp, down=True)
        self.layer4 = self._make_sweet_layer(SweetBlock, layers[3], depth * expand[3], inter[3], downexp, down=downlast)
        if downlast:
            indepth = depth * expand[3] * downexp
        else:
            indepth = depth * expand[3]
        self.classifier = SumaryBlock(indepth, classify, avgpool=True, active=active, nclass=nclass)

    def _make_sweet_layer(self, block, nums, depth, inter=1, downexp=2, down=True):
        layers = []
        for i in range(nums - 1):
            layers.append(block(depth, inter, downexp, downsize=False))
        layers.append(block(depth, inter, downexp, downsize=down))
        return nn.Sequential(*layers)

    def _make_trans_layer(self, block, indepth, outdepth):
        return block(indepth, outdepth)

    def forward(self, x):
        x = self.layer0(x)
        # utils.print_size(x)
        x = self.layer1(x)
        # utils.print_size(x)
        x = self.layer2(x)
        # utils.print_size(x)
        x = self.layer3(x)
        # utils.print_size(x)
        x = self.layer4(x)
        # utils.print_size(x)
        x = self.classifier(x)
        return x


class CifarSweetNet(nn.Module):
    def __init__(self, branch=2, depth=16, layers=(2, 3, 3), expand=(1, 2, 4), downexp=2, downlast=False,
                 inter=(1, 1, 1), classify=1, active='relu', nclass=10):
        super(CifarSweetNet, self).__init__()
        self.layers = layers
        self.layer0 = RockBlock(depth, branch, dataset='cifar')
        self.layer1 = self._make_sweet_layer(SweetBlock, layers[0], depth * expand[0], inter[0], downexp, down=True)
        self.layer2 = self._make_sweet_layer(SweetBlock, layers[1], depth * expand[1], inter[1], downexp, down=True)
        self.layer3 = self._make_sweet_layer(SweetBlock, layers[2], depth * expand[2], inter[2], downexp, down=downlast)
        if downlast:
            indepth = depth * expand[2] * downexp
        else:
            indepth = depth * expand[2]
        self.classifier = SumaryBlock(indepth, classify, avgpool=True, active=active, nclass=nclass)

    def _make_sweet_layer(self, block, nums, depth, inter=1, downexp=2, down=True):
        layers = []
        for i in range(nums - 1):
            layers.append(block(depth, inter, downexp, downsize=False))
        layers.append(block(depth, inter, downexp, downsize=down))
        return nn.Sequential(*layers)

    def _make_trans_layer(self, block, indepth, outdepth):
        return block(indepth, outdepth)

    def forward(self, x):
        x = self.layer0(x)
        # utils.print_size(x)
        x = self.layer1(x)
        # utils.print_size(x)
        x = self.layer2(x)
        # utils.print_size(x)
        x = self.layer3(x)
        # utils.print_size(x)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    import xtils

    torch.manual_seed(9528)
    criterion = nn.CrossEntropyLoss()

    # model = SweetNet(branch=2, depth=64, layers=(2, 5, 3, 2), expand=(1, 2, 4, 8), downexp=2, downlast=True,
    #                  inter=(1, 1, 1, 1), classify=2, active='relu', nclass=1000)
    # print('\n', model, '\n')
    # x = torch.randn(4, 3, 256, 256)
    # # utils.tensorboard_add_model(model, x)
    # utils.calculate_params_scale(model, format='million')
    # utils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'linear'))
    # y = model(x)
    # print(sum(model.layers), len(y), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])

    arch_kwargs = {}
    model = CifarSweetNet(branch=2, depth=16, layers=(2, 2, 2), expand=(1, 2, 4), downexp=2, downlast=False,
                          inter=(1, 1, 1), classify=1, active='relu', nclass=10)
    print('\n', model, '\n')
    x = torch.randn(4, 3, 32, 32)
    # utils.tensorboard_add_model(model, x)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'linear'))
    y = model(x)
    # loss = [criterion(o, torch.randint(0, 10, o.size()).long()) for o in y]
    # optimizer = torch.optim.Adam(params=model.parameters(), lr=0.1)
    # optimizer.zero_grad()
    # sum(loss).backward()
    # optimizer.step()
    print(sum(model.layers), len(y), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])
