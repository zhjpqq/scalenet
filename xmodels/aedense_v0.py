# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F

"""
只有 双层 的 WaveDenseNet
"""


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
            self.method = nn.ReLU()
        elif method == 'sigmoid':
            self.method = nn.Sigmoid()
        elif method == 'leaky_relu':
            self.method = nn.LeakyReLU(negative_slope=0.02)
        else:
            raise NotImplementedError('--->%s' % method)

    def forward(self, x):
        return self.method(x)


class DownUpBlock(nn.Module):
    def __init__(self, depth, growth, classify=0, active='relu', nclass=1000):
        super(DownUpBlock, self).__init__()
        self.depth = depth
        self.growth = growth
        self.classify = classify
        self.nclass = nclass
        self.active = getattr(nn.functional, active)

        self.bn1 = nn.BatchNorm2d(depth)
        self.conv1 = nn.Conv2d(depth, growth, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth + growth)
        self.conv2 = nn.Conv2d(depth + growth, growth, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(depth + growth)
        self.deconv3 = nn.ConvTranspose2d(depth + growth, growth, 3, stride=2, padding=1,
                                          output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(depth + growth)
        self.deconv4 = nn.ConvTranspose2d(depth + growth, growth, 3, stride=2, padding=1,
                                          output_padding=1, bias=False)

        if self.classify > 0:
            self.classifier = nn.Sequential(
                nn.BatchNorm2d(depth + growth),
                Activate(active),
                AdaAvgPool(),
                ViewLayer(dim=-1),
                nn.Linear(depth + growth, nclass)
            )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x1, x2, x3, x4, pred = x
        else:
            x1, x2, x3, x4, pred = x, None, None, None, None
        res1 = self.conv1(self.active(self.bn1(x1)))
        res1 = torch.cat((res1, x4), 1)
        res2 = self.conv2(self.active(self.bn2(res1)))
        res2 = torch.cat((res2, x3), 1)
        out = res2
        res3 = self.deconv3(self.active(self.bn3(res2)))
        res3 = torch.cat((res3, x2), 1)
        res4 = self.deconv4(self.active(self.bn4(res3)))
        res4 = torch.cat((res4, x1), 1)

        if self.classify > 0:
            out = self.classifier(out)
        else:
            out = None
        pred.append(out)
        # print('up & down -------->')
        # utils.print_size([res4, res3, res2, res1])
        return res4, res3, res2, res1, pred


class TransBlock(nn.Module):
    exp = 2

    def __init__(self, indepth, outdepth, growth=None, pool='avg', active='relu'):
        super(TransBlock, self).__init__()
        if pool == 'avg':
            self.pool2d = F.avg_pool2d
        elif pool == 'max':
            self.pool2d = F.max_pool2d
        self.active = getattr(nn.functional, active)
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(indepth)
        self.conv2 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(indepth)
        self.conv3 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(indepth)
        self.conv4 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x1, x2, x3, x4, pred = x
        else:
            x1, x2, x3, x4, pred = x, None, None, None, None
        x1 = self.conv1(self.active(self.bn1(x1)))
        x1 = self.pool2d(x1, 2)
        x2 = self.conv2(self.active(self.bn2(x2)))
        x2 = self.pool2d(x2, 2)
        x3 = self.conv3(self.active(self.bn3(x3)))
        x3 = self.pool2d(x3, 2)
        x4 = self.conv4(self.active(self.bn4(x4)))
        x4 = self.pool2d(x4, 2)
        return x1, x2, x3, x4, pred


class TransLinkBlock(nn.Module):
    def __init__(self, indepth, outdepth, growth):
        super(TransLinkBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(indepth + growth)
        self.conv2 = nn.Conv2d(indepth + growth, outdepth, 1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(indepth + growth)
        self.conv3 = nn.Conv2d(indepth + growth, outdepth, 1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(indepth + growth)
        self.conv4 = nn.Conv2d(indepth + growth, outdepth, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x1, x2, x3, x4, pred = x
        else:
            x1, x2, x3, x4, pred = x, None, None, None, None
        res1 = self.conv1(F.relu(self.bn1(x1)))
        res1 = F.avg_pool2d(res1)
        x2 = torch.cat((res1, x2), 1)
        res2 = self.conv2(F.relu(self.bn2(x2)))
        res2 = F.avg_pool2d(res2)
        x3 = torch.cat((res2, x3), 1)
        res3 = self.conv3(F.relu(self.bn3(x3)))
        res3 = F.avg_pool2d(res3)
        x4 = torch.cat((res3, x4), 1)
        res4 = self.conv3(F.relu(self.bn4(x4)))
        res4 = F.avg_pool2d(res4)
        return res1, res2, res3, res4, pred


class SummaryBlock(nn.Module):
    def __init__(self, depth, classify=3, avgpool=True, active='relu', nclass=1000):
        super(SummaryBlock, self).__init__()
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
        if self.classify >= 3:
            self.classifier3 = nn.Sequential(
                nn.BatchNorm2d(depth),
                Activate(active),
                AdaAvgPool(),
                ViewLayer(),
                nn.Linear(depth, nclass)
            )
        if self.classify >= 4:
            self.classifier4 = nn.Sequential(
                nn.BatchNorm2d(depth),
                Activate(active),
                AdaAvgPool(),
                ViewLayer(),
                nn.Linear(depth, nclass)
            )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x1, x2, x3, x4, pred = x
        else:
            x1, x2, x3, x4, pred = x, None, None, None, None
        if self.classify == 1:
            x1 = self.classifier1(x1)
            pred.extend([x1])
        elif self.classify == 2:
            x1 = self.classifier1(x1)
            x2 = self.classifier2(x2)
            pred.extend([x2, x1])
        elif self.classify == 3:
            x1 = self.classifier1(x1)
            x2 = self.classifier2(x2)
            x3 = self.classifier3(x3)
            pred.extend([x3, x2, x1])
        elif self.classify == 4:
            x1 = self.classifier1(x1)
            x2 = self.classifier2(x2)
            x3 = self.classifier3(x3)
            x4 = self.classifier4(x4)
            pred.extend([x4, x3, x2, x1])
        else:
            raise NotImplementedError
        return pred


class RockBlock(nn.Module):
    def __init__(self, outdepth=16, branch=4, expand=(1, 1, 1), dataset='cfar'):
        super(RockBlock, self).__init__()
        self.outdepth = outdepth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, outdepth, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(depth),
                # nn.ReLU(inplace=True),
            )
            if branch >= 2:
                self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if branch >= 3:
                self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            if branch >= 4:
                self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                # self.branch4 = nn.RandomPool2d()

        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, outdepth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(outdepth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            if expand[0] == 1:
                self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            elif expand[0] > 1:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(outdepth, outdepth * expand[0], kernel_size=7, stride=2, padding=3, bias=False),
                )
            if expand[1] == 1:
                self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            elif expand[1] > 1:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(outdepth, outdepth * expand[1], kernel_size=7, stride=2, padding=3, bias=False),
                )
            if expand[2] == 1:
                self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            elif expand[2] > 1:
                self.branch4 = nn.Sequential(
                    nn.Conv2d(outdepth, outdepth * expand[2], kernel_size=7, stride=2, padding=3, bias=False),
                )

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, None, pred
        elif self.branch == 2:
            x = self.branch1(x)
            x2 = self.branch2(x)
            return x, x2, None, None, pred
        elif self.branch == 3:
            x = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x2)
            return x, x2, x3, None, pred
        elif self.branch == 4:
            x1 = self.branch1(x)  # 64,32,16,32
            x2 = self.branch2(x1)
            x3 = self.branch3(x2)
            x4 = self.branch4(x1)  # x3
            return x1, x2, x3, x4, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class AEDenseNet(nn.Module):
    def __init__(self, block=DownUpBlock, trans=TransBlock, branch=4, depth=64, growth=12, reduction=0.5,
                 layers=(2, 3, 3, 2), classify=(0, 0, 0, 0, 4), poolMode='avg', active='relu', lastTrans=False,
                 nclass=1000):
        super(AEDenseNet, self).__init__()
        self.branch = branch
        self.depth = depth
        self.layers = layers
        self.trans = trans
        self.classify = classify
        self.poolMode = poolMode
        self.active = active
        self.lastTrans = lastTrans
        self.nclass = nclass

        self.layer0 = RockBlock(outdepth=depth, branch=branch, dataset='imagenet')  # 128*128*64

        indepth = depth
        self.layer1 = self._make_dense_layer(block, layers[0], indepth, growth, classify[0])
        indepth += layers[0] * growth
        outdepth = int(math.floor(indepth * reduction))
        self.trans1 = self._make_trans_layer(trans, indepth, outdepth, growth, poolMode)

        indepth = outdepth
        self.layer2 = self._make_dense_layer(block, layers[1], indepth, growth, classify[1])
        indepth += layers[1] * growth
        outdepth = int(math.floor(indepth * reduction))
        self.trans2 = self._make_trans_layer(trans, indepth, outdepth, growth, poolMode)

        indepth = outdepth
        self.layer3 = self._make_dense_layer(block, layers[2], indepth, growth, classify[2])
        indepth += layers[2] * growth
        outdepth = int(math.floor(indepth * reduction))
        self.trans3 = self._make_trans_layer(trans, indepth, outdepth, growth, poolMode)

        indepth = outdepth
        self.layer4 = self._make_dense_layer(block, layers[3], indepth, growth, classify[3])
        indepth += layers[3] * growth
        if lastTrans:
            outdepth = int(math.floor(indepth * reduction * trans.exp))
            self.trans4 = self._make_trans_layer(trans, indepth, outdepth, growth)
            indepth = outdepth
        else:
            indepth = indepth

        self.summary = SummaryBlock(indepth, classify=classify[4], nclass=nclass)

    def _make_dense_layer(self, block, nums, indepth, growth, cfy):
        layers = []
        for i in range(nums):
            # print(block(indepth, growth, cfy, self.nclass))
            layers.append(block(indepth, growth, cfy, self.active, self.nclass))
            indepth += growth
        return nn.Sequential(*layers)

    def _make_trans_layer(self, block, indepth, outdepth, growth=None, pool='avg'):
        return block(indepth, outdepth, growth, pool, self.active)

    def forward(self, x):
        x = self.layer0(x)
        # utils.print_size(x)
        x = self.layer1(x)
        # utils.print_size(x)
        x = self.trans1(x)
        # utils.print_size(x)
        x = self.layer2(x)
        # utils.print_size(x)
        x = self.trans2(x)
        # utils.print_size(x)
        x = self.layer3(x)
        # utils.print_size(x)
        x = self.trans3(x)
        # utils.print_size(x)
        x = self.layer4(x)
        # utils.print_size(x)
        if self.lastTrans:
            x = self.trans4(x)
        # utils.print_size(x)
        x = self.summary(x)
        x = [p for p in x if p is not None]
        return x


class CifarAEDenseNet(nn.Module):
    def __init__(self, block=DownUpBlock, trans=TransBlock, branch=4, depth=24, growth=12, reduction=0.5,
                 layers=(2, 3, 3), classify=(0, 0, 0, 4), poolMode='avg', active='relu', lastTrans=False, nclass=10):
        super(CifarAEDenseNet, self).__init__()
        self.branch = branch
        self.depth = depth
        self.layers = layers
        self.trans = trans
        self.classify = classify
        self.poolMode = poolMode
        self.active = active
        self.lastTrans = lastTrans
        self.nclass = nclass

        self.layer0 = RockBlock(outdepth=depth, branch=branch, dataset='cifar')  # 128*128*64

        indepth = depth
        self.layer1 = self._make_dense_layer(block, layers[0], indepth, growth, classify[0])
        indepth += layers[0] * growth
        outdepth = int(math.floor(indepth * reduction))
        self.trans1 = self._make_trans_layer(trans, indepth, outdepth, growth, poolMode)

        indepth = outdepth
        self.layer2 = self._make_dense_layer(block, layers[1], indepth, growth, classify[1])
        indepth += layers[1] * growth
        outdepth = int(math.floor(indepth * reduction))
        self.trans2 = self._make_trans_layer(trans, indepth, outdepth, growth, poolMode)
        indepth = outdepth

        indepth = outdepth
        self.layer3 = self._make_dense_layer(block, layers[2], indepth, growth, classify[2])
        indepth += layers[2] * growth
        if lastTrans:
            outdepth = int(math.floor(indepth * reduction * trans.exp))
            self.trans3 = self._make_trans_layer(trans, indepth, outdepth, growth, poolMode)
            indepth = outdepth
        else:
            indepth = indepth

        self.summary = SummaryBlock(indepth, classify=classify[3], active=active, nclass=nclass)

    def _make_dense_layer(self, block, nums, indepth, growth, cfy):
        layers = []
        for i in range(nums):
            # print(block(indepth, growth, cfy, self.nclass))
            layers.append(block(indepth, growth, cfy, self.active, self.nclass))
            indepth += growth
        return nn.Sequential(*layers)

    def _make_trans_layer(self, block, indepth, outdepth, growth=None, poolMode='avg'):
        return block(indepth, outdepth, growth, poolMode, self.active)

    def forward(self, x):
        x = self.layer0(x)
        # utils.print_size(x)
        x = self.layer1(x)
        # utils.print_size(x)
        x = self.trans1(x)
        # utils.print_size(x)
        x = self.layer2(x)
        # utils.print_size(x)
        x = self.trans2(x)
        # utils.print_size(x)
        x = self.layer3(x)
        # utils.print_size(x)
        if self.lastTrans:
            x = self.trans3(x)
        # utils.print_size(x)
        x = self.summary(x)
        x = [p for p in x if p is not None]
        return x


if __name__ == '__main__':
    import xtils

    torch.manual_seed(9528)

    model = AEDenseNet(block=DownUpBlock, trans=TransBlock, branch=4, depth=64, growth=12, reduction=0.5,
                       layers=(2, 3, 3, 3), classify=(0, 0, 0, 0, 2), poolMode='avg',
                       active='sigmoid', lastTrans=False, nclass=1000)
    x = torch.randn(4, 3, 256, 256)
    # utils.tensorboard_add_model(model, x)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d'))
    y = model(x)
    print(sum(model.layers), len(y), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])

    arch_kwargs = {'block': DownUpBlock, 'trans': TransBlock,
                   'branch': 4, 'depth': 36, 'growth': 24, 'reduction': 0.5,
                   'layers': (5, 5, 5), 'classify': (0, 0, 0, 1), 'poolMode': 'max', 'active': 'relu',
                   'lastTrans': False}
    model = CifarAEDenseNet(**arch_kwargs)
    print('\n', model, '\n')

    x = torch.randn(4, 3, 32, 32)
    # utils.tensorboard_add_model(model, x)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d'))
    y = model(x)
    print(sum(model.layers), len(y), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])
