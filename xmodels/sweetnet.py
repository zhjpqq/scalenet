# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/15 12:17'

"""
只有双层耦合的 WaveResNet, 只有 SingleCouple 模块
"""

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


class AvgMaxPool(nn.Module):
    def __init__(self, method='avg', ksize=2):
        super(AvgMaxPool, self).__init__()
        if method == 'avg':
            self.method = nn.AvgPool2d(ksize)
        elif method == 'max':
            self.method = nn.MaxPool2d(ksize)

    def forward(self, x):
        return self.method(x)


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


class DownSampleA(nn.Module):
    def __init__(self, indepth, outdepth, pool='avg', double=False):
        super(DownSampleA, self).__init__()
        self.double = double
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)
        self.pool = AvgMaxPool(pool, 2)
        if double:
            self.bn2 = nn.BatchNorm2d(outdepth)
            self.conv2 = nn.Conv2d(outdepth, outdepth, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x), inplace=True))
        x = self.pool(x)
        if self.double:
            x = self.conv2(F.relu(self.bn2(x), inplace=True))
        return x


class DownSampleB(nn.Module):
    def __init__(self, indepth, outdepth, pool='avg', double=False):
        super(DownSampleB, self).__init__()
        self.double = double
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=2, padding=1, bias=False)
        if double:
            self.bn2 = nn.BatchNorm2d(outdepth)
            self.conv2 = nn.Conv2d(outdepth, outdepth, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x), inplace=True))
        if self.double:
            x = self.conv2(F.relu(self.bn2(x), inplace=True))
        return x


class DownSampleC(nn.Module):
    def __init__(self, indepth, outdepth, pool='avg', double=True):
        super(DownSampleC, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdepth)
        self.conv2 = nn.Conv2d(outdepth, outdepth, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x), inplace=True))
        x = self.conv2(F.relu(self.bn2(x), inplace=True))
        return x


class SweetBlock(nn.Module):
    down_func = {'A': DownSampleA, 'B': DownSampleB, 'C': DownSampleC}
    # performace: C >> B > A, C: extra parameters

    def __init__(self, depth, inter=1, classify=0, nclass=1000,
                 downsamp=('A', False), downexp=2, downsize=False,
                 slink='A', pool='avg'):
        super(SweetBlock, self).__init__()
        self.down_func = self.down_func[downsamp[0]]
        self.downsize = downsize
        self.classify = classify
        self.slink = slink     # perfomance: A ≥ C >> B
        self.pool = pool

        self.bn1 = nn.BatchNorm2d(depth)
        self.conv1 = nn.Conv2d(depth, depth * inter, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth * inter)
        self.deconv2 = nn.ConvTranspose2d(depth * inter, depth, 3, stride=2, padding=1, output_padding=1, bias=False)

        if downsize:
            self.down1 = self.down_func(depth, depth * downexp, pool, double=downsamp[1])
            self.down2 = self.down_func(depth, depth * downexp, pool, double=downsamp[1])

        if classify > 0:
            self.classifier = nn.Sequential(
                nn.BatchNorm2d(depth * inter),
                nn.ReLU(),
                AdaAvgPool(),
                ViewLayer(dim=-1),
                nn.Linear(depth * inter, nclass)
            )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            assert len(x) == 3, 'len of x is: %s ...' % len(x)
            x1, x2, pred = x  # (big, small, pred)
        else:
            x1, x2, pred = x, None, None
        if self.slink == 'A':
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.deconv2(F.relu(self.bn2(res1)))
            if self.classify > 0:
                pred.append(self.classifier(res1))
            res1 = res1 + x2
            res2 = res2 + x1
            if self.downsize:
                res2 = self.down2(res2)
                res1 = self.down1(res1)
            # utils.print_size([res2, res1])
            return res2, res1, pred
        elif self.slink == 'B':
            res1 = self.conv1(F.relu(self.bn1(x1)))
            if self.classify > 0:
                pred.append(self.classifier(res1))
            res1 = res1 + x2
            res2 = self.deconv2(F.relu(self.bn2(res1)))
            res2 = res2 + x1
            if self.downsize:
                res2 = self.down2(res2)
                res1 = self.down1(res1)
            # utils.print_size([res2, res1])
            return res2, res1, pred
        elif self.slink == 'C':
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.deconv2(F.relu(self.bn2(res1)))
            if self.classify > 0:
                pred.append(self.classifier(res1))
            res2 = res2 + x1
            if self.downsize:
                res2 = self.down2(res2)
                res1 = self.down1(res1)
            # utils.print_size([res2, res1])
            return res2, res1, pred
        else:
            raise NotImplementedError('check the slink: %s ...' % self.slink)


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
    def __init__(self, branch=2, depth=64, layers=(2, 3, 3, 3), expand=(1, 2, 4, 8), inter=(1, 1, 1, 1),
                 classify=(1, 1, 1, 1, 2), downsamp=('A', False), downexp=2, downlast=False,
                 slink='A', pool='avg', active='relu', nclass=1000):
        super(SweetNet, self).__init__()
        self.layers = layers
        self.layer0 = RockBlock(depth, branch, dataset='imagenet')
        self.layer1 = self._make_sweet_layer(SweetBlock, layers[0], depth * expand[0], inter[0], classify[0],
                                             nclass, downsamp, downexp, True, slink, pool)
        self.layer2 = self._make_sweet_layer(SweetBlock, layers[1], depth * expand[1], inter[1], classify[1],
                                             nclass, downsamp, downexp, True, slink, pool)
        self.layer3 = self._make_sweet_layer(SweetBlock, layers[2], depth * expand[2], inter[2], classify[2],
                                             nclass, downsamp, downexp, True, slink, pool)
        self.layer4 = self._make_sweet_layer(SweetBlock, layers[3], depth * expand[3], inter[3], classify[3],
                                             nclass, downsamp, downexp, downlast, slink, pool)
        if downlast:
            indepth = depth * expand[3] * downexp
        else:
            indepth = depth * expand[3]
        self.classifier = SumaryBlock(indepth, classify[4], avgpool=True, active=active, nclass=nclass)

    def _make_sweet_layer(self, block, nums, depth, inter, cfy, nclass,
                          downsamp, downexp, down=True, slink='A', pool='avg'):
        layers = []
        for i in range(nums - 1):
            layers.append(block(depth, inter, cfy, nclass,
                                downsamp, downexp, downsize=False, slink=slink, pool=pool))
        layers.append(block(depth, inter, cfy, nclass,
                            downsamp, downexp, downsize=down, slink=slink, pool=pool))
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
    def __init__(self, branch=2, depth=16, layers=(2, 3, 3), expand=(1, 2, 4), inter=(1, 1, 1),
                 classify=(1, 1, 1, 2), downsamp=('A', False), downexp=2, downlast=False,
                 slink='A', pool='avg', active='relu', nclass=10):
        super(CifarSweetNet, self).__init__()
        assert branch <= 2 and branch >= 1, 'branch !!!'
        self.layers = layers
        self.layer0 = RockBlock(depth, branch, dataset='cifar')
        self.layer1 = self._make_sweet_layer(SweetBlock, layers[0], depth * expand[0], inter[0], classify[0],
                                             nclass, downsamp, downexp, True, slink, pool)
        self.layer2 = self._make_sweet_layer(SweetBlock, layers[1], depth * expand[1], inter[1], classify[1],
                                             nclass, downsamp, downexp, True, slink, pool)
        self.layer3 = self._make_sweet_layer(SweetBlock, layers[2], depth * expand[2], inter[2], classify[2],
                                             nclass, downsamp, downexp, downlast, slink, pool)
        if downlast:
            indepth = depth * expand[2] * downexp
        else:
            indepth = depth * expand[2]
        self.classifier = SumaryBlock(indepth, classify[3], avgpool=True, active=active, nclass=nclass)

        self.softmax = nn.Softmax(dim=1)
        self.cross = nn.CrossEntropyLoss()
        self.kldiv = nn.KLDivLoss()

    def _make_sweet_layer(self, block, nums, depth, inter, cfy, nclass,
                          downsamp, downexp, down=True, slink='A', pool='avg'):
        layers = []
        for i in range(nums - 1):
            layers.append(block(depth, inter, cfy, nclass,
                                downsamp, downexp, downsize=False, slink=slink, pool=pool))
        layers.append(block(depth, inter, cfy, nclass,
                            downsamp, downexp, downsize=down, slink=slink, pool=pool))
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
        # utils.print_size(x)
        return x

    def kldiv_loss(self, x, sumit=False):
        pred = [self.softmax(o) for o in x if o is not None]
        loss = [self.kldiv(o, pred[-1]) for o in pred[:-1]]
        if sumit:
            loss = sum(loss)
        return loss

    def kldiv_cross_loss(self, x, labels, sumit=False):
        klloss = self.kldiv_loss(x, False)
        cross = self.cross(x[-1], labels)
        loss = klloss.append(cross)
        if sumit:
            loss = sum(loss)
        return loss


if __name__ == '__main__':
    import xtils

    torch.manual_seed(9528)
    criterion = nn.CrossEntropyLoss()
    kldiv = nn.KLDivLoss()

    # model = SweetNet(branch=2, depth=64, layers=(2, 3, 3, 3), expand=(1, 2, 4, 8), inter=(1, 1, 1, 1),
    #                  downsamp=('A', False), downexp=2, downlast=False,
    #                  classify=(0, 0, 1, 1, 1), slink='A', pool='avg', active='relu', nclass=1000)
    # print('\n', model, '\n')
    # x = torch.randn(4, 3, 256, 256)
    # # utils.tensorboard_add_model(model, x)
    # utils.calculate_params_scale(model, format='million')
    # utils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'linear'))
    # y = model(x)
    # print(sum(model.layers), len(y), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])

    arch_kwargs = {}
    model = CifarSweetNet(branch=2, depth=16, layers=(15, 15, 15), expand=(1, 2, 4), inter=(1, 1, 1),
                          downsamp=('C', False), downexp=2, downlast=False,
                          classify=(0, 0, 0, 1), slink='A', pool='avg', active='relu', nclass=10)
    print('\n', model, '\n')
    x = torch.randn(4, 3, 32, 32)
    # utils.tensorboard_add_model(model, x)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'linear'))
    y = model(x)
    print(sum(model.layers), len(y), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])

    # pred = [nn.Softmax(dim=1)(o) for o in y if o is not None]
    # print(len(pred), pred[0].size(), '\n', pred)
    # loss = [kldiv(o, pred[-1]) for o in pred[:-1] if o is not None]
    # print(loss)
    # print(model.kldiv_loss(y))
