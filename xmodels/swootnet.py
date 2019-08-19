# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/15 12:17'

"""
只有双层耦合的 WaveDenseNet, 只有 SingleCouple 模块
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
        self.deconv2 = nn.ConvTranspose2d(depth + growth, growth, 3, stride=2, padding=1,
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
            x1, x2, pred = x
        else:
            x1, x2, pred = x, None, None
        res1 = self.conv1(self.active(self.bn1(x1)))
        res1 = torch.cat((res1, x2), 1)
        res2 = self.deconv2(self.active(self.bn2(res1)))
        res2 = torch.cat((res2, x1), 1)
        out = res1

        if self.classify > 0:
            out = self.classifier(out)
        else:
            out = None
        pred.append(out)
        # print('up & down -------->')
        # utils.print_size([res4, res3, res2, res1])
        return res2, res1, pred


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

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x1, x2, pred = x
        else:
            x1, x2, pred = x, None, None
        x1 = self.conv1(self.active(self.bn1(x1)))
        x1 = self.pool2d(x1, 2)
        x2 = self.conv2(self.active(self.bn2(x2)))
        x2 = self.pool2d(x2, 2)
        return x1, x2, pred


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
            raise NotImplementedError('classify is too big, %s' % self.classify)
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
            raise ValueError('check branch must be in [1, 2]!')


class SwootNet(nn.Module):
    def __init__(self, block=DownUpBlock, trans=TransBlock, branch=2, depth=64, growth=12, reduction=0.5,
                 layers=(2, 3, 3, 3), classify=(0, 0, 0, 0, 2), poolMode='avg', active='relu', lastTrans=False,
                 nclass=1000):
        super(SwootNet, self).__init__()
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

        self.summary = SumaryBlock(indepth, classify=classify[4], nclass=nclass)

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


class CifarSwootNet(nn.Module):
    def __init__(self, block=DownUpBlock, trans=TransBlock, branch=2, depth=64, growth=12, reduction=0.5,
                 layers=(2, 3, 3), classify=(0, 0, 0, 2), poolMode='avg', active='relu', lastTrans=False,
                 nclass=10):
        super(CifarSwootNet, self).__init__()
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
        if lastTrans:
            outdepth = int(math.floor(indepth * reduction * trans.exp))
            self.trans3 = self._make_trans_layer(trans, indepth, outdepth, growth, poolMode)
            indepth = outdepth
        else:
            indepth = indepth

        self.summary = SumaryBlock(indepth, classify=classify[3], nclass=nclass)

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
        if self.lastTrans:
            x = self.trans3(x)
        # utils.print_size(x)
        x = self.summary(x)
        x = [p for p in x if p is not None]
        return x


if __name__ == '__main__':
    import xtils

    torch.manual_seed(9528)
    criterion = nn.CrossEntropyLoss()

    model = SwootNet(block=DownUpBlock, trans=TransBlock, branch=2, depth=64, growth=12, reduction=0.5,
                     layers=(2, 3, 3, 3), classify=(0, 1, 1, 1, 2), poolMode='avg', active='relu', lastTrans=False,
                     nclass=1000)
    # print('\n', model, '\n')
    x = torch.randn(4, 3, 256, 256)
    # utils.tensorboard_add_model(model, x)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'linear'))
    y = model(x)
    print(sum(model.layers), len(y), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])

    arch_kwargs = {}
    model = CifarSwootNet(block=DownUpBlock, trans=TransBlock, branch=2, depth=16, growth=16, reduction=0.5,
                          layers=(5, 5, 5), classify=(0, 0, 0, 1), poolMode='avg', active='relu', lastTrans=False,
                          nclass=10)
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
