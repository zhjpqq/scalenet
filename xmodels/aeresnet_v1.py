# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F


"""
只有 双层 的 WaveResNet
"""


class DownSampleA(nn.Module):
    def __init__(self, indepth, outdepth):
        super(DownSampleA, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdepth)
        self.conv2 = nn.Conv2d(outdepth, outdepth, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))
        return x


class DownSampleB(nn.Module):
    def __init__(self, indepth, outdepth, pool='max'):
        super(DownSampleB, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)
        if pool == 'max':
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif pool == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.pool(x)
        return x


class DownSampleC(nn.Module):
    def __init__(self, indepth, outdepth):
        super(DownSampleC, self).__init__()
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x)))
        return x


class ChannelSample(nn.Module):
    def __init__(self, indepth, outdepth, double=True):
        super(ChannelSample, self).__init__()
        self.double = double
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(outdepth)
        self.conv2 = nn.Conv2d(outdepth, outdepth, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(F.relu(self.bn1(x)))
        if self.double:
            x = self.conv2(F.relu(self.bn2(x)))
        return x


class ViewLayer(nn.Module):
    def __init__(self, dim=-1):
        super(ViewLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        # print('view-layer -> ', x.size())
        x = x.view(x.size(0), self.dim)
        return x


class AdaAvgPool(nn.AvgPool2d):
    def __init__(self):
        super(AdaAvgPool, self).__init__(kernel_size=1)

    def forward(self, x):
        # print('AdaAvg-layer -> ', x.size())
        h, w = x.size(2), x.size(3)
        assert h == w
        # super(AdaAvgPool, self).__init__(kernel_size=(h, w))
        return F.avg_pool2d(x, kernel_size=(h, w))


class AEBlock(nn.Module):
    exp1 = 1  # exp1, exp2 is the internal expand of this block, set it >=1 will be ok.
    exp2 = 1
    exp3 = 2  # must==2, double the channel for the next down-size layer
    exp4 = 1  # maybe no channel expand for the last fc layer

    down_func = {'A': DownSampleA, 'B': DownSampleB, 'C': DownSampleC}

    # A >> B >>> C , exp:22>>26>>>25, A extra Parameters

    def __init__(self, depth, after=True, down=False, last_down=False, dfunc='A', last_branch=1, classify=0,
                 nclass=1000, blockexp=(1, 1), slink='A'):
        super(AEBlock, self).__init__()
        self.depth = depth
        self.after = after  # dose still have another AEBlock behind of this AEBlock. ?
        self.down = down  # dose connect to a same-size AEBlock or down-sized AEBlock. ?
        self.last_branch = last_branch  # how many output-ways (branches) for the classifier ?
        self.last_down = last_down  # dose down size for the classifier?
        self.slink = slink  # A >> C > B
        self.dfunc = dfunc
        self.down_func = self.down_func[dfunc]
        self.classify = classify
        self.exp1, self.exp2 = blockexp

        self.bn1 = nn.BatchNorm2d(depth)
        self.conv1 = nn.Conv2d(depth, depth * self.exp1, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth * self.exp1)
        self.conv2 = nn.Conv2d(depth * self.exp1, depth * self.exp2, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(depth * self.exp2)
        self.deconv3 = nn.ConvTranspose2d(depth * self.exp2, depth * self.exp1, 3, stride=2, padding=1,
                                          output_padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(depth * self.exp1)
        self.deconv4 = nn.ConvTranspose2d(depth * self.exp1, depth, 3, stride=2, padding=1, output_padding=1,
                                          bias=False)

        if self.classify > 0:
            self.classifier = nn.Sequential(
                nn.BatchNorm2d(depth * self.exp2),
                nn.ReLU(),
                # nn.AvgPool2d(self.classify),
                AdaAvgPool(),
                ViewLayer(dim=-1),
                nn.Linear(depth * self.exp2, nclass)
            )

        if self.after:
            if self.down:
                self.down_a = self.down_func(depth, depth * self.exp3)
                self.down_b = self.down_func(depth * self.exp1, depth * self.exp1 * self.exp3)
                self.down_c = self.down_func(depth * self.exp2, depth * self.exp2 * self.exp3)
        else:
            if self.last_down:
                if self.last_branch >= 1:
                    self.down_d = self.down_func(depth, depth * self.exp4)
                if self.last_branch >= 2:
                    self.down_e = self.down_func(depth * self.exp1, depth * self.exp4)
                if self.last_branch >= 3:
                    self.down_f = self.down_func(depth * self.exp2, depth * self.exp4)

    def forward(self, x):
        if isinstance(x, tuple):
            x1, x2, x3, pred = x
            # print('x1, x2, x3: ', x1.size(), x2.size(), x3.size(), pred)
        else:
            x1, x2, x3, pred = x, None, None, None

        if self.slink == 'A':
            res1 = self.conv1(F.relu(self.bn1(x1)))
            out = res1 if x2 is None else res1 + x2
            res2 = self.conv2(F.relu(self.bn2(out)))
            out = res2 if x3 is None else res2 + x3
            res3 = self.deconv3(F.relu(self.bn3(out)))
            res4 = self.deconv4(F.relu(self.bn4(res3 + res1)))  # + res1
            res4 = res4 + x1

        elif self.slink == 'B':
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.conv2(F.relu(self.bn2(res1)))
            res3 = self.deconv3(F.relu(self.bn3(res2)))
            res4 = self.deconv4(F.relu(self.bn4(res3)))
            out = res2

            res1 = res1 if x2 is None else res1 + x2
            res2 = res2 if x3 is None else res2 + x3
            res3 = res3 + res1
            res4 = res4 + x1

        elif self.slink == 'C':
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.conv2(F.relu(self.bn2(res1)))
            res3 = self.deconv3(F.relu(self.bn3(res2)))
            res4 = self.deconv4(F.relu(self.bn4(res3)))
            out = res2

            # res1 = res1 if x2 is None else res1 + x2
            res2 = res2 if x3 is None else res2 + x3
            res3 = res3 if x3 is None else res3 + x2
            res4 = res4 + x1

        else:
            raise NotImplementedError

        if self.classify > 0:
            out = self.classifier(out)
        else:
            out = None
        pred.append(out)

        if self.after:
            if self.down:
                res4 = self.down_a(res4)
                res3 = self.down_b(res3)
                res2 = self.down_c(res2)
                return res4, res3, res2, pred
            else:
                return res4, res3, res2, pred
        else:
            if self.last_branch == 1:
                if self.last_down:
                    res4 = self.down_d(res4)
                return res4, pred
            elif self.last_branch == 2:
                if self.last_down:
                    res4 = self.down_d(res4)
                    res3 = self.down_e(res3)
                return res3, res4, pred
            elif self.last_branch == 3:
                if self.last_down:
                    res4 = self.down_d(res4)
                    res3 = self.down_e(res3)
                    res2 = self.down_f(res2)
                return res3, res4, res2, pred
            else:
                raise ValueError('check self.last_branch value...')


class RockBlock(nn.Module):
    def __init__(self, depth=16, branch=3, expand=(1, 1), dataset='cfar'):
        super(RockBlock, self).__init__()
        self.depth = depth
        self.branch = branch
        self.expand = expand
        if dataset == 'cifar':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.BatchNorm2d(depth),
                # nn.ReLU(inplace=True),
            )
            self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            # self.branch2 = nn.Sequential(
            #     nn.BatchNorm2d(depth),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(depth, depth, kernel_size=3, stride=2, padding=1, bias=False),
            # )
            # self.branch3 = nn.Sequential(
            #     nn.BatchNorm2d(depth),
            #     nn.ReLU(inplace=True),
            #     nn.Conv2d(depth, depth, kernel_size=3, stride=2, padding=1, bias=False),
            # )
        elif dataset == 'imagenet':
            self.branch1 = nn.Sequential(
                nn.Conv2d(3, depth, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(depth),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            if expand[0] == 1:
                self.branch2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            elif expand[0] > 1:
                self.branch2 = nn.Sequential(
                    nn.Conv2d(depth, depth * expand[0], kernel_size=7, stride=2, padding=3, bias=False),
                )
            if expand[0] == 1:
                self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            elif expand[0] > 1:
                self.branch3 = nn.Sequential(
                    nn.Conv2d(depth, depth * expand[1], kernel_size=7, stride=2, padding=3, bias=False),
                )

    def forward(self, x):
        pred = []
        if self.branch == 1:
            x = self.branch1(x)
            return x, None, None, pred
        elif self.branch == 2:
            x = self.branch1(x)
            x2 = self.branch2(x)
            return x, x2, None, pred
        elif self.branch == 3:
            x = self.branch1(x)
            x2 = self.branch2(x)
            x3 = self.branch3(x2)
            return x, x2, x3, pred
        else:
            raise ValueError('check branch must be in [1, 2, 3]!')


class AEResNet(nn.Module):
    def __init__(self, branch=3, depth=64, blockexp=(1, 1), slink='A',
                 expand=(1, 2, 4, 8),
                 layers=(2, 3, 3, 2),
                 dfunc=('C', 'C', 'C', 'C'),
                 classify=(16, 8, 4, 2),
                 last_down=True,
                 nclass=1000):
        super(AEResNet, self).__init__()
        self.branch = branch
        self.depth = depth
        self.slink = slink
        self.expand = expand
        self.layers = layers
        self.dfunc = dfunc
        self.classify = classify
        self.last_down = last_down
        self.nclass = nclass

        self.layer0 = RockBlock(depth=depth, branch=branch, dataset='imagenet')  # 128*128*64

        self.layer1 = self._make_aelayer(AEBlock, layers[0], expand[0] * depth, slink, after=True, last_down=False,
                                         dfunc=dfunc[0], cfy=classify[0], blockexp=blockexp)  # 64*64*64
        self.layer2 = self._make_aelayer(AEBlock, layers[1], expand[1] * depth, slink, after=True, last_down=False,
                                         dfunc=dfunc[1], cfy=classify[1], blockexp=blockexp)  # 32*32*128
        self.layer3 = self._make_aelayer(AEBlock, layers[2], expand[2] * depth, slink, after=True, last_down=False,
                                         dfunc=dfunc[2], cfy=classify[2], blockexp=blockexp)  # 16*16*256
        self.layer4 = self._make_aelayer(AEBlock, layers[3], expand[3] * depth, slink, after=False, last_down=last_down,
                                         dfunc=dfunc[3], cfy=classify[3], blockexp=blockexp)  # 8*8*(512*exp4)

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(expand[3] * depth * AEBlock.exp4),
            nn.ReLU(),
            # nn.AvgPool2d(4 || 8),
            AdaAvgPool(),
            ViewLayer(dim=-1),
            nn.Linear(expand[3] * depth * AEBlock.exp4, nclass)
        )

    def _make_aelayer(self, block, block_nums, indepth, slink, after, last_down, dfunc, cfy=0, blockexp=(1, 1)):
        layers = []
        for i in range(block_nums - 1):
            layers.append(block(indepth, after=True, down=False, last_down=False, dfunc=dfunc, classify=cfy,
                                nclass=self.nclass, blockexp=blockexp, slink=slink))
        layers.append(block(indepth, after=after, down=True, last_down=last_down, dfunc=dfunc, classify=cfy,
                            nclass=self.nclass, blockexp=blockexp, slink=slink))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x, pred = self.layer4(x)
        x = self.classifier(x)
        pred.append(x)
        pred = [p for p in pred if p is not None]
        return pred


class CifarAEResNet(nn.Module):
    def __init__(self, branch=3, depth=16, blockexp=(1, 1), slink='A',
                 expand=(1, 2, 4),
                 layers=(2, 3, 3),
                 dfunc=('C', 'C', 'C'),
                 classify=(8, 4, 2),
                 last_down=True,
                 nclass=10):
        super(CifarAEResNet, self).__init__()
        self.branch = branch
        self.depth = depth
        self.slink = slink
        self.expand = expand
        self.layers = layers
        self.dfunc = dfunc
        self.classify = classify
        self.last_down = last_down
        self.nclass = nclass

        self.layer0 = RockBlock(depth=depth, branch=branch, dataset='cifar')  # 32*32*1d
        self.layer1 = self._make_aelayer(AEBlock, layers[0], expand[0] * depth, slink, after=True, last_down=False,
                                         dfunc=dfunc[0], cfy=classify[0], blockexp=blockexp)  # 16*16*2d
        self.layer2 = self._make_aelayer(AEBlock, layers[1], expand[1] * depth, slink, after=True, last_down=False,
                                         dfunc=dfunc[1], cfy=classify[1], blockexp=blockexp)  # 8*8*4d
        self.layer3 = self._make_aelayer(AEBlock, layers[2], expand[2] * depth, slink, after=False, last_down=last_down,
                                         dfunc=dfunc[2], cfy=classify[2], blockexp=blockexp)  # 4*4*(4d*exp4)

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(expand[2] * depth * AEBlock.exp4),
            nn.ReLU(),
            # nn.AvgPool2d(4),
            AdaAvgPool(),
            ViewLayer(dim=-1),
            nn.Linear(expand[2] * depth * AEBlock.exp4, nclass)
        )

    def _make_aelayer(self, block, block_nums, indepth, slink, after, last_down, dfunc, cfy=0, blockexp=(1, 1)):
        layers = []
        for i in range(block_nums - 1):
            layers.append(block(depth=indepth, after=True, down=False, last_down=last_down, dfunc=dfunc, classify=cfy,
                                nclass=self.nclass, blockexp=blockexp, slink=slink))
        layers.append(block(depth=indepth, after=after, down=True, last_down=last_down, dfunc=dfunc, classify=cfy,
                            nclass=self.nclass, blockexp=blockexp, slink=slink))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x, pred = self.layer3(x)
        x = self.classifier(x)
        pred.append(x)
        pred = [p for p in pred if p is not None]
        return pred


if __name__ == '__main__':
    import xtils

    torch.manual_seed(1)

    dfunc1 = ('A', 'A', 'A', 'A')
    dfunc2 = ('B', 'B', 'B', 'B')
    dfunc3 = ('C', 'C', 'C', 'C')
    model = AEResNet(branch=3, depth=64, blockexp=(2, 2), slink='B', expand=(1, 2, 4, 8), layers=(2, 2, 2, 2),
                     dfunc=dfunc3, classify=(2, 2, 2, 2), last_down=True, nclass=1000)
    x = torch.randn(4, 3, 256, 256)
    # utils.tensorboard_add_model(model, x)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d'))
    y = model(x)
    print(len(y), sum(model.layers), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])

    # dfunc1 = ('A', 'A', 'A')
    # dfunc2 = ('B', 'B', 'B')
    # dfunc3 = ('C', 'C', 'C')
    # model = CifarAEResNet(branch=3, depth=16, blockexp=(1, 1), slink='A', expand=(1, 2, 4), layers=(1, 1, 1),
    #                       dfunc=dfunc1, classify=(1, 1, 1), last_down=True, nclass=10)
    # print('\n', model, '\n')
    # x = torch.randn(4, 3, 32, 32)
    # # utils.tensorboard_add_model(model, x)
    # utils.calculate_params_scale(model, format='million')
    # utils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
    # y = model(x)
    # print(len(y), sum(model.layers), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])
