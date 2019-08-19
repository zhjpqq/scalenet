# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from xmodules.downsample import DownSampleA, DownSampleB, DownSampleC, DownSampleD, DownSampleE, DownSampelH
from xmodules.rockblock import RockBlock, RockBlockM, RockBlockU, RockBlockV, RockBlockR
from xmodules.classifier import ViewLayer, AdaAvgPool, Activate, XClassifier


"""
  WaveResNet => + DoubleCouple + SingleCouple + Summary(+merge +split) + Boost + BottleNeck
"""


class DoubleCouple(nn.Module):
    exp3 = 2  # must==2, double the channel for the next down-size layer
    down_func = {'A': DownSampleA, 'B': DownSampleB, 'C': DownSampleC,
                 'D': DownSampleD, 'E': DownSampleE, 'H': DownSampelH}

    def __init__(self, depth, growth, slink='A', after=True, down=False, dfunc='A',
                 classify=0, nclass=1000, last_branch=1, last_down=False):
        super(DoubleCouple, self).__init__()
        assert last_branch <= 3, '<last_branch> of DoubleCouple should be <= 3...'
        assert len(growth) == 2, 'len of <growth> of DoubleCouple should be 2'
        self.depth = depth
        self.growth = growth  # how much channels will be added or minus, when plain size is halved or doubled.
        growtha, growthb = growth
        self.slink = slink
        self.after = after  # dose still have another DoubleCouple behind of this DoubleCouple. ?
        self.down = down  # dose connect to a same-size DoubleCouple or down-sized DoubleCouple. ?
        self.dfunc = dfunc
        self.down_func = self.down_func[dfunc]
        self.last_branch = last_branch  # how many output-ways (branches) for the classifier ?
        self.last_down = last_down  # dose down size for the classifier?
        self.classify = classify
        self.nclass = nclass
        self.active_fc = False

        self.bn1 = nn.BatchNorm2d(depth)
        self.conv1 = nn.Conv2d(depth, depth + growtha, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth + growtha)
        self.conv2 = nn.Conv2d(depth + growtha, depth + 2 * growtha, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(depth + 2 * growtha)
        self.deconv3 = nn.ConvTranspose2d(depth + 2 * growtha, depth + growtha, 4, stride=2, padding=1, bias=False,
                                          output_padding=0, dilation=1)
        self.bn4 = nn.BatchNorm2d(depth + growtha)
        self.deconv4 = nn.ConvTranspose2d(depth + growtha, depth, 4, stride=2, padding=1, bias=False,
                                          output_padding=0, dilation=1)

        if self.classify > 0:
            self.classifier = XClassifier(depth + 2 * growtha, nclass)

        if self.after:
            if self.down:
                self.down_res4 = self.down_func(depth, depth * self.exp3)
                self.down_res3 = self.down_func(depth + growtha, depth * self.exp3 + growthb)
                self.down_res2 = self.down_func(depth + 2 * growtha, depth * self.exp3 + 2 * growthb)
        else:
            if self.last_down:
                if self.last_branch >= 1:
                    self.down_last4 = self.down_func(depth, depth + growthb)
                if self.last_branch >= 2:
                    self.down_last3 = self.down_func(depth + growtha, depth + growthb)  # + growtha
                if self.last_branch >= 3:
                    self.down_last2 = self.down_func(depth + 2 * growtha, depth + growthb)  # + 2 * growtha
            else:
                if self.classify > 0 and self.last_branch == 3:
                    # 最后一个Couple的中间层被当做branch输出而对接在Summary上.
                    # 因此，删除此Couple自带的Classifier,以免与Summary中的Classifier重复.
                    delattr(self, 'classifier')
                    self.classify = 0
                    print('Note: 1 xfc will be deleted because of duplicate with the last-fc!')

    def forward(self, x):
        if isinstance(x, tuple):
            x1, x2, x3, pred = x
            # print('x1, x2, x3: ', x1.size(), x2.size(), x3.size(), pred)
        else:
            x1, x2, x3, pred = x, None, None, None

        # add-style
        if self.slink == 'A':
            res1 = self.conv1(F.relu(self.bn1(x1)))
            out = res1 if x2 is None else res1 + x2
            res2 = self.conv2(F.relu(self.bn2(out)))
            out = res2 if x3 is None else res2 + x3
            res3 = self.deconv3(F.relu(self.bn3(out)))
            res4 = self.deconv4(F.relu(self.bn4(res3 + res1)))
            res4 = res4 + x1

        elif self.slink == 'X':
            # first add, then shortcut, low < 'A'
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res1 = res1 if x2 is None else res1 + x2
            res2 = self.conv2(F.relu(self.bn2(res1)))
            res2 = res2 if x3 is None else res2 + x3
            out = res2
            res3 = self.deconv3(F.relu(self.bn3(res2)))
            res3 = res3 + res1
            res4 = self.deconv4(F.relu(self.bn4(res3)))
            res4 = res4 + x1

        elif self.slink == 'B':
            # A的简化版，只有每个block内部的2个内连接
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.conv2(F.relu(self.bn2(res1)))
            res3 = self.deconv3(F.relu(self.bn3(res2)))
            res4 = self.deconv4(F.relu(self.bn4(res3 + res1)))
            res4 = res4 + x1
            out = res2

        elif self.slink == 'C':
            # A的简化版，只有每个block内内部最大尺寸的那1个内连接， 类似resnet
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.conv2(F.relu(self.bn2(res1)))
            res3 = self.deconv3(F.relu(self.bn3(res2)))
            res4 = self.deconv4(F.relu(self.bn4(res3)))
            res4 = res4 + x1
            out = res2

        elif self.slink == 'D':
            # A的简化版，2个夸block连接，1个block内连接(res4+x1)
            res1 = self.conv1(F.relu(self.bn1(x1)))
            out = res1 if x2 is None else res1 + x2
            res2 = self.conv2(F.relu(self.bn2(out)))
            out = res2 if x3 is None else res2 + x3
            res3 = self.deconv3(F.relu(self.bn3(out)))
            res4 = self.deconv4(F.relu(self.bn4(res3)))
            res4 = res4 + x1

        elif self.slink == 'E':
            # A的简化版，只有2个夸block连接
            res1 = self.conv1(F.relu(self.bn1(x1)))
            out = res1 if x2 is None else res1 + x2
            res2 = self.conv2(F.relu(self.bn2(out)))
            out = res2 if x3 is None else res2 + x3
            res3 = self.deconv3(F.relu(self.bn3(out)))
            res4 = self.deconv4(F.relu(self.bn4(res3)))
            res4 = res4

        elif self.slink == 'F':
            # A的简化版，1个夸block连接，1个block内连接
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.conv2(F.relu(self.bn2(res1)))
            out = res2 if x3 is None else res2 + x3
            res3 = self.deconv3(F.relu(self.bn3(out)))
            res4 = self.deconv4(F.relu(self.bn4(res3)))
            res4 = res4 + x1

        # cat-style
        elif self.slink == 'G':
            # Note: x2, x3 在当前block内不生效，全部累加到本stage的最后一个block内, 在downsize的时候生效
            # only x1 for calculate, x2 / x3 all moved to next block until the last block
            # Not good! x2, x3 be wasted .
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.conv2(F.relu(self.bn2(res1)))
            res3 = self.deconv3(F.relu(self.bn3(res2)))
            res4 = self.deconv4(F.relu(self.bn4(res3)))
            out = res2

            res1 = res1 if x2 is None else res1 + x2
            res2 = res2 if x3 is None else res2 + x3
            res3 = res3 + res1
            res4 = res4 + x1

        elif self.slink == 'H':
            # B的变体，但在向后累加时，丢掉本block的res1，只有res3.
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.conv2(F.relu(self.bn2(res1)))
            res3 = self.deconv3(F.relu(self.bn3(res2)))
            res4 = self.deconv4(F.relu(self.bn4(res3)))
            out = res2

            # res1 = res1 if x2 is None else res1 + x2
            res2 = res2 if x3 is None else res2 + x3
            res3 = res3 if x3 is None else res3 + x2
            res4 = res4 + x1

        elif self.slink == 'N':
            # No Shortcuts Used
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.conv2(F.relu(self.bn2(res1)))
            res3 = self.deconv3(F.relu(self.bn3(res2)))
            res4 = self.deconv4(F.relu(self.bn4(res3)))
            out = res2

        else:
            raise NotImplementedError('Unknown Slink for DoubleCouple : %s ' % self.slink)

        if self.classify > 0 and self.active_fc:
            out = self.classifier(out)
        else:
            out = None
        pred.append(out)

        if self.after:
            if self.down:
                res4 = self.down_res4(res4)
                res3 = self.down_res3(res3)
                res2 = self.down_res2(res2)
                return res4, res3, res2, pred
            else:
                return res4, res3, res2, pred
        else:
            if self.last_branch == 1:
                if self.last_down:
                    res4 = self.down_last4(res4)
                return res4, pred
            elif self.last_branch == 2:
                if self.last_down:
                    res4 = self.down_last4(res4)
                    res3 = self.down_last3(res3)
                return res4, res3, pred
            elif self.last_branch == 3:
                if self.last_down:
                    res4 = self.down_last4(res4)
                    res3 = self.down_last3(res3)
                    res2 = self.down_last2(res2)
                return res4, res3, res2, pred
            else:
                raise ValueError('<last_branch> of DoubleCouple should be <= 3!')


class SingleCouple(nn.Module):
    exp3 = 2  # must==2, double the channel for the next down-size layer
    down_func = {'A': DownSampleA, 'B': DownSampleB, 'C': DownSampleC,
                 'D': DownSampleD, 'E': DownSampleE, 'H': DownSampelH}

    def __init__(self, depth, growth, slink='A', after=True, down=False, dfunc='A',
                 classify=0, nclass=1000, last_branch=1, last_down=False):
        super(SingleCouple, self).__init__()
        assert last_branch <= 2, '<last_branch> of SingleCouple should be <= 2'
        assert len(growth) == 2, 'len of <growth> of SingleCouple should be 2'
        self.depth = depth
        self.growth = growth
        growtha, growthb = growth
        self.slink = slink
        self.after = after  # dose still have another DoubleCouple behind of this DoubleCouple. ?
        self.down = down    # dose connect to a same-size DoubleCouple or down-sized DoubleCouple. ?
        self.dfunc = dfunc
        self.down_func = self.down_func[dfunc]
        self.last_branch = last_branch  # how many output-ways (branches) for the classifier ?
        self.last_down = last_down      # dose down size for the classifier?
        self.classify = classify
        self.nclass = nclass
        self.active_fc = False

        self.bn1 = nn.BatchNorm2d(depth)
        self.conv1 = nn.Conv2d(depth, depth + growtha, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth + growtha)
        self.deconv2 = nn.ConvTranspose2d(depth + growtha, depth, 4, stride=2, padding=1, bias=False,
                                          output_padding=0)
        if self.classify > 0:
            self.classifier = XClassifier(depth + growtha, nclass)

        if self.after:
            if self.down:
                self.down_res2 = self.down_func(depth, depth * self.exp3)
                self.down_res1 = self.down_func(depth + growtha, depth * self.exp3 + growthb)
        else:
            if self.last_down:
                if self.last_branch >= 1:
                    self.down_last2 = self.down_func(depth, depth + growthb)
                if self.last_branch >= 2:
                    self.down_last1 = self.down_func(depth + growtha, depth + growthb)  # + growtha
            else:
                if self.classify > 0 and self.last_branch == 2:
                    # 此时，最后一个Couple的中间层被当做branch输出而对接在Summary上.
                    # 因此，删除此Couple自带的Classifier,以免与Summary中的Classifier重复.
                    delattr(self, 'classifier')
                    self.classify = 0
                    print('Note: 1 xfc will be deleted  because of duplicate with the last-fc!')

    def forward(self, x):
        # x3/res3 will be not used, but output 0 for parameters match between 2 layers
        if isinstance(x, tuple):
            x1, x2, x3, pred = x
            # print('x1, x2, x3: ', x1.size(), x2.size(), x3.size(), pred)
        else:
            x1, x2, x3, pred = x, None, None, None

        # add-style for use
        if self.slink == 'A':
            # 共包含1个夸Block连接，1个Block内连接.
            # first shorcut, then add,
            res1 = self.conv1(F.relu(self.bn1(x1)))
            out = res1 if x2 is None else res1 + x2
            res2 = self.deconv2(F.relu(self.bn2(out)))
            res2 = res2 if x1 is None else res2 + x1
            res3 = torch.Tensor(0).type_as(x3)

        elif self.slink == 'X':
            # 共包含1个夸Block连接，1个Block内连接
            # first add, then shorcut
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res1 = res1 if x2 is None else res1 + x2
            res2 = self.deconv2(F.relu(self.bn2(res1)))
            res2 = res2 if x1 is None else res2 + x1
            # res3 = torch.zeros_like(x3, dtype=x3.dtype, device=x3.device)
            res3 = torch.Tensor(0).type_as(x3)
            out = res1

        elif self.slink == 'B':
            # A的简化版，只有1个block内连接
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.deconv2(F.relu(self.bn2(res1)))
            res2 = res2 if x1 is None else res2 + x1
            res3 = torch.Tensor(0).type_as(x3)
            out = res1

        elif self.slink == 'C':
            # A的简化版，只有1个跨block连接
            res1 = self.conv1(F.relu(self.bn1(x1)))
            out = res1 if x2 is None else res1 + x2
            res2 = self.deconv2(F.relu(self.bn2(out)))
            res3 = torch.Tensor(0).type_as(x3)

        elif self.slink == 'D':
            # 夸Block的链接全部累加到最后一个Block内生效
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.deconv2(F.relu(self.bn2(res1)))
            res1 = res1 if x2 is None else res1 + x2
            res2 = res2 if x1 is None else res2 + x1
            res3 = torch.Tensor(0).type_as(x3)
            out = res1

        # cat-style no use
        elif self.slink == 'E':
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.deconv2(F.relu(self.bn2(res1)))
            res1 = res1 if x2 is None else torch.cat((res1, x2), 1)
            res2 = res2 if x1 is None else torch.cat((res2, x1), 1)
            res3 = torch.Tensor(0).type_as(x3)
            out = res1

        elif self.slink == 'N':
            # No Shortcuts Used
            res1 = self.conv1(F.relu(self.bn1(x1)))
            res2 = self.deconv2(F.relu(self.bn2(res1)))
            res3 = torch.Tensor(0).type_as(x3)
            out = res1

        else:
            raise NotImplementedError('Unknown Slink for SingleCouple : %s ' % self.slink)

        if self.classify > 0 and self.active_fc:
            out = self.classifier(out)
        else:
            out = None
        pred.append(out)

        if self.after:
            if self.down:
                res2 = self.down_res2(res2)
                res1 = self.down_res1(res1)
                return res2, res1, res3, pred   # todo ?
            else:
                return res2, res1, res3, pred
        else:
            if self.last_branch == 1:
                if self.last_down:
                    res2 = self.down_last2(res2)
                return res2, pred
            elif self.last_branch == 2:
                if self.last_down:
                    res2 = self.down_last2(res2)
                    res1 = self.down_last1(res1)
                return res2, res1, pred
            else:
                raise ValueError('<last_branch> of SingleCouple should be <= 2!')


class RockSummary(nn.Module):
    METHOD = ['split', 'merge']

    def __init__(self, indepth, branch, active='relu', pool='avg', nclass=1000, method='split'):
        super(RockSummary, self).__init__()
        assert len(indepth) == branch, '各输出分支的通道数必须全部给定，len of <indepth> == branch.'
        assert method in self.METHOD, 'Unknown <methond> %s.' % method
        self.indepth = indepth
        self.branch = branch
        self.active_fc = True
        self.nclass = nclass
        self.method = method
        if method == 'split':
            for b in range(1, branch + 1):
                layer = nn.Sequential(
                    nn.BatchNorm2d(indepth[b-1]),
                    Activate(active),
                    AdaAvgPool(),
                    ViewLayer(),
                    nn.Linear(indepth[b-1], nclass))
                setattr(self, 'classifier%s' % b, layer)
        elif method == 'merge':
            for b in range(1, branch + 1):
                layer = nn.Sequential(
                    nn.BatchNorm2d(indepth[b - 1]),
                    Activate(active),
                    AdaAvgPool(),
                    ViewLayer())
                setattr(self, 'pool_view%s' % b, layer)
            self.classifier = nn.Linear(sum(indepth), nclass)
        else:
            raise NotImplementedError

    def forward(self, x):
        # x1, x2, x3 extracted form x is big, media, small respectively.
        # 为确保fc(xi)的顺序与layer_i在model内的顺序相一致和相对应，
        # so the output order should be [fc(x3), fc(x2), fc(x1)] or [fc([x3, x2, x1])]
        if not self.active_fc:
            return x
        assert isinstance(x, (tuple, list)), 'x must be tuple, but %s' % type(x)
        assert len(x) == self.branch + 1, 'pred should be input together with x'
        x, pred = x[:-1][::-1], x[-1]
        # utils.print_size(x)
        if self.method == 'split':
            for i, xi in enumerate(x):
                xi = getattr(self, 'classifier%s' % (len(x)-i))(xi)
                pred.append(xi)
        elif self.method == 'merge':
            x = [getattr(self, 'pool_view%s' % (len(x)-i))(xi) for i, xi in enumerate(x)]
            x = torch.cat(x, dim=1)
            x = self.classifier(x)
            pred.append(x)
        return pred


class AutoBoost(nn.Module):
    # (xfc+lfc) x batchSize x nclass -> nfc x batchSize x nclass
    # bsize x nfc x nclass -> bsize x 1 x nclass -> bsize x nclass
    # '+': train xfc + boost together
    # '-': train first only xfc, then only boost, then tuning together
    # '*': no train only val, train only xfc, then 根据xfc的输出进行投票 boost
    METHOD = ('none', 'conv', 'soft', 'hard')
    # none-voting, conv-voting, soft-voting  hard-voting

    def __init__(self, xfc, lfc, ksize=1, nclass=1000, method='none'):
        super(AutoBoost, self).__init__()
        assert method in self.METHOD, 'Unknown Param <method> %s' % method
        self.xfc = xfc
        self.lfc = lfc
        self.ksize = ksize
        self.nclass = nclass
        self.method = method
        self.nfc = xfc + lfc
        self.active_fc = False
        if self.method == 'none':
            # 不做处理，直接返回
            pass
        elif self.method == 'conv':
            # 线性加权，Σnfc = fc
            self.conv = nn.Conv1d(self.nfc, 1, ksize, stride=1, padding=0, bias=False)
        elif self.method == 'soft':
            # 所有xfc按均值投票
            pass
        elif self.method == 'hard':
            # 所有xfc票多者胜
            pass

    def forward(self, x):
        if not self.active_fc:
            return x
        assert isinstance(x, (list, tuple))
        pred = []
        if self.method == 'none':
            return x
        elif self.method == 'conv':
            x = [xi.view(xi.size(0), 1, xi.size(1)) for xi in x]
            x = torch.cat(x, dim=1)
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            pred.append(x)
        elif self.method == 'soft':
            x = sum(x) / len(x)
            pred.append(x)
        elif self.method == 'hard':
            raise NotImplementedError('Not Implemented !')
        else:
            raise NotImplementedError('Unknown Param <method> : %s' % self.method)
        return pred


class WaveResNet(nn.Module):
    couple = {'D': DoubleCouple, 'S': SingleCouple}
    rocker = {'M': RockBlockM, 'U': RockBlockU, 'V': RockBlockV, 'R': RockBlockR}

    def __init__(self, branch=3, rock='U', depth=64, stages=4,
                 layers=(2, 2, 2, 2),
                 blocks=('D', 'D', 'D', 'D'),
                 slink=('A', 'A', 'A', 'A'),
                 expand=(1, 2, 4, 8),
                 growth=(10, 10, 15, 15),
                 dfunc=('D', 'D', 'D', 'D'),
                 classify=(1, 1, 1, 1),
                 fcboost='none',
                 last_branch=1,
                 last_down=True,
                 last_expand=30,
                 kldloss=False,
                 summer='split',
                 nclass=1000):
        super(WaveResNet, self).__init__()
        assert stages <= min(len(blocks), len(slink), len(expand),
                             len(layers), len(dfunc), len(classify)), \
            'Hyper Pameters Not Enough to Match Stages Nums:%s!' % stages
        assert sorted(blocks[:stages]) == list(blocks[:stages]), \
            'DoubleCouple must be ahead of SingleCouple! %s' % blocks[:stages]
        assert stages == len(growth), 'len of <growth> must be == <stage>'
        dataset = ['imagenet', 'cifar'][nclass != 1000]
        if dataset == 'cifar':
            assert stages <= 4, 'cifar stages should <= 4'
        elif dataset == 'imagenet':
            assert stages <= 5, 'imagenet stages should <= 5'

        self.branch = branch
        self.rock = self.rocker[rock]
        self.depth = depth
        self.stages = stages
        self.layers = layers
        self.blocks = blocks  # [self.couple[b] for b in blocks]
        self.slink = slink
        self.expand = expand
        growth = list(growth)
        growth.append(last_expand)
        self.growth = growth
        self.dfunc = dfunc
        self.classify = classify
        self.fcboost = fcboost
        self.last_branch = last_branch
        self.last_down = last_down
        self.last_expand = last_expand
        self.kldloss = kldloss
        self.summer = summer
        self.nclass = nclass

        self.after = [True for _ in range(stages - 1)] + [False]
        self.stage0 = self.rock(3, depth, branch, expand=(growth[0], growth[0]), dataset=dataset)
        for i in range(stages):
            layer = self._make_aelayer(self.couple[blocks[i]], layers[i], depth * expand[i],
                                       (growth[i], growth[i + 1]), slink[i], self.after[i],
                                       dfunc[i], classify[i], last_branch, last_down)
            setattr(self, 'stage%s' % (i + 1), layer)
        if last_down:
            fc_indepth = [last_expand] * last_branch
        else:
            if last_branch == 1: fc_indepth = [0]
            elif last_branch == 2: fc_indepth = [0, growth[stages-1]]
            elif last_branch == 3: fc_indepth = [0, growth[stages-1], 2 * growth[stages-1]]
            else: raise NotImplementedError
        fc_indepth = np.array([depth * expand[stages - 1]] * last_branch) + np.array(fc_indepth)
        fc_indepth = fc_indepth.tolist()
        self.summary = RockSummary(fc_indepth, last_branch, active='relu', nclass=nclass, method=summer)
        xfc_nums = sum([1 for n, m in self.named_modules()
                        if isinstance(m, (DoubleCouple, SingleCouple))
                        and hasattr(m, 'classifier')])
        lfc_nums = last_branch if summer == 'split' else 1
        self.boost = AutoBoost(xfc=xfc_nums, lfc=lfc_nums, ksize=1, nclass=nclass, method=fcboost)
        if kldloss:  self.kld_criterion = nn.KLDivLoss()

        self.train_which_now = {'conv+rock': False, 'xfc+boost': False,
                                'xfc-only': False, 'boost-only': False}
        self.eval_which_now = {'conv+rock': False, 'conv+rock+xfc': False,
                               'conv+rock+boost': False, 'conv+rock+xfc+boost': False}
        self._init_params()

    def _make_aelayer(self, block, block_nums, indepth, growth, slink, after,
                      dfunc, cfy, last_branch, last_down):
        # last_branch & last_down & last_expand work only when after=False;
        # if after=True, they don't work and their values have no influence.
        layers = []
        for i in range(block_nums - 1):
            layers.append(
                block(depth=indepth, growth=growth, slink=slink, after=True,
                      down=False, dfunc=dfunc, classify=cfy, nclass=self.nclass,
                      last_branch=last_branch, last_down=False))
        layers.append(
            block(depth=indepth, growth=growth, slink=slink, after=after,
                  down=True, dfunc=dfunc, classify=cfy, nclass=self.nclass,
                  last_branch=last_branch, last_down=last_down))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            # elif isinstance(m, nn.Conv1d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.stage0(x)
        for s in range(self.stages):
            x = getattr(self, 'stage%s' % (s + 1))(x)
        x = self.summary(x)  # x <=> pred
        x = [p for p in x if p is not None]
        x = self.boost(x)
        return x

    def forward2(self, x):
        # deperecated: the same logical to self.forward
        x = self.stage0(x)
        if self.stages >= 1:
            x = self.stage1(x)
        if self.stages >= 2:
            x = self.stage2(x)
        if self.stages >= 3:
            x = self.stage3(x)
        if self.stages >= 4:
            x = self.stage4(x)
        if self.stages >= 5:
            x = self.stage5(x)
        x = self.summary(x)
        x = [p for p in x if p is not None]
        x = self.boost(x)
        return x

    def kld_loss(self, pred, by='last', issum=True):
        if by == 'last':
            refer = pred[-1]
        elif by == 'mean':
            mean = pred[0]
            for p in pred[1:]:
                mean += p
            refer = mean / len(pred)
        else:
            raise NotImplementedError
        kld_loss = [self.kld_criterion(p, refer) for p in pred]
        if issum:
            kld_loss = sum(kld_loss)
        return kld_loss

    def train_which(self, part='conv+rock'):
        # if self.train_which_now[part]:
        #     return
        if part == 'conv+rock':
            self.train()
            for name, module in self.named_modules():
                if isinstance(module, (DoubleCouple, SingleCouple)):
                    module.active_fc = False
                    fclayer = getattr(module, 'classifier', nn.ReLU())
                    fclayer.eval()
                    for p in fclayer.parameters():
                        p.requires_grad = False
                if isinstance(module, AutoBoost):
                    module.active_fc = False
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
        elif part == 'xfc+boost':
            for name, module in self.named_modules():
                if isinstance(module, (DoubleCouple, SingleCouple)):
                    module.active_fc = True
                    for n, m in module.named_modules():
                        if 'classifier' in n:
                            m.train()
                        else:
                            m.eval()
                    for n, p in module.named_parameters():
                        if 'classifier' in n:
                            p.requires_grad = True
                        else:
                            p.requires_grad = False
                if isinstance(module, AutoBoost):
                    module.train()
                    module.active_fc = True
                    for p in module.parameters():
                        p.requires_grad = True
                if isinstance(module, (RockBlock, RockSummary)):
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
        elif part == 'xfc-only':
            for name, module in self.named_modules():
                if isinstance(module, (DoubleCouple, SingleCouple)):
                    module.active_fc = True
                    for n, m in module.named_modules():
                        if 'classifier' in n:
                            m.train()
                        else:
                            m.eval()
                    for n, p in module.named_parameters():
                        if 'classifier' in n:
                            p.requires_grad = True
                        else:
                            p.requires_grad = False
                if isinstance(module, AutoBoost):
                    module.eval()
                    module.active_fc = False
                    for p in module.parameters():
                        p.requires_grad = False
                if isinstance(module, (RockBlock, RockSummary)):
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
        elif part == 'boost-only':
            for name, module in self.named_modules():
                if isinstance(module, (DoubleCouple, SingleCouple)):
                    module.active_fc = True
                    module.eval()
                    for n, p in module.named_parameters():
                        p.requires_grad = False
                if isinstance(module, AutoBoost):
                    module.train()
                    module.active_fc = True
                    for p in module.parameters():
                        p.requires_grad = True
                if isinstance(module, (RockBlock, RockSummary)):
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
        else:
            raise NotImplementedError('Unknown Param <part> : %s' % part)
        for key in self.train_which_now.keys():
            self.train_which_now[key] = True if key == part else False
        print('model.train_which_now', self.train_which_now)

    def eval_which(self, part='conv+rock'):
        # if self.eval_which_now[part]:
        #     return
        if part == 'conv+rock':
            self.eval()
            for name, module in self.named_modules():
                if isinstance(module, (DoubleCouple, SingleCouple)):
                    module.active_fc = False
                if isinstance(module, AutoBoost):
                    module.active_fc = False
        elif part == 'conv+rock+xfc':
            self.eval()
            for name, module in self.named_modules():
                if isinstance(module, (DoubleCouple, SingleCouple)):
                    module.active_fc = True
                if isinstance(module, AutoBoost):
                    module.active_fc = False
        elif part == 'conv+rock+boost':
            assert sum(self.classify) == 0, '此情况下xfc数量必须为零，否则<xfc_nums>无法唯一确定！'
            self.eval()
            for name, module in self.named_modules():
                if isinstance(module, (DoubleCouple, SingleCouple)):
                    module.active_fc = False
                if isinstance(module, AutoBoost):
                    module.active_fc = True
        elif part == 'conv+rock+xfc+boost':
            self.eval()
            for name, module in self.named_modules():
                if isinstance(module, (DoubleCouple, SingleCouple)):
                    module.active_fc = True
                if isinstance(module, AutoBoost):
                    module.active_fc = True
        for key in self.eval_which_now.keys():
            self.eval_which_now[key] = True if key == part else False
        print('model.eval_which_now', self.eval_which_now)


if __name__ == '__main__':
    import xtils

    torch.manual_seed(1)

    # # imageNet
    # exp5 = {'stages': 5, 'branch': 3, 'rock': 'R', 'depth': 16, 'kldloss': False,
    #         'layers': (6, 5, 4, 3, 2), 'blocks': ('D', 'D', 'D', 'D', 'S'), 'slink': ('A', 'A', 'A', 'A', 'A'),
    #         'expand': (1, 2, 4, 8, 16), 'growth': (12, 12, 12, 12, 12), 'dfunc': ('D', 'D', 'D', 'D', 'D'),
    #         'classify': (1, 1, 1, 1, 1), 'fcboost': 'none', 'nclass': 1000, 'summer': 'merge',
    #         'last_branch': 2, 'last_down': False, 'last_expand': 11}
    #
    # exp4 = {'stages': 4, 'branch': 3, 'rock': 'R', 'depth': 16, 'kldloss': False,
    #         'layers': (6, 5, 4, 3), 'blocks': ('D', 'D', 'D', 'S'), 'slink': ('A', 'A', 'A', 'A'),
    #         'expand': (1, 2, 4, 8), 'growth': (10, 15, 20, 30), 'dfunc': ('D', 'D', 'D', 'D'),
    #         'classify': (0, 1, 1, 1), 'fcboost': 'conv', 'nclass': 1000, 'summer': 'merge',
    #         'last_branch': 2, 'last_down': False, 'last_expand': 15}
    #
    # exp3 = {'stages': 3, 'branch': 3, 'rock': 'V', 'depth': 64, 'kldloss': False,
    #         'layers': (3, 3, 1), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
    #         'expand': (1, 2, 4), 'growth': (5, 5, 5), 'dfunc': ('D', 'D', 'D'),
    #         'classify': (0, 0, 0), 'fcboost': 'none', 'nclass': 1000, 'summer': 'merge',
    #         'last_branch': 1, 'last_down': True, 'last_expand': 256}
    #
    # model = WaveResNet(**exp3)
    # print('\n', model, '\n')
    #
    # # train_which & eval_which 在组合上必须相互匹配
    # # model.train_which(part=['conv+rock', 'xfc+boost', 'xfc-only', 'boost-only'][1])
    # model.eval_which(part=['conv+rock', 'conv+rock+xfc', 'conv+rock+boost', 'conv+rock+xfc+boost'][1])
    # # print(model.stage1[1].conv1.training)
    # # print(model.stage1[1].classifier.training)
    # # print(model.stage2[0].classifier.training)
    # # print(model.summary.classifier1.training)
    #
    # x = torch.randn(4, 3, 256, 256)
    # # utils.tensorboard_add_model(model, x)
    # utils.calculate_params_scale(model, format='million')
    # utils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
    # y = model(x)
    # print('有效分类支路：', len(y), '\t共有blocks：', sum(model.layers))
    # print(':', [yy.shape for yy in y if yy is not None])
    # print(':', [yy.max(1) for yy in y if yy is not None])

    # print('\n xxxxxxxxxxxxxxxxx \n')
    # for n, m in model.named_modules(prefix='stage'):
    #     print(n,'-->', m)
    #     if hasattr(m, 'active_fc'):
    #         print('OK-->', n, m)
    #     if isinstance(m, (DoubleCouple, SingleCouple)):
    #         m.active_fc = False

    # cifar10
    exp4 = {'stages': 4, 'branch': 3, 'rock': 'U', 'depth': 16, 'kldloss': False,
            'layers': (6, 5, 4, 3), 'blocks': ('D', 'D', 'D', 'S'), 'slink': ('A', 'A', 'A', 'A'),
            'expand': (1, 2, 4, 8), 'growth': (10, 15, 20, 30), 'dfunc': ('D', 'D', 'D', 'D'),
            'classify': (1, 1, 1, 1), 'fcboost': 'none', 'nclass': 10, 'summer': 'merge',
            'last_branch': 1, 'last_down': True, 'last_expand': 5}
    exp3 = {'stages': 3, 'branch': 3, 'rock': 'R', 'depth': 16, 'kldloss': False,
            'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
            'expand': (1, 2, 4), 'growth': (5, 3, 7), 'dfunc': ('D', 'D', 'D'),
            'classify': (1, 1, 1), 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
            'last_branch': 2, 'last_down': False, 'last_expand': 30}
    exp2 = {'stages': 2, 'branch': 3, 'rock': 'R', 'depth': 5, 'kldloss': False,
            'layers': (2, 2), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
            'expand': (1, 2), 'growth': (4, 4), 'dfunc': ('D', 'D'),
            'classify': (1, 1), 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
            'last_branch': 2, 'last_down': True, 'last_expand': 30}
    exp1 = {'stages': 1, 'branch': 3, 'rock': 'U', 'depth': 32, 'kldloss': False,
            'layers': (10, 2), 'blocks': ('D', 'S'), 'slink': ('A', 'A'),
            'expand': (1, 2), 'growth': (4,), 'dfunc': ('D', 'D'),
            'classify': (0, 0), 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
            'last_branch': 1, 'last_down': True, 'last_expand': 32}

    model = WaveResNet(**exp2)
    print('\n', model, '\n')
    # model.train_which(part=['conv+rock', 'xfc+boost', 'xfc-only', 'boost-only'][1])
    model.eval_which(part=['conv+rock', 'conv+rock+xfc', 'conv+rock+boost', 'conv+rock+xfc+boost'][1])
    x = torch.randn(4, 3, 32, 32)
    # utils.tensorboard_add_model(model, x)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
    y = model(x)
    print('有效分类支路：', len(y), '\t共有blocks：', sum(model.layers))
    print(len(y), sum(model.layers), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])
