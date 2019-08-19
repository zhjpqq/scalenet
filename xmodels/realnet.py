# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F
from xmodules.classifier import ViewLayer, AdaAvgPool, Activate
from xmodules.transition import TransitionA
from xmodules.rockblock import RockBlock, RockBlockQ, RockBlockX, RockBlockO
polate = F.interpolate


"""
  RealNet => + DoubleCouple + SingleCouple + Summary(+merge +split) + BottleNeck
"""


class DoubleCouple(nn.Module):
    _trans = {'A': TransitionA}

    def __init__(self, indepth, growth, active='relu', first=False, after=True, down=False,
                 trans='A', reduce=0.5, convkp='T', inmode='nearest', classify=0, nclass=1000,
                 last_branch=1, last_down=False, last_expand=0):
        super(DoubleCouple, self).__init__()
        assert last_branch <= 4, '<last_branch> of DoubleCouple should be <= 4...'
        self.indepth = indepth
        self.growth = growth
        self.classify = classify
        self.nclass = nclass
        self.active = getattr(nn.functional, active)
        self.first = first
        self.after = after
        self.down = down
        self.trans = trans
        self.reduce = reduce
        self.trans_func = self._trans[trans]
        self.last_branch = last_branch
        self.last_down = last_down
        self.last_expand = last_expand
        self.inmode = inmode
        self.convkp = convkp
        kp = {'T': (3, 1), 'O': (1, 0)}[convkp]

        first_outdepth = indepth + growth if self.first else growth
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, first_outdepth, 3, stride=2, padding=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(indepth + growth)
        self.conv2 = nn.Conv2d(indepth + growth, growth, 3, stride=2, padding=1, bias=False, dilation=1)
        self.bn3 = nn.BatchNorm2d(indepth + growth)
        self.conv3 = nn.Conv2d(indepth + growth, growth, kp[0], stride=1, padding=kp[1], bias=False, dilation=1)
        self.bn4 = nn.BatchNorm2d(indepth + growth)
        self.conv4 = nn.Conv2d(indepth + growth, growth, kp[0], stride=1, padding=kp[1], bias=False, dilation=1)

        if self.classify > 0:
            self.classifier = nn.Sequential(
                nn.BatchNorm2d(indepth + growth),
                Activate(active),
                AdaAvgPool(),
                ViewLayer(dim=-1),
                nn.Linear(indepth + growth, nclass)
            )

        if self.after:
            if self.down:
                outdepth = int(math.floor((indepth + growth) * reduce))
                self.down_res4 = self.trans_func(indepth + growth, outdepth)
                self.down_res3 = self.trans_func(indepth + growth, outdepth)
                self.down_res2 = self.trans_func(indepth + growth, outdepth)
                self.down_res1 = self.trans_func(indepth + growth, outdepth)
        else:
            if self.last_down:
                outdepth = indepth + growth + last_expand
                if self.last_branch >= 1:
                    self.down_last4 = self.trans_func(indepth + growth, outdepth)
                if self.last_branch >= 2:
                    self.down_last3 = self.trans_func(indepth + growth, outdepth)
                if self.last_branch >= 3:
                    self.down_last2 = self.trans_func(indepth + growth, outdepth)
                if self.last_branch >= 4:
                    self.down_last1 = self.trans_func(indepth + growth, outdepth)
            else:
                if self.classify > 0 and self.last_branch == 4:
                    # 最后一个Couple的中间层被当做branch输出而对接在Summary上.
                    # 因此，删除此Couple自带的Classifier,以免与Summary中的Classifier重复.
                    delattr(self, 'classifier')
                    self.classify = 0
                    print('Note: 1 xfc will be deleted because of duplicated with the lfc!')

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x1, x2, x3, x4, pred = x  # 大 中 小 中
        else:
            x1, x2, x3, x4, pred = x, None, None, None, None
        res1 = self.conv1(self.active(self.bn1(x1)))
        res1 = torch.cat((res1, x4), 1) if not self.first else res1
        res2 = self.conv2(self.active(self.bn2(res1)))
        res2 = torch.cat((res2, x3), 1)
        out = res2
        res3 = self.conv3(self.active(self.bn3(polate(res2, scale_factor=2, mode=self.inmode))))
        res3 = torch.cat((res3, x2), 1)
        res4 = self.conv4(self.active(self.bn4(polate(res3, scale_factor=2, mode=self.inmode))))
        res4 = torch.cat((res4, x1), 1)

        if self.classify > 0:
            out = self.classifier(out)
        else:
            out = None
        pred.append(out)

        if self.after:
            if self.down:
                res4 = self.down_res4(res4)
                res3 = self.down_res3(res3)
                res2 = self.down_res2(res2)
                res1 = self.down_res1(res1)
            return res4, res3, res2, res1, pred
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
            elif self.last_branch == 4:
                if self.last_down:
                    res4 = self.down_last4(res4)
                    res3 = self.down_last3(res3)
                    res2 = self.down_last2(res2)
                    res1 = self.down_last1(res1)
                return res4, res3, res2, res1, pred
            else:
                raise ValueError('<last_branch> of DoubleCouple should be <= 3!')


class SingleCouple(nn.Module):
    _trans = {'A': TransitionA}

    def __init__(self, indepth, growth, active='relu', first=False, after=True, down=False,
                 trans='A', reduce=0.5, convkp='T', inmode='nearest', classify=0, nclass=1000,
                 last_branch=1, last_down=False, last_expand=0):
        super(SingleCouple, self).__init__()
        assert last_branch <= 2, '<last_branch> of SingleCouple should be <= 2...'
        self.indepth = indepth
        self.growth = growth
        self.active = getattr(nn.functional, active)
        self.first = first
        self.after = after
        self.down = down
        self.trans = trans
        self.reduce = reduce
        self.trans_func = self._trans[trans]
        self.classify = classify
        self.nclass = nclass
        self.last_branch = last_branch
        self.last_down = last_down
        self.last_expand = last_expand
        self.inmode = inmode
        self.convkp = convkp
        kp = {'T': (3, 1), 'O': (1, 0)}[convkp]

        first_outdepth = indepth + growth if self.first else growth
        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, first_outdepth, 3, stride=2, padding=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(indepth + growth)
        self.conv2 = nn.Conv2d(indepth + growth, growth, kp[0], stride=1, padding=kp[1], bias=False, dilation=1)

        if self.classify > 0:
            self.classifier = nn.Sequential(
                nn.BatchNorm2d(indepth + growth),
                Activate(active),
                AdaAvgPool(),
                ViewLayer(dim=-1),
                nn.Linear(indepth + growth, nclass)
            )
        if self.after:
            if self.down:
                outdepth = int(math.floor((indepth + growth) * reduce))
                self.down_res2 = self.trans_func(indepth + growth, outdepth)
                self.down_res1 = self.trans_func(indepth + growth, outdepth)
        else:
            if self.last_down:
                outdepth = indepth + growth + last_expand
                if self.last_branch >= 1:
                    self.down_last2 = self.trans_func(indepth + growth, outdepth)
                if self.last_branch >= 2:
                    self.down_last1 = self.trans_func(indepth + growth, outdepth)
            else:
                if self.classify > 0 and self.last_branch == 2:
                    # 此时，最后一个Couple的中间层被当做branch输出而对接在Summary上.
                    # 因此，删除此Couple自带的Classifier,以免与Summary中的Classifier重复.
                    delattr(self, 'classifier')
                    self.classify = 0
                    print('Note: 1 xfc will be deleted  because of duplicate with the last-fc!')

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x1, x2, x3, x4, pred = x  # 大 中 小 中
        else:
            x1, x2, x3, x4, pred = x, None, None, None, None
        res1 = self.conv1(self.active(self.bn1(x1)))
        res1 = torch.cat((res1, x2), 1) if not self.first else res1
        out = res1
        res2 = self.conv2(self.active(self.bn2(polate(res1, scale_factor=2, mode=self.inmode))))
        res2 = torch.cat((res2, x1), 1)

        if self.classify > 0:
            out = self.classifier(out)
        else:
            out = None
        pred.append(out)

        if self.after:
            if self.down:
                res2 = self.down_res2(res2)
                res1 = self.down_res1(res1)
            return res2, res1, None, None, pred
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


class SummaryBlock(nn.Module):
    METHOD = ['split', 'merge']

    def __init__(self, indepth, branch=1, active='relu', nclass=1000, method='split'):
        super(SummaryBlock, self).__init__()
        assert len(indepth) == branch, '各分类分支的通道数必须全部给定，so， len of <indepth> == branch.'
        assert method in self.METHOD, 'Unknown <method> %s for SummaryBlock.' % method
        self.indepth = indepth
        self.branch = branch
        self.active = active
        self.nclass = nclass
        self.method = method
        self.active_fc = True

        if method == 'split':
            for i in range(branch):
                fc_layer = nn.Sequential(
                    nn.BatchNorm2d(indepth[i]),
                    Activate(active),
                    AdaAvgPool(),
                    ViewLayer(),
                    nn.Linear(indepth[i], nclass))
                setattr(self, 'classifier%s' % (i + 1), fc_layer)
        elif method == 'merge':
            for i in range(branch):
                view_layer = nn.Sequential(
                    nn.BatchNorm2d(indepth[i]),
                    Activate(active),
                    AdaAvgPool(),
                    ViewLayer())
                setattr(self, 'pool_view%s' % (i + 1), view_layer)
            self.classifier = nn.Linear(sum(indepth), nclass)
        else:
            raise NotImplementedError

    def forward(self, x):
        # x1, x2, x3, x4 extracted form x is "big, media, small, media" respectively.
        # 为确保fc(xi)的顺序与layer_i在model内的顺序相一致和相对应，
        # so the output order should be [fc(x4), fc(x3), fc(x2), fc(x1)] or [fc([x4, x3, x2, x1])]
        if not self.active_fc:
            return x
        assert isinstance(x, (tuple, list)), 'x must be tuple, but %s' % type(x)
        assert len(x) == self.branch + 1, 'pred should be input together with x'
        x, pred = x[:self.branch], x[-1]
        if self.method == 'split':
            xx = []
            for i, xi in enumerate(x):
                xi = getattr(self, 'classifier%s' % (i + 1))(xi)
                xx.append(xi)
            pred.extend(xx[::-1])
        elif self.method == 'merge':
            x = [getattr(self, 'pool_view%s' % (i + 1))(xi) for i, xi in enumerate(x)]
            x = torch.cat(x[::-1], dim=1)
            x = self.classifier(x)
            pred.append(x)
        return pred


class RealNet(nn.Module):
    _couple = {'D': DoubleCouple, 'S': SingleCouple}
    _rocker = {'X': RockBlockX, 'Q': RockBlockQ, 'O': RockBlockO}

    def __init__(self, stages=4, rock='O', branch=4, indepth=16, growth=12, multiway=4,
                 layers=(3, 3, 3, 3), blocks=('D', 'D', 'D', 'D'), classify=(0, 0, 0, 0),
                 trans=('A', 'A', 'A', 'A'), reduction=(0.5, 0.5, 0.5, 0.5), convkp=('T', 'T', 'T', 'O'),
                 last_branch=4, last_down=False, last_expand=0, poolmode='avg', active='relu',
                 summer='split', nclass=1000, inmode='nearest'):
        super(RealNet, self).__init__()
        assert stages <= min(len(layers), len(blocks), len(classify), len(trans),
                             len(reduction), len(convkp)), \
            'Hyper Pameter Not Enough to Match Stages Nums:%s!' % stages
        assert stages == sum([bool(l) for l in layers[:stages]]), \
            'Hyper Pameter <stages:%s> and <layers:%s> cannot match, ' \
            'number of no-zero value in <layers> should be == <stage> !' % (stages, layers)
        assert stages == sum([bool(r) for r in reduction[:stages - 1]]) + 1, \
            'Hyper Pameter <stages:%s> and <reduction:%s> cannot match, ' \
            'number of no-zero value in <reduction> should be == <stages-1>!' % (stages, reduction)
        assert sorted(blocks[:stages]) == list(blocks[:stages]), \
            'DoubleCouple must be ahead of SingleCouple! But your %s' % 'is ->'.join(blocks[:stages])
        assert (blocks[0] == 'D' and branch == 3) or (blocks[0] == 'S' and branch == 2), \
            'DoubleCouple need <branch>==3, SingleCouple need <branch>==2, ' \
            'but your %s <branch> is %s' % (blocks[0], branch)
        assert len(convkp) == sum([1 for k in convkp if k in ['T', 'O']]), \
            'Hyper Pameter <convkp:%s> only can contain T or O.' % (convkp,)
        assert multiway in [3, 4], '<multiway> of dense connections now only support [3 or 4]!'

        dataset = ['imagenet', 'cifar'][nclass != 1000]
        if dataset == 'cifar':
            assert stages <= 4, 'cifar stages should <= 4'
        elif dataset == 'imagenet':
            assert stages <= 5, 'imagenet stages should <= 5'

        self.stages = stages
        self.rock = self._rocker[rock]
        self.branch = branch
        self.indepth = indepth
        self.growth = growth
        self.multiway = multiway
        self.layers = layers
        self.classify = classify
        self.trans = trans
        self.reduction = reduction
        self.convkp = convkp
        self.poolmode = poolmode
        self.active = active
        self.last_branch = last_branch
        self.last_down = last_down
        self.last_expand = last_expand
        self.nclass = nclass
        self.inmode = inmode

        self.after = [True for _ in range(stages - 1)] + [False]
        self.layer0 = self.rock(indepth=3, outdepth=indepth, branch=branch, dataset=dataset)
        for i in range(stages):
            dense_stage = self._make_dense_stage(self._couple[blocks[i]], layers[i], indepth, growth,
                                                 classify[i], self.after[i], trans[i], reduction[i],
                                                 convkp[i], last_branch, last_down, last_expand, i)
            setattr(self, 'dense%s' % (i + 1), dense_stage)
            indepth += layers[i] * growth
            if i < stages - 1:
                indepth = int(math.floor(indepth * reduction[i]))
            elif i == stages - 1:
                indepth = indepth + last_expand if last_down else indepth  # + growth

        indepth = [indepth] * last_branch
        self.summary = SummaryBlock(indepth, last_branch, active, nclass=nclass, method=summer)

    def _make_dense_stage(self, block, nums, indepth, growth, cfy, after, trans, reduce, convkp,
                          last_branch, last_down, last_expand, stage):
        layers = []
        for i in range(nums - 1):
            if self.multiway == 3:
                first = True  # 固定所有block内第1个conv1层的outdepth
            elif self.multiway == 4:
                first = bool(stage == 0 and i == 0)  # 固定第1个block内的第1个conv1层的outdepth
            layers.append(block(indepth=indepth, growth=growth, active=self.active, first=first,
                                after=True, down=False, trans=trans, reduce=reduce, convkp=convkp,
                                classify=cfy, nclass=self.nclass, inmode=self.inmode,
                                last_branch=last_branch, last_down=last_down, last_expand=last_expand))
            indepth += growth

        if self.multiway == 3:
            first = True
        elif self.multiway == 4:
            if stage == 0 and nums == 1: first = True  # 第1个stage的第1个block
            if stage == 0 and nums > 1: first = False  # 第1个stage的第2+个block
            if stage > 0: first = False  # 第2+个stage内的blocks
        layers.append(block(indepth=indepth, growth=growth, active=self.active, first=first,
                            after=after, down=True, trans=trans, reduce=reduce, convkp=convkp,
                            classify=cfy, nclass=self.nclass, inmode=self.inmode,
                            last_branch=last_branch, last_down=last_down, last_expand=last_expand))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer0(x)
        for i in range(self.stages):
            x = getattr(self, 'dense%s' % (i + 1))(x)
            # utils.print_size(x, True)
        x = self.summary(x)  # x <=> [x, pred]
        x = [p for p in x if p is not None]
        return x

    def forward2(self, x):
        # deprecated !!!!
        ok = False
        x = self.layer0(x)
        xtils.print_size(x, ok)
        x = self.dense1(x)
        xtils.print_size(x, ok)
        x = self.trans1(x)
        xtils.print_size(x, ok)
        x = self.dense2(x)
        xtils.print_size(x, ok)
        x = self.trans2(x)
        xtils.print_size(x, ok)
        x = self.dense3(x)
        xtils.print_size(x, ok)
        x = self.trans3(x)
        xtils.print_size(x, ok)
        x = self.dense4(x)
        xtils.print_size(x, ok)
        if self.last_trans:
            x = self.trans4(x)
        xtils.print_size(x, ok)
        x = self.summary(x)
        x = [p for p in x if p is not None]
        return x


if __name__ == '__main__':
    import xtils

    torch.manual_seed(9528)

    # # imagenet
    # exp = {'stages': 4, 'rock': 'Q', 'branch': 3, 'indepth': 5, 'growth': 3, 'multiway': 4,
    #        'layers': (6, 5, 4, 3), 'blocks': ('D', 'D', 'D', 'S'), 'classify': (1, 1, 1, 1),
    #        'trans': ('A', 'A', 'A', 'A'), 'reduction': (0.5, 0.5, 0.5, 0.5), 'convkp': ('T', 'T', 'T', 'O'),
    #        'last_branch': 2, 'last_down': False, 'last_expand': 10, 'inmode': 'nearest',
    #        'poolmode': 'avg', 'active': 'relu', 'summer': 'merge', 'nclass': 1000}
    #
    # model = RealNet(**exp)
    # print(model)
    # x = torch.randn(4, 3, 256, 256)
    # # utils.tensorboard_add_model(model, x)
    # utils.calculate_params_scale(model, format='million')
    # utils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
    # y = model(x)
    # print('共有dense blocks数：', sum(model.layers), '最后实际输出分类头数：', len(y),)
    # print('每个输出预测的尺寸:', [(yy.shape,) for yy in y if yy is not None])
    # print('每个输出预测的得分:', [(yy.max(1),) for yy in y if yy is not None])

    # cifar
    exp2 = {'stages': 2, 'rock': 'O', 'branch': 3, 'indepth': 24, 'growth': 12, 'multiway': 4,
            'layers': (7, 1, 0), 'blocks': ('D', 'D', '-'), 'classify': (0, 0, 0),
            'trans': ('A', 'A', '-'), 'reduction': (0.5, 0, 0), 'convkp': ('T', 'T', 'T', 'O'),
            'last_branch': 1, 'last_down': True, 'last_expand': 0, 'inmode': 'nearest',
            'poolmode': 'avg', 'active': 'relu', 'summer': 'split', 'nclass': 10}

    exp3 = {'stages': 3, 'rock': 'O', 'branch': 3, 'indepth': 24, 'growth': 12, 'multiway': 4,
            'layers': (4, 3, 2), 'blocks': ('D', 'D', 'D'), 'classify': (1, 1, 1),
            'trans': ('A', 'A', 'A'), 'reduction': (0.5, 0.5, 0), 'convkp': ('T', 'T', 'T', 'T'),
            'last_branch': 1, 'last_down': False, 'last_expand': 10, 'inmode': 'nearest',
            'poolmode': 'avg', 'active': 'relu', 'summer': 'split', 'nclass': 10}

    model = RealNet(**exp3)
    print(model)
    x = torch.randn(4, 3, 32, 32)
    # utils.tensorboard_add_model(model, x)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
    y = model(x)
    print('共有dense blocks数：', sum(model.layers), '最后实际输出分类头数：', len(y), )
    print('每个输出预测的尺寸:', [(yy.shape,) for yy in y if yy is not None])
    print('每个输出预测的得分:', [(yy.max(1),) for yy in y if yy is not None])

