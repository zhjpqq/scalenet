# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F
from xmodules.classifier import ViewLayer, AdaAvgPool, Activate
from xmodules.transition import TransitionB, TransitionC
from xmodules.preprocess import PreProcess

"""
  Namei => PreProcess + PyramidBlock + DoubleCouple + SingleCouple 
           + Summary(+merge +split) + BottleNeck(ok!) + Plain(ok!)
           
  可为每个stage独立指定工作模式（Bottlneck or Plain）, 以及BottlNeck值(bottle*growth)
"""

inplace = [False, True][1]


class DoubleCouple(nn.Module):
    _trans = {'B': TransitionB, 'C': TransitionC}

    def __init__(self, indepth, growth, bottle=4, active='relu', after=True, down=False, trans='B', pool='avg',
                 reduce=0.5, classify=0, nclass=1000, last_branch=1, last_down=False, last_expand=0):
        super(DoubleCouple, self).__init__()
        assert last_branch <= 4, '<last_branch> of DoubleCouple should be <= 4...'
        self.indepth = indepth
        self.growth = growth
        self.bottle = bottle
        self.classify = classify
        self.nclass = nclass
        self.active = getattr(nn.functional, active)
        self.after = after
        self.down = down
        self.trans = trans
        self.pool = pool
        self.reduce = reduce
        self.trans_func = self._trans[trans]
        self.last_branch = last_branch
        self.last_down = last_down
        self.last_expand = last_expand

        if bottle > 0:
            interdepth = int(bottle * growth)
            assert interdepth == math.ceil(bottle * growth), \
                '<bottle> * <growth> cannot be a fraction number (带小数位), but %s !\n' % bottle * growth

            self.bn1_b = nn.BatchNorm2d(indepth)
            self.conv1_b = nn.Conv2d(indepth, interdepth, 1, stride=1, padding=0, dilation=1, bias=False)
            self.bn1 = nn.BatchNorm2d(interdepth)
            self.conv1 = nn.Conv2d(interdepth, growth, 3, stride=2, padding=1, bias=False, dilation=1)

            self.bn2_b = nn.BatchNorm2d(indepth + growth)
            self.conv2_b = nn.Conv2d(indepth + growth, interdepth, 1, stride=1, padding=0, dilation=1, bias=False)
            self.bn2 = nn.BatchNorm2d(interdepth)
            self.conv2 = nn.Conv2d(interdepth, growth, 3, stride=2, padding=1, bias=False, dilation=1)

            self.bn3_b = nn.BatchNorm2d(indepth + growth)
            self.conv3_b = nn.Conv2d(indepth + growth, interdepth, 1, stride=1, padding=0, dilation=1, bias=False)
            self.bn3 = nn.BatchNorm2d(interdepth)
            self.deconv3 = nn.ConvTranspose2d(interdepth, growth, 3, stride=2, padding=1,
                                              output_padding=1, bias=False, dilation=1)

            self.bn4_b = nn.BatchNorm2d(indepth + growth)
            self.conv4_b = nn.Conv2d(indepth + growth, interdepth, 1, stride=1, padding=0, dilation=1, bias=False)
            self.bn4 = nn.BatchNorm2d(interdepth)
            self.deconv4 = nn.ConvTranspose2d(interdepth, growth, 3, stride=2, padding=1,
                                              output_padding=1, bias=False, dilation=1)

        elif bottle == 0:

            self.bn1 = nn.BatchNorm2d(indepth)
            self.conv1 = nn.Conv2d(indepth, growth, 3, stride=2, padding=1, bias=False, dilation=1)
            self.bn2 = nn.BatchNorm2d(indepth + growth)
            self.conv2 = nn.Conv2d(indepth + growth, growth, 3, stride=2, padding=1, bias=False, dilation=1)
            self.bn3 = nn.BatchNorm2d(indepth + growth)
            self.deconv3 = nn.ConvTranspose2d(indepth + growth, growth, 3, stride=2, padding=1,
                                              output_padding=1, bias=False, dilation=1)
            self.bn4 = nn.BatchNorm2d(indepth + growth)
            self.deconv4 = nn.ConvTranspose2d(indepth + growth, growth, 3, stride=2, padding=1,
                                              output_padding=1, bias=False, dilation=1)

        else:
            raise NotImplementedError('<bottle> should be >= 0, but %s' % bottle)

        if self.classify > 0:
            self.classifier = nn.Sequential(
                nn.BatchNorm2d(indepth + growth),
                Activate(active),
                AdaAvgPool(),
                ViewLayer(dim=-1),
                nn.Linear(indepth + growth, nclass))

        if self.after:
            if self.down and self.reduce != 1:
                outdepth = int(math.floor((indepth + growth) * reduce))
                self.down_res4 = self.trans_func(indepth + growth, outdepth, pool=pool)
                self.down_res3 = self.trans_func(indepth + growth, outdepth, pool=pool)
                self.down_res2 = self.trans_func(indepth + growth, outdepth, pool=pool)
                self.down_res1 = self.trans_func(indepth + growth, outdepth, pool=pool)

        else:
            if self.last_down:
                outdepth = indepth + growth + last_expand
                if self.last_branch >= 1:
                    self.down_last4 = self.trans_func(indepth + growth, outdepth, pool=pool)
                if self.last_branch >= 2:
                    self.down_last3 = self.trans_func(indepth + growth, outdepth, pool=pool)
                if self.last_branch >= 3:
                    self.down_last2 = self.trans_func(indepth + growth, outdepth, pool=pool)
                if self.last_branch >= 4:
                    self.down_last1 = self.trans_func(indepth + growth, outdepth, pool=pool)
            else:
                if self.classify > 0 and self.last_branch == 4:
                    # 最后一个Couple的中间层被当做branch输出而对接在Summary上.
                    # 因此，删除此Couple自带的Classifier,以免与Summary中的Classifier重复.
                    delattr(self, 'classifier')
                    self.classify = 0
                    print('\nNote: 1 xfc will be deleted because of duplicated with the lfc!\n')

    def forward(self, x):
        # assert isinstance(x, (tuple, list))
        x1, x2, x3, x4, pred = x  # 大 中 小 中
        # print('x1, x2, x3, x4: ', x1.size(), x2.size(), x3.size(), x4.size(), type(self))

        if self.bottle > 0:
            res1 = self.conv1(self.active(self.bn1(self.conv1_b(self.active(self.bn1_b(x1), inplace))), inplace))
            res1 = torch.cat((res1, x4), 1)
            res2 = self.conv2(self.active(self.bn2(self.conv2_b(self.active(self.bn2_b(res1), inplace))), inplace))
            res2 = torch.cat((res2, x3), 1)
            out = res2
            res3 = self.deconv3(self.active(self.bn3(self.conv3_b(self.active(self.bn3_b(res2), inplace))), inplace))
            res3 = torch.cat((res3, x2), 1)
            res4 = self.deconv4(self.active(self.bn4(self.conv4_b(self.active(self.bn4_b(res3), inplace))), inplace))
            res4 = torch.cat((res4, x1), 1)
        else:
            res1 = self.conv1(self.active(self.bn1(x1), inplace))
            res1 = torch.cat((res1, x4), 1)
            res2 = self.conv2(self.active(self.bn2(res1), inplace))
            res2 = torch.cat((res2, x3), 1)
            out = res2
            res3 = self.deconv3(self.active(self.bn3(res2), inplace))
            res3 = torch.cat((res3, x2), 1)
            res4 = self.deconv4(self.active(self.bn4(res3), inplace))
            res4 = torch.cat((res4, x1), 1)

        if self.classify > 0:
            out = self.classifier(out)
        else:
            out = None
        pred.append(out)

        if self.after:
            if self.down and self.reduce != 1:
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
    _trans = {'B': TransitionB, 'C': TransitionC}

    def __init__(self, indepth, growth, bottle=4, active='relu', after=True, down=False, trans='B', pool='avg',
                 reduce=0.5, classify=0, nclass=1000, last_branch=1, last_down=False, last_expand=0):
        super(SingleCouple, self).__init__()
        assert last_branch <= 2, '<last_branch> of SingleCouple should be <= 2...'
        self.indepth = indepth
        self.growth = growth
        self.bottle = bottle
        self.active = getattr(nn.functional, active)
        self.after = after
        self.down = down
        self.trans = trans
        self.pool = pool
        self.reduce = reduce
        self.trans_func = self._trans[trans]
        self.classify = classify
        self.nclass = nclass
        self.last_branch = last_branch
        self.last_down = last_down
        self.last_expand = last_expand

        if bottle > 0:
            interdepth = int(bottle * growth)
            assert interdepth == math.ceil(bottle * growth), \
                '<bottle> * <growth> cannot be a fraction number (带小数位), but %s !\n' % bottle * growth
            self.bn1_b = nn.BatchNorm2d(indepth)
            self.conv1_b = nn.Conv2d(indepth, interdepth, 1, stride=1, padding=0, dilation=1, bias=False)
            self.bn1 = nn.BatchNorm2d(interdepth)
            self.conv1 = nn.Conv2d(interdepth, growth, 3, stride=2, padding=1, bias=False, dilation=1)

            self.bn2_b = nn.BatchNorm2d(indepth + growth)
            self.conv2_b = nn.Conv2d(indepth + growth, interdepth, 1, stride=1, padding=0, dilation=1, bias=False)
            self.bn2 = nn.BatchNorm2d(interdepth)
            self.deconv2 = nn.ConvTranspose2d(interdepth, growth, 3, stride=2, padding=1,
                                              output_padding=1, bias=False, dilation=1)

        elif bottle == 0:
            self.bn1 = nn.BatchNorm2d(indepth)
            self.conv1 = nn.Conv2d(indepth, growth, 3, stride=2, padding=1, bias=False, dilation=1)
            self.bn2 = nn.BatchNorm2d(indepth + growth)
            self.deconv2 = nn.ConvTranspose2d(indepth + growth, growth, 3, stride=2, padding=1,
                                              output_padding=1, bias=False, dilation=1)

        else:
            raise NotImplementedError('<bottle> should be >= 0, but %s' % bottle)

        if self.classify > 0:
            self.classifier = nn.Sequential(
                nn.BatchNorm2d(indepth + growth),
                Activate(active),
                AdaAvgPool(),
                ViewLayer(dim=-1),
                nn.Linear(indepth + growth, nclass))

        if self.after:
            if self.down and self.reduce != 1:
                outdepth = int(math.floor((indepth + growth) * reduce))
                self.down_res2 = self.trans_func(indepth + growth, outdepth, pool=pool)
                self.down_res1 = self.trans_func(indepth + growth, outdepth, pool=pool)
        else:
            if self.last_down:
                outdepth = indepth + growth + last_expand
                if self.last_branch >= 1:
                    self.down_last2 = self.trans_func(indepth + growth, outdepth, pool=pool)
                if self.last_branch >= 2:
                    self.down_last1 = self.trans_func(indepth + growth, outdepth, pool=pool)
            else:
                if self.classify > 0 and self.last_branch == 2:
                    # 此时，最后一个Couple的中间层被当做branch输出而对接在Summary上.
                    # 因此，删除此Couple自带的Classifier,以免与Summary中的Classifier重复.
                    delattr(self, 'classifier')
                    self.classify = 0
                    print('\n Note: 1 xfc will be deleted  because of duplicate with the last-fc!!!!! \n')

    def forward(self, x):
        # assert isinstance(x, (tuple, list))
        x1, x2, x3, x4, pred = x  # 大 中 小 中
        # print('x1, x2, x3, x4: ', x1.size(), x2.size(), type(x3), type(x4), type(self))

        if self.bottle > 0:
            res1 = self.conv1(self.active(self.bn1(self.conv1_b(self.active(self.bn1_b(x1), inplace))), inplace))
            res1 = torch.cat((res1, x2), 1)
            out = res1
            res2 = self.deconv2(self.active(self.bn2(self.conv2_b(self.active(self.bn2_b(res1), inplace))), inplace))
            res2 = torch.cat((res2, x1), 1)
        else:
            res1 = self.conv1(self.active(self.bn1(x1), inplace))
            res1 = torch.cat((res1, x2), 1)
            out = res1
            res2 = self.deconv2(self.active(self.bn2(res1), inplace))
            res2 = torch.cat((res2, x1), 1)

        if self.classify > 0:
            out = self.classifier(out)
        else:
            out = None
        pred.append(out)

        if self.after:
            if self.down and self.reduce != 1:
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


class PyramidBlock(nn.Module):
    def __init__(self, indepth, outdepth, active='relu', nextb='D'):
        super(PyramidBlock, self).__init__()
        assert nextb in ('D', 'S'), 'next block is DoubleCouple or SingleCouple?'
        self.indepth = indepth
        self.outdepth = outdepth
        self.nextb = nextb
        self.active = getattr(nn.functional, active)

        self.bn1 = nn.BatchNorm2d(indepth)
        self.conv1 = nn.Conv2d(indepth, outdepth, 3, stride=2, padding=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(outdepth)
        self.conv2 = nn.Conv2d(outdepth, outdepth, 3, stride=2, padding=1, bias=False, dilation=1)
        self.bn3 = nn.BatchNorm2d(outdepth)
        self.deconv3 = nn.ConvTranspose2d(outdepth, outdepth, 3, stride=2, padding=1,
                                          output_padding=1, bias=False, dilation=1)
        self.bn4 = nn.BatchNorm2d(outdepth)
        self.deconv4 = nn.ConvTranspose2d(outdepth, outdepth, 3, stride=2, padding=1,
                                          output_padding=1, bias=False, dilation=1)

    def forward(self, x):
        x, pred = x
        res1 = self.conv1(self.active(self.bn1(x), inplace))
        res2 = self.conv2(self.active(self.bn2(res1), inplace))
        res3 = self.deconv3(self.active(self.bn3(res2), inplace))
        res4 = self.deconv4(self.active(self.bn4(res3), inplace))
        if self.nextb == 'D':
            return res4, res3, res2, res1, pred
        elif self.nextb == 'S':
            return res4, res3, None, None, pred
        else:
            raise NotImplementedError


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
        # x1, x2, x3 extracted form x is big, media, small respectively.
        # 为确保fc(xi)的顺序与layer_i在model内的顺序相一致和相对应，
        # so the output order should be [fc(x3), fc(x2), fc(x1)] or [fc([x3, x2, x1])]
        if not self.active_fc:
            return x
        # # 关闭数据检查
        # assert isinstance(x, (tuple, list)), 'x must be tuple, but %s' % type(x)
        # assert len(x) == self.branch + 1, 'pred should be input together with x'
        x, pred = x[:self.branch][::-1], x[-1]
        # utils.print_size(x)
        if self.method == 'split':
            for i, xi in enumerate(x):
                xi = getattr(self, 'classifier%s' % (len(x) - i))(xi)
                pred.append(xi)
        elif self.method == 'merge':
            x = [getattr(self, 'pool_view%s' % (len(x) - i))(xi) for i, xi in enumerate(x)]
            x = torch.cat(x, dim=1)
            x = self.classifier(x)
            pred.append(x)
        return pred


class NameiNet(nn.Module):
    _couple = {'D': DoubleCouple, 'S': SingleCouple}

    def __init__(self,
                 stages=4,
                 indepth=16,
                 growth=12,
                 layers=(3, 3, 3, 3),
                 blocks=('D', 'D', 'D', 'D'),
                 bottle=(4, 4, 4, 4),
                 classify=(0, 0, 0, 0),
                 trans=('B', 'B', 'B', 'B'),
                 reduction=(0.5, 0.5, 0.5, 0.5),
                 last_branch=4,
                 last_down=False,
                 last_expand=0,
                 poolmode='avg',
                 active='relu',
                 summer='split',
                 nclass=1000):
        super(NameiNet, self).__init__()
        assert stages <= min(len(layers), len(blocks), len(classify),
                             len(trans), len(reduction), len(bottle)), \
            'Hyper Pameters Not Enough to Match Stages Nums:%s!' % stages
        assert stages == sum([bool(l) for l in layers[:stages]]), \
            'Hyper Pameters <stages:%s> and <layers:%s> cannot match, ' \
            'number of no-zero value in <layers> should be == <stage> !' % (stages, layers)
        assert stages == sum([bool(r) for r in reduction[:stages - 1]]) + 1, \
            'Hyper Pameters <stages:%s> and <reduction:%s> cannot match, ' \
            'number of no-zero value in <reduction> should be == <stages-1>!' % (stages, reduction)
        assert sorted(blocks[:stages]) == list(blocks[:stages]), \
            'DoubleCouple must be ahead of SingleCouple! But your is %s' % ' ->'.join(blocks[:stages])

        dataset = ['imagenet', 'cifar'][nclass != 1000]
        if dataset == 'cifar':
            assert stages <= 4, 'cifar stages should <= 4'
        elif dataset == 'imagenet':
            assert stages <= 5, 'imagenet stages should <= 5'

        self.stages = stages
        self.indepth = indepth
        self.growth = growth
        self.bottle = bottle
        self.layers = layers
        self.classify = classify
        self.trans = trans
        self.poolmode = poolmode
        self.active = active
        self.last_branch = last_branch
        self.last_down = last_down
        self.last_expand = last_expand
        self.nclass = nclass

        self.preproc = PreProcess(indepth=3, outdepth=indepth, dataset=dataset)

        self.pyramid = PyramidBlock(indepth=indepth, outdepth=indepth, active=active, nextb=blocks[0])

        self.after = [True for _ in range(stages - 1)] + [False]
        for i in range(stages):
            dense_stage = self._make_dense_stage(self._couple[blocks[i]], layers[i], indepth, growth, bottle[i],
                                                 classify[i], self.after[i], trans[i], reduction[i],
                                                 last_branch, last_down, last_expand, i)
            setattr(self, 'dense%s' % (i + 1), dense_stage)
            indepth += layers[i] * growth
            if i < stages - 1:
                indepth = int(math.floor(indepth * reduction[i]))
            elif i == stages - 1:
                indepth = indepth + last_expand if last_down else indepth  # + growth

        indepth = [indepth] * last_branch
        self.summary = SummaryBlock(indepth, last_branch, active, nclass=nclass, method=summer)

    def _make_dense_stage(self, block, nums, indepth, growth, bottle, cfy, after, trans, reduce,
                          last_branch, last_down, last_expand, stage):
        layers = []
        for i in range(nums - 1):
            layers.append(block(indepth=indepth, growth=growth, bottle=bottle, active=self.active,
                                after=True, down=False, trans=trans, reduce=reduce,
                                classify=cfy, nclass=self.nclass, pool=self.poolmode,
                                last_branch=last_branch, last_down=last_down, last_expand=last_expand))
            indepth += growth

        layers.append(block(indepth=indepth, growth=growth, bottle=bottle, active=self.active, pool=self.poolmode,
                            after=after, down=True, trans=trans, reduce=reduce, classify=cfy, nclass=self.nclass,
                            last_branch=last_branch, last_down=last_down, last_expand=last_expand))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.preproc(x)
        x = self.pyramid(x)
        for i in range(self.stages):
            x = getattr(self, 'dense%s' % (i + 1))(x)
        x = self.summary(x)  # x <=> [x, pred]
        x = [p for p in x if p is not None]
        return x


if __name__ == '__main__':
    import xtils, time

    torch.manual_seed(9528)

    # # imagenet

    # nm1d = {'stages': 1, 'indepth': 16, 'growth': 16, 'poolmode': 'max', 'active': 'relu',
    #         'layers': (20, 0), 'blocks': ('D', '-'), 'bottle': (0, 0), 'classify': (0, 0),
    #         'trans': ('B', '-'), 'reduction': (0, 0),
    #         'last_branch': 1, 'last_down': True, 'last_expand': 0,
    #         'summer': 'split', 'nclass': 1000}
    #
    # nm2d = {'stages': 2, 'indepth': 6, 'growth': 6, 'poolmode': 'max', 'active': 'relu',
    #         'layers': (3, 3), 'blocks': ('D', 'S'), 'bottle': (0, 0), 'classify': (0, 0),
    #         'trans': ('B', 'B'), 'reduction': (0.5, 0),
    #         'last_branch': 1, 'last_down': True, 'last_expand': 0,
    #         'summer': 'split', 'nclass': 1000}
    #
    # nm3d = {'stages': 3, 'indepth': 6, 'growth': 3, 'poolmode': 'max', 'active': 'relu',
    #         'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'bottle': (0, 0, 0),
    #         'classify': (0, 0, 0), 'trans': ('B', 'B', 'B'), 'reduction': (0.5, 0.5, 0),
    #         'last_branch': 1, 'last_down': True, 'last_expand': 10,
    #         'summer': 'split', 'nclass': 1000}
    #
    # model = NameiNet(**nm1d)
    # print(model)
    # x = torch.randn(4, 3, 224, 224)
    # # utils.tensorboard_add_model(model, x)
    #
    # utils.calculate_params_scale(model, format='million')
    # utils.calculate_FLOPs_scale(model, 224, use_gpu=False, multiply_adds=True)
    # utils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
    # y = model(x)
    # print('共有dense blocks数：', sum(model.layers), '最后实际输出分类头数：', len(y), )
    # print('每个输出预测的尺寸:', [(yy.shape,) for yy in y if yy is not None])
    # print('每个输出预测的得分:', [(yy.max(1),) for yy in y if yy is not None])

    # cifar
    nm1s = {'stages': 1, 'indepth': 24, 'growth': 12,
            'layers': (3, 0), 'blocks': ('D', '-'), 'bottle': (0, 0), 'classify': (0, 0),
            'trans': ('B', '-'), 'reduction': (0, 0),
            'last_branch': 1, 'last_down': True, 'last_expand': 0,
            'poolmode': 'max', 'active': 'relu', 'summer': 'split', 'nclass': 10}

    nm2s = {'stages': 2, 'indepth': 24, 'growth': 12,
            'layers': (3, 3), 'blocks': ('D', 'S'), 'bottle': (0, 0), 'classify': (0, 0),
            'trans': ('B', 'B'), 'reduction': (0.5, 0),
            'last_branch': 1, 'last_down': False, 'last_expand': 0,
            'poolmode': 'max', 'active': 'relu', 'summer': 'split', 'nclass': 10}

    nm3s = {'stages': 3, 'indepth': 24, 'growth': 8,
            'layers': (5, 5, 5), 'blocks': ('D', 'D', 'S'), 'bottle': (0, 0, 0), 'classify': (0, 0, 0),
            'trans': ('C', 'B', 'B'), 'reduction': (0.5, 0.5, 0),
            'last_branch': 1, 'last_down': False, 'last_expand': 10,
            'poolmode': 'max', 'active': 'relu', 'summer': 'split', 'nclass': 10}

    nm4s = {'stages': 4, 'indepth': 24, 'growth': 8,
            'layers': (3, 3, 3, 3), 'blocks': ('D', 'D', 'D', 'D'), 'bottle': (0, 0, 0, 0),
            'classify': (1, 1, 1, 1), 'trans': ('C', 'C', 'C', 'C'), 'reduction': (0.5, 0.5, 0.5, 0),
            'last_branch': 1, 'last_down': False, 'last_expand': 10,
            'poolmode': 'max', 'active': 'relu', 'summer': 'split', 'nclass': 10}

    model = NameiNet(**nm4s)
    print('\n', model, '\n')
    # utils.tensorboard_add_model(model, x)
    xtils.calculate_params_scale(model, format='million')
    # utils.calculate_FLOPs_scale(model, input_size=32, use_gpu=False, multiply_adds=True)
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
    x = torch.randn(4, 3, 32, 32)
    tic, toc = time.time(), 1
    y = [model(x) for _ in range(toc)][0]
    toc = (time.time() - tic) / toc
    print('有效分类支路：', len(y), '\t共有blocks：', sum(model.layers), '\t处理时间: %.5f 秒' % toc)
    print('每个输出预测的尺寸:', [(yy.shape,) for yy in y if yy is not None])
    print('每个输出预测的得分:', [(yy.max(1),) for yy in y if yy is not None])
