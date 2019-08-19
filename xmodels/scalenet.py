# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import numpy as np
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from xmodules.downsample import DownSampleA, DownSampleB, DownSampleC, DownSampleD, \
    DownSampleE, DownSampelH, DownSampleO
from xmodules.rockblock import RockBlock, RockBlockM, RockBlockU, RockBlockV, RockBlockR, RockBlockG, RockBlockN
from xmodules.classifier import ViewLayer, AdaAvgPool, Activate, ReturnX, XClassifier, ConvClassifier
from xmodules.affineblock import AfineBlock

"""
  ScaleNet => AffineBlock + RockBlock + DoubleCouple + SingleCouple 
               + Summary(+merge +split) + Boost + BottleNeck 
               + Cat/Para-Transform + Channels(++expand)
"""

inplace = [False, True][1]


class DoubleCouple(nn.Module):
    down_func_dict = {'A': DownSampleA, 'B': DownSampleB, 'C': DownSampleC,
                      'D': DownSampleD, 'E': DownSampleE, 'H': DownSampelH,
                      'O': DownSampleO, 'None': None}
    down_style = {'avgpool', 'maxpool', 'convk2', 'convk3', 'convk2a', 'convk2m', 'convk3a', 'convk3m'}

    """
     version 1: deconv_kernel=4, output_padding=0 ; conv @ TransDown 
     version 2: deconv_kernel=4, output_padding=0 ; cocat @ TransDown 
     version 3: deconv_kernel=3, output_padding=1 ; cocat @ TransDown
    """

    def __init__(self, depth, expand, growth, slink='A', after=True, down=False, dfunc='A',
                 dstyle=('maxpool', 'convk3m', 'convk2'), classify=0, nclass=1000,
                 last_branch=1, last_down=False, last_dfuc='D', version=1):

        super(DoubleCouple, self).__init__()
        assert last_branch <= 3, '<last_branch> of DoubleCouple should be <= 3...'
        assert len(growth) == 2, 'len of <growth> of DoubleCouple should be 2'
        assert last_dfuc != 'O', '<last_dfuc> of DoubleCouple should not be "O", ' \
                                 'choose others in <down_func_dict>'
        assert set(dstyle).issubset(self.down_style), '<dstyle> should be in <down_style>, but %s.' % dstyle
        assert version in [1, 2, 3], '<version> now expected in [1, 2, 3], but %s.' % version
        self.depth = depth
        self.expand = expand
        self.growth = growth  # how much channels will be added or minus, when plain size is halved or doubled.
        growtha, growthb = growth
        self.slink = slink
        self.after = after  # dose still have another DoubleCouple behind of this DoubleCouple. ?
        self.down = [down, ['independent', 'interactive'][dfunc == 'O']][down]
        # dose connect to a same-size DoubleCouple or down-sized DoubleCouple. ?
        self.dfunc = dfunc
        self.down_func = self.down_func_dict[dfunc]
        self.dstyle = dstyle

        self.last_branch = last_branch  # how many output-ways (branches) for the classifier ?
        self.last_down = last_down  # dose down size for the classifier?
        self.last_dfuc = self.down_func_dict[last_dfuc]
        self.classify = classify
        self.nclass = nclass
        self.version = version
        self.active_fc = False

        if version in [1, 2]:
            deck, decop = 4, 0
        else:
            deck, decop = 3, 1

        self.bn1 = nn.BatchNorm2d(depth)
        self.conv1 = nn.Conv2d(depth, depth + growtha, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth + growtha)
        self.conv2 = nn.Conv2d(depth + growtha, depth + 2 * growtha, 3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(depth + 2 * growtha)
        self.deconv3 = nn.ConvTranspose2d(depth + 2 * growtha, depth + growtha, deck, stride=2, padding=1, bias=False,
                                          output_padding=decop, dilation=1)
        self.bn4 = nn.BatchNorm2d(depth + growtha)
        self.deconv4 = nn.ConvTranspose2d(depth + growtha, depth, deck, stride=2, padding=1, bias=False,
                                          output_padding=decop, dilation=1)

        if self.classify > 0:
            self.classifier = XClassifier(depth + 2 * growtha, nclass)

        if self.after:
            if self.down == 'independent':
                self.down_res4 = self.down_func(depth, depth + expand)
                self.down_res3 = self.down_func(depth + growtha, depth + expand + growthb)
                self.down_res2 = self.down_func(depth + 2 * growtha, depth + expand + 2 * growthb)
            elif self.down == 'interactive':
                self.down_res4 = self.down_func(depth, depth, 2, dstyle[0])
                self.down_res3 = self.down_func(depth + growtha, depth + growtha, 2, dstyle[0])
                self.down_res2 = self.down_func(depth + 2 * growtha, depth + 2 * growtha, 2, dstyle[0])
                self.down_res4x = self.down_func(depth, depth, 4, dstyle[1])
                if version == 1:
                    self.comp_res4 = nn.Conv2d(depth * 2 + growtha, depth + expand, 1, 1, 0, bias=False)
                    self.comp_res3 = nn.Conv2d(depth * 2 + 3 * growtha, depth + expand + growthb, 1, 1, 0, bias=False)
                    self.comp_res2 = nn.Conv2d(depth * 2 + 2 * growtha, depth + expand + 2 * growthb, 1, 1, 0,
                                               bias=False)
                else:
                    self.comp_res4 = [nn.Conv2d(depth * 2 + growtha, depth + expand, 1, 1, 0, bias=False),
                                      ReturnX()][depth * 2 + growtha == depth + expand]
                    self.comp_res3 = [nn.Conv2d(depth * 2 + 3 * growtha, depth + expand + growthb, 1, 1, 0, bias=False),
                                      ReturnX()][depth * 2 + 3 * growtha == depth + expand + growthb]
                    self.comp_res2 = \
                        [nn.Conv2d(depth * 2 + 2 * growtha, depth + expand + 2 * growthb, 1, 1, 0, bias=False),
                         ReturnX()][depth * 2 + 2 * growtha == depth + expand + 2 * growthb]
        else:
            if self.last_down is True:
                if self.last_branch >= 1:
                    self.down_last4 = self.last_dfuc(depth, depth + growthb)
                if self.last_branch >= 2:
                    self.down_last3 = self.last_dfuc(depth + growtha, depth + growthb)
                if self.last_branch >= 3:
                    self.down_last2 = self.last_dfuc(depth + 2 * growtha, depth + growthb)
            elif self.last_down is False:
                if self.classify > 0 and self.last_branch == 3:
                    # 最后一个Couple的中间层被当做branch输出而对接在Summary上.
                    # 因此，删除此Couple自带的Classifier,以免与Summary中的Classifier重复.
                    delattr(self, 'classifier')
                    self.classify = 0
                    print('\nNote ****: 1 xfc will be deleted because of duplicate with the last-fc!\n')
            elif self.last_down is None:
                # 删除整个网络最末端的DoubleCouple中最后的2个DeConv层，分类器将直接扣在保留下的Conv层的输出上.
                units_in_last_couple = ['bn3', 'deconv3', 'bn4', 'deconv4']
                for unit in units_in_last_couple:
                    # setattr(self, unit, ReturnX())
                    delattr(self, unit)

    def forward(self, x):
        # assert isinstance(x, (tuple, list))
        x1, x2, x3, pred = x
        # print('x1, x2, x3: ', x1.size(), x2.size(), x3.size(), type(self))

        # add-style
        if self.slink == 'A':
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            out = res1 if x2 is None else res1 + x2
            res2 = self.conv2(F.relu(self.bn2(out), inplace))
            out = res2 if x3 is None else res2 + x3
            if self.after:
                res3 = self.deconv3(F.relu(self.bn3(out), inplace))
                res4 = self.deconv4(F.relu(self.bn4(res3 + res1), inplace))
                res4 = res4 + x1
                # utils.tensor_to_img(res4, ishow=False, istrans=True,
                # file_name='xxxxx', save_dir='C://Users//xo//Pictures//xxx')
            else:
                if self.last_down in [True, False]:
                    res3 = self.deconv3(F.relu(self.bn3(out), inplace))
                    res4 = self.deconv4(F.relu(self.bn4(res3 + res1), inplace))
                    res4 = res4 + x1
                elif self.last_down is None:
                    res3 = None
                    res4 = None
                else:
                    raise NotImplementedError

        elif self.slink == 'X':
            # first add, then shortcut, low < 'A'
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res1 = res1 if x2 is None else res1 + x2
            res2 = self.conv2(F.relu(self.bn2(res1), inplace))
            res2 = res2 if x3 is None else res2 + x3
            out = res2
            res3 = self.deconv3(F.relu(self.bn3(res2), inplace))
            res3 = res3 + res1
            res4 = self.deconv4(F.relu(self.bn4(res3), inplace))
            res4 = res4 + x1

        elif self.slink == 'B':
            # A的简化版，只有每个block内部的2个内连接
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res2 = self.conv2(F.relu(self.bn2(res1), inplace))
            res3 = self.deconv3(F.relu(self.bn3(res2), inplace))
            res4 = self.deconv4(F.relu(self.bn4(res3 + res1), inplace))
            res4 = res4 + x1
            out = res2

        elif self.slink == 'C':
            # A的简化版，只有每个block内内部最大尺寸的那1个内连接， 类似resnet
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res2 = self.conv2(F.relu(self.bn2(res1), inplace))
            out = res2
            if self.after:
                res3 = self.deconv3(F.relu(self.bn3(res2), inplace))
                res4 = self.deconv4(F.relu(self.bn4(res3), inplace))
                res4 = res4 + x1
            else:
                if self.last_down in [False, True]:
                    res3 = self.deconv3(F.relu(self.bn3(res2), inplace))
                    res4 = self.deconv4(F.relu(self.bn4(res3), inplace))
                    res4 = res4 + x1
                elif self.last_down is None:
                    res3 = None
                    res4 = None
                else:
                    raise NotImplementedError

        elif self.slink == 'D':
            # A的简化版，2个夸block连接，1个block内连接(res4+x1)
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            out = res1 if x2 is None else res1 + x2
            res2 = self.conv2(F.relu(self.bn2(out), inplace))
            out = res2 if x3 is None else res2 + x3
            res3 = self.deconv3(F.relu(self.bn3(out), inplace))
            res4 = self.deconv4(F.relu(self.bn4(res3), inplace))
            res4 = res4 + x1

        elif self.slink == 'E':
            # A的简化版，只有2个夸block连接
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            out = res1 if x2 is None else res1 + x2
            res2 = self.conv2(F.relu(self.bn2(out), inplace))
            out = res2 if x3 is None else res2 + x3
            res3 = self.deconv3(F.relu(self.bn3(out), inplace))
            res4 = self.deconv4(F.relu(self.bn4(res3), inplace))
            res4 = res4

        elif self.slink == 'F':
            # A的简化版，1个夸block连接，1个block内连接
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res2 = self.conv2(F.relu(self.bn2(res1), inplace))
            out = res2 if x3 is None else res2 + x3
            res3 = self.deconv3(F.relu(self.bn3(out), inplace))
            res4 = self.deconv4(F.relu(self.bn4(res3), inplace))
            res4 = res4 + x1

        # cat-style
        elif self.slink == 'G':
            # Note: x2, x3 在当前block内不生效，全部累加到本stage的最后一个block内, 在downsize的时候生效
            # only x1 for calculate, x2 / x3 all moved to next block until the last block
            # Not good! x2, x3 be wasted .
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res2 = self.conv2(F.relu(self.bn2(res1), inplace))
            res3 = self.deconv3(F.relu(self.bn3(res2), inplace))
            res4 = self.deconv4(F.relu(self.bn4(res3), inplace))
            out = res2

            res1 = res1 if x2 is None else res1 + x2
            res2 = res2 if x3 is None else res2 + x3
            res3 = res3 + res1
            res4 = res4 + x1

        elif self.slink == 'H':
            # B的变体，但在向后累加时，丢掉本block的res1，只有res3.
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res2 = self.conv2(F.relu(self.bn2(res1), inplace))
            res3 = self.deconv3(F.relu(self.bn3(res2), inplace))
            res4 = self.deconv4(F.relu(self.bn4(res3), inplace))
            out = res2

            # res1 = res1 if x2 is None else res1 + x2
            res2 = res2 if x3 is None else res2 + x3
            res3 = res3 if x3 is None else res3 + x2
            res4 = res4 + x1

        elif self.slink == 'N':
            # No Shortcuts Used
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res2 = self.conv2(F.relu(self.bn2(res1), inplace))
            res3 = self.deconv3(F.relu(self.bn3(res2), inplace))
            res4 = self.deconv4(F.relu(self.bn4(res3), inplace))
            out = res2

        else:
            raise NotImplementedError('Unknown Slink for DoubleCouple : %s ' % self.slink)

        if self.classify > 0 and self.active_fc:
            out = self.classifier(out)
        else:
            out = None
        pred.append(out)

        if self.after:
            if self.down == 'independent':
                res4 = self.down_res4(res4)
                res3 = self.down_res3(res3)
                res2 = self.down_res2(res2)
                return res4, res3, res2, pred
            elif self.down == 'interactive':
                res4 = self.down_res4(res4)
                rex4 = self.down_res4x(res4)
                res4 = torch.cat((res4, res3), 1)
                res4 = self.comp_res4(res4)

                res3 = self.down_res3(res3)
                res3 = torch.cat((res3, res2), 1)
                res3 = self.comp_res3(res3)

                res2 = self.down_res2(res2)
                res2 = torch.cat((res2, rex4), 1)
                res2 = self.comp_res2(res2)
                return res4, res3, res2, pred
            else:
                return res4, res3, res2, pred
        else:
            if self.last_down in [False, True]:
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
            elif self.last_down is None:
                return res2, pred
            else:
                raise NotImplementedError


class SingleCouple(nn.Module):
    down_func_dict = {'A': DownSampleA, 'B': DownSampleB, 'C': DownSampleC,
                      'D': DownSampleD, 'E': DownSampleE, 'H': DownSampelH,
                      'O': DownSampleO, 'None': None}
    down_style = {'avgpool', 'maxpool', 'convk2', 'convk3', 'convk2a', 'convk2m', 'convk3a', 'convk3m'}

    def __init__(self, depth, expand, growth, slink='A', after=True, down=False, dfunc='A',
                 dstyle=('maxpool', 'convk3m', 'convk2'), classify=0, nclass=1000, last_branch=1,
                 last_down=False, last_dfuc='D', version=1):
        super(SingleCouple, self).__init__()
        assert last_branch <= 2, '<last_branch> of SingleCouple should be <= 2'
        assert len(growth) == 2, 'len of <growth> of SingleCouple should be 2'
        assert last_dfuc != 'O', '<last_dfuc> of SingleCouple should not be "O", ' \
                                 'choose others in <down_func_dict>'
        assert set(dstyle).issubset(self.down_style), '<dstyle> should be in <down_style>, but %s.' % dstyle
        assert version in [1, 2, 3], '<version> now expected in [1, 2, 3], but %s.' % version
        self.depth = depth
        self.expand = expand
        self.growth = growth
        growtha, growthb = growth
        self.slink = slink
        self.after = after  # dose still have another DoubleCouple behind of this DoubleCouple. ?
        # dose connect to a same-size DoubleCouple or down-sized DoubleCouple. ?
        self.down = [down, ['independent', 'interactive'][dfunc == 'O']][down]
        self.dfunc = dfunc
        self.down_func = self.down_func_dict[dfunc]
        self.dstyle = dstyle
        self.last_branch = last_branch  # how many output-ways (branches) for the classifier ?
        self.last_down = last_down  # dose down size for the classifier?
        self.last_dfuc = self.down_func_dict[last_dfuc]
        self.classify = classify
        self.nclass = nclass
        self.version = version
        self.active_fc = False

        if version in [1, 2]:
            deck, decop = 4, 0
        else:
            deck, decop = 3, 1

        self.bn1 = nn.BatchNorm2d(depth)
        self.conv1 = nn.Conv2d(depth, depth + growtha, 3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(depth + growtha)
        self.deconv2 = nn.ConvTranspose2d(depth + growtha, depth, deck, stride=2, padding=1, bias=False,
                                          output_padding=decop)

        if self.classify > 0:
            self.classifier = XClassifier(depth + growtha, nclass)

        if self.after:
            if self.down == 'independent':
                self.down_res2 = self.down_func(depth, depth + expand)
                self.down_res1 = self.down_func(depth + growtha, depth + expand + growthb)
            elif self.down == 'interactive':
                self.down_res2 = self.down_func(depth, depth, 2, dstyle[0])
                self.down_res1 = self.down_func(depth + growtha, depth + growtha, 2, dstyle[0])
                self.down_res2x = self.down_func(depth, depth, 2, dstyle[2])
                if version == 1:
                    self.comp_res2 = nn.Conv2d(depth * 2 + growtha, depth + expand, 1, 1, 0, bias=False)
                    self.comp_res1 = nn.Conv2d(depth * 2 + growtha, depth + expand + growthb, 1, 1, 0, bias=False)
                else:
                    self.comp_res2 = [nn.Conv2d(depth * 2 + growtha, depth + expand, 1, 1, 0, bias=False),
                                      ReturnX()][depth * 2 + growtha == depth + expand]
                    self.comp_res1 = [nn.Conv2d(depth * 2 + growtha, depth + expand + growthb, 1, 1, 0, bias=False),
                                      ReturnX()][depth * 2 + growtha == depth + expand + growthb]
        else:
            if self.last_down is True:
                if self.last_branch >= 1:
                    self.down_last2 = self.last_dfuc(depth, depth + growthb)
                if self.last_branch >= 2:
                    self.down_last1 = self.last_dfuc(depth + growtha, depth + growthb)
            elif self.last_down is False:
                if self.classify > 0 and self.last_branch == 2:
                    # 此时，最后一个Couple的中间层被当做branch输出而对接在Summary上.
                    # 因此，删除此Couple自带的Classifier,以免与Summary中的Classifier重复.
                    delattr(self, 'classifier')
                    self.classify = 0
                    print('\nNote ****: 1 xfc will be deleted  because of duplicate with the last-fc!\n')
            elif self.last_down is None:
                # 删除整个网络最末端的DoubleCouple中最后的2个DeConv层，分类器将直接扣在保留下的Conv层的输出上.
                units_in_last_couple = ['bn2', 'deconv2']
                for unit in units_in_last_couple:
                    # setattr(self, unit, ReturnX())
                    delattr(self, unit)

    def forward(self, x):
        # x3/res3 will be not used, but output 0 for parameters match between 2 layers
        # assert isinstance(x, (tuple, list))
        x1, x2, x3, pred = x
        # print('x1, x2, x3: ', x1.size(), x2.size(), x3.size(), type(self))

        # add-style for use
        if self.slink == 'A':
            # 共包含1个夸Block连接，1个Block内连接.
            # first shorcut, then add,
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            out = res1 if x2 is None else res1 + x2
            if self.after:
                res2 = self.deconv2(F.relu(self.bn2(out), inplace))
                res2 = res2 if x1 is None else res2 + x1
                res3 = None  # torch.Tensor(0).type_as(x3)
            else:
                if self.last_down in [True, False]:
                    res2 = self.deconv2(F.relu(self.bn2(out), inplace))
                    res2 = res2 if x1 is None else res2 + x1
                    res3 = None  # torch.Tensor(0).type_as(x3)
                elif self.last_down is None:
                    res2 = None
                    res3 = None
                else:
                    raise NotImplementedError

        elif self.slink == 'X':
            # 共包含1个夸Block连接，1个Block内连接
            # first add, then shorcut
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res1 = res1 if x2 is None else res1 + x2
            res2 = self.deconv2(F.relu(self.bn2(res1), inplace))
            res2 = res2 if x1 is None else res2 + x1
            # res3 = torch.zeros_like(x3, dtype=x3.dtype, device=x3.device)
            res3 = torch.Tensor(0).type_as(x3)
            out = res1

        elif self.slink == 'B':
            # A的简化版，只有1个block内连接
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res2 = self.deconv2(F.relu(self.bn2(res1), inplace))
            res2 = res2 if x1 is None else res2 + x1
            res3 = torch.Tensor(0).type_as(x3)
            out = res1

        elif self.slink == 'C':
            # A的简化版，只有1个跨block连接
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            out = res1 if x2 is None else res1 + x2
            if self.after:
                res2 = self.deconv2(F.relu(self.bn2(out), inplace))
                res3 = torch.Tensor(0).type_as(x3)
            else:
                if self.last_down in [True, False]:
                    res2 = self.deconv2(F.relu(self.bn2(out), inplace))
                    res3 = torch.Tensor(0).type_as(x3)
                elif self.last_down is None:
                    res2 = None
                    res3 = None
                else:
                    raise NotImplementedError

        elif self.slink == 'D':
            # 夸Block的链接全部累加到最后一个Block内生效
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res2 = self.deconv2(F.relu(self.bn2(res1), inplace))
            res1 = res1 if x2 is None else res1 + x2
            res2 = res2 if x1 is None else res2 + x1
            res3 = torch.Tensor(0).type_as(x3)
            out = res1

        # cat-style no use
        elif self.slink == 'E':
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res2 = self.deconv2(F.relu(self.bn2(res1), inplace))
            res1 = res1 if x2 is None else torch.cat((res1, x2), 1)
            res2 = res2 if x1 is None else torch.cat((res2, x1), 1)
            res3 = torch.Tensor(0).type_as(x3)
            out = res1

        elif self.slink == 'N':
            # No Shortcuts Used
            res1 = self.conv1(F.relu(self.bn1(x1), inplace))
            res2 = self.deconv2(F.relu(self.bn2(res1), inplace))
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
            if self.down == 'independent':
                res2 = self.down_res2(res2)
                res1 = self.down_res1(res1)
                return res2, res1, res3, pred  # todo let res3=None ?
            elif self.down == 'interactive':
                res2 = self.down_res2(res2)
                rex2 = self.down_res2x(res2)
                res2 = torch.cat((res2, res1), 1)
                res2 = self.comp_res2(res2)
                res1 = self.down_res1(res1)
                res1 = torch.cat((res1, rex2), 1)
                res1 = self.comp_res1(res1)
                return res2, res1, res3, pred
            else:
                return res2, res1, res3, pred
        else:
            if self.last_down in [True, False]:
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
            elif self.last_down is None:
                return res1, pred
            else:
                raise NotImplementedError


class RockSummary(nn.Module):
    METHOD = ['split', 'merge', 'convf', 'convs', 'convt', 'convx']

    def __init__(self, indepth, insize, branch, active='relu', pool='avg', nclass=1000, method='split'):
        super(RockSummary, self).__init__()
        assert len(indepth) == branch, '各输出分支的通道数必须全部给定，len of <indepth> == branch.'
        assert method in self.METHOD, 'Unknown <methond> %s.' % method
        if method in ['convf', 'convs', 'convt']:
            assert insize >= 1, '进行卷积分类，输入特征图尺寸<insize> >= 1.'
        if method == 'convf':
            assert branch >= 1, '对last-1进行卷积分类，输出分支<last-branch>>=1'
        elif method == 'convs':
            assert branch >= 2, '对last-2进行卷积分类，输出分支<last-branch>必须>=2'
        elif method == 'convt':
            assert branch >= 3, '对last-3进行卷积分类，输出分支<last-branch>必须>=3'

        self.indepth = indepth
        self.branch = branch
        self.active_fc = True
        self.nclass = nclass
        insize = int(insize)
        self.insize = insize
        self.method = method
        if method == 'split':
            for b in range(1, branch + 1):
                layer = nn.Sequential(
                    nn.BatchNorm2d(indepth[b - 1]),
                    Activate(active),
                    AdaAvgPool(),
                    ViewLayer(),
                    nn.Linear(indepth[b - 1], nclass))
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
        elif method == 'convx':
            for b in range(1, branch + 1):
                ksize = int(insize * ((1 / 2) ** (b - 1)))
                layer = ConvClassifier(indepth[b - 1], nclass, ksize=ksize, stride=1, padding=0)
                setattr(self, 'classifier%s' % b, layer)
        elif method == 'convf':
            self.classifier = ConvClassifier(indepth[0], nclass, ksize=insize // 1, stride=1, padding=0)
        elif method == 'convs':
            self.classifier = ConvClassifier(indepth[1], nclass, ksize=insize // 2, stride=1, padding=0)
        elif method == 'convt':
            self.classifier = ConvClassifier(indepth[2], nclass, ksize=insize // 4, stride=1, padding=0)
        else:
            raise NotImplementedError('Unknow <method> for SummaryBlock, %s' % method)

    def forward(self, x):
        # x1, x2, x3 extracted form x is big, media, small respectively.
        # 为确保fc(xi)的顺序与layer_i在model内的顺序相一致和相对应，
        # so the output order should be [fc(x3), fc(x2), fc(x1)] or [fc([x3, x2, x1])]
        if not self.active_fc:
            return x
        # assert isinstance(x, (tuple, list)), 'x must be tuple, but %s' % type(x)
        # assert len(x) == self.branch + 1, 'pred should be input together with x'
        x, pred = x[:-1][::-1], x[-1]
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
        elif self.method == 'convf':  # 大
            x = self.classifier(x[-1])
            pred.append(x)
        elif self.method == 'convs':  # 中
            x = self.classifier(x[-2])
            pred.append(x)
        elif self.method == 'convt':  # 小
            x = self.classifier(x[-3])
            pred.append(x)
        elif self.method == 'convx':
            for i, xi in enumerate(x):
                xi = getattr(self, 'classifier%s' % (len(x) - i))(xi)
                pred.append(xi)
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


class ScaleNet(nn.Module):
    couple = {'D': DoubleCouple, 'S': SingleCouple}
    rocker = {'M': RockBlockM, 'U': RockBlockU, 'V': RockBlockV,
              'R': RockBlockR, 'G': RockBlockG, 'N': RockBlockN}

    def __init__(self,
                 stages=4,
                 afisok=False,
                 afkeys=('af1', 'af2', 'af3'),
                 convon=True,
                 convlayers=1,
                 convdepth=3,
                 rock='U',
                 branch=3,
                 depth=64,
                 layers=(2, 2, 2, 2),
                 blocks=('D', 'D', 'S', 'S'),
                 slink=('A', 'A', 'A', 'A'),
                 growth=(2, 3, 4, 5),
                 classify=(1, 1, 1, 1),
                 expand=(1 * 64, 2 * 64, 4 * 64),  # stage 间的通道扩增数量
                 dfunc=('D', 'A', 'O'),  # stage 间的transform function
                 dstyle=('avgpool', 'convk3a', 'convk2'),  # 0: 1/2 for DCouple, 1:1/4 for DCopule, 2:1/2 for SCouple
                 fcboost='none',
                 last_branch=2,
                 last_down=True,
                 last_dfuc='D',
                 last_expand=30,
                 kldloss=False,
                 summer='split',
                 insize=0,
                 nclass=1000,
                 version=1):
        super(ScaleNet, self).__init__()
        assert sum(np.array([len(layers), len(blocks), len(slink), len(classify),
                             len(expand) + 1, len(dfunc) + 1]) - stages) == 0, \
            'Hyper Pameters Number Cannot Match Stages Nums:%s!' % stages
        assert set(blocks[:stages]) in [{'D'}, {'S'}, {'D', 'S'}], \
            '<blocks> must be in [D, S]! But %s' % ('-'.join(blocks[:stages]),)
        assert sorted(blocks[:stages]) == list(blocks[:stages]), \
            'DoubleCouple("D") must be ahead of SingleCouple("S")! But %s' % ('->'.join(blocks[:stages]),)
        assert stages == len(growth), 'len of <growth> must be == <stage>'
        assert rock in self.rocker.keys(), '<rock> must be in <self.rocker.keys()>, but your %s ' % rock
        assert last_down in [True, False, None], '<last_down> must be in [True, False, None], but %s .' % last_down
        if last_down is None:
            assert last_branch == 1, 'when <last_down==None>, must let <last_branch==1>'
        assert version in [1, 2, 3], '<version> now should be in [1, 2, 3].'
        dataset = ['imagenet', 'cifar'][nclass != 1000]
        if dataset == 'cifar':
            self.insize = 32 if insize == 0 else insize
            assert stages <= 4, 'cifar stages should <= 4'
        elif dataset == 'imagenet':
            self.insize = 64 if insize == 0 else insize
            assert stages <= 5, 'imagenet stages should <= 5'

        self.branch = branch
        self.rock = self.rocker[rock]
        self.depth = depth
        self.stages = stages
        self.layers = layers
        self.blocks = blocks  # [self.couple[b] for b in blocks]
        self.slink = slink
        growth = list(growth)
        growth.append(last_expand)
        self.growth = growth
        self.classify = classify
        expand = list(expand) + [0]  # 最后一个stage不扩增通道，或使用last_expand扩增通道
        self.expand = expand
        dfunc = list(dfunc) + ['None']  # 最后一个stage不降维，或使用last_down/last_dfuc降维
        self.dfunc = dfunc
        if not isinstance(dstyle, (tuple, list)):
            # 兼容旧版本，dstyle只设一个值,后续2个值为固定值
            dstyle = [dstyle, 'convk3a', 'convk2']
        self.dstyle = dstyle
        self.fcboost = fcboost
        self.last_branch = last_branch
        self.last_down = last_down
        self.last_dfuc = last_dfuc
        self.last_expand = last_expand
        self.kldloss = kldloss
        self.summer = summer
        self.nclass = nclass
        self.version = version
        self.afisok = afisok

        self.afdict = {'afkeys': afkeys, 'convon': convon,
                       'convlayers': convlayers, 'convdepth': convdepth}
        self.affine = AfineBlock(**self.afdict) if afisok else None

        indepth = 3 if not afisok else (len(afkeys) + 1) * [3, convdepth][convon]
        self.pyramid = self.rock(indepth=indepth, outdepth=depth, branch=branch,
                                 expand=(growth[0], growth[0]), dataset=dataset)

        self.after = [True for _ in range(stages - 1)] + [False]  # after: 最后一个stage的最后一个Couple是False
        indepth = depth
        for i in range(stages):
            layer = self._make_stage(self.couple[blocks[i]], layers[i], indepth, expand[i],
                                     (growth[i], growth[i + 1]), slink[i], self.after[i],
                                     dfunc[i], classify[i], dstyle, last_branch, last_down, last_dfuc)
            setattr(self, 'stage%s' % (i + 1), layer)
            indepth += expand[i]

        if last_down is True:
            fc_indepth = [last_expand] * last_branch
        elif last_down is False:
            if last_branch == 1:
                fc_indepth = [0]
            elif last_branch == 2:
                fc_indepth = [0, growth[stages - 1]]
            elif last_branch == 3:
                fc_indepth = [0, growth[stages - 1], 2 * growth[stages - 1]]
            else:
                raise NotImplementedError
        elif last_down is None:
            if blocks[-1] == 'D':
                fc_indepth = [2 * growth[stages - 1]]
            elif blocks[-1] == 'S':
                fc_indepth = [growth[stages - 1]]
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        fc_indepth = np.array([indepth] * last_branch) + np.array(fc_indepth)
        fc_indepth = fc_indepth.tolist()

        conv_insize = self.insize * ((1 / 2) ** (stages - 1))
        if last_down is True:
            conv_insize = conv_insize // 2
        elif last_down is False:
            conv_insize = conv_insize // 1
        elif last_down is None:
            if blocks[-1] == 'D':
                conv_insize = conv_insize // 4
            elif blocks[-1] == 'S':
                conv_insize = conv_insize // 2

        self.summary = RockSummary(fc_indepth, conv_insize, last_branch, 'relu', nclass=nclass, method=summer)

        xfc_nums = sum([1 for n, m in self.named_modules()
                        if isinstance(m, (DoubleCouple, SingleCouple))
                        and hasattr(m, 'classifier')])
        lfc_nums = last_branch if summer == 'split' else 1
        self.boost = AutoBoost(xfc=xfc_nums, lfc=lfc_nums, ksize=1, nclass=nclass, method=fcboost)

        if kldloss:
            self.kld_criterion = nn.KLDivLoss()

        self.train_which_now = {'conv': False, 'rock': False, 'conv+rock': False, 'xfc+boost': False,
                                'xfc-only': False, 'boost-only': False}
        self.eval_which_now = {'conv': False, 'rock': False, 'conv+rock': False, 'conv+rock+xfc': False,
                               'conv+rock+boost': False, 'conv+rock+xfc+boost': False}
        self._init_params()

    def _make_stage(self, block, block_nums, indepth, expand, growth, slink, after,
                    dfunc, cfy, dstyle, last_branch, last_down, last_dfuc):
        # if after=True, last_down must be False !!
        layers = []
        for i in range(block_nums - 1):
            layers.append(
                block(depth=indepth, expand=0, growth=growth, slink=slink, after=True,
                      down=False, dfunc=dfunc, dstyle=dstyle, classify=cfy, nclass=self.nclass,
                      last_branch=last_branch, last_down=False, last_dfuc='None', version=self.version))
        layers.append(
            block(depth=indepth, expand=expand, growth=growth, slink=slink, after=after,
                  down=True, dfunc=dfunc, dstyle=dstyle, classify=cfy, nclass=self.nclass,
                  last_branch=last_branch, last_down=last_down, last_dfuc=last_dfuc, version=self.version))
        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
                # nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.normal_(m.weight, mean=0, std=1)
                # nn.init.xavier_normal_(m.weight, gain=1)
                m.bias.data.zero_()

    def _init_weights(self):
        # forked from FishNet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def init_optimizer(self, cfg, visual=False):

        if cfg.decay_fly['flymode'] == 'nofly':
            # 所有参数的优化超参数相同，通常方法
            if cfg.optim_type == 'Adam':
                optimizer = torch.optim.Adam(params=self.parameters(), lr=cfg.lr_start,
                                             weight_decay=cfg.weight_decay)
            elif cfg.optim_type == 'SGD':
                optimizer = torch.optim.SGD(self.parameters(), cfg.lr_start, momentum=cfg.momentum,
                                            weight_decay=cfg.weight_decay)
            else:
                raise NotImplementedError('Unkwon optimizer type %s ...' % (cfg.optim_type,))

            if cfg.resume and cfg.resume_optimizer:
                optimizer.load_state_dict(cfg.checkpoint['optimizer'])
                if 'weight_decay' in cfg.exclude_keys:
                    print('\n当前 weight_decay: %s' % optimizer.param_groups[0]['weight_decay'])
                    for group in optimizer.param_groups:
                        group['weight_decay'] = cfg.weight_decay
                    print('更新后 weight_decay: %s \n' % optimizer.param_groups[0]['weight_decay'])
            if cfg.resume and not cfg.resume_optimizer:
                cfg.start_iter = 0

            if visual:
                wd_list = []
                for pgroup in optimizer.param_groups:
                    print('只有1个值，无曲线--->', len(pgroup['params']), pgroup['lr'], pgroup['weight_decay'])
                    wd_list.append(pgroup['weight_decay'])
                return optimizer, wd_list

            return optimizer

        elif cfg.decay_fly['flymode'] == 'stepall':
            # 递减的 or 递增的 or 不变的 weight_decay
            # wdecay_start > or < or = wdecay_end
            assert cfg.optim_type in ['SGD', 'Adam'], 'Only SGD Adam are support now!'
            wd_start = cfg.decay_fly.get('wd_start', None)
            wd_end = cfg.decay_fly.get('wd_end', None)
            wd_bn = cfg.decay_fly.get('wd_bn', None)
            assert wd_start is not None and wd_end is not None
            a = cfg.decay_fly.get('a', None)
            b = cfg.decay_fly.get('b', None)
            f = cfg.decay_fly.get('f', None)
            assert a is not None and b is not None and f is not None, 'a, b, f must not be None!'

            optimizer = None

            if wd_start != wd_end:
                # wd 逐层递增 or 递减
                assert b != a  # a<b wd递增，a>b wd递减
                fvals = [f(x) for x in np.arange(a, b, (b - a) / len(list(self.parameters())))]
                wdvals = xtils.linear_map(wd_start, wd_end, np.array(fvals))

                for i, (n, p) in enumerate(self.named_parameters()):
                    pgroup = {'params': [p], 'lr': cfg.lr_start, 'momentum': cfg.momentum, 'weight_decay': wdvals[i]}
                    if ('.bn' in n) and wd_bn is not None:  # batchnorm
                        pgroup = {'params': [p], 'lr': cfg.lr_start, 'momentum': cfg.momentum, 'weight_decay': wd_bn}
                    if i == 0:
                        if cfg.optim_type == 'Adam':
                            pgroup.pop('momentum')
                            optimizer = torch.optim.Adam(**pgroup)
                        else:
                            optimizer = torch.optim.SGD(**pgroup)
                    else:
                        if cfg.optim_type == 'Adam':
                            pgroup.pop('momentum')
                            optimizer.add_param_group(pgroup)
                        else:
                            optimizer.add_param_group(pgroup)
            else:
                # 所有层wd相同
                for i, (n, p) in enumerate(self.named_parameters()):
                    pgroup = {'params': [p], 'lr': cfg.lr_start,
                              'momentum': cfg.momentum, 'weight_decay': cfg.weight_decay}
                    if ('.bn' in n) and wd_bn is not None:  # batchnorm
                        pgroup = {'params': [p], 'lr': cfg.lr_start, 'momentum': cfg.momentum, 'weight_decay': wd_bn}
                    if i == 0:
                        if cfg.optim_type == 'Adam':
                            pgroup.pop('momentum')
                            optimizer = torch.optim.Adam(**pgroup)
                        else:
                            optimizer = torch.optim.SGD(**pgroup)
                    else:
                        if cfg.optim_type == 'Adam':
                            pgroup.pop('momentum')
                            optimizer.add_param_group(pgroup)
                        else:
                            optimizer.add_param_group(pgroup)

            if cfg.resume and cfg.resume_optimizer:
                optimizer.load_state_dict(cfg.checkpoint['optimizer'])
            if cfg.resume and not cfg.resume_optimizer:
                cfg.start_iter = 0

            if visual:
                wdvals = []
                for i, pgroup in enumerate(optimizer.param_groups):
                    # print('%s--->' % (i+1), len(pgroup['params']), pgroup['lr'],
                    #       pgroup['weight_decay'], pgroup['params'][0].size())
                    wdvals.append(pgroup['weight_decay'])
                return optimizer, wdvals

            assert optimizer is not None
            return optimizer

    def visual_weight_decay(self, cfg=None, visual=True):
        # 可视化 递增/递减的weight-decay
        from config.configure import Config
        from matplotlib import pyplot as plt
        if cfg is None:
            cfg = Config()
            cfg.decay_fly = {'flymode': ['nofly', 'stepall'][1],
                             'a': 0, 'b': 1, 'f': lambda x: x, 'wd_bn': None,
                             'weight_decay': 0.0001, 'wd_start': 0, 'wd_end': 1}
        optimizer, wdvals = self.init_optimizer(cfg, visual=True)
        plt.plot(wdvals)
        plt.xticks(range(0, len(wdvals) + 1, 5))
        # plt.ylim((0, 0.1))
        plt.grid()
        plt.show()
        print('visual done!')

    def forward(self, x):
        if self.afisok:
            x = self.affine(x)
        x = self.pyramid(x)
        for s in range(self.stages):
            x = getattr(self, 'stage%s' % (s + 1))(x)
        x = self.summary(x)  # x <=> pred
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

    def train_mode(self, ite, cfg):
        # 当迭代次数 ite 超过设定值，开启对应的训练阶段
        for key in sorted(cfg.train_which.keys())[::-1]:
            if ite >= key:
                which = cfg.train_which[key]
                break
        self.set_train_which(part=which)

    def eval_mode(self, ite, cfg):
        # 当迭代次数 ite 超过设定值，开启对应的测试阶段
        for key in sorted(cfg.eval_which.keys())[::-1]:
            if ite >= key:
                which = cfg.eval_which[key]
                break
        self.set_eval_which(part=which)

    def get_train_which(self):
        for k, v in self.train_which_now.items():
            if v:
                return k
        raise ValueError('NO MODULE IS TRAINING NOW!')

    def get_eval_which(self):
        for k, v in self.eval_which_now.items():
            if v:
                return k
        raise ValueError('NO MODULE IS EVALUATE NOW!')

    def set_train_which(self, part='conv+rock'):
        assert part in self.train_which_now, '设定超出可选项范围--> %s' % part
        # if self.train_which_now[part]:
        #     return
        if part == 'conv':
            raise NotImplementedError
        elif part == 'rock':
            raise NotImplementedError
        elif part == 'conv+rock':
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
        # print('model.train_which_now', self.train_which_now)

    def set_eval_which(self, part='conv+rock'):
        assert part in self.eval_which_now, '设定超出可选项范围--> %s' % part
        # if self.eval_which_now[part]:
        #     return
        if part == 'conv':
            raise NotImplementedError
        elif part == 'rock':
            raise NotImplementedError
        elif part == 'conv+rock':
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
        # print('model.eval_which_now', self.eval_which_now)


if __name__ == '__main__':
    import xtils, time

    torch.manual_seed(1)

    # # imageNet
    # exp5 = {'stages': 5, 'branch': 3, 'rock': 'U', 'depth': 16, 'kldloss': False,
    #         'layers': (6, 5, 4, 3, 2), 'blocks': ('D', 'D', 'D', 'S', 'S'), 'slink': ('A', 'A', 'A', 'A', 'A'),
    #         'growth': (12, 12, 12, 12, 12), 'classify': (1, 1, 1, 1, 1),
    #         'expand': (1 * 16, 2 * 16, 4 * 16, 8 * 16), 'dfunc': ('O', 'D', 'O', 'A'),
    #         'fcboost': 'none', 'nclass': 1000, 'summer': 'merge',
    #         'last_branch': 1, 'last_down': True, 'last_dfuc': 'A', 'last_expand': 15,
    #         'afisok': False, 'afkeys': ('af1', 'af2'), 'convon': True,
    #         'convlayers': 1, 'convdepth': 4}
    #
    # exp4 = {'stages': 4, 'branch': 3, 'rock': 'U', 'depth': 16, 'kldloss': False,
    #         'layers': (6, 5, 4, 3), 'blocks': ('D', 'D', 'D', 'S'), 'slink': ('A', 'A', 'A', 'A'),
    #         'growth': (10, 15, 20, 30), 'classify': (0, 0, 1, 1), 'expand': (10, 20, 30),
    #         'dfunc': ('O', 'D', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'split',
    #         'last_branch': 2, 'last_down': False, 'last_dfuc': 'A', 'last_expand': 15,
    #         'afisok': True, 'afkeys': ('af1', 'af2'), 'convon': True,
    #         'convlayers': 1, 'convdepth': 4}
    #
    # exp3 = {'stages': 3, 'branch': 3, 'rock': 'U', 'depth': 16, 'kldloss': False,
    #         'layers': (3, 3, 1), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
    #         'growth': (5, 5, 5), 'classify': (0, 0, 0), 'expand': (1 * 16, 2 * 16),
    #         'dfunc': ('D', 'O'), 'fcboost': 'none', 'nclass': 1000, 'summer': 'merge',
    #         'last_branch': 1, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 256,
    #         'afisok': False, 'afkeys': ('af1', 'af2'), 'convon': True,
    #         'convlayers': 2, 'convdepth': 4}
    #
    # model = ScaleNet(**exp4)
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
    exp4 = {'stages': 4, 'depth': 16, 'branch': 3, 'rock': 'U', 'kldloss': False,
            'layers': (6, 5, 4, 3), 'blocks': ('D', 'D', 'S', 'S'), 'slink': ('A', 'A', 'A', 'A'),
            'growth': (3, 4, 5, 6), 'classify': (1, 1, 1, 1), 'dfunc': ('O', 'O', 'D'), 'expand': (10, 20, 40),
            'fcboost': 'none', 'nclass': 10, 'summer': 'merge',
            'last_branch': 2, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 5,
            'afisok': True, 'afkeys': ('af1', 'af2'), 'convon': True,
            'convlayers': 1, 'convdepth': 4}

    exp3 = {'stages': 3, 'depth': 16, 'branch': 3, 'rock': 'R', 'kldloss': False,
            'layers': (3, 3, 3), 'blocks': ('D', 'D', 'D'), 'slink': ('A', 'A', 'A'),
            'growth': (2, 3, 5), 'classify': (0, 0, 0), 'expand': (1 * 16, 2 * 16),
            'dfunc': ('A', 'O'), 'fcboost': 'none', 'nclass': 10, 'summer': 'split',
            'last_branch': 3, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 30,
            'afisok': True, 'afkeys': ('af1', 'af2'), 'convon': True,
            'convlayers': 1, 'convdepth': 4}

    exp2 = {'stages': 2, 'depth': 16, 'branch': 3, 'rock': 'U', 'kldloss': False,
            'layers': (2, 2), 'blocks': ('D', 'S'), 'slink': ('A', 'A'), 'growth': (3, 5),
            'classify': (0, 0), 'dfunc': ('O',), 'expand': (16,),
            'fcboost': 'none', 'nclass': 10, 'summer': 'split',
            'last_branch': 2, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 111,
            'afisok': False, 'afkeys': ('af1', 'af2'), 'convon': True,
            'convlayers': 1, 'convdepth': 4}

    exp1 = {'stages': 1, 'depth': 16, 'branch': 3, 'rock': 'R', 'kldloss': False,
            'layers': (300,), 'blocks': ('D',), 'slink': ('A',), 'growth': (4,),
            'classify': (0,), 'dfunc': (), 'expand': (),
            'fcboost': 'none', 'nclass': 10, 'summer': 'convt',
            'last_branch': 3, 'last_down': True, 'last_dfuc': 'D', 'last_expand': 32,
            'afisok': False}

    model = ScaleNet(**exp3)
    print('\n', model, '\n')
    # model.set_train_which(part=['conv+rock', 'xfc+boost', 'xfc-only', 'boost-only'][1])
    model.set_eval_which(part=['conv+rock', 'conv+rock+xfc', 'conv+rock+boost', 'conv+rock+xfc+boost'][3])
    # utils.tensorboard_add_model(model, x)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
    x = torch.randn(4, 3, 32, 32)
    tic, toc = time.time(), 3
    y = [model(x) for _ in range(toc)][0]
    toc = (time.time() - tic) / toc
    print('有效分类支路：', len(y), '\t共有blocks：', sum(model.layers), '\t处理时间: %.5f 秒' % toc)
    print(len(y), sum(model.layers), ':', [(yy.shape, yy.max(1)) for yy in y if yy is not None])

    # 查看模型的权值衰减曲线
    # from config.configure import Config
    #
    # cfg = Config()
    # cfg.decay_fly = {'flymode': ['nofly', 'stepall'][1],
    #                  'a': 0, 'b': 1, 'f': utils.Curves(3).func2, 'wd_bn': None,
    #                  'weight_decay': 0.0001, 'wd_start': 0.0001, 'wd_end': 0.0005}
    # model.visual_weight_decay(cfg=cfg, visual=True)

    # # 查看模型的可学习参数列表
    # for n, p in model.named_parameters():
    #     print('---->', n, '\t', p.size())
