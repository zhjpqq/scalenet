# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/6/9 12:17'

"""
Multi-Resolution Net  2019-6-20  20:58
"""
from collections import OrderedDict
import math
import torch
from torch import nn
from torch.nn import functional as F
from xmodules.classifier import AdaPoolView, ReturnX
import xtils
from xtils import GCU


class HSigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, self.inplace) / 6
        return out


def hsigmoid(x):
    out = F.relu6(x + 3, inplace=True) / 6
    return out


class SeModule(nn.Module):
    _ActiveFuc = {'relu': nn.ReLU, 'hsig': HSigmoid, 'relu6': nn.ReLU6}

    def __init__(self, indepth, reduction=4, active='hsig'):
        super(SeModule, self).__init__()
        """
         Squeeze-> x ->Expand, x => [batch, channels, 1, 1]
        """
        assert active in self._ActiveFuc.keys()
        Active = self._ActiveFuc[active]

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(indepth, indepth // reduction, 1, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(indepth // reduction, indepth, 1, 1, 0, bias=False),
            Active(inplace=True)
        )

    def forward(self, x):
        return x * self.se(x)


class BranchDownsize(nn.Module):
    def __init__(self, factor=None, size=None, mode='nearest', align_corners=False):
        super(BranchDownsize, self).__init__()
        self.downsize = nn.Upsample(size, factor, mode, align_corners)

    def forward(self, x):
        if isinstance(x, (tuple, list)):
            x3, x2, x1 = x
            x3 = self.downsize(x3)
            x2 = self.downsize(x2)
            x1 = self.downsize(x1)
            x = (x3, x2, x1)
        else:
            x = self.downsize(x)
        # print('---->', x[0].size())
        return x


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training or p == 0:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)  # uniform [0,1)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class PreProc(nn.Module):

    def __init__(self, indepth=3, outdepth=16, outnums=1, stride_dilate='1/4-1'):
        super(PreProc, self).__init__()
        assert outnums in [1, 2, 3]
        assert stride_dilate in ['1/2-1', '1/2-2', '1/4-1', '1/4-2']
        stride, dilate = stride_dilate.split('-')
        self.stride = stride
        self.outnums = outnums

        # stride = 1/2
        if dilate == '1':
            self.conv1 = nn.Conv2d(indepth, outdepth, 3, 2, 1, dilation=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(indepth, outdepth, 3, 2, 2, dilation=2, bias=False)
        self.bn1 = nn.BatchNorm2d(outdepth)
        self.act1 = nn.ReLU()

        if stride == '1/4':
            if dilate == '1':
                self.conv2 = nn.Conv2d(outdepth, outdepth, 3, 2, 1, dilation=1, bias=False)
            else:
                self.conv2 = nn.Conv2d(outdepth, outdepth, 3, 2, 2, dilation=2, bias=False)
            self.bn2 = nn.BatchNorm2d(outdepth)
            self.act2 = nn.ReLU()

    def forward(self, x):
        # stride = '1/2'
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.stride == '1/4':
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.act2(x)
        if self.outnums == 1:
            return x
        elif self.outnums == 2:
            return x, None
        elif self.outnums == 3:
            return x, None, None


class MoBlock(nn.Module):
    _ActiveFuc = {'relu': nn.ReLU, 'hsig': HSigmoid, 'relu6': nn.ReLU6}

    def __init__(self, indepth, outdepth, growth, pre_ind_grow, ksp='3.1.1', pre_ksp_half=False, groups='auto',
                 skgroups='gcu', active='relu', dropout=0.0, isse=1, seactive='hsig', first=False, idx=1):
        """
        - indepth: 当前block的输入通道数.
        - outdepth: 当前block的输出通道数.
        - growth: 当前block的通道增长数.
        - pre_ind_grow: 上一个block内, 输入通道数indepth + 通道增长数growth 的值.
        - ksp: kernel_size, stride, padding in Depth-Wise Convolution. cur_ksp_half, 当前block内是否将特征图尺寸减半.
        - pre_ksp_half: 上一个block内, 是否对特征图进行了尺寸减半.
        - groups: groups 值 in Depth-Wise Convolution.
        - skgroups: 所有 skip 连接的groups值, skip-groups
        - active:
        - dropout:
        - isse: 是否包含SeModule. =1: no-SeModule ; >1: has-SeModule(reduction=isse) 默认值4
        - first:
        - idx:
        """
        super(MoBlock, self).__init__()
        Active = self._ActiveFuc[active]
        ksp = [int(x) for x in ksp.split(sep='.')]
        assert len(ksp) == 3
        cur_ksp_half = bool(ksp[1] == 2)
        self.ksp = ksp
        assert dropout * (0.5 - dropout) >= 0, '<dropout> must be in [0, 0.5], but get %s .' % dropout
        self.dropout = dropout
        assert isse >= 1 and isinstance(isse, int), '<isse> must be a int >=1, but get %s .' % isse
        self.isse = isse
        self.first = first
        self.idx = idx

        if groups == '1x':
            groups = 1
        elif groups == 'auto':
            groups = indepth + growth

        curr_ind_grow = indepth + growth

        self.conv1 = nn.Conv2d(indepth, curr_ind_grow, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(curr_ind_grow, eps=1e-05, momentum=0.1)
        self.act1 = Active(inplace=True)

        # depth-wise conv
        self.conv2 = nn.Conv2d(curr_ind_grow, curr_ind_grow, ksp[0], ksp[1], ksp[2], groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(curr_ind_grow, eps=1e-05, momentum=0.1)
        self.act2 = Active(inplace=True)
        self.segate2 = [nn.Sequential(), SeModule(curr_ind_grow, isse, active=seactive)][isse != 1]

        # 计算 skip1 & skip2
        if self.first:
            self.skip1 = nn.Sequential()
            self.skip2 = nn.Sequential()
        else:
            if curr_ind_grow == pre_ind_grow:
                skip_group = GCU(pre_ind_grow, curr_ind_grow) if skgroups == 'gcu' else 1
                if not pre_ksp_half:
                    # print('init---> idx %s .' % idx)
                    self.skip1 = nn.Sequential()
                else:
                    skip1_ksp = (2, 2, 0)
                    self.skip1 = nn.Sequential(
                        nn.Conv2d(pre_ind_grow, curr_ind_grow, skip1_ksp[0], skip1_ksp[1], skip1_ksp[2],
                                  bias=False, groups=skip_group),
                        nn.BatchNorm2d(curr_ind_grow, eps=1e-05, momentum=0.1))
                if not cur_ksp_half:
                    self.skip2 = nn.Sequential()
                else:
                    skip2_ksp = (2, 2, 0)
                    self.skip2 = nn.Sequential(
                        nn.Conv2d(pre_ind_grow, curr_ind_grow, skip2_ksp[0], skip2_ksp[1], skip2_ksp[2],
                                  bias=False, groups=skip_group),
                        nn.BatchNorm2d(curr_ind_grow, eps=1e-05, momentum=0.1))

            elif curr_ind_grow != pre_ind_grow:
                skip_group = GCU(pre_ind_grow, curr_ind_grow) if skgroups == 'gcu' else 1
                skip1_ksp = (2, 2, 0) if pre_ksp_half else (1, 1, 0)
                skip2_ksp = (2, 2, 0) if cur_ksp_half else (1, 1, 0)
                self.skip1 = nn.Sequential(
                    nn.Conv2d(pre_ind_grow, curr_ind_grow, skip1_ksp[0], skip1_ksp[1], skip1_ksp[2],
                              bias=False, groups=skip_group),
                    nn.BatchNorm2d(curr_ind_grow, eps=1e-05, momentum=0.1))
                self.skip2 = nn.Sequential(nn.Conv2d(pre_ind_grow, curr_ind_grow, skip2_ksp[0], skip2_ksp[1],
                                                     skip2_ksp[2], bias=False, groups=skip_group),
                                           nn.BatchNorm2d(curr_ind_grow, eps=1e-05, momentum=0.1))

        # 计算skip3
        if outdepth == indepth and not cur_ksp_half:
            self.skip3 = nn.Sequential()
        else:
            skip3_ksp = (2, 2, 0) if cur_ksp_half else (1, 1, 0)
            skip_group = GCU(indepth, outdepth) if skgroups == 'gcu' else 1
            self.skip3 = nn.Sequential(nn.Conv2d(indepth, outdepth, skip3_ksp[0], skip3_ksp[1], skip3_ksp[2],
                                                 bias=False, groups=skip_group),
                                       nn.BatchNorm2d(outdepth, eps=1e-05, momentum=0.1))

        # point-wise conv
        self.conv3 = nn.Conv2d(curr_ind_grow, outdepth, kernel_size=1, stride=1, padding=0, groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outdepth, eps=1e-05, momentum=0.1)
        self.act3 = Active(inplace=True)
        self.drop3 = nn.Dropout2d(p=dropout, inplace=False)

    def forward(self, x):
        # print('\n-----> %s' % self.idx)
        assert isinstance(x, (list, tuple)) and len(x) == 3
        x3, x2, x1 = x  # c3, c2, c1
        if self.first:
            c1 = self.act1(self.bn1(self.conv1(x3)))
            c2 = self.act2(self.bn2(self.conv2(c1)))
            c2 = self.segate2(c2)
            c3 = self.act3(self.bn3(self.conv3(c2)))
            c3 = self.drop3(c3)
            c3 = c3 + self.skip3(x3)
            # return c3, c2, c1
        else:
            c1 = self.act1(self.bn1(self.conv1(x3)))
            c2 = self.act2(self.bn2(self.conv2(c1 + self.skip1(x1))))
            c2 = self.segate2(c2)
            c3 = self.act3(self.bn3(self.conv3(c2 + self.skip2(x2))))
            c3 = self.drop3(c3)
            c3 = c3 + self.skip3(x3)
            # return c3, c2, c1
        # xtils.print_size(c3)
        return c3, c2, c1


class Clssifier(nn.Module):
    _ActiveFuc = {'relu': nn.ReLU, 'hsig': HSigmoid, 'relu6': nn.ReLU6}

    def __init__(self, indepth, middepth=0, outdepth=1000, dropout=(0,), active='relu'):
        super(Clssifier, self).__init__()
        assert isinstance(dropout, (list, tuple))

        self.dropout = dropout
        self.middepth = middepth

        if middepth == 0:
            assert len(self.dropout) >= 1
            self.drop = nn.Dropout(p=self.dropout[0], inplace=False)
            self.fc = nn.Linear(indepth, outdepth)
        elif middepth > 0:
            assert len(self.dropout) == 2
            self.drop1 = nn.Dropout(p=self.dropout[0], inplace=False)
            self.fc1 = nn.Linear(indepth, middepth)
            self.drop2 = nn.Dropout(p=self.dropout[1], inplace=False)
            self.fc2 = nn.Linear(middepth, outdepth)

    def forward(self, x):
        if self.middepth == 0:
            x = self.drop(x)
            x = self.fc(x)
        elif self.middepth > 0:
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.drop2(x)
            x = self.fc2(x)
        return x


class ConcatSummary(nn.Module):
    """
        汇总多个xfc的输出到一个fc; 或 汇总多个squeeze的输出到一个fc.
    """

    def __init__(self, indepth, middepth=0, outdepth=1000, dropout=(0, 0), active='relu', with_fc=True):
        """
         - indepth:  对所有输入x, 进行拼接后的输入通道数
         - middepth: fc 层的中间隐藏层，=0 则无隐藏层
         - outdepth: 输出通道数 => nlabels
         - dropout: fc 层的辍学率
         - active: fc 层的激活函数
         - withfc: when indepth==outdepth， False => 不添加fc层，直接输出拼接向量进行分类.
        """
        super(ConcatSummary, self).__init__()
        if not with_fc:
            assert indepth == outdepth, '<withfc> can be False only under <indepth>==<outdepth>.'
            self.classifier = nn.Sequential()
        else:
            self.classifier = Clssifier(indepth, middepth, outdepth, dropout, active)

    def forward(self, x):
        # assert isinstance(x, (tuple, list))
        x = torch.cat(x, dim=1)
        x = self.classifier(x)
        return x

    def __repr__(self):
        strme = '(\n  (concat): torch.cat(dim=1)()\n' + \
                '  (classifier): ' + self.classifier.__repr__() + '\n)'
        return strme


class PollSummary(nn.Module):
    """
        汇总多个xfc的输出, 进行投票 ==> 平均投票法 & 最大投票法.
        投票前可选择是否先进行归一化 F.softmax() or F.normalize().
    """

    def __init__(self, method='avg', isnorm='none'):
        super(PollSummary, self).__init__()
        assert isnorm in ['none', 'softmax', 'normal', 'minmax']
        self.isnorm = isnorm
        self.method = method

        if isnorm == 'none':
            self.normalize = None
        elif isnorm == 'softmax':
            self.normalize = F.softmax
        elif isnorm == 'normal':
            self.normalize = F.normalize
        elif isnorm == 'minmax':
            self.normalize = self.minmax
        else:
            raise NotImplementedError

        if method == 'avg':
            self.reduce = torch.mean
        elif method == 'max':
            self.reduce = torch.max
        elif method == 'sum':
            self.reduce = torch.sum
        else:
            raise NotImplementedError

    def minmax(self, x, dim=-1):
        assert x.ndimension() == 2
        min_x, max_x = x.min(dim)[0], x.max(dim)[0]
        factor = (max_x - min_x).unsqueeze(dim)
        x = (x - min_x.unsqueeze(dim)) / factor
        return x

    def forward(self, x):
        assert isinstance(x, (tuple, list))
        if self.isnorm != 'none':
            x = [self.normalize(z, dim=-1) for z in x]
        x = [z.unsqueeze_(dim=-1) for z in x]
        x = torch.cat(x, dim=-1)
        x = self.reduce(x, dim=-1)
        return x

    def __repr__(self):
        string = self.__class__.__name__ + \
                 '(method={}, isnorm={})'.format(self.method, self.isnorm)
        return string


class MultiScaleNet(nn.Module):

    def __init__(self, depth, growth, blocks=9,
                 expand=((2, 30), (3, 50), (5, 70)),
                 exksp=((2, '3.2.1'), (3, '3.2.1'), (5, '3.2.1')),
                 exisse='ft-0:7', seactive='hsig', groups='auto', skgroups='gcu', prestride='1/4-2',
                 conv_drop='all-0.2', conv_act='relu', dataset='imagenet',
                 summar='concat', sfc_poll=('avg', 'minmax'),
                 sfc_with=True, sfc_indep=256, sfc_middep=512, sfc_drop=(0.3, 0.3), sfc_active='relu',
                 head_key_cfg={'2-4-4': {}, '3-8-8': {}, '3-8-12': {}, '9-32-32': {}}):
        """
         - depth:
         - growth:
         - blocks:
         - expand:
         - exksp:
         - exisse:
         - groups:
         - skgroups:
         - prestride:
         - conv_drop: 卷积骨架层中的辍学率，'all-0.2': 全部block都为0.2;
                      'bet-0.1-0.2': between 0.1 & 0.2, 逐block递增从0.1到0.2
         - conv_act: 卷积骨架层网络的激活函数.
         - dataset:

         - summar:     Summary 中执行Summary操作的方式.
         - sfc_poll:   Summary 中按投票方式操作时，具体的投票规则. only useful when summar == 'fcpoll'.
         - sfc_with:   Summary 中安拼接方式操作时，当输入/输出通道数相等，是否还需要fc层. 若不需要fc即直接squeeze出lables.
         - sfc_indep:  Summary 中Linear层的输入通道数.
         - sfc_middep: Summary 中Linear层的中间隐藏层，为0则无隐藏层.
         - sfc_drop:   Summary 中Linear层的辍学率.
         - sfc_active: Summary 中Linear层的激活函数.
         - head_key_cfg: {pos-scale-key: cfg-dict, ...} => {'block_id-scale1-scale2': {}, ...}.
        """

        super(MultiScaleNet, self).__init__()
        assert groups in ['auto', '1x'], '<groups> must be in ["1x", "auto"], but get %s .' % groups
        assert skgroups in ['gcu', '1x'], '<skgroups> must be in ["gcu", "1x"], but get %s.' % skgroups
        assert (sfc_middep == 0 and len(sfc_drop) >= 1) or (sfc_middep != 0 and len(sfc_drop) == 2)
        assert conv_drop[:3] in ['all', 'bet'], '<conv_drop> must be "all-a" or "bet-a-b".'
        assert summar in ['concat', 'independ', 'fcpoll']

        if dataset == 'imagenet':
            nlabels = 1000
        elif dataset == 'cifar10':
            nlabels = 10
        elif dataset == 'cifar100':
            nlabels = 100
        else:
            raise NotImplementedError('Unknown <dataset: %s>' % dataset)
        self.nlabels = nlabels
        self.summar = summar

        expand = OrderedDict(expand)
        exksp = OrderedDict(exksp)
        exisse = self.get_semodule(exisse, blocks)
        conv_drop = self.get_dropout(conv_drop, blocks)
        sfc_drop = sfc_drop

        # 预处理
        self.preproc = PreProc(indepth=3, outdepth=depth, outnums=3, stride_dilate=prestride)

        # 骨架特征提取 backbone feature-extractor
        self.features = nn.ModuleList([])
        indepth = depth
        pre_ind_grow = 0
        pre_ksp_half = False
        for i in range(1, blocks + 1):
            outdepth = indepth + expand.get(i, 0)
            curr_ksp = exksp.get(i, '3.1.1')
            curr_se = exisse.get(i, 1)
            curr_drop = conv_drop.get(i, 0)
            block = MoBlock(indepth, outdepth, growth, pre_ind_grow, ksp=curr_ksp, pre_ksp_half=pre_ksp_half,
                            groups=groups, skgroups=skgroups, active=conv_act, isse=curr_se, seactive=seactive,
                            dropout=curr_drop, first=bool(i == 1), idx=i)
            pre_ind_grow = indepth + growth
            pre_ksp_half = curr_ksp.split('.')[1] == '2'
            indepth = outdepth
            self.features.add_module('mo_%s' % i, block)

        # 添加分类头 fc-head.
        # 方法：用key衔接骨架fmap和分类head (key->pos.fmap & key->model.head)
        # 运行前将head以key为属性注册到网络中, head=model.key.
        # 运行前将fmap以{key, pos}进行标记, 运行时缓存对应pos上的fmap即可
        self.bone_feat_maps = {}  # 用于缓存骨架特征图    backbone-fmaps
        self.head_key_pos = {}  # 用于关联特征图和分类头 fcHead-pose
        for key, cfg in head_key_cfg.items():
            print('**** ->', key)
            head = self.get_fc_head(key=key, **cfg)
            setattr(self, 'head-%s' % key, head)  # 注册head
            moid = key.split('-')[0]  # 标记fmap
            self.head_key_pos.setdefault('head-%s' % key, moid)
        print(self.head_key_pos)

        # 汇总各个分类头的输出
        if summar == 'independ':
            self.summary = nn.Sequential()
        elif summar == 'concat':
            self.summary = ConcatSummary(sfc_indep, sfc_middep, nlabels, sfc_drop, sfc_active, sfc_with)
        elif summar == 'fcpoll':
            self.summary = PollSummary(method=sfc_poll[0], isnorm=sfc_poll[1])

        self._init_params()

    def get_semodule(self, exisse, blocks):
        """
            以多种方式添加SeModule到MoBlock上，默认reduction=4
        """
        if exisse == () or isinstance(exisse[0], tuple):
            # 指定要添加的block-id 和 reduce-val
            exisse = OrderedDict(exisse)
        elif isinstance(exisse, (list, tuple)):
            # 指定要添加的block-id
            exisse_dict = OrderedDict()
            for id in exisse:
                exisse_dict.setdefault(id, 4)
            exisse = exisse_dict
        elif exisse == 'all':
            # 所有block都添加SeModule
            exisse = OrderedDict()
            for i in range(1, blocks + 1):
                exisse.setdefault(i, 4)
        elif exisse.startswith('ft-'):
            # 指定block添加SeModule
            # ft-3:5-7:10-15:18, from 3-5, 7-10, 15-18.
            # ft-0:0 不添加
            ft = exisse[3:].split('-')
            exisse = []
            for idx in ft:
                idx = [int(x) for x in idx.split(':')]
                for i in range(idx[0], idx[1] + 1):
                    exisse.append((i, 4))
            exisse = OrderedDict(exisse)
        return exisse

    def get_dropout(self, conv_drop='', blocks=0):
        if conv_drop.startswith('all-'):
            dropout = float(conv_drop.split('-')[1])
            conv_drop = OrderedDict()
            for i in range(1, blocks + 1):
                conv_drop.setdefault(i, dropout)
            return conv_drop
        elif conv_drop.startswith('bet-'):
            import numpy as np
            dropout = conv_drop.split('-')
            dropout = [float(dropout[1]), float(dropout[2])]
            dropout = np.arange(dropout[0], dropout[1],
                                (dropout[1] - dropout[0]) / blocks)
            conv_drop = OrderedDict()
            for i in range(1, blocks + 1):
                conv_drop.setdefault(i, round(dropout[i - 1], 3))
            return conv_drop

    def get_fc_head(self, key, blocks, indepth, expand, exksp, exisse, growth, pre_ind_grow,
                    pre_ksp_half, groups='auto', skgroups='gcu', conv_drop='all-0', conv_active='relu',
                    seactive='hsig', fc_middep=0, fc_drop=0, fc_active='relu', with_fc=True):
        # fc_head内的各MoBlock中不进行尺寸削减stride==1, 以固定的分辨率对接到fc层.

        # key="moid-scale1-scale2".
        # moid=> 使用骨架中的第moid个MoBlock的特征图作为当前fchead的输入.
        # scale1=> 骨架挂载点(moid)处的MoBlock相对原图的尺寸系数.
        # scale2=> 当前fc_head内，MoBlock相对原图的尺寸系数.
        # scale1=> scale2, 当需要更精细的分辨率时，将输入特征图的分辨率(scale1)降采样到当前head的分辨率(scale2).
        # (moid=5, scale1=4, scale2=6): 在骨架中第5个MoBlock所产生的1/4特征图上，接一个1/6的分支头，需要先对其特征图降采样(4/6).

        moid, scale1, scale2 = [float(x) for x in key.split('-')]
        expand = OrderedDict(expand)
        exksp = OrderedDict(exksp)
        exisse = self.get_semodule(exisse, blocks)
        conv_drop = self.get_dropout(conv_drop, blocks)

        head = nn.ModuleList([])
        if scale2 != scale1:
            head.add_module('downsize',
                            BranchDownsize(factor=round(scale1 / scale2, 7), mode='bilinear', align_corners=True))
        for i in range(1, blocks + 1):
            outdepth = indepth + expand.get(i, 0)
            curr_ksp = exksp.get(i, '3.1.1')
            curr_se = exisse.get(i, 1)
            curr_drop = conv_drop.get(i, 0)
            block = MoBlock(indepth, outdepth, growth, pre_ind_grow, ksp=curr_ksp, pre_ksp_half=pre_ksp_half,
                            groups=groups, skgroups=skgroups, active=conv_active, isse=curr_se, seactive=seactive,
                            dropout=curr_drop, first=bool(i == 1), idx=i)
            pre_ind_grow = indepth + growth
            pre_ksp_half = curr_ksp.split('.')[1] == '2'
            indepth = outdepth
            head.add_module('mo_%s' % i, block)

        head.add_module('squeeze', AdaPoolView(pool='avg', dim=-1, which=0))
        if with_fc:
            head.add_module('classifier', Clssifier(indepth, fc_middep, self.nlabels, fc_drop, fc_active))
        return head

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                if getattr(m, 'bias', None) is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight)
                # nn.init.normal_(m.weight, mean=0, std=1)
                # nn.init.xavier_normal_(m.weight, gain=1)
                if getattr(m, 'bias', None) is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        bone_feat_maps = {}
        x = self.preproc(x)
        # print('pre->', x[0].size())
        for id, mo in enumerate(self.features):
            x = mo(x)
            if str(id + 1) in self.head_key_pos.values():
                bone_feat_maps.setdefault(str(id + 1), x)
            # print('id->%s' % id, x[0].size())

        logits = []
        for key, pos in self.head_key_pos.items():
            # print('*** --> ', key)
            head = getattr(self, key, None)
            x = bone_feat_maps.get(pos, None)
            assert (x is not None) and (head is not None)
            for id, mo in enumerate(head):
                x = mo(x)
            logits.append(x)
        logits = self.summary(logits)
        return logits


if __name__ == '__main__':
    # 主要特性如下
    # 1. growth ==> channel-growth可正/可负
    # 2. expand & expksp ==> feature-size & feature-channels 之间 可同步调节 & 可异步调节
    # 3. expksp ==> 任意Bottleneck-Block的核尺寸可任意指定，'3.1.1' or '3.2.1' or '5.2.2'
    # 4. exisse ==> 任意MoBlock上可附加一个SeModule.

    # get_fc_head(self, key, blocks, indepth, expand, exksp, exisse, growth, pre_ind_grow,
    #             pre_ksp_half, groups, skgroups, conv_drop, fc_middep, fc_drop)
    #
    # f-size 减小5次
    with_fc = False
    net1_hkc = {
        '4-4-4': dict(blocks=3, indepth=90, growth=5, pre_ind_grow=96,
                      expand=((1, 10), (2, 10), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                      groups='auto', skgroups='gcu', conv_drop='all-0.1', conv_active='relu',
                      fc_middep=512, fc_drop=(0.2, 0.2), fc_active='relu', with_fc=with_fc),
        # '4-4-5.3': dict(blocks=3, indepth=90, growth=5, pre_ind_grow=96,
        #                 expand=((1, 10), (2, 10), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
        #                 groups='auto', skgroups='gcu', conv_drop='all-0.1', conv_active='relu',
        #                 fc_middep=512, fc_drop=(0.2, 0.2), fc_active='relu', with_fc=with_fc),
        '4-4-4.6': dict(blocks=3, indepth=90, growth=5, pre_ind_grow=96,
                        expand=((1, 10), (2, 15), (3, 15)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                        groups='auto', skgroups='gcu', conv_drop='all-0.1', conv_active='relu',
                        fc_middep=512, fc_drop=(0.2, 0.2), fc_active='relu', with_fc=with_fc),
        '4-4-6': dict(blocks=3, indepth=90, growth=5, pre_ind_grow=96,
                      expand=((1, 15), (2, 15), (3, 20)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                      groups='auto', skgroups='gcu', conv_drop='all-0.1', conv_active='relu',
                      fc_middep=512, fc_drop=(0.2, 0.2), fc_active='relu', with_fc=with_fc),
        '8-8-8': dict(blocks=3, indepth=150, growth=5, pre_ind_grow=156,
                      expand=((1, 0), (2, 0), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                      groups='auto', skgroups='gcu', conv_drop='all-0.1', conv_active='relu6',
                      fc_middep=512, fc_drop=(0.3, 0.3), fc_active='relu', with_fc=with_fc),
        '9-8-10.5': dict(blocks=3, indepth=150, growth=5, pre_ind_grow=156,
                         expand=((1, 10), (2, 20), (3, 20)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                         groups='auto', skgroups='gcu', conv_drop='all-0.1', conv_active='hsig',
                         fc_middep=512, fc_drop=(0.3, 0.3), fc_active='relu', with_fc=with_fc),
        '13-16-16': dict(blocks=3, indepth=210, growth=5, pre_ind_grow=216,
                         expand=((1, 10), (2, 10), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                         groups='auto', skgroups='gcu', conv_drop='all-0.1', conv_active='hsig',
                         fc_middep=512, fc_drop=(0.4, 0.4), fc_active='relu', with_fc=with_fc),
        '14-16-22': dict(blocks=3, indepth=210, growth=5, pre_ind_grow=216,
                         expand=((1, 10), (2, 20), (3, 40)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                         groups='auto', skgroups='gcu', conv_drop='all-0.1', conv_active='hsig',
                         fc_middep=512, fc_drop=(0.4, 0.4), fc_active='relu', with_fc=with_fc),
        '20-32-32': dict(blocks=0, indepth=470, growth=5, pre_ind_grow=476,
                         expand=(), exksp=(), exisse='ft-1:3', pre_ksp_half=False,
                         groups='auto', skgroups='gcu', conv_drop='all-0.1', conv_active='relu6',
                         fc_middep=1024, fc_drop=(0.5, 0.5), fc_active='relu', with_fc=with_fc)
    }

    net1 = OrderedDict(depth=30, growth=5, blocks=20, expand=((2, 60), (5, 60), (10, 60), (15, 290), (18, 0)),
                       exksp=((2, '3.2.1'), (5, '3.2.1'), (10, '3.2.1'), (15, '3.2.1')), exisse='ft-0:1',
                       groups='auto', skgroups='gcu', prestride='1/2-1', conv_drop='all-0.1', conv_act='relu',
                       summar='concat', sfc_with=True, sfc_poll=('avg', 'minmax'), sfc_indep=1770, sfc_middep=2014,
                       sfc_drop=(0.3, 0.3), sfc_active='relu', seactive='hsig', head_key_cfg=net1_hkc)

    net2_hkc = {
        '20-32-32': dict(blocks=0, indepth=530, growth=5, pre_ind_grow=250,
                         expand=(), exksp=(), exisse='ft-1:3', pre_ksp_half=False,
                         groups='auto', skgroups='gcu', conv_drop='all-0.1', conv_active='relu',
                         fc_middep=0, fc_drop=(0.5, 0.5), fc_active='relu', seactive='hsig', with_fc=True)
    }  # indepth = 4*60 + 290 = 530 ; pre_ind_grow = 4*60+10=250 ;
    net2 = OrderedDict(depth=60, growth=10, blocks=20, expand=((2, 60), (5, 60), (10, 60), (15, 290), (18, 0)),
                       exksp=((2, '3.2.1'), (5, '3.2.1'), (10, '3.2.1'), (15, '3.2.1')), exisse='ft-1:20',
                       groups='auto', skgroups='gcu', prestride='1/2-2', conv_drop='all-0.1', conv_act='relu',
                       summar='independ', sfc_with=True, sfc_poll=('avg', 'minmax'), sfc_indep=1770, sfc_middep=2014,
                       sfc_drop=(0.3, 0.3), sfc_active='relu', seactive='relu', head_key_cfg=net2_hkc)

    model = MultiScaleNet(**net2)
    model.eval()
    print(model)

    # x = torch.randn(1, 3, 224, 224)
    # z = model(x)
    # print('z-->', len(z), '\n')

    xtils.calculate_layers_num(model)
    xtils.calculate_FLOPs_scale(model, input_size=224, use_gpu=False, multiply_adds=True)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_time_cost(model, insize=224, toc=1, pritout=True, use_gpu=False)
