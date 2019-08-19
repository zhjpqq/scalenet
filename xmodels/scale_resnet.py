# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2018/12/15 12:17'
import torch.nn as nn
import math, os
import torch.utils.model_zoo as model_zoo
from torchvision.models.resnet import model_urls
import torch
import xtils
from xmodules.classifier import AdaPoolView, ReturnX
from collections import OrderedDict


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreProc(nn.Module):
    def __init__(self, indep=3, outdep=64):
        super(PreProc, self).__init__()
        self.conv1 = nn.Conv2d(indep, outdep, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(outdep)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class Features(nn.ModuleList):

    def __init__(self):
        super(Features, self).__init__()


class MoHead(nn.ModuleList):

    def __init__(self, active_me=True, active_fc=True, with_fc=True, main_aux='main'):
        """
         - active_me: 是否激活整个head, MoBlock + fc.
         - active_fc: 是否去激活Head中的fc，否则截断计算，只输出squeeze()后的特征向量.
         - with_fc:   head头中是否带有fc，否则直接输出squeeze()后的特征向量. must be False when with_fc=False.
         - main_aux:  是否是主分类头 or 辅分类头.
        """
        super(MoHead, self).__init__()
        self.active_me = active_me
        self.active_fc = active_fc
        self.with_fc = with_fc
        self.main_aux = main_aux
        if not with_fc:
            assert not active_fc, '不存在fc时，无法截断fc.'
        else:
            assert active_fc or not active_fc, 'head中有fc时，可用/可不用fc.'


class AdaConvView(nn.Module):
    _ActiveFuc = {'relu': nn.ReLU, 'relu6': nn.ReLU6, 'none': ReturnX}

    def __init__(self, indepth, outdepth, ksize=1, stride=1, padding=0,
                 dilation=1, groups='gcu||1', active='relu', isview=True, which=0):
        """
            用conv_KxK提升fmap通道，并转换为特征向量fvector。ksize为当前特征图的平面尺寸。
            self.view ==> AdaPoolView(). 比如224x224训练，ksize=28x28，可获得1x1的特征图;
            但当改变输入尺寸为320x320时，ksize却不能随之而变为40x40，仍然是固定的28x28，
            因而获得的fmap不是1x1，需要AdaPoolView()。
        """
        super(AdaConvView, self).__init__()
        self.which = which
        if groups == 'gcu':
            groups = xtils.GCU(indepth, outdepth)
        self.conv = nn.Conv2d(indepth, outdepth, ksize, stride, padding, dilation, groups, bias=False)
        # can not work on 1x1-fmap
        # self.bn = nn.BatchNorm1d(outdepth)
        active = self._ActiveFuc[active]
        self.active = active(inplace=True)
        self.view = [ReturnX(), AdaPoolView('avg', -1, 0)][isview]

    def forward(self, x):
        if isinstance(x, tuple):
            x = x[self.which]
        x = self.conv(x)
        # x = self.bn(x)
        x = self.active(x)
        x = self.view(x)
        return x


class Clssifier(nn.Module):
    _ActiveFuc = {'relu': nn.ReLU, 'sig': nn.Sigmoid, 'relu6': nn.ReLU6}

    def __init__(self, indepth, middepth=0, outdepth=1000, expansion=1, dropout=(0,), active='relu'):
        super(Clssifier, self).__init__()
        assert isinstance(dropout, (list, tuple))

        self.dropout = dropout
        self.middepth = middepth
        self.expansion = expansion  # block.expansion

        if middepth == 0:
            assert len(self.dropout) >= 1
            self.drop = nn.Dropout(p=self.dropout[0], inplace=False)
            self.fc = nn.Linear(indepth * expansion, outdepth)
        elif middepth > 0:
            assert len(self.dropout) == 2
            self.drop1 = nn.Dropout(p=self.dropout[0], inplace=False)
            self.fc1 = nn.Linear(indepth, middepth)
            self.active1 = self._ActiveFuc[active]
            self.drop2 = nn.Dropout(p=self.dropout[1], inplace=False)
            self.fc2 = nn.Linear(middepth, outdepth)

    def forward(self, x):
        if self.middepth == 0:
            x = self.drop(x)
            x = self.fc(x)
        elif self.middepth > 0:
            x = self.drop1(x)
            x = self.fc1(x)
            x = self.active1(x)
            x = self.drop2(x)
            x = self.fc2(x)
        return x


class Summary(nn.Module):

    def __init__(self, active_me=True):
        super(Summary, self).__init__()
        self.active_me = active_me


class ConcatSummary(Summary):
    """
        汇总多个xfc的输出到一个fc; 或 汇总多个squeeze的输出到一个fc.
    """

    def __init__(self, indepth, middepth=0, outdepth=1000, dropout=(0, 0),
                 active='relu', with_fc=True, active_me=True):
        """
         - indepth:  对所有输入x, 进行拼接后的输入通道数
         - middepth: fc 层的中间隐藏层，=0 则无隐藏层
         - outdepth: 输出通道数 => nlabels
         - dropout:  fc 层的辍学率
         - active:   fc 层的激活函数
         - withfc:   when indepth==outdepth， False => 不添加fc层，直接输出拼接向量进行分类.
         - active_me: 是否激活当前模块，不激活则计算时绕过此模块
        """
        super(ConcatSummary, self).__init__(active_me)
        if not with_fc:
            assert indepth == outdepth, '<withfc> can be False only under <indepth>==<outdepth>.'
            self.classifier = ReturnX()
        else:
            self.classifier = Clssifier(indepth, middepth, outdepth, 1, dropout, active)

    def forward(self, x):
        # assert isinstance(x, (tuple, list))
        x = torch.cat(x, dim=1)
        x = self.classifier(x)
        return x

    def __repr__(self):
        strme = self.__class__.__name__ + '(\n  (concat): torch.cat(dim=1)()\n' + \
                '  (classifier): ' + self.classifier.__repr__() + ')'
        return strme


class ScaleResNet(nn.Module):
    _XBlock = {'basic': BasicBlock, 'bottle': Bottleneck}

    def __init__(self, depth, btype, layers, dataset='imagenet', summar='concat', sum_active=True,
                 sfc_with=True, sfc_indep=256, sfc_middep=512, sfc_drop=(0, 0), sfc_active='relu',
                 head_key_cfg={'id-scal1-scale2': {"cfg": ""}}):
        super(ScaleResNet, self).__init__()

        if dataset == 'imagenet':
            nlabels = 1000
        elif dataset == 'cifar10':
            nlabels = 10
        elif dataset == 'cifar100':
            nlabels = 100
        else:
            raise NotImplementedError('Unknown <dataset: %s>' % dataset)
        xblock = self._XBlock[btype]
        self.inplanes = depth
        self.nlabels = nlabels
        self.dataset = dataset
        self.summar = summar
        self.sum_active = sum_active

        # self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.preproc = PreProc(indep=3, outdep=self.inplanes)

        layer1 = self._make_layer(xblock, 64, layers[0])
        layer2 = self._make_layer(xblock, 128, layers[1], stride=2)
        layer3 = self._make_layer(xblock, 256, layers[2], stride=2)
        layer4 = self._make_layer(xblock, 512, layers[3], stride=2)
        self.features = Features()
        for l in range(1, 5):
            idx = 1 if l == 1 else idx
            block_list = eval('layer%s' % l)
            for i, block in enumerate(block_list):
                self.features.add_module('bo_%s' % idx, block)
                idx += 1
        del layer1, layer2, layer3, layer4
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * xblock.expansion, nlabels)

        self.bone_feat_maps = {}  # 用于缓存骨架特征图
        self.head_key_pos = {}  # 用于关联特征图和分类头
        for key, cfg in head_key_cfg.items():
            print('**** -> head-%s' % key)
            head = self.get_fc_head(key=key, **cfg)
            setattr(self, 'head-%s' % key, head)  # 注册head
            boid = key.split('-')[0]  # 标记fmap
            self.head_key_pos.setdefault('head-%s' % key, boid)
        print(self.head_key_pos)

        self.summary = ConcatSummary(sfc_indep, sfc_middep, nlabels, sfc_drop, sfc_active, sfc_with, sum_active)

        self.train_which_now = {'bone+mhead': False, 'auxhead': False, 'summary': False}
        self.eval_which_now = {'bone+mhead': False, 'bone+mhead+auxhead': False, 'bone+mhead+auxhead+summary': False}

        self.init_params()

    def _make_layer(self, block, planes, blocks, stride=1, inhead=False):
        if not inhead:
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            layers = list()
            layers.append(block(self.inplanes, planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            # layers = nn.Sequential(*layers)
            return layers
        else:
            layers = list()
            self.inplanes = planes * block.expansion
            for i in range(1, blocks + 1):
                layers.append(block(self.inplanes, planes))
            # layers = nn.Sequential(*layers)
            return layers

    def get_fc_head(self, key, indepth, bnums, btype='bottle', main_aux='main',
                    fc_middep=0, fc_drop=(0,), fc_actfuc='relu', with_fc=True, active_fc=False, active_me=True,
                    squeeze='pool', sq_outdep=1, sq_ksize=1, sq_groups='gcu', sq_active='relu', sq_isview=True):
        # assert isinstance(expand, tuple), ((3, 64), (5, 64))
        key = key.replace('@', '.')
        boid, scale1, scale2 = [float(x) for x in key.split('-')]
        xblock = self._XBlock[btype]
        # expand = OrderedDict(expand)

        head = MoHead(active_me, active_fc, with_fc, main_aux)
        if scale2 != scale1:
            head.add_module('downsize',
                            BranchDownsize(factor=round(scale1 / scale2, 7), mode='bilinear', align_corners=True))

        block_list = self._make_layer(xblock, indepth, bnums, stride=1, inhead=True)
        for i, block in enumerate(block_list):
            head.add_module('bo_%s' % (i + 1), block)

        if squeeze == 'pool':
            squeeze = AdaPoolView(pool='avg', dim=-1, which=0)
        elif squeeze == 'conv':
            squeeze = AdaConvView(indepth, sq_outdep, sq_ksize, stride=1, padding=0,
                                  groups=sq_groups, active=sq_active, isview=sq_isview, which=0)
            indepth = sq_outdep
        else:
            raise NotImplementedError('<squeeze> must be <pool || conv>, but get %s.' % squeeze)
        head.add_module('squeeze', squeeze)

        if with_fc:
            head.add_module('classifier',
                            Clssifier(indepth, fc_middep, self.nlabels, xblock.expansion, fc_drop, fc_actfuc))
        return head

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def train_mode(self, ite, cfg):
        # 当迭代次数 ite 超过设定值，开启对应的训练阶段
        which = None
        for key in sorted(cfg.train_which.keys())[::-1]:
            if ite >= key:
                which = cfg.train_which[key]
                break
        self.set_train_which(part=which, name_which=cfg.name_which)

    def eval_mode(self, ite, cfg):
        # 当迭代次数 ite 超过设定值，开启对应的测试阶段
        which = None
        for key in sorted(cfg.eval_which.keys())[::-1]:
            if ite >= key:
                which = cfg.eval_which[key]
                break
        self.set_eval_which(part=which, name_which=cfg.name_which)

    def reset_mode(self, mode='train'):
        if mode == 'train':
            for k, v in self.train_which_now.items():
                self.train_which_now[k] = False
        elif mode == 'val':
            for k, v in self.eval_which_now.items():
                self.eval_which_now[k] = False

    def set_train_which(self, part, name_which='none'):
        """
         -part: 基于Module类的控制, eg. PreProc, Features, MoHead
         -name_which: 基于Block实例的控制, eg. MoHead.head-3-2-2, MoHead.head-3-2-2
         要控制哪一类Module下的哪一个模块Block => eg. MoHead 下的 MoHead.head-3-2-2
        """
        assert part in self.train_which_now, '设定超出可选项范围--> %s' % part
        self.reset_mode(mode='val')
        if self.train_which_now[part]:
            return
        else:
            self.reset_mode(mode='train')
            self.train_which_now[part] = True

        if part == 'bone+mhead':
            for name, module in self.named_modules():
                if isinstance(module, (PreProc, Features)):
                    module.train()
                    for p in module.parameters():
                        p.requires_grad = True
                if isinstance(module, MoHead):
                    if module.main_aux == 'main':
                        module.active_me = True
                        module.active_fc = True
                        module.train()
                        for p in module.parameters():
                            p.requires_grad = True
                    elif module.main_aux == 'aux':
                        module.active_me = False
                        module.active_fc = False
                        module.eval()
                        for p in module.parameters():
                            p.requires_grad = False
                if isinstance(module, Summary):
                    module.active_me = False
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
        elif part == 'auxhead':
            for name, module in self.named_modules():
                if isinstance(module, (PreProc, Features)):
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
                if isinstance(module, MoHead):
                    if module.main_aux == 'main':
                        # 其实可以输出，但不能反传梯度
                        module.active_me = True
                        module.active_fc = True
                        module.eval()
                        for p in module.parameters():
                            p.requires_grad = False
                    elif module.main_aux == 'aux':
                        if name == name_which:
                            module.active_me = True
                            module.active_fc = True
                            module.train()
                            for p in module.parameters():
                                p.requires_grad = True
                        else:
                            module.active_me = False
                            module.active_fc = False
                            module.eval()
                            for p in module.parameters():
                                p.requires_grad = False
                if isinstance(module, Summary):
                    module.active_me = False
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
        elif part == 'summary':
            for name, module in self.named_modules():
                if isinstance(module, (PreProc, Features)):
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
                if isinstance(module, MoHead):
                    module.active_me = True
                    module.active_fc = False
                    module.eval()
                    for p in module.parameters():
                        p.requires_grad = False
                if isinstance(module, Summary):
                    module.active_me = True
                    module.train()
                    for p in module.parameters():
                        p.requires_grad = True

    def set_eval_which(self, part, name_which='none'):
        assert part in self.eval_which_now, '设定超出可选项范围--> %s' % part
        self.reset_mode(mode='train')
        if self.eval_which_now[part]:
            return
        else:
            self.reset_mode(mode='val')
            self.eval_which_now[part] = True

        if part == 'bone+mhead':
            self.eval()
            for name, module in self.named_modules():
                if isinstance(module, (PreProc, Features)):
                    pass
                elif isinstance(module, MoHead):
                    if module.main_aux == 'main':
                        module.active_me = True
                        module.active_fc = True
                    elif module.main_aux == 'aux':
                        module.active_me = False
                        module.active_fc = False
                if isinstance(module, Summary):
                    module.active_me = False
        elif part == 'bone+mhead+auxhead':
            self.eval()
            for name, module in self.named_modules():
                if isinstance(module, (PreProc, Features)):
                    pass
                elif isinstance(module, MoHead):
                    if module.main_aux == 'main':
                        module.active_me = True
                        module.active_fc = True
                    elif module.main_aux == 'aux':
                        if name == name_which:
                            module.active_me = True
                            module.active_fc = True
                        else:
                            module.active_me = False
                            module.active_fc = False
                elif isinstance(module, Summary):
                    module.active_me = False
        elif part == 'bone+mhead+auxhead+summary':
            self.eval()
            for name, module in self.named_modules():
                if isinstance(module, (PreProc, Features)):
                    pass
                elif isinstance(module, MoHead):
                    module.active_me = True
                    module.active_fc = False
                elif isinstance(module, Summary):
                    module.active_me = True

    def forward(self, x):
        bone_feat_maps = {}
        x = self.preproc(x)
        # print('pre->', x[0].size(), len(x))
        for id, mo in enumerate(self.features):
            x = mo(x)
            if str(id + 1) in self.head_key_pos.values():
                bone_feat_maps.setdefault(str(id + 1), x)
            # print('id->%s' % (id + 1), x.size())
        logits = []
        for key, pos in self.head_key_pos.items():
            # print('*** --> ', key)
            head = getattr(self, key, None)
            # assert head is not None
            if not head.active_me:  # 跳过未激活头head
                continue
            if head.with_fc and not head.active_fc:
                head = head[:-1]  # 去掉head中未激活的分类器
            x = bone_feat_maps.get(pos, None)
            # assert x is not None
            for id, mo in enumerate(head):
                x = mo(x)
            logits.append(x)
        if self.summary.active_me:
            logits = [self.summary(logits)]
        return logits


def resnet50(pretrained=False, model_path=None, **kwargs):
    """Constructs a ResNet-50 model. top1-acc-76.130%  parameter-25.56M

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ScaleResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    print(model)
    if pretrained:
        if model_path is not None:
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict)
        else:
            model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


if __name__ == '__main__':
    # model_dir = xtils.get_pretrained_models()
    # model_ckpt_map = {
    #     'resnet18': 'resnet18-5c106cde.pth',
    #     'resnet34': 'resnet34-333f7ec4.pth',
    #     'resnet50': 'resnet50-19c8e357.pth',
    #     'resnet101': 'resnet101-5d3b4d8f.pth',
    #     'resnet152': 'resnet152-b121ed2d.pth',
    # }
    #
    # model = resnet50(pretrained=True, model_path=os.path.join(model_dir, model_ckpt_map['resnet50']))
    # # model = RESNets(**resx)
    # print(model)

    # imagenet
    sr1_cfg = {
        '3-4-4': dict(indepth=64, bnums=1, btype='bottle', main_aux='aux',
                      fc_middep=0, fc_drop=(0, 0), fc_actfuc='relu', with_fc=True, active_fc=False, active_me=True,
                      squeeze='pool', sq_outdep=256, sq_ksize=7, sq_groups='gcu', sq_active='relu', sq_isview=True),
        '7-8-8': dict(indepth=128, bnums=1, btype='bottle', main_aux='aux',
                      fc_middep=0, fc_drop=(0, 0), fc_actfuc='relu', with_fc=True, active_fc=False, active_me=True,
                      squeeze='pool', sq_outdep=512, sq_ksize=7, sq_groups='gcu', sq_active='relu', sq_isview=True),
        '13-16-16': dict(indepth=256, bnums=1, btype='bottle', main_aux='aux',
                         fc_middep=0, fc_drop=(0, 0), fc_actfuc='relu', with_fc=True, active_fc=False, active_me=True,
                         squeeze='pool', sq_outdep=1024, sq_ksize=7, sq_groups='gcu', sq_active='relu', sq_isview=True),
        '16-32-32': dict(indepth=512, bnums=0, btype='bottle', main_aux='main',
                         fc_middep=0, fc_drop=(0, 0), fc_actfuc='relu', with_fc=True, active_fc=False, active_me=True,
                         squeeze='pool', sq_outdep=2048, sq_ksize=7, sq_groups='gcu', sq_active='relu', sq_isview=True)
    }
    sr1 = OrderedDict(depth=64, btype='bottle', layers=[3, 4, 6, 3], dataset='imagenet',
                      summar='concat', sum_active=True, sfc_with=True, sfc_indep=3840,
                      sfc_middep=0, sfc_drop=(0, 0), sfc_active='relu', head_key_cfg=sr1_cfg)

    model = ScaleResNet(**sr1)
    print(model)

    model.set_train_which(part=['bone+mhead', 'auxhead', 'summary'][0], name_which='none')
    # model.set_eval_which(part=['bone+mhead', 'bone+mhead+auxhead', 'bone+mhead+auxhead+summary'][1], name_which='head-16-32-32')
    # paramsx = [p for p in model.parameters() if p.requires_grad]

    # model.set_train_which(part=['bone+mhead', 'auxhead', 'summary'][1], name_which='head-3-4-4')
    # paramsy = [p for p in model.parameters() if p.requires_grad]

    # share = list(set(paramsx).intersection(set(paramsy)))

    insize = [32, 224][model.dataset == 'imagenet']
    x = torch.randn(1, 3, insize, insize)
    z = model(x)
    print('z-->', len(z), '\n')

    # xtils.calculate_layers_num(model)
    # xtils.calculate_FLOPs_scale(model, input_size=224, use_gpu=False, multiply_adds=False)
    # xtils.calculate_params_scale(model, 'million')
    # xtils.calculate_time_cost(model, insize=224, use_gpu=False, toc=1, pritout=True)
