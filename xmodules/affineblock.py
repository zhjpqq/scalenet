# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2019/1/9 12:17'

import math, torch
import torch.nn as nn
import torch.nn.functional as F

inplace = [False, True][1]


class AfineBlock(nn.Module):
    """
    传入af-keys列表，按其顺序依次仿射图像，再按其顺序组合放射图像.
    输出通道数 = (3 or convdepth) * (len(afkeys) + 1).
    """
    af1 = {'alpha': 20, 'tx': 0, 'ty': 0, 'scale': 1}
    af2 = {'alpha': -20, 'tx': 0, 'ty': 0, 'scale': 1}
    af3 = {'alpha': 160, 'tx': 0, 'ty': 0, 'scale': 1}
    af4 = {'alpha': -160, 'tx': 0, 'ty': 0, 'scale': 1}
    af5 = {'alpha': 0, 'tx': 0, 'ty': 0, 'scale': 0.8}
    af6 = {'alpha': 0, 'tx': 0.5, 'ty': 0.5, 'scale': 1}

    afdict = {'af1': af1, 'af2': af2, 'af3': af3, 'af4': af4,
              'af5': af5, 'af6': af6}

    def __init__(self, afkeys, convon=False, convlayers=1, convdepth=3):
        super(AfineBlock, self).__init__()
        assert isinstance(afkeys, (tuple, list)), '<afkeys> must be a list or tuple!'
        self.afkeys = afkeys  # 各种变换及其组合顺序
        self.totals = len(afkeys)  # 总变换次数
        self.convon = convon  # 是否先卷积再输出
        self.convlayers = convlayers  # 卷积层的层数
        self.convdepth = convdepth  # 卷积变换后的通道数

        self.conveval = ['s-eval', 'x-eval'][1]  # 测试评估阶段是否继续组合
        self.training = True

        if self.totals >= 1:
            for i, key in enumerate(afkeys):
                theta = self._init_theta(key)
                setattr(self, 'theta%s' % (i + 1), theta)

        if self.convon:
            for i in range(self.totals + 1):  # n+1 包括原图和仿射图
                observer = self._init_observer(convlayers, convdepth)
                setattr(self, 'observe%s' % (i + 1), observer)

        # print(self)

    def _init_theta(self, afkey):
        assert afkey in self.afdict.keys(), 'Unknown <af> key for afdict, %s' % afkey
        afargs = self.afdict[afkey]
        theta = torch.zeros(2, 3)
        theta[0, 0] = math.sin(afargs['alpha']) * afargs['scale']
        theta[1, 1] = -math.sin(afargs['alpha']) * afargs['scale']
        theta[0, 1] = math.cos(afargs['alpha']) * afargs['scale']
        theta[1, 0] = math.cos(afargs['alpha']) * afargs['scale']
        theta[0, 2] = afargs['tx']
        theta[1, 2] = afargs['ty']
        theta = nn.Parameter(theta, requires_grad=False)
        return theta

    def _init_observer(self, layer_nums, out_depth):
        if layer_nums == 1:
            conv = nn.Sequential(
                nn.Conv2d(3, out_depth, kernel_size=3, stride=1, padding=1)
            )
        elif layer_nums == 2:
            conv = nn.Sequential(
                nn.Conv2d(3, out_depth, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_depth),
                nn.ReLU(inplace)
            )
        else:
            raise NotImplementedError
        return conv

    def forward(self, x):
        if self.totals == 0:
            return x
        res = [x]
        for i in range(self.totals):
            theta = getattr(self, 'theta%s' % (i + 1))
            theta = theta.expand(x.size(0), 2, 3)
            grid = F.affine_grid(theta, x.size())
            afx = F.grid_sample(x, grid)
            res.append(afx)
        if self.convon:
            for i in range(self.totals + 1):
                observe = getattr(self, 'observe%s' % (i + 1))
                res[i] = observe(res[i])
        # if not self.training and self.conveval == 's-eval':
        #     x = res[0]
        #     return x
        x = torch.cat(res, dim=1)
        return x


class AfineBlock2(nn.Module):
    af1 = {'alpha': 50, 'tx': 0, 'ty': 0, 'scale': 1}
    af2 = {'alpha': 30, 'tx': 0, 'ty': 0, 'scale': 1.5}
    af3 = {'alpha': 0, 'tx': 0, 'ty': 0, 'scale': 0.8}
    af4 = {'alpha': 0, 'tx': 0.3, 'ty': 0, 'scale': 1}

    afdict = {'af1': af1, 'af2': af2, 'af3': af3, 'af4': af4}

    def __init__(self, af):
        super(AfineBlock2, self).__init__()
        self.af = self.afdict[af]
        self.theta = torch.zeros(2, 3)
        self.theta[0, 0] = math.sin(self.af['alpha']) * self.af['scale']
        self.theta[1, 1] = -math.sin(self.af['alpha']) * self.af['scale']
        self.theta[0, 1] = math.cos(self.af['alpha']) * self.af['scale']
        self.theta[1, 0] = math.cos(self.af['alpha']) * self.af['scale']
        self.theta[0, 2] = self.af['tx']
        self.theta[1, 2] = self.af['ty']

    def forward(self, x):
        theta = self.theta.expand(x.size(0), 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


if __name__ == '__main__':
    from factory.data_factory_bp import data_factory
    from matplotlib import pyplot as plt
    import numpy as np
    import torchvision

    torch.manual_seed(111)

    affine = AfineBlock2('af4')
    aftrans = AfineBlock(afkeys=['af1', 'af2', 'af3', 'af4'],
                         convon=True, convlayers=1, convdepth=5)

    imgnet_aug = {'train': ['rresize-1crop', '1resize-1crop', '2resize-1crop', '1resize-0crop'][2],
                  'val': ['1resize-1crop', '1resize-10crop'][0]}
    cifar10_aug = {'train': ['no-aug', '1crop-flip', '10crop-flip'][1], 'val': ['no-aug'][0]}

    idx = 0
    dataset = ['cifar10', 'imagenet'][idx]
    dataroot = ['F:\DataSets\cifar10', '/data0/ImageNet_ILSVRC2012/'][idx]
    augment = [cifar10_aug, imgnet_aug][idx]
    batch_size = 1
    train_loader, _, _ = data_factory(dataset, dataroot, batch_size, batch_size, augment, num_workers=4)


    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        plt.pause(30)


    # get some random training images
    # dataiter = iter(train_loader)
    # images, labels = next(train_loader)
    for batch, (images, labels) in enumerate(train_loader):
        # show images
        # images = affine(images)
        images = aftrans(images)
        print(images.shape)
        # images = images[:, 0:3, :, :]
        images = images.view(-1, 3, 32, 32)
        # 脱离求导链
        images = images.detach()
        images = torchvision.utils.make_grid(images)
        imshow(images)
        # print labels
        print(' '.join('%5s' % labels[j] for j in range(batch_size)))
        print('.....')
