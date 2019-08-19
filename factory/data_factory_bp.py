# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2018/12/15 12:17'

import time
import argparse
import shutil
import os
import random
from datetime import datetime
from collections import namedtuple
from PIL import Image
import warnings

import torch
from torch import nn
from torch.utils import data
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import functional as F


class ColorAugmentation(object):
    # fork from FishNet
    def __init__(self):
        self.eig_vec = torch.Tensor([
            [0.4009, 0.7192, -0.5675],
            [-0.8140, -0.0045, -0.5808],
            [0.4203, -0.6948, -0.5836],
        ])
        self.eig_val = torch.Tensor([[0.2175, 0.0188, 0.0045]])

    def __call__(self, tensor):
        # assert tensor.size(0) == 3
        alpha = torch.normal(mean=torch.zeros_like(self.eig_val)) * 0.1
        quatity = torch.mm(self.eig_val * alpha, self.eig_vec)
        tensor = tensor + quatity.view(3, 1, 1)
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '()'
        return format_string


class MinMaxResize(object):

    def __init__(self, min_size, max_size, interpolation=Image.BILINEAR):
        """
        调节图片尺寸到一个指定的数值范围内[min_size, max_size]
        """
        super(MinMaxResize, self).__init__()
        assert isinstance(min_size, int) and isinstance(max_size, int)
        assert (max_size > min_size) and (min_size > 0)
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def _get_size(self):
        return round(random.uniform(self.min_size, self.max_size))

    def __call__(self, img):
        return F.resize(img, self._get_size(), self.interpolation)

    def __repr__(self):
        interpolate_str = transforms.transforms._pil_interpolation_to_str[self.interpolation]
        format_string = self.__class__.__name__ + '(min_size={0}'.format(self.min_size)
        format_string += ', max_size={0}'.format(self.max_size)
        format_string += ', interpolation={0})'.format(interpolate_str)
        return format_string


class ReturnOrigin(object):

    def __init__(self):
        super(ReturnOrigin, self).__init__()

    def __call__(self, img):
        return img


def ipil_loader(path):
    print(path, '\n')
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def load_cifar10(data_root, batch_size_train, batch_size_val, augment, num_workers=8, ret='loader', **kwargs):
    # titanxp_root /data/dataset/cifar-10-batches-py/
    # 1080ti_root  /data0/cifar10/
    # K40_root     /home/zhjm/PycharmProjects/cifar10
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    if augment['train'] == 'no-aug':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-flip':
        # used by paper: DSN, ResNet, DenseNet, ResNeXt
        train_transform = transforms.Compose([
            # transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-only':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-1affine-flip':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-raffine-flip':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomApply([transforms.RandomAffine(degrees=8)]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '10crop-flip':
        raise NotImplementedError
    else:
        raise ValueError('Unknown data augment policy: %s ...' % (augment['train'],))

    if augment['val'] == 'no-aug':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Unknown data augment policy: %s ...' % (augment['val'],))

    train_dataset = datasets.CIFAR10(root=data_root, train=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=data_root, train=False, transform=val_transform)

    # train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, (45000, 5000))
    # train_dataset.dataset.transform = train_transform   # no work
    # val_dataset.dataset.transform = val_transform       # can work

    test_dataset = val_dataset

    if ret == 'dataset':
        return train_dataset, val_dataset, test_dataset
    elif ret == 'loader':
        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size_train,
                                       num_workers=num_workers,
                                       shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset,
                                     batch_size=batch_size_val,
                                     num_workers=num_workers,
                                     shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size_val,
                                      num_workers=num_workers,
                                      shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        raise NotImplementedError('<ret> must be loader or dataset')


def load_cifar100(data_root, batch_size_train, batch_size_val, augment, num_workers=8, ret='loader', **kwargs):
    # titanxp_root /data/dataset/cifar-100-python/
    # 1080ti_root  /data0/cifar100/
    # K40_root     ?

    mean = [x / 255 for x in [129.3, 124.1, 112.4]]
    std = [x / 255 for x in [68.2, 65.4, 70.4]]

    if augment['train'] == 'no-aug':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-flip':
        # used by paper: DSN, ResNet, DenseNet, ResNeXt
        train_transform = transforms.Compose([
            # transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-only':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-1affine-flip':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-raffine-flip':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomApply([transforms.RandomAffine(degrees=8)]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '10crop-flip':
        raise NotImplementedError
    else:
        raise ValueError('Unknown data augment policy: %s ...' % (augment['train'],))

    if augment['val'] == 'no-aug':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Unknown data augment policy: %s ...' % (augment['val'],))

    train_dataset = datasets.CIFAR100(root=data_root, train=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(root=data_root, train=False, transform=val_transform)
    test_dataset = val_dataset

    if ret == 'dataset':
        return train_dataset, val_dataset, test_dataset
    elif ret == 'loader':
        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size_train,
                                       num_workers=num_workers,
                                       shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset,
                                     batch_size=batch_size_val,
                                     num_workers=num_workers,
                                     shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size_val,
                                      num_workers=num_workers,
                                      shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        raise NotImplementedError('<ret> must be loader or dataset!')


def load_svhn(data_root, batch_size_train, batch_size_val, augment, num_workers=8, ret='loader', **kwargs):
    # train:73257  test:26032 extra:531131
    # TODO CIFAR10 BELLOW mean std
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    if augment['train'] == 'no-aug':
        # used by paper: DSN, ResNet, DenseNet, ResNeXt
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-flip':
        train_transform = transforms.Compose([
            # transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-only':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-1affine-flip':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-raffine-flip':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomApply([transforms.RandomAffine(degrees=8)]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '10crop-flip':
        raise NotImplementedError
    else:
        raise ValueError('Unknown data augment policy: %s ...' % (augment['train'],))

    if augment['val'] == 'no-aug':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Unknown data augment policy: %s ...' % (augment['val'],))

    if kwargs.get('train', 'train-only') == 'train-only':
        print('svhn-training with split <train>.')
        train_dataset = datasets.SVHN(root=data_root, split='train', transform=train_transform)
    elif kwargs.get('train', None) == 'extra-only':
        print('svhn-training with split <extra>.')
        train_dataset = datasets.SVHN(root=data_root, split='extra', transform=train_transform)
    elif kwargs.get('train', None) == 'train-extra':
        print('svhn-training with split <train> + <extra>.')
        train_dataset = datasets.SVHN(root=data_root, split='train', transform=train_transform)
        extra_dataset = datasets.SVHN(root=data_root, split='extra', transform=train_transform)
        train_dataset = data.dataset.ConcatDataset((train_dataset, extra_dataset))
    else:
        raise NotImplementedError('Unkown <train: %s> for SVHN.' % kwargs.get('train'))
    val_dataset = datasets.SVHN(root=data_root, split='test', transform=val_transform)
    test_dataset = val_dataset

    if ret == 'dataset':
        return train_dataset, val_dataset, test_dataset
    elif ret == 'loader':
        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size_train,
                                       num_workers=num_workers,
                                       shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset,
                                     batch_size=batch_size_val,
                                     num_workers=num_workers,
                                     shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size_val,
                                      num_workers=num_workers,
                                      shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        raise NotImplementedError('<ret> must be loader or dataset')


def load_stl10(data_root, batch_size_train, batch_size_val, augment, num_workers=8, ret='loader', **kwargs):
    # titanxp_root /data/dataset/cifar-10-batches-py/
    # 1080ti_root  /data0/cifar10/
    # K40_root     /home/zhjm/PycharmProjects/cifar10
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    if augment['train'] == 'no-aug':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-flip':
        # used by paper: DSN, ResNet, DenseNet, ResNeXt
        train_transform = transforms.Compose([
            # transforms.Pad(4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-only':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-1affine-flip':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(degrees=8),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '1crop-raffine-flip':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomApply([transforms.RandomAffine(degrees=8)]),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif augment['train'] == '10crop-flip':
        raise NotImplementedError
    else:
        raise ValueError('Unknown data augment policy: %s ...' % (augment['train'],))

    if augment['val'] == 'no-aug':
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError('Unknown data augment policy: %s ...' % (augment['val'],))

    train_dataset = datasets.STL10(root=data_root, split='train', transform=train_transform)
    val_dataset = datasets.STL10(root=data_root, split='test', transform=val_transform)
    test_dataset = val_dataset

    if ret == 'dataset':
        return train_dataset, val_dataset, test_dataset
    elif ret == 'loader':
        train_loader = data.DataLoader(dataset=train_dataset,
                                       batch_size=batch_size_train,
                                       num_workers=num_workers,
                                       shuffle=True)
        val_loader = data.DataLoader(dataset=val_dataset,
                                     batch_size=batch_size_val,
                                     num_workers=num_workers,
                                     shuffle=False)
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=batch_size_val,
                                      num_workers=num_workers,
                                      shuffle=False)
        return train_loader, val_loader, test_loader
    else:
        raise NotImplementedError('<ret> must be loader or dataset')


def load_imagenet(data_root, batch_size_train, batch_size_val, augement, num_workers=12, ret='loader', **kwargs):
    """
    augment: {'train': ['rresize-1crop', '1resize-1crop', '2resize-1crop', '1resize-0crop'][0],
              'val': ['center-crop'][0]}

    'rresize-1crop' random resized and crop, used in torch.transforms.RandomResizedCrop().
    '2resize-1crop' used in resnet paper, resize to (256 or 480), then crop to 224, then flip.

    MinMaxResizedCrop: 先缩放再裁切 Size'=Size*scale*1
    minmax: (256, 480) (300, 500)
        下界太小，导致物体有效像素占比太小，物体圆满性被破坏！ 切太稀！
        上界太大，会导致图片被切的太碎，物体完整性被破坏！ 切太碎！

    RandomResizedCrop: 先裁切再缩放 Size'=Size*scale*β_auto_crop
    scale： (1/5, 4/5)=(0.2~0.8)
        下界太小，会导致图片被切的太碎，物体完整性被破坏！ 切太碎！
        上界太大，导致物体有效像素占比太小，物体圆满性被破坏！ 切太稀！
    """
    # titanxp_root /data/dataset/ImageNetDownSample/   64x64  32x32
    # 1080ti_root  /data0/ImageNet_ILSVRC2012/
    # K40_root     no

    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    insize = augement.get('insize', None)
    xscale = augement.get('xscale', [])  # for train
    vscale = augement.get('vscale', [])  # for validate
    assert insize is not None, '输入图片的尺寸 <insize @cfg.data_augment> 未设置.'
    assert isinstance(xscale, (list, tuple)), '<xscale @cfg.data_augment> must be list or tuple'
    assert isinstance(vscale, (list, tuple)), '<vscale @cfg.data_augment> must be list or tuple'

    if augement['train'] == 'rresize-1crop':
        # +随机尺寸变换 + 随机形状变换 +随机平移变换
        scale = augement.get('scale', (0.08, 1))
        ratio = augement.get('ratio', (3. / 4, 4. / 3))
        resize_crop = [transforms.RandomResizedCrop(insize, scale, ratio)]
    elif augement['train'] == 'rotate-rresize-1crop':
        degree = augement.get('degree', (-5, 5))
        scale = augement.get('scale', (0.08, 1))
        ratio = augement.get('ratio', (3. / 4, 4. / 3))
        if tuple(degree) == (0, 0):
            resize_crop = [transforms.RandomResizedCrop(insize, scale, ratio)]
        else:
            resize_crop = [transforms.RandomRotation(degree),
                           transforms.RandomResizedCrop(insize, scale, ratio)]
    elif augement['train'] == 'rotate-rresize-1crop-color':
        degree = augement.get('degree', (-5, 5))
        scale = augement.get('scale', (0.08, 1))
        ratio = augement.get('ratio', (3. / 4, 4. / 3))
        brightness = augement.get('brightness', 0.2)
        contrast = augement.get('contrast', 0.2)
        saturation = augement.get('saturation', 0.2)
        hue = augement.get('hue', 0.2)
        if tuple(degree) == (0, 0):
            resize_crop = [transforms.RandomResizedCrop(insize, scale, ratio)]
        else:
            resize_crop = [transforms.RandomRotation(degree),
                           transforms.RandomResizedCrop(insize, scale, ratio)]
        color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
        resize_crop.append(color_jitter)
    elif augement['train'] == 'xresize-1crop':
        # +固定X尺寸变换 -形状变换 +随机平移变换
        assert len(xscale) >= 1 and min(xscale) >= insize
        resize_list = [transforms.Resize(x) for x in xscale]
        resize_crop = [transforms.RandomChoice(resize_list),
                       transforms.RandomCrop(insize)]
    elif augement['train'] == 'minmax-resize-1crop':
        assert len(xscale) == 2 and min(xscale) > insize
        resize_crop = [MinMaxResize(min(xscale), max(xscale)),
                       transforms.RandomCrop(insize)]
    elif augement['train'] == 'rotate-minmax-resize-1crop':
        assert len(xscale) == 2 and min(xscale) > insize
        degree = augement.get('degree', (-5, 5))
        if tuple(degree) == (0, 0):
            resize_crop = [MinMaxResize(min(xscale), max(xscale)),
                           transforms.RandomCrop(insize)]
        else:
            resize_crop = [transforms.RandomRotation(degree),
                           MinMaxResize(min(xscale), max(xscale)),
                           transforms.RandomCrop(insize)]
    elif augement['train'] == '1resize-0crop':
        resize_crop = [transforms.Resize(insize)]
    elif augement['train'] == '0resize-1crop':
        resize_crop = [transforms.RandomCrop(insize)]
    elif augement['train'] == 'xresize-1affine-1crop':
        # +固定X尺寸变换 + 仿射变换 -形状变换 +随机平移变换
        assert len(xscale) >= 1 and min(xscale) >= insize
        resize_list = [transforms.Resize(x) for x in xscale]
        resize_crop = [transforms.RandomChoice(resize_list),
                       transforms.RandomAffine(degrees=(-15, 15), translate=None, scale=None, shear=(-5, 5)),
                       transforms.RandomCrop(insize)]
    elif augement['train'] == 'xresize-raffine-1crop':
        # +固定X尺寸变换 + 随机仿射变换 -形状变换 +随机平移变换
        assert len(xscale) >= 1 and min(xscale) >= insize
        resize_list = [transforms.Resize(x) for x in xscale]
        resize_crop = [transforms.RandomChoice(resize_list),
                       transforms.RandomApply([transforms.RandomAffine(degrees=(-15, 15), shear=(-5, 5))]),
                       transforms.RandomCrop(insize)]
    elif augement['train'] == 'xresize-caffine-1crop':
        # +固定X尺寸变换 + 仿射变换 -形状变换 +随机平移变换
        assert len(xscale) >= 1 and min(xscale) >= insize
        resize_list = [transforms.Resize(x) for x in xscale]
        resize_crop = [transforms.RandomChoice(resize_list),
                       transforms.RandomChoice([transforms.RandomAffine(degrees=15, shear=None),
                                                transforms.RandomAffine(degrees=0, shear=10)]),
                       transforms.RandomCrop(insize)]
    else:
        raise NotImplementedError('Unknown Resize & Crop Method %s ...' % (augement,))

    if augement.get('color', None):
        other_process = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ]
    else:
        other_process = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    train_transform = transforms.Compose(resize_crop + other_process)

    if augement['val'] == '1resize-1crop':
        assert len(vscale) == 1 and vscale[0] >= insize
        val_transform = transforms.Compose([
            transforms.Resize(vscale[0]),
            transforms.CenterCrop(insize),
            transforms.ToTensor(),
            normalize,
        ])
    elif augement['val'] == '1resize-0crop':
        assert insize > 0
        val_transform = transforms.Compose([
            transforms.Resize(insize),
            transforms.ToTensor(),
            normalize,
        ])
    elif augement['val'] == '1resize-1crop-color':
        assert len(vscale) == 1 and vscale[0] >= insize
        val_transform = transforms.Compose([
            transforms.Resize(vscale[0]),
            transforms.CenterCrop(insize),
            transforms.ToTensor(),
            ColorAugmentation(),
            normalize,
        ])
    elif augement['val'] == '1resize-5crop':
        assert len(vscale) == 1 and vscale[0] >= insize
        val_transform = transforms.Compose([
            transforms.Resize(vscale[0]),
            transforms.FiveCrop(insize),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    elif augement['val'] == '1resize-10crop':
        assert len(vscale) == 1 and vscale[0] >= insize
        val_transform = transforms.Compose([
            transforms.Resize(vscale[0]),
            transforms.TenCrop(insize),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    else:
        raise NotImplementedError('Unknown Resize & Crop Method %s ...' % (augement,))

    print('\nWarning: Please Assure Train-Transform is -->\n' + repr(train_transform))
    print('\nWarning: Please Assure Val-Transform is -->\n' + repr(val_transform))

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    val_dataset = datasets.ImageFolder(valdir, val_transform)
    test_dataset = val_dataset
    if ret == 'dataset':
        return train_dataset, val_dataset, test_dataset
    elif ret == 'loader':
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size_train, shuffle=True,
            num_workers=num_workers, pin_memory=True)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=batch_size_val, shuffle=False,
            num_workers=16, pin_memory=False)
        test_loader = val_loader
        return train_loader, val_loader, test_loader
    else:
        raise NotImplementedError('<ret> must be loader or dataset!')


def data_factory(dataset, data_root, batch_size_train, batch_size_val, augment, num_workers, ret='loader', **kwargs):
    if dataset == 'cifar10':
        train, val, test = \
            load_cifar10(data_root, batch_size_train, batch_size_val, augment, num_workers, ret, **kwargs)
    elif dataset == 'cifar100':
        train, val, test = \
            load_cifar100(data_root, batch_size_train, batch_size_val, augment, num_workers, ret, **kwargs)
    elif dataset == 'svhn':
        train, val, test = \
            load_svhn(data_root, batch_size_train, batch_size_val, augment, num_workers, ret, **kwargs)
    elif dataset == 'stl10':
        train, val, test = \
            load_svhn(data_root, batch_size_train, batch_size_val, augment, num_workers, ret, **kwargs)
    elif dataset == 'imagenet':
        train, val, test = \
            load_imagenet(data_root, batch_size_train, batch_size_val, augment, num_workers, ret, **kwargs)
    else:
        raise NotImplementedError('Unknow <dataset: %s>, check data_factory.py' % dataset)

    if ret == 'loader':
        train_loader, val_loader, test_loader = train, val, test
        return train_loader, val_loader, test_loader
    elif ret == 'dataset':
        train_dataset, val_dataset, test_dataset = train, val, test
        return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    pass
    train, val, test = load_svhn('F:\数据集\SVHN', 1, 1,
                                 augment={'train': 'no-aug', 'val': 'no-aug'},
                                 num_workers=8, ret='loader', train='train-extra')
    print([len(x) for x in [train, val, test]])
