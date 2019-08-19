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
        调节图片最小边尺寸到一个指定的数值范围内[min_size, max_size]
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


class DataPrefetcher(object):
    """
    https://github.com/NVIDIA/apex/blob/
    f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1, 3, 1, 1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1, 3, 1, 1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        self.preload()
        return input, target


def load_cifar10(data_root, bsize_train, bsize_val, augment, num_workers=8, result='loader', **kwargs):
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
    test_dataset = val_dataset

    if result == 'dataset':
        return train_dataset, val_dataset, test_dataset

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=bsize_train,
                                   num_workers=num_workers,
                                   shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=bsize_val,
                                 num_workers=num_workers,
                                 shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=bsize_val,
                                  num_workers=num_workers,
                                  shuffle=False)
    return train_loader, val_loader, test_loader


def load_cifar100(data_root, bsize_train, bsize_val, augment, num_workers=8, result='loader', **kwargs):
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

    if result == 'dataset':
        return train_dataset, val_dataset, test_dataset
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=bsize_train,
                                   num_workers=num_workers,
                                   shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=bsize_val,
                                 num_workers=num_workers,
                                 shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=bsize_val,
                                  num_workers=num_workers,
                                  shuffle=False)
    return train_loader, val_loader, test_loader


def load_svhn(data_root, bsize_train, bsize_val, augment, num_workers=8, result='loader', **kwargs):
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

    if result == 'dataset':
        return train_dataset, val_dataset, test_dataset

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=bsize_train,
                                   num_workers=num_workers,
                                   shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=bsize_val,
                                 num_workers=num_workers,
                                 shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=bsize_val,
                                  num_workers=num_workers,
                                  shuffle=False)
    return train_loader, val_loader, test_loader


def load_stl10(data_root, bsize_train, bsize_val, augment, num_workers=8, result='loader', **kwargs):
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

    if result == 'dataset':
        return train_dataset, val_dataset, test_dataset

    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=bsize_train,
                                   num_workers=num_workers,
                                   shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset,
                                 batch_size=bsize_val,
                                 num_workers=num_workers,
                                 shuffle=False)
    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=bsize_val,
                                  num_workers=num_workers,
                                  shuffle=False)
    return train_loader, val_loader, test_loader


def load_imagenet(data_root, bsize_train, bsize_val, augement, num_workers=12, result='loader', **kwargs):
    """
    augment: {'train': ['rresize-1crop', '1resize-1crop', '2resize-1crop', '1resize-0crop'][0],
              'val': ['center-crop'][0]}
    imsize / insize = 0.875, eg. 256/0.875=224
    """

    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    imsize = augement.get('imsize', None)
    insize = augement.get('insize', None)
    interp = augement.get('interp', 'bilinear')
    assert imsize is not None, '数据扩增的图片尺寸 <imsize @cfg.data_augment> 未设置.'
    assert insize is not None, '网络输入的图片尺寸 <insize @cfg.data_augment> 未设置.'
    assert imsize >= insize, 'imsize must > insize.'
    assert interp in ['linear', 'bilinear', 'bicubic']
    interp = getattr(Image, interp.upper())

    if augement['train'] == 'rotate-rresize-1crop':
        degree = augement.get('degree', (0, 0))
        scale = augement.get('scale', (0.08, 1))
        ratio = augement.get('ratio', (3. / 4, 4. / 3))
        resize_crop = []
        if tuple(degree) != (0, 0):
            resize_crop.append(transforms.RandomRotation(degree))
        resize_crop.append(transforms.RandomResizedCrop(insize, scale, ratio, interp))

    elif augement['train'] == 'rotate-rresize-1crop-colorj':
        degree = augement.get('degree', (0, 0))
        scale = augement.get('scale', (0.08, 1))
        ratio = augement.get('ratio', (3. / 4, 4. / 3))
        brightness = augement.get('brightness', 0.2)
        contrast = augement.get('contrast', 0.2)
        saturation = augement.get('saturation', 0.2)
        hue = augement.get('hue', 0.2)
        resize_crop = []
        if tuple(degree) != (0, 0):
            resize_crop.append(transforms.RandomRotation(degree))
        resize_crop.append(transforms.RandomResizedCrop(insize, scale, ratio, interp))
        color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)
        resize_crop.append(color_jitter)

    elif augement['train'] == 'rotate-mmresize-1crop':
        mmsize = augement.get('mmsize', [])
        assert len(mmsize) == 2 and min(mmsize) > insize
        degree = augement.get('degree', (0, 0))
        resize_crop = []
        if tuple(degree) != (0, 0):
            resize_crop.append(transforms.RandomRotation(degree))
        resize_crop.append(MinMaxResize(min(mmsize), max(mmsize), interp))
        resize_crop.append(transforms.RandomCrop(insize))

    elif augement['train'] == 'mmresize-raffine-1crop':
        mmsize = augement.get('mmsize', [])
        assert len(mmsize) == 2 and min(mmsize) > insize
        resize_crop = [MinMaxResize(min(mmsize), max(mmsize), interp),
                       transforms.RandomApply([transforms.RandomAffine(degrees=(-15, 15), shear=(-5, 5))]),
                       transforms.RandomCrop(insize)]
    else:
        raise NotImplementedError('Unknown Resize & Crop Method %s ...' % (augement,))

    other_process = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]
    if augement.get('color', None):
        other_process.insert(-1, ColorAugmentation())
    train_transform = transforms.Compose(resize_crop + other_process)

    if augement['val'] == '1resize-0crop':
        assert insize > 0
        val_transform = transforms.Compose([
            transforms.Resize(insize, interp),
            transforms.ToTensor(),
            normalize,
        ])
    elif augement['val'] == '1resize-1crop':
        val_transform = transforms.Compose([
            transforms.Resize(imsize, interp),
            transforms.CenterCrop(insize),
            transforms.ToTensor(),
            normalize,
        ])
    elif augement['val'] == '1resize-10crop':
        val_transform = transforms.Compose([
            transforms.Resize(imsize, interp),
            transforms.TenCrop(insize),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        ])
    else:
        raise NotImplementedError('Unknown Resize & Crop Method %s ...' % (augement,))

    # print('\nWarning: Please Assure Train-Transform is -->\n' + repr(train_transform))
    # print('\nWarning: Please Assure Val-Transform is -->\n' + repr(val_transform))

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    val_dataset = datasets.ImageFolder(valdir, val_transform)
    test_dataset = val_dataset

    if result == 'dataset':
        return train_dataset, val_dataset, test_dataset

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=bsize_train, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=bsize_val, shuffle=False,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=bsize_val, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def data_factory(dataset, data_root, bsize_train, bsize_val, augment, num_workers, result='loader', **kwargs):
    assert result in ['loader', 'dataset'], 'result must == loader || dataset'

    if dataset == 'cifar10':
        train, val, test = \
            load_cifar10(data_root, bsize_train, bsize_val, augment, num_workers, result, **kwargs)
    elif dataset == 'cifar100':
        train, val, test = \
            load_cifar100(data_root, bsize_train, bsize_val, augment, num_workers, result, **kwargs)
    elif dataset == 'svhn':
        train, val, test = \
            load_svhn(data_root, bsize_train, bsize_val, augment, num_workers, result, **kwargs)
    elif dataset == 'stl10':
        train, val, test = \
            load_svhn(data_root, bsize_train, bsize_val, augment, num_workers, result, **kwargs)
    elif dataset == 'imagenet':
        train, val, test = \
            load_imagenet(data_root, bsize_train, bsize_val, augment, num_workers, result, **kwargs)
    else:
        raise NotImplementedError('Unknow <dataset: %s>, check data_factory.py' % dataset)

    return train, val, test


if __name__ == '__main__':
    pass
    train, val, test = load_svhn('F:\数据集\SVHN', 1, 1,
                                 augment={'train': 'no-aug', 'val': 'no-aug'},
                                 num_workers=8, result='loader', train='train-extra')
    print([len(x) for x in [train, val, test]])
