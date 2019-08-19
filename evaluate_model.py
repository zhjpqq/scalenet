# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Ke Sun (sunk@mail.ustc.edu.cn)
# MOdified by OOO
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import random

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from arch_params.scalenet_imagenet_archs import vo69, vo72
from arch_params.res_dense_fish_mobile_hrnet import hrw18, hrw30, hrw32
from factory.model_factory import model_factory
from factory.data_factory import data_factory
from xtils import accuracy, AverageMeter
import xtils


class Config(object):
    gpu_ids = [0, 1, 2, 3]

    device = torch.device('cuda:{}'.format(gpu_ids[0]))

    dataset = 'imagenet'

    data_root = xtils.get_data_root(data='imagenet')

    data_augment = {'train': 'rotate-rresize-1crop', 'val': '1resize-1crop',
                    'imsize': 256, 'insize': 224, 'color': True,
                    'degree': (0, 0), 'scale': (0.08, 1), 'ratio': (3. / 4, 4. / 3)}

    arch_name = ['hrnet', 'resnet50', 'resnet152', 'fishnet150', 'scalenet'][-1]

    arch_kwargs = [hrw18, hrw30, hrw32, vo69, vo72][-2]

    ckpt_file = os.path.join(xtils.get_pretrained_models(),
                             ['hrnetv2_w18_imagenet_pretrained.pth',
                              'fishnet150_ckpt.tar',
                              'resnet50-19c8e357.pth', 'resnet152-b121ed2d.pth',
                              'scale103-5.09M-71.63-vo69.pth',
                              'scale59-30.52M-74.86-vo72.pth'][-2])

    mgpus_to_sxpu = ['m2s', 's2m', 'none'][-1]

    image_size = 256

    input_size = int(0.875 * image_size)

    bsize_val = 256

    num_workers = 8

    print_freq = 20

    valid_total_time = 0

    random_seed = random.randint(0, 100) or None

    # 特殊模型的补充字段
    mode_custom = False
    xfc_which = -1


def valid_model(val_loader, model, criterion, it, cfg=Config(), writer=None):
    """
    1-crop & 10-crop & single-output & list-output
    """
    valid_tic = time.time()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if cfg.mode_custom:
        if isinstance(model, torch.nn.DataParallel):
            model.module.eval_mode(it, cfg)
        else:
            model.eval_mode(it, cfg)
    else:
        model.eval()

    with torch.no_grad():
        btic = time.time()
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

            if images.dim() == 4:
                outputs = model(images)
            elif images.dim() == 5:
                bh, ncrop, c, h, w = images.size()
                outputs = model(images.view(-1, c, h, w))
                if isinstance(outputs, (list, tuple)):
                    outputs = [ot.view(bh, ncrop, -1).mean(1) for ot in outputs]
                else:
                    outputs = outputs.view(bh, ncrop, -1).mean(1)
            else:
                raise NotImplementedError('exptect image.dim in [4, 5], but %s' % images.dim())

            if isinstance(outputs, (list, tuple)):
                all_loss = [criterion(out, labels) for out in outputs]
                # 只验证哪一个xfc抽头  或   # 验证所有xfc抽头的平均
                if isinstance(cfg.xfc_which, int):
                    loss = all_loss[cfg.xfc_which]
                    outputs = outputs[cfg.xfc_which]
                elif cfg.xfc_which == 'all-avg':
                    loss = sum(all_loss) / len(outputs)
                    outputs = sum(outputs) / len(outputs)
                elif cfg.xfc_which == 'aux-avg':
                    loss = sum(all_loss[:-1]) / len(outputs[:-1])
                    outputs = sum(outputs[:-1]) / len(outputs[:-1])
                else:
                    raise NotImplementedError
            else:
                loss = criterion(outputs, labels)

            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - btic)
            btic = time.time()

            if it == 0 and i % 20 == 0:
                print('Valid: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} \t Loss {loss.val:.4f} \t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1, top5=top5))

    if writer is not None:
        writer.add_scalar(tag='val-loss', scalar_value=losses.avg, global_step=it)
        writer.add_scalar(tag='val-prec1', scalar_value=top1.avg, global_step=it)
        writer.add_scalar(tag='val-prec5', scalar_value=top5.avg, global_step=it)

    current_time = (time.time() - valid_tic) / 60
    cfg.valid_total_time += current_time
    print('\nValidate Iteration <{it:d}> --> '
          'Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.3f}\t '
          'CurrentValidTime {vtime:.4f} minutes\t TotalValidTime {ttime:.4f} minutes ****** \n'
          .format(it=it, top1=top1, top5=top5, losses=losses,
                  vtime=current_time, ttime=cfg.valid_total_time))

    return top1.avg, top5.avg, losses.avg


def main():
    cfg = Config()

    if torch.cuda.is_available():
        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enabled = True

    if cfg.random_seed is not None:
        random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        cudnn.deterministic = True

    # model = HRNet(cfg.arch_name)
    model = model_factory(cfg.arch_name, cfg.arch_kwargs, cfg.dataset, with_info=True)
    model = model.to(cfg.device)
    if len(cfg.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=cfg.gpu_ids)

    if cfg.ckpt_file:
        print('=> loading model from {}'.format(cfg.ckpt_file))
        xtils.load_ckpt_weights(model, cfg.ckpt_file, cfg.device, cfg.mgpus_to_sxpu)

    criterion = torch.nn.CrossEntropyLoss()

    # val_dataset = datasets.ImageFolder(os.path.join(cfg.data_root, 'val'),
    #                                    transforms.Compose([
    #                                        transforms.Resize(cfg.image_size),
    #                                        transforms.CenterCrop(cfg.input_size),
    #                                        transforms.ToTensor(),
    #                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                             std=[0.229, 0.224, 0.225]),
    #                                    ]))
    # val_loader = torch.utils.data.DataLoader(
    #     dataset=val_dataset,
    #     bsize_val=cfg.bsize_val,
    #     shuffle=False,
    #     num_workers=cfg.num_workers,
    #     pin_memory=True)

    _, val_loader, _ = data_factory(cfg.dataset, cfg.data_root, 256, cfg.bsize_val,
                                    cfg.data_augment, cfg.num_workers, result='loader')

    valid_model(val_loader, model, criterion, it=0, cfg=cfg, writer=None)


if __name__ == '__main__':
    main()
