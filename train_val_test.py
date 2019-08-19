# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2018/12/15 12:17'
#  适用所有数据, 所有模型, 带冻结的训练过程x

import time
import os
import torch
from torch import nn
from tensorboardX import SummaryWriter

import xtils
from xtils import AverageMeter, accuracy, adjust_learning_rate
from factory.data_factory import data_factory
from factory.model_factory import model_factory
from config.configure import Config


# train-flow
def train_model(cfg=Config()):

    # resumer-flow
    if cfg.resume:
        if cfg.resume_config:
            assert os.path.isfile(cfg.resume), FileNotFoundError('\nCan not find the .ckpt file: %s ...' % cfg.resume)
            print("\nloading config from checkpoint file at %s ..." % (cfg.resume,))
            # checkpoint = torch.load(f=cfg.resume, map_location=cfg.device)
            cfg.dict_to_class(torch.load(f=cfg.resume)['config'], exclude=cfg.exclude_keys)
            cfg.start_iter = cfg.current_iter + 1
            print('loaded done at epoch {0} ......\n'.format(cfg.current_epoch))
            print('current training state: train_prec1-%0.4f val_prec1-%0.4f val-prec5-%0.4f\n' % (
                cfg.best_train_prec1, cfg.best_val_prec1, cfg.best_val_prec5))
            print('model will be saved at ckpt_dir: {0} ......\n'.format(cfg.ckpt_dir))
            print('log will be saved at log_dir: {0} ......\n'.format(cfg.log_dir))
            checkpoint = None
    print('\nExp-%s start ... \n' % (cfg.exp_version,))

    # data-flow
    train_loader, val_loader, test_loader = \
        data_factory(cfg.dataset, cfg.data_root, cfg.bsize_train, cfg.bsize_val,
                     cfg.data_augment, cfg.data_workers, result='loader', **cfg.data_kwargs)
    assert cfg.batch_nums == len(train_loader), '<batch_num> must == <data_info>[train_size]>/<batch_size>'
    assert cfg.max_iters % cfg.batch_nums == 0, '迭代次数不能整除数据集批次数,最后epoch中数据不完整'
    cfg.max_epochs = cfg.max_iters // cfg.batch_nums
    print('\nTrain BatchNums-> %s : Val BatchNums-> %s : Test BatchNums-> %s\n' %
          (len(train_loader), len(val_loader), len(test_loader)))

    # model-flow
    model, params, gflops, mdepth = model_factory(cfg.arch_name, cfg.arch_kwargs, cfg.dataset, with_info='return')
    model = model.to(cfg.device)
    if len(cfg.gpu_ids) > 1:
        model = nn.DataParallel(model, device_ids=cfg.gpu_ids)
    if cfg.resume:
        model = xtils.load_ckpt_weights(model, cfg.resume, cfg.device, cfg.mgpus_to_sxpu, strict=cfg.resume_strict)

    # logger-flow
    if cfg.train_val_test[0]:
        writer = SummaryWriter(cfg.log_dir)
        # writer.add_graph(model, torch.zeros(4, 3, 224, 224).to(cfg.device))

    # judger
    criterion = nn.CrossEntropyLoss()

    # optimizer
    if cfg.optim_custom:
        if isinstance(model, nn.DataParallel):
            optimizer = model.module.init_optimizer(cfg=cfg)
        else:
            optimizer = model.init_optimizer(cfg=cfg)
    else:
        optimizer = get_optimizer(model, cfg=cfg)
    if cfg.resume and cfg.resume_optimizer:
        optimizer.load_state_dict(torch.load(cfg.resume)['optimizer'])

    # batch iterations flow
    epoch = 0
    current_lr, data_iter = None, None
    epoch_tic = time.time()
    epoch_time = AverageMeter()
    if cfg.resume:
        epoch = cfg.start_iter // cfg.batch_nums
        cfg.current_epoch = epoch
        data_iter = iter(train_loader)
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()

    print('\n=> Warning: Train Val Test iterations will be start, please confirm your config at last!\n')
    cfg.show_config()

    # iteration-flow
    if cfg.train_val_test[0]:
        for it in range(cfg.start_iter, cfg.max_iters):

            cfg.current_iter = it
            batch_tic = time.time()

            if it == 0 or it % cfg.batch_nums == 0:
                epoch = it // cfg.batch_nums
                cfg.current_epoch = epoch
                if cfg.data_shuffle:
                    train_loader, _, _ = data_factory(cfg.dataset, cfg.data_root, cfg.bsize_train,
                                                      cfg.bsize_val, cfg.data_augment, cfg.data_workers,
                                                      result='loader', **cfg.data_kwargs)
                data_iter = iter(train_loader)
                losses = AverageMeter()
                top1 = AverageMeter()
                top5 = AverageMeter()
                batch_time = AverageMeter()
                data_time = AverageMeter()
                epoch_time.update(time.time() - epoch_tic)
                epoch_tic = time.time()

            if cfg.mode_custom:
                if isinstance(model, nn.DataParallel):
                    model.module.train_mode(it, cfg)
                else:
                    model.train_mode(it, cfg)
            else:
                model.train()

            current_lr = adjust_learning_rate(optimizer, it, cfg)

            # data to device
            tic = time.time()
            images, labels = next(data_iter)
            data_time.update(time.time() - tic)
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

            # forward
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                all_loss = [criterion(out, labels) for out in outputs]
                loss = sum(all_loss)
            else:
                loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if isinstance(outputs, (list, tuple)):
                loss = all_loss[cfg.xfc_which]
                outputs = outputs[cfg.xfc_which]

            # measure accuracy and record loss of
            # current batch(.val) and current epoch(.avg)
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - batch_tic)
            cfg.train_total_time += batch_time.val

            cfg.curr_train_prec1, cfg.curr_train_prec5 = top1.val, top5.val

            # 信息打印
            if (it + 1) % cfg.print_frequency == 0:
                print('Epoch: [{0}/{1}][{2}/{3} {4:.2f}%] '
                      'BatchTime:{batch_time.val:.4f}s({batch_time.avg:.4f}s) '
                      'DataTime:{data_time.val:.5f}s({data_time.avg:.5f}s) '
                      'Loss-{loss.val:.4f}({loss.avg:.4f}) '
                      'Prec@1-{top1.val:.3f}({top1.avg:.4f}) '
                      'Prec@5-{top5.val:.3f}({top5.avg:.4f}) '
                      'lr-{lr:.5f} {cost:.1f}Hours EpochTime:{epoch_time:.2f}Minutes'.format(
                    epoch, cfg.max_epochs, it, cfg.max_iters, 100 * it / cfg.max_iters,
                    batch_time=batch_time, data_time=data_time, loss=losses, top1=top1, top5=top5, lr=current_lr,
                    cost=cfg.train_total_time / 3600, epoch_time=epoch_time.val / 60))

            # 绘制曲线
            if (it + 1) % cfg.plot_frequency == 0:
                writer.add_scalar(tag='train-loss', scalar_value=losses.val, global_step=it)
                writer.add_scalar(tag='train-prec1', scalar_value=top1.val, global_step=it)
                writer.add_scalar(tag='train-prec5', scalar_value=top5.val, global_step=it)
                writer.add_scalar(tag='learning-rate', scalar_value=current_lr, global_step=it)

            # 模型验证
            if cfg.train_val_test[1] and ((it + 1) >= cfg.val_frequency[0] and (it + 1) % cfg.val_frequency[1] == 0):
                cfg.curr_val_prec1, cfg.curr_val_prec5, curr_val_loss \
                    = valid_model(val_loader, model, criterion, it, cfg, writer)

            # 模型测试
            if cfg.train_val_test[2] and ((it + 1) >= cfg.test_frequency[0] and (it + 1) % cfg.test_frequency[1] == 0):
                curr_test_prec1, curr_test_prec5 = test_model(test_loader, model, it, cfg)

            # 记录历史最佳值, 并保存其模型
            # 大于多少迭代次(或迭代回合)再开始发现最大值，防止开始时保存太多
            cfg.best_val_prec1 = max(cfg.curr_val_prec1, cfg.best_val_prec1)
            cfg.best_train_prec1 = max(cfg.curr_train_prec1, cfg.best_train_prec1)
            cfg.best_val_prec5 = max(cfg.curr_val_prec5, cfg.best_val_prec5)
            cfg.best_train_prec5 = max(cfg.curr_train_prec5, cfg.best_train_prec5)

            if (it + 1) >= cfg.batch_nums * cfg.best_prec['best_start'] and cfg.best_prec['val_prec1'] < cfg.curr_val_prec1:
                cfg.best_prec['val_prec1'] = cfg.curr_val_prec1
                cfg.best_prec['best_ok'] = True
                cfg.ckpt_suffix = '-best'
            else:
                cfg.best_prec['best_ok'] = False
                cfg.ckpt_suffix = '-norm'

            # 保存模型
            if ((it + 1) >= cfg.save_frequency[0] and (it + 1) % cfg.save_frequency[1] == 0) \
                    or it == cfg.max_iters - 1 or cfg.best_prec['best_ok']:
                checkpoint = {'model': model.state_dict(),
                              'config': cfg.class_to_dict(),
                              'optimizer': optimizer.state_dict()}
                filename = '%s-%s%s-ep%s-it%d-acc%.2f-best%.2f-topv%.2f-par%.2fM%s-%s.ckpt' % \
                           (cfg.dataset, cfg.arch_name, mdepth, epoch, it, cfg.curr_val_prec1,
                            cfg.best_val_prec1, cfg.best_val_prec5, params, cfg.ckpt_suffix, cfg.exp_version)
                if not os.path.exists(cfg.ckpt_dir):
                    os.makedirs(cfg.ckpt_dir)
                print('\n *** Model will be saved at: %s ..., %s \n' % (cfg.ckpt_dir, filename))
                torch.save(checkpoint, f=os.path.join(cfg.ckpt_dir, filename))

    if cfg.train_val_test[1]:
        valid_model(val_loader, model, criterion, 0, cfg, writer=None)

    if cfg.train_val_test[2]:
        test_model(test_loader, model, 0, cfg)


# val-flow
def valid_model(val_loader, model, criterion, it, cfg, writer=None):
    valid_tic = time.time()
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if cfg.mode_custom:
        if isinstance(model, nn.DataParallel):
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
          'prec@1-{top1.avg:.3f} prec@5-{top5.avg:.3f} Loss {losses.avg:.3f}\t '
          'CurrentValidTime {vtime:.4f} minutes\t TotalValidTime {ttime:.4f} minutes ****** \n'
          .format(it=it, top1=top1, top5=top5, losses=losses,
                  vtime=current_time, ttime=cfg.valid_total_time))

    return top1.avg, top5.avg, losses.avg


# test-flow
def test_model(test_loader, model, it, cfg):
    print('\nModel testing start ...\n')
    test_tic = time.time()
    top1 = AverageMeter()
    top5 = AverageMeter()

    if cfg.mode_custom:
        if isinstance(model, nn.DataParallel):
            model.module.eval_mode(it, cfg)
        else:
            model.eval_mode(it, cfg)
    else:
        model.eval()

    with torch.no_grad():
        cost = AverageMeter()  # 处理图片耗时
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(cfg.device)
            labels = labels.to(cfg.device)

            xtic = time.time()
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
            cost.update(time.time() - xtic, images.size(0))

            if isinstance(outputs, (list, tuple)):
                if isinstance(cfg.xfc_which, int):
                    outputs = outputs[cfg.xfc_which]
                elif cfg.xfc_which == 'all-avg':
                    outputs = sum(outputs) / len(outputs)
                elif cfg.xfc_which == 'aux-avg':
                    outputs = sum(outputs[:-1]) / len(outputs[:-1])
                else:
                    raise NotImplementedError

            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

    current_time = (time.time() - test_tic) / 60
    cfg.test_total_time += current_time

    print('\n*** Test Model at Iteration {it:d} --> '
          'Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\t '
          'CurrentTestTime {vtime:.4f} minutes\t TotalTestTime {ttime:.4f} minutes, '
          'SingleImage-AvgTestTime {sitime:.5f}*** \n'
          .format(it=it, top1=top1, top5=top5, vtime=current_time, ttime=cfg.test_total_time, sitime=cost.avg))
    return top1.avg, top5.avg


def get_optimizer(model, cfg):
    parameters = model.parameters()
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    if cfg.optim_type == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=cfg.lr_start, weight_decay=cfg.weight_decay)
    elif cfg.optim_type == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=cfg.lr_start, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay, nesterov=cfg.nesterov)
    elif cfg.optim_type == 'RMSPROP':
        optimizer = torch.optim.RMSprop(
            parameters,
            lr=cfg.lr_start,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            alpha=cfg.rmsprop_alpha,
            centered=cfg.rmsprop_centered
        )
    else:
        raise NotImplementedError
    return optimizer


# adjust-params-flow
def run_main_by_cfgs(cfg):
    mtic = time.time()
    train_model(cfg)
    print('\nExp-%s Time Cost Is %.4f Hours ... \n' % (cfg.exp_version, (time.time() - mtic) / 3600))
