# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2018/12/15 12:17'

import torch
import shutil
import os, time
import math
import numpy as np
from datetime import datetime
from collections import OrderedDict
from torchvision import transforms
from collections import namedtuple
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TopkAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, k=5):
        self.k = k
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.topk = [0]

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.topk = sorted(record_topk_value(self.topk, val, self.k))


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')


def adjust_learning_rate_org(optimizer, epoch, lr_start=0.01, decay_rate=0.1, decay_time=30):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr_start * (decay_rate ** (epoch // decay_time))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_learning_rate(optimizer, epoch, cfg=None):
    """
    Policy1: 'regular':
    lr will decay *rate from lr_start, for every n epoch, when the epoch > start

    lr_start = 0.01
    decay_policy = 'regular'
    decay_rate = 0.1
    decay_time = n
    decay_start = start

    Policy2: 'appoint':
    lr will decay *rate from lr_start, for the epoch appointed in [(rate1, ep1), (rate2, ep2), ...]
    """
    lr_start, lr_end, decay_policy, decay_rate, decay_time, decay_start, decay_appoint = \
        cfg.lr_start, cfg.lr_end, cfg.lr_decay_policy, cfg.lr_decay_rate, cfg.lr_decay_time, cfg.lr_decay_start, cfg.lr_decay_appoint

    current_lr = optimizer.param_groups[0]['lr']
    if decay_policy == 'regular':
        if epoch >= decay_start:
            current_lr = lr_start * (decay_rate ** ((epoch - decay_start) // decay_time + 1))
            if current_lr <= lr_end:
                current_lr = lr_end
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        return current_lr

    elif decay_policy == 'appoint':
        for ep, rate in decay_appoint:
            if epoch == ep:
                current_lr = current_lr * rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
        return current_lr

    elif decay_policy == 'original':
        start_epoch = 0 if cfg.start_epoch == 0 else 1
        current_lr = lr_start * (decay_rate ** ((epoch - start_epoch) // decay_time))
        if current_lr <= lr_end:
            current_lr = lr_end
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        return current_lr

    else:
        raise NotImplementedError


def adjust_batch_size(current_bsize, epoch, cfg):
    if cfg.bs_decay_policy == 'frozen':
        return current_bsize

    if cfg.bs_decay_policy == 'appoint':
        for ep, rate in cfg.bs_decay_appoint:
            if epoch == ep:
                current_bsize = current_bsize * rate
        return current_bsize

    if cfg.bs_decay_policy == 'regular':
        if epoch >= cfg.bs_decay_start:
            if current_bsize <= cfg.bsize_end:
                current_bsize = cfg.bsize_end
            else:
                decay_rate = cfg.bs_decay_rate ** ((epoch - cfg.bs_decay_start) // cfg.bs_decay_interval + 1)
                current_bsize = cfg.bsize_start * decay_rate
        return current_bsize


def resume_from_ckpt(model, optimizer, resume):
    if os.path.isfile(resume):
        print("\nloading checkpoint file from %s ..." % (resume,))
        checkpoint = torch.load(f=resume)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_prec1 = checkpoint['best_prec1']
        print('loaded done at epoch {} ...\n'.format(start_epoch))
        return start_epoch, best_prec1
    else:
        raise FileNotFoundError('\ncan not find the ckpt file @ %s ...' % resume)


def model_from_ckpt(model, ckpt):
    if os.path.isfile(ckpt):
        print("\nloading checkpoint file from %s ..." % (ckpt,))
        checkpoint = torch.load(f=ckpt)
        try:
            model.load_state_dict(checkpoint['state_dict'])
        except KeyError:
            model.load_state_dict(checkpoint['model'])
        except:
            raise KeyError('check model KEY name in ckpt file.')
        return model
    else:
        raise FileNotFoundError('check ckpt file exist!')


def calculate_params_scale(model, format=''):
    if isinstance(model, str) and model.endswith('.ckpt'):
        checkpoint = torch.load(model)
        try:
            model = checkpoint['state_dict']
        except KeyError:
            model = checkpoint['model']
        except:
            raise KeyError('Please check the model KEY in ckpt!')
    scale = 0

    if isinstance(model, torch.nn.Module):
        # method 1
        scale = sum([param.nelement() for param in model.parameters()])
        # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        # scale = sum([np.prod(p.size()) for p in model_parameters])

    elif isinstance(model, OrderedDict):
        # method 3
        for key, val in model.items():
            if not isinstance(val, torch.Tensor):
                continue
            scale += val.numel()

    if format == 'million':  # (百万)
        scale /= 1000000
        print("\n*** Number of params: " + str(scale) + '\tmillion...\n')
        return scale
    else:
        print("\n*** Number of params: " + str(scale) + '\t...')
        return scale


def calculate_FLOPs_scale(model, input_size, multiply_adds=False, use_gpu=False):
    """
    forked from FishNet @ github
    https://www.zhihu.com/question/65305385/answer/256845252
    https://blog.csdn.net/u011501388/article/details/81061024
    https://blog.csdn.net/xidaoliang/article/details/88191910

    no bias: K^2 * IO * HW
    multiply_adds : False in FishNet Paper, but True in DenseNet paper
    """
    assert isinstance(model, torch.nn.Module)

    USE_GPU = use_gpu and torch.cuda.is_available()

    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_conv.append(flops)

    def deconv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups) * (
            2 if multiply_adds else 1)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_deconv.append(flops)

    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement())

    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = output_channels * (kernel_ops + bias_ops)
        flops = batch_size * params * output_height * output_width

        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.ConvTranspose2d):
                net.register_forward_hook(deconv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            return
        for c in childrens:
            foo(c)

    multiply_adds = multiply_adds
    list_conv, list_deconv, list_bn, list_relu, list_linear, list_pooling = [], [], [], [], [], []
    foo(model)

    input = torch.rand(2, 3, input_size, input_size)
    if USE_GPU:
        input = input.cuda()
        model = model.cuda()
    _ = model(input)

    total_flops = (sum(list_conv) + sum(list_deconv) + sum(list_linear)
                   + sum(list_bn) + sum(list_relu) + sum(list_pooling))
    print('  + Number of FLOPs: %.5fG' % (total_flops / 1e9 / 2))


def calculate_layers_num(model, layers=('conv2d', 'classifier')):
    assert isinstance(model, torch.nn.Module)
    type_dict = {'conv2d': torch.nn.Conv2d,
                 'bnorm2d': torch.nn.BatchNorm2d,
                 'relu': torch.nn.ReLU,
                 'fc': torch.nn.Linear,
                 'classifier': torch.nn.Linear,
                 'linear': torch.nn.Linear,
                 'deconv2d': torch.nn.ConvTranspose2d}
    nums_list = []

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, type_dict[layer]):
                pass
            return 1
        count = 0
        for c in childrens:
            count += foo(c)
        return count

    def foo2(net, layer):
        count = 0
        for n, m in net.named_modules():
            if isinstance(m, type_dict[layer]):
                count += 1
        return count

    for layer in layers:
        # nums_list.append(foo(model))
        nums_list.append(foo2(model, layer))
    total = sum(nums_list)

    strtip = ''
    for layer, nums in zip(list(layers), nums_list):
        strtip += ', %s: %s' % (layer, nums)
    print('\n*** Number of layers: %s %s ...\n' % (total, strtip))
    return total


def calculate_time_cost(model, insize=32, toc=1, use_gpu=False, pritout=False):
    if not use_gpu:
        x = torch.randn(4, 3, insize, insize)
        tic, toc = time.time(), toc
        y = [model(x) for _ in range(toc)][0]
        toc = (time.time() - tic) / toc
        print('处理时间: %.5f 秒\t' % toc)
        if not isinstance(y, (list, tuple)):
            y = [y]
        if pritout:
            print('预测输出: %s 个xfc.' % len(y), [yy.max(1) for yy in y])
        return y
    else:
        assert torch.cuda.is_available()
        x = torch.randn(4, 3, insize, insize)
        model, x = model.cuda(), x.cuda()
        tic, toc = time.time(), toc
        y = [model(x) for _ in range(toc)][0]
        toc = (time.time() - tic) / toc
        print('处理时间: %.5f 秒\t' % toc)
        if not isinstance(y, (list, tuple)):
            y = [y]
        if pritout:
            print('预测输出: %s 个xfc.' % len(y), [yy.max(1) for yy in y])
        return y


def get_model_summary(model, insize=224, item_length=26, verbose=False):
    """
    forked from HRNet-cls
    """
    summary = []

    input_tensors = torch.rand((1, 3, insize, insize))

    ModuleDetails = namedtuple("Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
                    class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                        torch.prod(
                            torch.LongTensor(list(module.weight.data.size()))) *
                        torch.prod(
                            torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
                and not isinstance(module, nn.Sequential) \
                and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
                  os.linesep + \
                  "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                      ' ' * (space_len - len("Name")),
                      ' ' * (space_len - len("Input Size")),
                      ' ' * (space_len - len("Output Size")),
                      ' ' * (space_len - len("Parameters")),
                      ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                  + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                       + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
               + "Total Parameters: {:,}".format(params_sum) \
               + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(
        flops_sum / (1024 ** 3)) \
               + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details


def tensorboard_add_model(model, x, comment=''):
    assert isinstance(model, torch.nn.Module)
    assert isinstance(x, torch.Tensor)
    from tensorboardX import SummaryWriter
    current_time = datetime.now().strftime('%b%d_%H:%M-graph--')
    log_dir = os.path.join('runs', current_time + model._get_name() + comment)
    writer = SummaryWriter(log_dir)
    writer.add_graph(model, x)
    print('\n*** Model has been add to tensorboardX graph dir: %s ...\n' % (log_dir,))


def find_max_index(dir, sign1='-exp', sign2='.ckpt'):
    files = list(os.walk(dir))[0][2]
    index = [0]
    for f in files:
        if sign1 in f and sign2 in f:
            f = f.split(sign1)[1].split(sign2)[0]
            index.append(int(f))
    return max(index)


def find_max_index2(dir, sign1='-exp'):
    print('\n*** try to find max exp index in dir: %s ***\n' % dir)
    files = list(os.walk(dir))[0][1]
    index = [0]
    for f in files:
        if sign1 in f:
            f = f.split(sign1)[1]
            index.append(int(f))
    return max(index)


def print_size(x, ok=True):
    if not ok:
        return
    if isinstance(x, torch.Tensor):
        print(x.size())
    elif isinstance(x, (list, tuple)):
        for xx in x:
            if isinstance(xx, torch.Tensor):
                print(xx.size())


def record_topk_value(record, val, k=5):
    # record the max topk value
    assert isinstance(record, list)
    if len(record) < k:
        record.append(val)
        return record
    elif len(record) > k:
        record = sorted(record)[::-1]
        if min(record) > val:
            return record[:k]
        else:
            record = record[:k]
            record = record_topk_value(record, val, k)
            return record
    else:
        min_val = min(record)
        if min_val >= val:
            return record
        else:
            record.pop(record.index(min_val))
            record.append(val)
            return record


def plot_log(log_path='./logs/log.txt'):
    # forked from  https://github.com/prlz77/ResNeXt.pytorch
    import re
    import matplotlib.pyplot as plt

    file = open(log_path, 'r')
    accuracy = []
    epochs = []
    loss = []
    for line in file:
        test_accuracy = re.search('"test_accuracy": ([0]\.[0-9]+)*', line)
        if test_accuracy:
            accuracy.append(test_accuracy.group(1))

        epoch = re.search('"epoch": ([0-9]+)*', line)
        if epoch:
            epochs.append(epoch.group(1))

        train_loss = re.search('"train_loss": ([0-9]\.[0-9]+)*', line)
        if train_loss:
            loss.append(train_loss.group(1))
    file.close()
    plt.figure('test_accuracy vs epochs')
    plt.xlabel('epoch')
    plt.ylabel('test_accuracy')
    plt.plot(epochs, accuracy, 'b*')
    plt.plot(epochs, accuracy, 'r')
    plt.grid(True)

    plt.figure('train_loss vs epochs')
    plt.xlabel('epoch')
    plt.ylabel('train_loss')
    plt.plot(epochs, loss, 'b*')
    plt.plot(epochs, loss, 'y')
    plt.grid(True)

    plt.show()


def tensor_to_img(x, ishow=False, istrans=False, file_name='xxxx', save_dir=''):
    maxv, minv, meanv = x.max(), x.min(), x.mean()

    x = x[0, 0:3, :, :].squeeze(0)

    if istrans:
        x = ((x - minv) / (maxv - minv)) * 255
        maxv, minv, meanv = x.max(), x.min(), x.mean()

    img = transforms.ToPILImage()(x)

    if ishow:
        img.show()

    if save_dir:
        file_name = file_name + '_' + str(time.time()).replace('.', '')[0:14] + '.png'
        file_path = os.path.join(save_dir, file_name)
        img.save(file_path)
        print('img has been saved at %s ' % file_path)


def get_pretrained_path(model_dir, arch_name='resnet18'):
    """
    :model_dir:  存放预训练模型的文件夹，将所有预训练模型集中存放在此文件夹内
    :arch_name:  根据模型名称找到对应的路径
    :noraise:    找不到时不提示错误
    :return：    the path of the pretrained model for torch.load(model_path)
    """

    if os.path.isfile(model_dir):
        return model_dir

    elif '.pth' in arch_name or '.tar' in arch_name:
        model_path = os.path.join(model_dir, arch_name)
        if os.path.isfile(model_path):
            return model_path
        else:
            raise FileNotFoundError('%s' % model_path)

    arch_name_list = [
        'vgg11-bbd30ac9.pth',
        'vgg19-dcbb9e9d.pth',
        'resnet18-5c106cde.pth',
        'resnet34-333f7ec4.pth',
        'resnet50-19c8e357.pth',
        'resnet101-5d3b4d8f.pth',
        'resnet152-b121ed2d.pth',
        'densenet121-a639ec97.pth',
        'densenet169-b2777c0a.pth',
        'densenet201-c1103571.pth',
        'densenet161-8d451a50.pth',

        'fishnet99_ckpt.tar',
        'fishnet150_ckpt.tar',
        'fishnet15x_ckpt_welltrain-==fishnet150.tar',

        'mobilev3_y_large-657e7b3d.pth',
        'mobilev3_y_small-c7eb32fe.pth',
        'mobilev3_x_large.pth.tar',
        'mobilev3_x_small.pth.tar',
        'mobilev3_d_small.pth.tar',
    ]
    arch = arch_name
    arch_name = [name for name in arch_name_list if name.startswith(arch_name)]
    if len(arch_name) == 1:
        arch_name = arch_name[0]
    elif len(arch_name) > 1:
        raise Warning('too much choices for %s ... !' % arch)
    else:
        raise Warning('no checkpoint exist for %s... !' % arch)
    model_path = os.path.join(model_dir, arch_name)
    return model_path


def load_ckpt_weights(model, ckptf, device='cpu', mgpus_to_sxpu='none', noload=False, strict=True):
    """
    MultiGpu.ckpt/.model 与 SingleXpu.model/.ckpt 之间转换加载

    m2s: MultiGpu.ckpt -> SingleXpu.model ; remove prefix 'module.'
    s2m: SingleXpu.ckpt -> MultiGpu.model ; add prefix 'module.'
    none: MultiGpu -> MultiGpu or SingleXpu -> SingleXpu ; 无需转换直接加载.
    auto: 轮流选择上述三种情况，直到加载成功
    """

    def remove_module_dot(old_state_dict):
        # remove the prefix 'module.' of nn.DataParallel
        state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            state_dict[k[7:]] = v
        return state_dict

    def add_module_dot(old_state_dict):
        # add the prefix 'module.' to nn.DataParallel
        state_dict = OrderedDict()
        for k, v in old_state_dict.items():
            state_dict['module.' + k] = v
        return state_dict

    if isinstance(device, torch.device):
        pass
    elif device == 'cpu':
        device = torch.device(device)
    elif device == 'gpu':
        device = torch.device('cuda:0')
    elif device.startswith('cuda:'):
        device = torch.device(device)
    else:
        raise NotImplementedError

    model = model.to(device)
    if noload:
        return model

    print('\n=> loading model.pth from %s ' % ckptf)
    assert os.path.isfile(ckptf), '指定路径下的ckpt文件未找到. %s' % ckptf
    assert mgpus_to_sxpu in ['auto', 'm2s', 's2m', 'none']

    ckpt = torch.load(f=ckptf, map_location=device)

    if 'state_dict' in ckpt.keys():
        state_dict = ckpt['state_dict']
    elif 'model' in ckpt.keys():
        state_dict = ckpt['model']
    else:
        # ckpt is jus the state_dict.pth!
        state_dict = ckpt

    if mgpus_to_sxpu == 'auto':
        try:
            model.load_state_dict(state_dict, strict)
        except:
            try:
                model.load_state_dict(remove_module_dot(state_dict), strict)
            except:
                try:
                    model.load_state_dict(add_module_dot(state_dict), strict)
                except:
                    print('\n=> Error: key-in-model and key-in-ckpt not match, '
                          'not because of the  prefrex "module." eg. "." cannot be exist in key.\n')
                    model.load_state_dict(state_dict, strict)
        print('\nSuccess: loaded done from %s \n' % ckptf)
        return model

    elif mgpus_to_sxpu == 'm2s':
        state_dict = remove_module_dot(state_dict)
    elif mgpus_to_sxpu == 's2m':
        state_dict = add_module_dot(state_dict)
    elif mgpus_to_sxpu == 'none':
        state_dict = state_dict

    model.load_state_dict(state_dict, strict)
    print('\nSuccess: loaded done from %s \n' % ckptf)
    return model


def linear_map(a, b, x):
    """
    线性映射x到区间[a, b]
    :return:
    """
    assert max(x) != min(x)
    assert isinstance(x, np.ndarray)
    return (x - min(x)) / (max(x) - min(x)) * (b - a) + a


def startwithxyz(it, xyz=()):
    # it startwith x or y or z ?
    assert isinstance(it, str)
    assert isinstance(xyz, (tuple, list))
    isok = [it.startswith(x) for x in xyz]
    return bool(sum(isok))


class Curves(object):
    """
    为 weight-decay 提供取值曲线
    """

    def __init__(self, ep=None):
        self.ep = ep
        super(Curves, self).__init__()

    def func1(self, x):
        if self.ep is None:
            self.ep = 8
        return round(x, self.ep)

    def func2(self, x):
        if self.ep is None:
            self.ep = 3
        return x ** self.ep

    def func3(self, x):
        if self.ep is None:
            self.ep = 3
        return x ** (1 / self.ep)

    def func4(self, x):
        return math.exp(x)

    def func5(self, x):
        return math.exp(-x)

    def func6(self, x):
        return 1 - math.exp(x)


def GCU(m, n):
    # 欧几里得辗转相除法求最大公约数
    # https://www.cnblogs.com/todayisafineday/p/6115852.html
    if not n:
        return m
    else:
        return GCU(n, m % n)


def get_xfc_which(it, xfc_which):
    """
     - it: 当前迭代次数
     - xfc_which: {0 * BN: -1, 20 * BN: -2, 30 * BN: -3}
    """
    if isinstance(xfc_which, int):
        return xfc_which
    elif isinstance(xfc_which, str):
        return xfc_which
    elif isinstance(xfc_which, dict):
        which = None
        for ite in sorted(xfc_which.keys())[::-1]:
            if it >= ite:
                which = xfc_which[ite]
                break
        if which is None:
            raise NotImplementedError
        return which
    else:
        raise NotImplementedError


# 根据设备进行路径配置, 更换设备后直接在此配置即可
# including 预训练模型路径，数据路径，当前实验路径

def get_current_device(device=0):
    if isinstance(device, int):
        device_list = ['1080Ti', 'titan', 'mic251', 'dellcpu', 'new-device']
        device = device_list[device]
    elif isinstance(device, str):
        device = device
    else:
        raise NotImplementedError
    return device


def get_pretrained_models():
    device = get_current_device()

    model_dir = {'1080Ti': '/datah/zhangjp/PreTrainedModels',
                 'titan': '/data/zhangjp/PreTrainedModels',
                 'mic251': '/DATA/251/jpzhang/Projects/PreTrainedModels',
                 'dellcpu': 'E://PreTrainedModels',
                 'new-device': ''}
    model_dir = model_dir[device]
    return model_dir


def get_data_root(data='imagenet || cifar10 || cifar100 || svhn || ***'):
    device = get_current_device()

    class Dataset(object):
        imagenet = {
            '1080Ti': ['/ImageNet2012/', '/data0/ImageNet_ILSVRC2012'][0],
            'titan': '/data/dataset/ImageNet2012',
            'mic251': '/data1/jpzhang/datasets/imagenet/',
            'new-device': '',
        }
        cifar10 = {
            '1080Ti': '/data0/cifar10/',
            'titan': '/data/dataset/cifar-10-batches-py/',
            'mic251': '/data1/jpzhang/datasets/cifar10/',
            'new-device': '',
        }
        cifar100 = {
            '1080Ti': '/data0/cifar100/',
            'titan': '/data/dataset/cifar-100-python/',
            'mic251': '/data1/jpzhang/datasets/cifar100/',
            'new-device': '',
        }
        svhn = {
            '1080Ti': '',
            'titan': '',
            'mic251': '',
            'new-device': '',
        }

    data_root = getattr(Dataset(), data.lower())[device]
    return data_root


def get_base_dir(k='ckpt || log'):
    device = get_current_device()

    assert k in ['ckpt', 'log']
    ckpt_base_dir = {'local': '.',
                     '1080Ti': '/data1/zhangjp/classify/checkpoints',
                     'titan': '/backup/zhangjp/classify/checkpoints',
                     'mic251': '/DATA/251/jpzhang/Projects/checkpoints',
                     'new-device': ''}
    log_base_dir = {'local': '.',
                    '1080Ti': '/data1/zhangjp/classify/runs',
                    'titan': '/backup/zhangjp/classify/runs',
                    'mic251': '/DATA/251/jpzhang/Projects/runs',
                    'new-device': ''}
    if k == 'ckpt':
        return ckpt_base_dir[device]
    else:
        return log_base_dir[device]


if __name__ == '__main__':
    import torchvision as tv
    from xmodels.scalenet import ScaleNet

    imgnet = {'stages': 3, 'depth': 22, 'branch': 3, 'rock': 'U', 'kldloss': False,
              'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
              'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 22, 2 * 22),
              'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 1000,
              'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
              'summer': 'split', 'afisok': False, 'version': 2}

    model = tv.models.resnet18()
    x = torch.Tensor(1, 3, 224, 224)
    print(model._modules.keys())
    v = model.layer1
    pred = model(x)
    print(pred.max(1))

    model = ScaleNet(**imgnet)
    x = torch.Tensor(1, 3, 256, 256)
    pred = model(x)
    pred = pred[0]
    print(pred.max(1))
    print(model)
