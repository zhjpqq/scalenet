# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2018/12/15 12:17'

import os
import time
import random
import warnings
import argparse
import torch
import torch.backends.cudnn as cudnn
from config.configure import Config
from train_val_test import run_main_by_cfgs

# How To Use #########@########################
# cd to/your/project/dir
# nohup python run_main.py -name resnet -arch res50 -cfg cfgresxx -exp exp.resxx -gpu 1 3 1>resxx.out 2>&1 &


if __name__ == '__main__':

    # 声明设备配置
    seed = None and random.random()
    if torch.cuda.is_available():
        # 必须放在主程序入口，否则无法随意指定GPU_ID
        cudnn.benchmark = True
        cudnn.deterministic = False
        cudnn.enable = True
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        cudnn.deterministic = True
        warnings.warn('\nYou have set a fixed random seed: %s, which may slow down training speed!\n' % seed)

    # 声明参数配置
    cfg = Config()

    # 从命令行读取参数
    parser = argparse.ArgumentParser(description='Pytorch Cifar10 Training & Validation')
    parser.add_argument('-name', '--arch_name', type=str, default='', help='model name')
    parser.add_argument('-arch', '--arch_list', type=str, nargs='+', default='',
                        help='the keys list of <arch_kwargs> from exp_list')
    parser.add_argument('-cfg', '--cfg_dict', type=str, default='', help='the key of <config> from cfg_params')
    parser.add_argument('-exp', '--exp_version', type=str, default='', help='named as uxp.xxx, experiment version')
    parser.add_argument('-gpu', '--gpu_ids', type=int, nargs='+', default=[], help='which gpus used to train')
    args = parser.parse_args()

    # Pycharm模式下，直接给定参数，无需命令行输入
    # 命令行模式下，必须注释掉此处，使用命令行输入
    # args.arch_name = 'msnet'
    # args.arch_list = ['ms9']
    # args.cfg_dict = 'cfgms9'
    # args.exp_version = 'exp.ms9'
    # args.gpu_ids = [0]

    # args.arch_name = 'srnet'
    # args.arch_list = ['sr1']
    # args.cfg_dict = 'cfgsr1'
    # args.exp_version = 'exp.sr1'
    # args.gpu_ids = [0, 1, 2, 3]
    # print('\n=> Your Args is :', args, '\n')

    # args.arch_name = 'scalenet'
    # args.arch_list = ['ci7']
    # args.cfg_dict = 'cfgci7'
    # args.exp_version = 'exp.ci7'
    # args.gpu_ids = [0, 1]
    print('\n=> Your Args is :', args, '\n')

    # 从配置文件 <cfg_dict> 中读取参数
    import cfg_params as training_cfg

    cfg_dict = getattr(training_cfg, args.cfg_dict, None)
    del training_cfg
    assert isinstance(cfg_dict, dict)
    cfg.dict_to_class(cfg_dict, exclude=())

    # 用命令行参数替换<cfg_dict>中的旧参数
    if args.arch_name:
        cfg.arch_name = args.arch_name
    if args.exp_version:
        cfg.exp_version = args.exp_version
    if args.gpu_ids:
        cfg.gpu_ids = args.gpu_ids

    # 从构造文件 <arch_dict> 中读取模型参数
    import arch_params as training_arch

    arch_kwargs_list = [getattr(training_arch, arch, None) for arch in args.arch_list]
    del training_arch
    assert None not in arch_kwargs_list

    # 运行主训练程序
    for i, arch_kwargs in enumerate(arch_kwargs_list):
        cfg.arch_kwargs = arch_kwargs
        cfg.check_configs()
        run_main_by_cfgs(cfg)
