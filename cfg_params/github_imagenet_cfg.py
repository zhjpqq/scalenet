import math, os, time
from math import ceil, floor
import xtils

# how-to-use: test a model.pth by run_main.py #####

# args.arch_name = 'scalenet'
# args.arch_list = ['vo69']
# args.cfg_dict = 'cfgscale'
# args.exp_version = 'exp.test'
# args.gpu_ids = [1, 0]

# args.arch_name = 'resnet'
# args.arch_list = ['res50']
# args.cfg_dict = 'cfgtest'
# args.exp_version = 'exp.test'
# args.gpu_ids = [0, 1]

# args.arch_name = 'fishnet'
# args.arch_list = ['fish150']
# args.cfg_dict = 'cfgtest'
# args.exp_version = 'exp.test'
# args.gpu_ids = [1, 0]

# args.arch_name = 'mobilev3'
# args.arch_list = ['mbvxs']
# args.cfg_dict = 'cfgtest'
# args.exp_version = 'exp.test'
# args.gpu_ids = [0, 1]

# args.arch_name = 'hrnet'
# args.arch_list = ['hrw18']
# args.cfg_dict = 'cfgtest'
# args.exp_version = 'exp.test'

# args.arch_name = 'effnet'
# args.arch_list = ['effb3']
# args.cfg_dict = 'cfgtest'
# args.exp_version = 'exp.test'
# args.gpu_ids = [1]

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 1281167
batch_size = 256
batch_size_val = 12
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit    #5005

cfgtest = {
    # experiment config
    'exp_version': 'exp.xxx',
    'train_val_test': (False, True, False),

    # device config
    'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8][0:4],

    # model config
    'arch_name': 'xxx',
    'arch_kwargs': {},
    'resume': '', # os.path.join(xtils.get_pretrained_models(), 'resnet50-19c8e357.pth'),
    'resume_config': False,
    'resume_optimizer': False,
    'mgpus_to_sxpu': ['m2s', 's2m', 'none', 'auto'][3],

    # data config
    'dataset': 'imagenet',
    'data_info': {'train_size': train_size, 'val_size': 50000, 'test_size': 50000},
    'data_root': xtils.get_data_root(data='imagenet'),
    'data_augment': {'train': 'rotate-rresize-1crop', 'val': '1resize-1crop',
                     'imsize': 256, 'insize': 224, 'color': True,
                     'degree': (0, 0), 'scale': (0.08, 1), 'ratio': (3. / 4, 4. / 3)},
    'data_kwargs': {},
    'data_workers': 4,

    # path config
    'current_time': '',
    'ckpt_suffix': '',  # when save a ckpt, u can add a special mark to its filename.
    'ckpt_base_dir': xtils.get_base_dir(k='ckpt'),
    'ckpt_dir': 'auto-setting',
    'log_base_dir': xtils.get_base_dir(k='log'),
    'log_dir': 'auto-setting',

    # iter config
    'start_iter': 0,
    'max_iters': [100 * BN, 90 * BN, 60 * BN, 40 * BN, 120 * BN][1],
    'start_epoch': 0,
    'max_epochs': 0,
    'bsize_train': batch_size,
    'bsize_val': batch_size_val,
    'batch_nums': batch_nums,
    'Unit': {'epoch': batch_nums, 'iter': 1}[['epoch', 'iter'][0]],  # 按epoch为单位调节 还是按iteration为单位调节lr/bs？
    'BN': BN,

    # lr config
    'optim_type': ['Adam', 'SGD'][1],
    'lr_start': {'Adam': 0.01, 'SGD': 0.1}['SGD'],
    'lr_end': 0.0,
    'lr_decay_policy': ['regular', 'appoint', 'original', 'trace_prec'][1],
    'lr_decay_start': 1 * BN,
    'lr_decay_rate': 1 / 10,
    'lr_decay_time': 15 * BN,
    'lr_decay_appoint': ((30 * BN, 1 / 10), (60 * BN, 1 / 10), (90 * BN, 1 / 10)),
    'momentum': 0.9,
    'weight_decay': [0.0001, 0.0005, 0.00017, 0.00002][0],

    # frequency config
    # # Note: if val_freq: (0, plot_freq)
    # # the loss-curve/prec-curve of train and val can have same x-axis point.
    'best_prec': {'train_prec1': 0, 'train_prec5': 0, 'val_prec1': 0, 'val_prec5': 0,
                  'best_start': 3, 'best_ok': False},
    'print_frequency': BN // 100,
    'plot_frequency': BN // 100,  # 5005/100=50
    'val_frequency': (0 * BN, BN // 1),
    'test_frequency': (999 * BN, BN // 1),
    'save_frequency': (0 * BN, BN // 1),

    # forzen config
    'mode_custom': False,
    'train_which': [],
    'eval_which': [],
    'xfc_which': -1,

    # time config
    'valid_total_time': 0,
    'test_total_time': 0,
    'exp_tic': time.time(),

    'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume'),
}

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 1281167
batch_size = 256
batch_size_val = 128
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit    #5005

cfgres1 = {
    # experiment config
    'exp_version': 'exp.xxx',
    'train_val_test': (True, True, True),

    # device config
    'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8][0:4],

    # model config
    'arch_name': 'xxx',
    'arch_kwargs': {},
    'resume': None or '/data1/zhangjp/classify/checkpoints/imagenet/resnet/resnet-exp.res2/'
              + 'imagenet-resnet54-ep81-it410409-acc75.76-best75.81-topv92.80-par25.56M-norm-exp.res2.ckpt',
    'resume_config': True,
    'resume_optimizer': True,
    'mgpus_to_sxpu': ['m2s', 's2m', 'none', 'auto'][3],

    # data config
    'dataset': 'imagenet',
    'data_info': {'train_size': train_size, 'val_size': 50000, 'test_size': 50000},
    'data_root': xtils.get_data_root(data='imagenet'),
    'data_augment': {'train': 'rotate-rresize-1crop', 'val': '1resize-1crop',
                     'imsize': 256, 'insize': 224, 'color': True, 'interp': 'bilinear',
                     'degree': (0, 0), 'scale': (0.08, 1), 'ratio': (3. / 4, 4. / 3)},
    'data_kwargs': {},
    'data_workers': 16,

    # path config
    'current_time': '',
    'ckpt_suffix': '',  # when save a ckpt, u can add a special mark to its filename.
    'ckpt_base_dir': xtils.get_base_dir(k='ckpt'),
    'ckpt_dir': 'auto-setting',
    'log_base_dir': xtils.get_base_dir(k='log'),
    'log_dir': 'auto-setting',

    # iter config
    'start_iter': 0,
    'max_iters': [90 * BN, 100 * BN, 60 * BN, 40 * BN, 120 * BN][1],
    'start_epoch': 0,
    'max_epochs': 0,
    'bsize_train': batch_size,
    'bsize_val': batch_size_val,
    'batch_nums': batch_nums,
    'Unit': {'epoch': batch_nums, 'iter': 1}[['epoch', 'iter'][0]],  # 按epoch为单位调节 还是按iteration为单位调节lr/bs？
    'BN': BN,

    # lr config
    'optim_type': ['Adam', 'SGD', 'RMSPROP'][1],
    'lr_start': {'Adam': 0.01, 'SGD': 0.1, 'RMSPROP': 0.1}['SGD'],
    'lr_decay_policy': ['regular', 'appoint', 'original', 'trace_prec'][1],
    'lr_decay_appoint': ((30 * BN, 1 / 10), (60 * BN, 1 / 10), (90 * BN, 1 / 10)),
    'momentum': 0.9,
    'weight_decay': [0.0001, 0.0005, 0.00017, 0.00002][0],
    'nesterov': False,
    'rmsprop_alpha': '',
    'rmsprop_centered': '',

    # frequency config
    # # Note: if val_freq: (0, plot_freq)
    # # the loss-curve/prec-curve of train and val can have same x-axis point.
    'best_prec': {'train_prec1': 0, 'train_prec5': 0, 'val_prec1': 0, 'val_prec5': 0,
                  'best_start': 3, 'best_ok': False},
    'print_frequency': BN // 100,
    'plot_frequency': BN // 100,  # 5005/100=50
    'val_frequency': (0 * BN, BN // 1),
    'test_frequency': (999 * BN, BN // 1),
    'save_frequency': (0 * BN, BN // 1),

    # forzen config
    'mode_custom': False,
    'train_which': [],
    'eval_which': [],
    'xfc_which': -1,

    # time config
    'valid_total_time': 0,
    'test_total_time': 0,
    'exp_tic': time.time(),

    'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume'),
}

cfgres2 = cfgres1