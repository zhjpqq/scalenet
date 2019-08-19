import math, os, time
from math import ceil, floor
import xtils

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 1281167
batch_size = 256
batch_size_val = 4
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit    #5005

cfgmsnet = {
    # experiment config
    'exp_version': 'exp.xxx',
    'train_val_test': (False, True, False),

    # device config
    'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8][0:4],

    # model config
    'arch_name': 'msnet',
    'arch_kwargs': {},
    'resume': os.path.join(xtils.get_pretrained_models(), ''),
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
    'data_workers': 12,

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
    'mode_custom': True,
    'train_which': [],
    'eval_which': [],
    'xfc_which': -1,

    # time config
    'valid_total_time': 0,
    'test_total_time': 0,
    'exp_tic': time.time(),

    'exclude_keys': ('exclude_keys', 'gpu_ids', 'device'),
}

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 1281167
batch_size = 256
batch_size_val = 64
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit    #5005

cfgms5 = {
    # experiment config
    'exp_version': 'exp.xxx',
    'train_val_test': (True, True, True),

    # device config
    'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8][0:4],

    # model config
    'arch_name': 'msnet',
    'arch_kwargs': {},
    'resume': '/data1/zhangjp/classify/checkpoints/imagenet/msnet/msnet-exp.ms5' + '/'
              + 'imagenet-msnet118-ep68-it345344-acc70.80-best70.89-topv89.97-par5.80M-norm-exp.ms5.ckpt',
    'resume_config': True,
    'resume_optimizer': True,
    'mgpus_to_sxpu': ['m2s', 's2m', 'none', 'auto'][3],

    # data config
    'dataset': 'imagenet',
    'data_info': {'train_size': train_size, 'val_size': 50000, 'test_size': 50000},
    'data_root': xtils.get_data_root(data='imagenet'),
    'data_augment': {'train': 'rotate-rresize-1crop', 'val': '1resize-1crop',
                     'imsize': 256, 'insize': 224, 'color': True,
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
    'max_iters': [100 * BN, 90 * BN, 60 * BN, 40 * BN, 120 * BN][0],
    'start_epoch': 0,
    'max_epochs': 0,
    'bsize_train': batch_size,
    'bsize_val': batch_size_val,
    'batch_nums': batch_nums,
    'Unit': {'epoch': batch_nums, 'iter': 1}[['epoch', 'iter'][0]],  # 按epoch为单位调节 还是按iteration为单位调节lr/bs？
    'BN': BN,

    # lr config
    'optim_type': ['Adam', 'SGD'][1],
    'optim_custom': False,
    'lr_start': {'Adam': 0.01, 'SGD': 0.1}['SGD'],
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

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 50000
batch_size = 128
batch_size_val = 64
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit

cfgms2 = {
    # experiment config
    'exp_version': 'exp.xxx',
    'train_val_test': (True, True, True),

    # device config
    'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8][0:4],

    # model config
    'arch_name': 'msnet',
    'arch_kwargs': {},
    'resume': None or '/data1/zhangjp/classify/checkpoints/cifar10/msnet/msnet-exp.ms32' + '/' +
              'cifar10-msnet132-ep138-it54348-acc88.45-best88.58-topv99.72-par0.26M-norm-exp.ms32.ckpt',
    'resume_config': True,
    'resume_optimizer': True,
    'mgpus_to_sxpu': ['m2s', 's2m', 'none', 'auto'][3],

    # data config
    'dataset': 'cifar10',
    'data_info': {'train_size': train_size, 'val_size': 10000, 'test_size': 10000},
    'data_root': xtils.get_data_root(data='cifar10'),
    'data_augment': {'train': '1crop-flip', 'val': 'no-aug'},
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
    'max_iters': [100 * BN, 90 * BN, 140 * BN, 150 * BN][-1],
    'start_epoch': 0,
    'max_epochs': 0,
    'bsize_train': batch_size,
    'bsize_val': batch_size_val,
    'batch_nums': batch_nums,
    'Unit': {'epoch': batch_nums, 'iter': 1}[['epoch', 'iter'][0]],  # 按epoch为单位调节 还是按iteration为单位调节lr/bs？
    'BN': BN,

    # lr config
    'optim_type': ['Adam', 'SGD'][1],
    'optim_custom': False,
    'lr_start': {'Adam': 0.01, 'SGD': 0.1}['SGD'],
    'lr_decay_policy': ['regular', 'appoint', 'original', 'trace_prec'][1],
    # 'lr_decay_appoint': ((80 * BN, 1 / 10), (90 * BN, 1 / 10), (100 * BN, 1 / 10)),
    # 'lr_decay_appoint': ((15 * BN, 1 / 10), ),
    # 'lr_decay_appoint': ((20 * BN, 10), (35 * BN, 1 / 10)),
    # 'lr_decay_appoint': ((40 * BN, 10), (55 * BN, 1 / 10)),
    # 'lr_decay_appoint': ((60 * BN, 10), (75 * BN, 1 / 10)),
    # 'lr_decay_appoint': ((80 * BN, 10), (95 * BN, 1 / 10)),
    'lr_decay_appoint': ((100 * BN, 10), (130 * BN, 1 / 10), (139 * BN, 1 / 10)),
    'momentum': 0.9,
    'weight_decay': [0.0001, 0.0005, 0.00017, 0.00002][1],
    'nesterov': False,
    'rmsprop_alpha': '',
    'rmsprop_centered': '',

    # frequency config
    # # Note: if val_freq: (0, plot_freq)
    # # the loss-curve/prec-curve of train and val can have same x-axis point.
    'best_prec': {'train_prec1': 0, 'train_prec5': 0, 'val_prec1': 0, 'val_prec5': 0,
                  'best_start': 3, 'best_ok': False},
    'print_frequency': (BN // 97) * 8,
    'plot_frequency': (BN // 97) * 1,  # 391/97=4
    'val_frequency': (0 * BN, BN // 1),
    'test_frequency': (999 * BN, BN // 1),
    'save_frequency': (0 * BN, BN // 1),

    # forzen config
    'mode_custom': True,
    'xfc_which': -1,
    'name_which': [None, 'head-3-1-1', 'head-3-1-1@3', 'head-8-2-2', 'head-8-2-2@6', 'head-15-4-4'][0],
    'train_which': [{0 * BN: 'bone+mhead'},
                    {80 * BN: 'auxhead'},
                    {100 * BN: 'summary'},

                    {0 * BN: 'bone+mhead', 20 * BN: 'auxhead', 30 * BN: 'summary'}][2],

    'eval_which': [{0 * BN: 'bone+mhead'},
                   {80 * BN: 'bone+mhead+auxhead'},
                   {100 * BN: 'bone+mhead+auxhead+summary'},

                   {0 * BN: 'bone+mhead', 20 * BN: 'bone+mhead+auxhead', 30 * BN: 'bone+mhead+auxhead+summary'}][2],

    # time config
    'valid_total_time': 0,
    'test_total_time': 0,
    'exp_tic': time.time(),

    'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume', 'max_iters', 'lr_decay_appoint',
                     'xfc_which', 'name_which', 'train_which', 'eval_which'),
    # 'max_iters', 'lr_decay_appoint', 'xfc_which', 'name_which', 'train_which', 'eval_which',
    # 'ckpt_dir', 'log_dir', 'exp_version'
}

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 50000
batch_size = 128
batch_size_val = 64
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit

cfgms4 = {
    # experiment config
    'exp_version': 'exp.xxx',
    'train_val_test': (True, True, True),

    # device config
    'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8][0:4],

    # model config
    'arch_name': 'msnet',
    'arch_kwargs': {},
    'resume': None or '/data1/zhangjp/classify/checkpoints/cifar10/msnet/msnet-exp.ms4' + '/' +
              'cifar10-msnet132-ep779-it304979-acc89.19-best90.94-topv99.78-par0.26M-norm-exp.ms4.ckpt',
    'resume_config': True,
    'resume_optimizer': True,
    'mgpus_to_sxpu': ['m2s', 's2m', 'none', 'auto'][3],

    # data config
    'dataset': 'cifar10',
    'data_info': {'train_size': train_size, 'val_size': 10000, 'test_size': 10000},
    'data_root': xtils.get_data_root(data='cifar10'),
    'data_augment': {'train': '1crop-flip', 'val': 'no-aug'},
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
    'max_iters': [100 * BN, 90 * BN, 140 * BN, 980 * BN][-1],
    'start_epoch': 0,
    'max_epochs': 0,
    'bsize_train': batch_size,
    'bsize_val': batch_size_val,
    'batch_nums': batch_nums,
    'Unit': {'epoch': batch_nums, 'iter': 1}[['epoch', 'iter'][0]],  # 按epoch为单位调节 还是按iteration为单位调节lr/bs？
    'BN': BN,

    # lr config
    'optim_type': ['Adam', 'SGD'][1],
    'optim_custom': False,
    'lr_start': {'Adam': 0.01, 'SGD': 0.1}['SGD'],
    'lr_decay_policy': ['regular', 'appoint', 'original', 'trace_prec'][1],
    # 'lr_decay_appoint': ((230 * BN, 1 / 10), (260 * BN, 1 / 10), (290 * BN, 1 / 10)),  # 299
    # 'lr_decay_appoint': ((300 * BN, 1000), (380 * BN, 1 / 10), (400 * BN, 1 / 10), (410 * BN, 1 / 10)),  # 420
    # 'lr_decay_appoint': ((420 * BN, 1000), (500 * BN, 1 / 10), (520 * BN, 1 / 10), (530 * BN, 1 / 10)),  # 540
    # 'lr_decay_appoint': ((540 * BN, 1000), (620 * BN, 1 / 10), (640 * BN, 1 / 10), (650 * BN, 1 / 10)),  # 660
    # 'lr_decay_appoint': ((660 * BN, 1000), (740 * BN, 1 / 10), (760 * BN, 1 / 10), (770 * BN, 1 / 10)),  # 780
    'lr_decay_appoint': ((780 * BN, 1000), (900 * BN, 1 / 10), (940 * BN, 1 / 10), (970 * BN, 1 / 10)),  # 980
    'momentum': 0.9,
    'weight_decay': [0.0001, 0.0005, 0.00017, 0.00002][1],
    'nesterov': False,
    'rmsprop_alpha': '',
    'rmsprop_centered': '',

    # frequency config
    # # Note: if val_freq: (0, plot_freq)
    # # the loss-curve/prec-curve of train and val can have same x-axis point.
    'best_prec': {'train_prec1': 0, 'train_prec5': 0, 'val_prec1': 0, 'val_prec5': 0,
                  'best_start': 3, 'best_ok': False},
    'print_frequency': (BN // 97) * 8,
    'plot_frequency': (BN // 97) * 1,  # 391/97=4
    'val_frequency': (0 * BN, BN // 1),
    'test_frequency': (999 * BN, BN // 1),
    'save_frequency': (0 * BN, BN // 1),

    # forzen config
    'mode_custom': True,
    'xfc_which': -1,
    'name_which': [None, 'head-3-1-1', 'head-3-1-1@3', 'head-8-2-2', 'head-8-2-2@6', 'head-15-4-4'][0],
    'train_which': [{0 * BN: 'bone+mhead'},
                    {660 * BN: 'auxhead'},
                    {780 * BN: 'summary'},
                    {0 * BN: 'bone+mhead', 20 * BN: 'auxhead', 30 * BN: 'summary'}][2],

    'eval_which': [{0 * BN: 'bone+mhead'},
                   {660 * BN: 'bone+mhead+auxhead'},
                   {780 * BN: 'bone+mhead+auxhead+summary'},
                   {0 * BN: 'bone+mhead', 20 * BN: 'bone+mhead+auxhead', 30 * BN: 'bone+mhead+auxhead+summary'}][2],

    # time config
    'valid_total_time': 0,
    'test_total_time': 0,
    'exp_tic': time.time(),

    # 'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume'),
    'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume', 'max_iters', 'lr_decay_appoint',
                     'xfc_which', 'name_which', 'train_which', 'eval_which'),
    # 'max_iters', 'lr_decay_appoint', 'xfc_which', 'name_which', 'train_which', 'eval_which',
    # 'ckpt_dir', 'log_dir', 'exp_version'
}

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 50000
batch_size = 128
batch_size_val = 64
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit

cfgms6 = {
    # experiment config
    'exp_version': 'exp.xxx',
    'train_val_test': (True, True, True),

    # device config
    'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8][0:4],

    # model config
    'arch_name': 'msnet',
    'arch_kwargs': {},
    'resume': None and '/data1/zhangjp/classify/checkpoints/cifar10/msnet/msnet-exp.ms6' + '/' +
              '',
    'resume_config': True,
    'resume_optimizer': True,
    'mgpus_to_sxpu': ['m2s', 's2m', 'none', 'auto'][3],

    # data config
    'dataset': 'cifar10',
    'data_info': {'train_size': train_size, 'val_size': 10000, 'test_size': 10000},
    'data_root': xtils.get_data_root(data='cifar10'),
    'data_augment': {'train': '1crop-flip', 'val': 'no-aug'},
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
    'max_iters': [100 * BN, 90 * BN, 140 * BN, 300 * BN][-1],
    'start_epoch': 0,
    'max_epochs': 0,
    'bsize_train': batch_size,
    'bsize_val': batch_size_val,
    'batch_nums': batch_nums,
    'Unit': {'epoch': batch_nums, 'iter': 1}[['epoch', 'iter'][0]],  # 按epoch为单位调节 还是按iteration为单位调节lr/bs？
    'BN': BN,

    # lr config
    'optim_type': ['Adam', 'SGD'][1],
    'optim_custom': False,
    'lr_start': {'Adam': 0.01, 'SGD': 0.1}['SGD'],
    'lr_decay_policy': ['regular', 'appoint', 'original', 'trace_prec'][1],
    'lr_decay_appoint': ((230 * BN, 1 / 10), (260 * BN, 1 / 10), (290 * BN, 1 / 10)),  # 299
    # 'lr_decay_appoint': ((300 * BN, 1000), (380 * BN, 1 / 10), (400 * BN, 1 / 10), (410 * BN, 1 / 10)),  # 420
    # 'lr_decay_appoint': ((420 * BN, 1000), (500 * BN, 1 / 10), (520 * BN, 1 / 10), (530 * BN, 1 / 10)),  # 540
    # 'lr_decay_appoint': ((540 * BN, 1000), (620 * BN, 1 / 10), (640 * BN, 1 / 10), (650 * BN, 1 / 10)),  # 660
    # 'lr_decay_appoint': ((660 * BN, 1000), (740 * BN, 1 / 10), (760 * BN, 1 / 10), (770 * BN, 1 / 10)),  # 780
    # 'lr_decay_appoint': ((780 * BN, 1000), (900 * BN, 1 / 10), (940 * BN, 1 / 10), (970 * BN, 1 / 10)),  # 980
    # 'lr_decay_appoint': ((540 * BN, 1000), (660 * BN, 1 / 10), (710 * BN, 1 / 10), (740 * BN, 1 / 10)),  # 750
    'momentum': 0.9,
    'weight_decay': [0.0001, 0.0005, 0.00017, 0.00002][1],
    'nesterov': False,
    'rmsprop_alpha': '',
    'rmsprop_centered': '',

    # frequency config
    # # Note: if val_freq: (0, plot_freq)
    # # the loss-curve/prec-curve of train and val can have same x-axis point.
    'best_prec': {'train_prec1': 0, 'train_prec5': 0, 'val_prec1': 0, 'val_prec5': 0,
                  'best_start': 3, 'best_ok': False},
    'print_frequency': (BN // 97) * 8,
    'plot_frequency': (BN // 97) * 1,  # 391/97=4
    'val_frequency': (0 * BN, BN // 1),
    'test_frequency': (999 * BN, BN // 1),
    'save_frequency': (0 * BN, BN // 1),

    # forzen config
    'mode_custom': True,
    'xfc_which': -1,
    'name_which': [None, 'head-8-2-2', 'head-8-2-2@6', 'head-15-4-4'][0],
    'train_which': [{0 * BN: 'bone+mhead'},
                    {0 * BN: 'auxhead'},
                    {0 * BN: 'summary'},
                    {0 * BN: 'bone+mhead', 20 * BN: 'auxhead', 30 * BN: 'summary'}][0],

    'eval_which': [{0 * BN: 'bone+mhead'},
                   {0 * BN: 'bone+mhead+auxhead'},
                   {0 * BN: 'bone+mhead+auxhead+summary'},
                   {0 * BN: 'bone+mhead', 20 * BN: 'bone+mhead+auxhead', 30 * BN: 'bone+mhead+auxhead+summary'}][0],

    # time config
    'valid_total_time': 0,
    'test_total_time': 0,
    'exp_tic': time.time(),

    'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume'),
    # 'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume', 'max_iters', 'lr_decay_appoint',
    #                  'xfc_which', 'name_which', 'train_which', 'eval_which'),
    # 'max_iters', 'lr_decay_appoint', 'xfc_which', 'name_which', 'train_which', 'eval_which',
    # 'ckpt_dir', 'log_dir', 'exp_version'
}

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 50000
batch_size = 128
batch_size_val = 64
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit

cfgms9 = {
    # experiment config
    'exp_version': 'exp.xxx',
    'train_val_test': (True, True, True),

    # device config
    'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8][0:4],

    # model config
    'arch_name': 'msnet',
    'arch_kwargs': {},
    'resume': None or '/data1/zhangjp/classify/checkpoints/cifar10/msnet/msnet-exp.ms9' + '/' +
              'cifar10-msnet82-ep539-it211139-acc85.99-best90.42-topv99.73-par0.22M-norm-exp.ms9.ckpt',
    'resume_config': True,
    'resume_optimizer': True,
    'mgpus_to_sxpu': ['m2s', 's2m', 'none', 'auto'][3],

    # data config
    'dataset': 'cifar10',
    'data_info': {'train_size': train_size, 'val_size': 10000, 'test_size': 10000},
    'data_root': xtils.get_data_root(data='cifar10'),
    'data_augment': {'train': '1crop-flip', 'val': 'no-aug'},
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
    'max_iters': [100 * BN, 90 * BN, 140 * BN, 650 * BN][-1],
    'start_epoch': 0,
    'max_epochs': 0,
    'bsize_train': batch_size,
    'bsize_val': batch_size_val,
    'batch_nums': batch_nums,
    'Unit': {'epoch': batch_nums, 'iter': 1}[['epoch', 'iter'][0]],  # 按epoch为单位调节 还是按iteration为单位调节lr/bs？
    'BN': BN,

    # lr config
    'optim_type': ['Adam', 'SGD'][1],
    'optim_custom': False,
    'lr_start': {'Adam': 0.01, 'SGD': 0.1}['SGD'],
    'lr_decay_policy': ['regular', 'appoint', 'original', 'trace_prec'][1],
    # 'lr_decay_appoint': ((230 * BN, 1 / 10), (260 * BN, 1 / 10), (290 * BN, 1 / 10)),  # 299
    # 'lr_decay_appoint': ((300 * BN, 1000), (380 * BN, 1 / 10), (400 * BN, 1 / 10), (410 * BN, 1 / 10)),  # 420
    # 'lr_decay_appoint': ((420 * BN, 1000), (500 * BN, 1 / 10), (520 * BN, 1 / 10), (530 * BN, 1 / 10)),  # 540
    'lr_decay_appoint': ((540 * BN, 1000), (620 * BN, 1 / 10), (640 * BN, 1 / 10), (650 * BN, 1 / 10)),  # 660
    # 'lr_decay_appoint': ((660 * BN, 1000), (740 * BN, 1 / 10), (760 * BN, 1 / 10), (770 * BN, 1 / 10)),  # 780
    # 'lr_decay_appoint': ((780 * BN, 1000), (900 * BN, 1 / 10), (940 * BN, 1 / 10), (970 * BN, 1 / 10)),  # 980
    # 'lr_decay_appoint': ((540 * BN, 1000), (660 * BN, 1 / 10), (710 * BN, 1 / 10), (740 * BN, 1 / 10)),  # 750
    'momentum': 0.9,
    'weight_decay': [0.0001, 0.0005, 0.00017, 0.00002][1],
    'nesterov': False,
    'rmsprop_alpha': '',
    'rmsprop_centered': '',

    # frequency config
    # # Note: if val_freq: (0, plot_freq)
    # # the loss-curve/prec-curve of train and val can have same x-axis point.
    'best_prec': {'train_prec1': 0, 'train_prec5': 0, 'val_prec1': 0, 'val_prec5': 0,
                  'best_start': 3, 'best_ok': False},
    'print_frequency': (BN // 97) * 8,
    'plot_frequency': (BN // 97) * 1,  # 391/97=4
    'val_frequency': (0 * BN, BN // 1),
    'test_frequency': (999 * BN, BN // 1),
    'save_frequency': (0 * BN, BN // 1),

    # forzen config
    'mode_custom': True,
    'xfc_which': -1,
    'name_which': [None, 'head-3-1-1', 'head-3-1-1@3', 'head-8-2-2', 'head-8-2-2@6', 'head-15-4-4'][0],
    'train_which': [{0 * BN: 'bone+mhead'},
                    {0 * BN: 'auxhead'},
                    {540 * BN: 'summary'},
                    {0 * BN: 'bone+mhead', 20 * BN: 'auxhead', 30 * BN: 'summary'}][2],
    'eval_which': [{0 * BN: 'bone+mhead'},
                   {0 * BN: 'bone+mhead+auxhead'},
                   {540 * BN: 'bone+mhead+auxhead+summary'},
                   {0 * BN: 'bone+mhead', 20 * BN: 'bone+mhead+auxhead', 30 * BN: 'bone+mhead+auxhead+summary'}][2],

    # time config
    'valid_total_time': 0,
    'test_total_time': 0,
    'exp_tic': time.time(),

    # 'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume'),
    'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume', 'max_iters', 'lr_decay_appoint',
                     'xfc_which', 'name_which', 'train_which', 'eval_which', 'arch_kwargs'),
    # 'max_iters', 'lr_decay_appoint', 'xfc_which', 'name_which', 'train_which', 'eval_which',
    # 'ckpt_dir', 'log_dir', 'exp_version'
}

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 50000
batch_size = 128
batch_size_val = 64
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit

cfgms3 = {
    # experiment config
    'exp_version': 'exp.xxx',
    'train_val_test': (True, True, True),

    # device config
    'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8][0:4],

    # model config
    'arch_name': 'msnet',
    'arch_kwargs': {},
    'resume': None or '/data1/zhangjp/classify/checkpoints/cifar10/msnet/msnet-exp.ms29/' +
              'cifar10-msnet75-ep29-it11729-acc77.68-best81.77-topv99.32-par0.20M-norm-exp.ms29.ckpt',
    'resume_config': True,
    'resume_optimizer': True,
    'mgpus_to_sxpu': ['m2s', 's2m', 'none', 'auto'][3],

    # data config
    'dataset': 'cifar10',
    'data_info': {'train_size': train_size, 'val_size': 10000, 'test_size': 10000},
    'data_root': xtils.get_data_root(data='cifar10'),
    'data_augment': {'train': '1crop-flip', 'val': 'no-aug'},
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
    'max_iters': [100 * BN, 90 * BN, 140 * BN, 40 * BN, 100 * BN][-2],
    'start_epoch': 0,
    'max_epochs': 0,
    'bsize_train': batch_size,
    'bsize_val': batch_size_val,
    'batch_nums': batch_nums,
    'Unit': {'epoch': batch_nums, 'iter': 1}[['epoch', 'iter'][0]],  # 按epoch为单位调节 还是按iteration为单位调节lr/bs？
    'BN': BN,

    # lr config
    'optim_type': ['Adam', 'SGD'][1],
    'optim_custom': False,
    'lr_start': {'Adam': 0.01, 'SGD': 0.1}['SGD'],
    'lr_decay_policy': ['regular', 'appoint', 'original', 'trace_prec'][1],
    # 'lr_decay_appoint': ((80 * BN, 1 / 10), (90 * BN, 1 / 10), (100 * BN, 1 / 10)),
    'lr_decay_appoint': ((83 * BN, 1 / 10), (113 * BN, 1 / 10), (133 * BN, 1 / 10)),
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
    'print_frequency': (BN // 97) * 8,
    'plot_frequency': (BN // 97) * 1,  # 391/97=4
    'val_frequency': (0 * BN, BN // 1),
    'test_frequency': (999 * BN, BN // 1),
    'save_frequency': (0 * BN, BN // 1),

    # forzen config
    'mode_custom': True,
    'xfc_which': -1,
    'train_which': [{0 * BN: 'bone+mhead'},
                    {0 * BN: 'summary'},
                    {0 * BN: 'bone+mhead', 6 * BN: 'auxhead'},
                    {0 * BN: 'bone+mhead', 20 * BN: 'auxhead', 30 * BN: 'summary'}][-1],

    'eval_which': [{0 * BN: 'bone+mhead'},
                   {0 * BN: 'bone+mhead+auxhead+summary'},
                   {0 * BN: 'bone+mhead', 6 * BN: 'bone+mhead+auxhead'},
                   {0 * BN: 'bone+mhead', 20 * BN: 'bone+mhead+auxhead', 30 * BN: 'bone+mhead+auxhead+summary'}][-1],

    # time config
    'valid_total_time': 0,
    'test_total_time': 0,
    'exp_tic': time.time(),

    'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume', 'max_iters', 'xfc_which'),
    # , 'exp_version', 'ckpt_dir', 'log_dir', 'xfc_which'
}

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 50000
batch_size = 128
batch_size_val = 64
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit

cfgms14 = {
    # experiment config
    'exp_version': 'exp.xxx',
    'train_val_test': (True, True, True),

    # device config
    'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8][0:4],

    # model config
    'arch_name': 'msnet',
    'arch_kwargs': {},
    'resume': '/data1/zhangjp/classify/checkpoints/cifar10/msnet/msnet-exp.ms14' + '/'
              + 'cifar10-msnet132-ep82-it32452-acc80.78-best80.78-topv99.04-par0.26M-best-exp.ms14.ckpt',
    'resume_config': True,
    'resume_optimizer': True,
    'mgpus_to_sxpu': ['m2s', 's2m', 'none', 'auto'][3],

    # data config
    'dataset': 'cifar10',
    'data_info': {'train_size': train_size, 'val_size': 10000, 'test_size': 10000},
    'data_root': xtils.get_data_root(data='cifar10'),
    'data_augment': {'train': '1crop-flip', 'val': 'no-aug'},
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
    'max_iters': [100 * BN, 90 * BN, 140 * BN, 40 * BN, 100 * BN][2],
    'start_epoch': 0,
    'max_epochs': 0,
    'bsize_train': batch_size,
    'bsize_val': batch_size_val,
    'batch_nums': batch_nums,
    'Unit': {'epoch': batch_nums, 'iter': 1}[['epoch', 'iter'][0]],  # 按epoch为单位调节 还是按iteration为单位调节lr/bs？
    'BN': BN,

    # lr config
    'optim_type': ['Adam', 'SGD'][1],
    'optim_custom': False,
    'lr_start': {'Adam': 0.01, 'SGD': 0.1}['SGD'],
    'lr_decay_policy': ['regular', 'appoint', 'original', 'trace_prec'][1],
    # 'lr_decay_appoint': ((80 * BN, 1 / 10), (90 * BN, 1 / 10), (100 * BN, 1 / 10)),
    'lr_decay_appoint': ((83 * BN, 1 / 10), (113 * BN, 1 / 10), (133 * BN, 1 / 10)),
    'momentum': 0.9,
    'weight_decay': [0.0001, 0.0005, 0.00017, 0.00002][1],
    'nesterov': False,
    'rmsprop_alpha': '',
    'rmsprop_centered': '',

    # frequency config
    # # Note: if val_freq: (0, plot_freq)
    # # the loss-curve/prec-curve of train and val can have same x-axis point.
    'best_prec': {'train_prec1': 0, 'train_prec5': 0, 'val_prec1': 0, 'val_prec5': 0,
                  'best_start': 3, 'best_ok': False},
    'print_frequency': (BN // 97) * 8,
    'plot_frequency': (BN // 97) * 1,  # 391/97=4
    'val_frequency': (0 * BN, BN // 1),
    'test_frequency': (999 * BN, BN // 1),
    'save_frequency': (0 * BN, BN // 1),

    # forzen config
    'mode_custom': True,
    'xfc_which': -1,
    'train_which': [{0 * BN: 'bone+mhead'},
                    {0 * BN: 'summary'},
                    {0 * BN: 'bone+mhead', 6 * BN: 'auxhead'},
                    {0 * BN: 'bone+mhead', 20 * BN: 'auxhead', 30 * BN: 'summary'}][1],

    'eval_which': [{0 * BN: 'bone+mhead'},
                   {0 * BN: 'bone+mhead+auxhead+summary'},
                   {0 * BN: 'bone+mhead', 6 * BN: 'bone+mhead+auxhead'},
                   {0 * BN: 'bone+mhead', 20 * BN: 'bone+mhead+auxhead', 30 * BN: 'bone+mhead+auxhead+summary'}][1],

    # time config
    'valid_total_time': 0,
    'test_total_time': 0,
    'exp_tic': time.time(),

    'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume', 'max_iters', 'lr_decay_appoint'),
}
