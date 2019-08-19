import math, xtils, time

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 1281167
batch_size = 256
batch_size_val = 128
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit    #5005

cfgsr1 = {
    # experiment config
    'exp_version': 'exp.xxx',
    'train_val_test': (True, True, True),

    # device config
    'gpu_ids': [0, 1, 2, 3, 4, 5, 6, 7, 8][0:4],

    # model config
    'arch_name': 'srnet',
    'arch_kwargs': {},
    'resume': None or '/data1/zhangjp/classify/checkpoints/imagenet/srnet/srnet-exp.sr1' + '/' +
              'imagenet-srnet67-ep339-it1701699-acc69.85-best76.20-topv93.11-par32.66M-norm-exp.sr1.ckpt',
    'resume_config': True,
    'resume_optimizer': True,
    'mgpus_to_sxpu': ['m2s', 's2m', 'none', 'auto'][3],

    # data config
    'dataset': 'imagenet',
    'data_info': {'train_size': train_size, 'val_size': 10000, 'test_size': 10000},
    'data_root': xtils.get_data_root(data='imagenet'),
    'data_augment': {'train': 'rotate-rresize-1crop', 'val': '1resize-1crop',
                     'imsize': 256, 'insize': 224, 'color': True, 'interp': 'bilinear',
                     'degree': (0, 0), 'scale': (0.08, 1), 'ratio': (3. / 4, 4. / 3)},
    'data_kwargs': {},
    'data_workers': 24,

    # path config
    'current_time': '',
    'ckpt_suffix': '',  # when save a ckpt, u can add a special mark to its filename.
    'ckpt_base_dir': xtils.get_base_dir(k='ckpt'),
    'ckpt_dir': 'auto-setting',
    'log_base_dir': xtils.get_base_dir(k='log'),
    'log_dir': 'auto-setting',

    # iter config
    'start_iter': 0,
    'max_iters': [100 * BN, 90 * BN, 140 * BN, 420 * BN][-1],
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
    # 'lr_decay_appoint': ((30 * BN, 1 / 10), (60 * BN, 1 / 10), (90 * BN, 1 / 10)),  # 100
    # 'lr_decay_appoint': ((100 * BN, 1000), (130 * BN, 1 / 10), (150 * BN, 1 / 10), (170 * BN, 1 / 10)),  # 180
    # 'lr_decay_appoint': ((180 * BN, 1000), (210 * BN, 1 / 10), (230 * BN, 1 / 10), (250 * BN, 1 / 10)),  # 260
    # 'lr_decay_appoint': ((260 * BN, 1000), (290 * BN, 1 / 10), (310 * BN, 1 / 10), (330 * BN, 1 / 10)),    # 340
    'lr_decay_appoint': ((340 * BN, 1000), (370 * BN, 1 / 10), (390 * BN, 1 / 10), (410 * BN, 1 / 10)),    # 420
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
    'mode_custom': True,
    'xfc_which': -1,
    'name_which': [None, 'head-3-4-4', 'head-7-8-8', 'head-13-16-16', 'head-16-32-32'][0],
    'train_which': [{0 * BN: 'bone+mhead'},
                    {260 * BN: 'auxhead'},
                    {340 * BN: 'summary'},
                    {0 * BN: 'bone+mhead', 20 * BN: 'auxhead', 30 * BN: 'summary'}][2],
    'eval_which': [{0 * BN: 'bone+mhead'},
                   {260 * BN: 'bone+mhead+auxhead'},
                   {340 * BN: 'bone+mhead+auxhead+summary'},
                   {0 * BN: 'bone+mhead', 20 * BN: 'bone+mhead+auxhead', 30 * BN: 'bone+mhead+auxhead+summary'}][2],

    # time config
    'valid_total_time': 0,
    'test_total_time': 0,
    'exp_tic': time.time(),

    # 'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume'),
    'exclude_keys': ('exclude_keys', 'gpu_ids', 'device', 'resume', 'max_iters', 'lr_decay_appoint',
                     'xfc_which', 'name_which', 'train_which', 'eval_which', 'arch_kwargs',
                     'train_val_test', 'data_workers'),
    # 'max_iters', 'lr_decay_appoint', 'xfc_which', 'name_which', 'train_which', 'eval_which',
    # 'ckpt_dir', 'log_dir', 'exp_version'
}
