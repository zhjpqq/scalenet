import os
import math
import time
import torch
from datetime import datetime

# batch_nums = math.ceil(data_info['train_size']/bsize_train)
train_size = 1281167
batch_size = 256
batch_nums = math.ceil(train_size / batch_size)
BN = batch_nums  # =>> Unit    #5005


# config-flow
class Config(object):
    # experiment config
    exp_version = 'exp.xxx'
    train_val_test = (True, True, True)

    # device config
    gpu_ids = [0, 1, 2, 3][0:4]

    # model config
    arch_name = 'xxmodel'
    arch_kwargs = {}
    resume = None  # None or path/to/the/model.ckpt
    resume_optimizer = True  # 复活ckpt文件时，是否同时复活optimizer，默认复活. 可用于 固定conv+lfc，训练xfc.
    resume_config = True  # 复活ckpt文件时，是否同时复活config，默认复活. validate-github-models.pth时需关闭.
    resume_strict = True     # 复活ckpt文件时，loading weights by strict rules.
    mgpus_to_sxpu = ['m2s', 's2m', 'none', 'auto'][3]  # MultiGpu.ckpt 与 SingleGpu.ckpt 之间转换加载，None：无需转换直接加载

    # data config
    dataset = 'xxdata'
    data_info = {'train_size': 45000, 'val_size': 5000, 'test_size': 10000}
    data_root = '/path/to/your/dataset/xxx-data'
    data_augment = {'train': '1crop-flip', 'val': 'no-aug'}
    # data_augment = {'train': 'rotate-rresize-1crop',  'val': '1resize-1crop',
    #                  'imsize': 256, 'insize': 224, 'color': True,
    #                  'degree': (0, 0), 'scale': (0.08, 1), 'ratio': (3. / 4, 4. / 3)},
    data_kwargs = {}
    data_workers = 12
    data_shuffle = False  # do shuffle agian at each epoch start.

    # path config
    current_time = [datetime.now().strftime('%b%d_%H_%M-'), ''][0]
    ckpt_base_dir = './checkpoints'  # 'path/to/your/dir to save ckpts.'
    ckpt_dir = 'os.path.join(ckpt_base_dir, dataset, arch_name, model_exp_version.ckpt'  # auto setting in bellow
    log_base_dir = './runs'  # 'path/to/your/dir to save logs.'
    log_dir = 'os.path.join(log_base_dir, dataset, current_time + arch_name)'  # auto setting in bellow
    ckpt_suffix = ''  # when save a ckpt, u can add a special mark to its filename.

    # iter config
    start_iter = 0          # resume时，start_iter = current_iter + 1, 即使用当前保存的current_iter更新start_iter
    current_iter = 0
    max_iters = 1000
    start_epoch = 0
    current_epoch = 0       # resume时，start_epoch = current_epoch + 1, 即使用当前保存的current_epoch更新start_epoch
    max_epochs = 'max_iters // batch_nums'
    bsize_train = 128
    bsize_val = 64
    batch_nums = math.ceil(data_info['train_size'] / bsize_train)
    Unit = {'epoch': batch_nums, 'iter': 1}[['epoch', 'iter'][0]]  # 按epoch为单位调节 还是按iteration为单位调节lr/bs？
    BN = {'epoch': batch_nums, 'iter': 1}[['epoch', 'iter'][0]]  # 按epoch为单位 还是 按iteration为单位进行调节？

    # lr config
    optim_type = ['Adam', 'SGD'][1]
    optim_custom = False  # 模型自带优化器
    lr_start = {'Adam': 0.01, 'SGD': 0.1}[optim_type]
    lr_end = 0.0
    lr_decay_policy = ['regular', 'appoint', 'original'][1]
    # # Note:  point = epoch * batch_nums
    lr_decay_start = 1
    lr_decay_rate = 1 / 10
    lr_decay_time = 15
    lr_decay_appoint = ((5, 1 / 10), (8, 1 / 10))
    momentum = 0.9
    weight_decay = [0.0001, 0.0005][0]
    nesterov = False
    rmsprop_alpha = ''
    rmsprop_centered = ''

    # frequency config
    # # Note: if val_freq = (0, plot_freq)
    # # the loss-curve/prec-curve of train and val can have same x-axis point.
    best_prec = {'train_prec1': 0, 'train_prec5': 0,
                 'val_prec1': 0, 'val_prec5': 0,
                 'best_start': 3, 'best_ok': False}  # best_start: 从第几回合开始保存最佳ckpt.
    print_frequency = 20
    plot_frequency = 10
    val_frequency = (0, 10)
    test_frequency = (0, 10)
    save_frequency = (0, 10)

    # WaveResNet 组合不同阶段进行训练策略调整
    # key = epoch * batch_nums
    # mode_custom = False: 关闭自定义的.train_model()/.val_model()方法，使用pytorch的.train()/.val()方法
    mode_custom = False
    xfc_which = -1      # 当有多个xfc输出时，需要看哪个输出的结果和曲线, 可用于固定conv+lfc 训练xfc. 默认=-1，即最后的分类器
    name_which = None   # 指定某类Module的具有name名称的模块
    train_which = []
    eval_which = []

    # time config
    train_total_time = 0
    valid_total_time = 0
    test_total_time = 0
    exp_tic = time.time()

    # record train/val state
    curr_val_prec1 = 0
    best_val_prec1 = 0
    curr_train_prec1 = 0
    best_train_prec1 = 0

    curr_val_prec5 = 0
    best_val_prec5 = 0
    curr_train_prec5 = 0
    best_train_prec5 = 0

    # 从cfg_dict中恢复cfg时，不希望恢复的属性，将使用当前设置.
    # 也可在 self.dict_to_class() 中使用 exclude 指定.
    # cfg_dict 可能来自配置文件，也可能来自ckpt文件.
    exclude_keys = ('exclude_keys',)  # eg. = ('gpu_ids', 'device')

    cfgs_check_done = False

    def class_to_dict(self):
        cfg_dict = {}
        [cfg_dict.setdefault(key, getattr(self, key))
         for key in self.__dir__()
         if not (key.startswith('__') or callable(getattr(self, key)))]
        return cfg_dict

    def dict_to_class(self, cfg_dict, exclude=()):
        # convert cfgs dict into a Config() class object, not include the key in exclude
        assert isinstance(exclude, (tuple, list))
        no_used_keys = []
        exclude_keys = set(list(exclude))
        for key, val in cfg_dict.items():
            if key not in self.__dir__():
                no_used_keys.append(key)
                setattr(self, key, val)
            elif key in exclude_keys:
                continue
            else:
                setattr(self, key, val)
        if len(no_used_keys) > 0:
            print('Warning: such keys in Dict() not exist in Config(), they will be added!:' + ', '.join(no_used_keys))
        if len(exclude_keys) > 0:
            print('Warning: such keys in Config() not set in Dict(), they will be default!:' + ', '.join(exclude_keys))
        self.add_new_args_for_old_ckpt()
        print('Success: config dict has been moved into Config() class !')
        return self

    def read_from_args(self, args):
        # move args into Config
        pass

    def read_from_file(self, cfg_file_path):
        # move file cfg into Config
        pass

    def show_config(self, ckpt_path=None, result=None):
        # show the configs saved in ckpt or in memeory
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path)
            cfgs_dict = ckpt['config']
        else:
            cfgs_dict = self.class_to_dict()
        print('\ncurrent config is: ...\n')
        for (k, v) in cfgs_dict.items():
            print(k, ':', v)
        print('\n----------------------\n')
        if result == dict:
            return cfgs_dict
        elif result is None:
            return None
        else:
            return self.dict_to_class(cfgs_dict)

    def config_gpus(self):
        if len(self.gpu_ids) >= 1:
            device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # ignore warning
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(id) for id in self.gpu_ids])  # use gpu gpu_ids
            # import torch.backends.cudnn as cudnn
            # cudnn.benchmark = True
            # cudnn.deterministic = False
            # cudnn.enable = True
        else:
            device = torch.device('cpu')
        self.device = device

    def config_path(self):
        """
        exp=='exp.xx', 为实验使用指定编号
        """
        assert 'exp.' in str(self.exp_version), '实验编号必须以exp.开头, eg. exp.xxx'
        save_path = self.arch_name + '-exp.' + str(self.exp_version).split('exp.')[1]
        log_path = self.arch_name + '-exp.' + str(self.exp_version).split('exp.')[1]
        self.ckpt_dir = os.path.join(self.ckpt_base_dir, self.dataset, self.arch_name, save_path)
        self.log_dir = os.path.join(self.log_base_dir, self.dataset, self.current_time + log_path)

    def check_configs(self):
        # 基于一次配置的二次配置
        self.config_gpus()
        self.config_path()

        # 安全检查
        assert self.batch_nums == math.ceil(self.data_info['train_size'] / self.bsize_train)
        assert self.BN == self.batch_nums
        assert getattr(self, 'device', None) is not None, \
            '<cfg.config_gpus()> must need do, after <cfg.gpu_ids> has been set'

        self.cfgs_check_done = True

    def add_new_args_for_old_ckpt(self):
        # 1. add cfg.best_prec['best_start'] for old version ckpt.
        if 'best_start' not in self.best_prec:
            self.best_prec.setdefault('best_start', 10)


if __name__ == '__main__':
    cfg = Config()
    cfg.check_configs()
    print(cfg.exp_version, cfg.ckpt_dir)
