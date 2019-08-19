'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.

'''

import torch
from xmodels.mobilev3x import MobileV3X
from xmodels.mobilev3y import MobileV3Y
from xmodels.mobilev3z import MobileV3Z
from xmodels.mobilev3d import MobileV3D
import os
import xtils

# MobileNet-V3

# 统一接口 interface for model_factory ##########################################

# 一、 官方模型配置
# small = {}
# large = {}

# 二、 重构入口参数
mbvdl = {'arch': 'mbvdl', 'cfg': '', 'model_path': ['local', 'download', ''][2]}

mbvds = {'arch': 'mbvds', 'cfg': '', 'model_path': ['local', 'download', ''][2]}

mbvxl = {'arch': 'mbvxl', 'cfg': '', 'model_path': ['local', 'download', ''][2]}  # prec@1-70.788  prec@5-89.410

mbvxs = {'arch': 'mbvxs', 'cfg': '', 'model_path': ['local', 'download', ''][2]}  # prec@1-64.926 prec@5-85.466

mbvyl = {'arch': 'mbvyl', 'cfg': '', 'model_path': ['local', 'download', ''][2]}

mbvys = {'arch': 'mbvys', 'cfg': '', 'model_path': ['local', 'download', ''][2]}


# 三、重构调用入口
def MobileV3(arch='mbvdl', cfg='', model_path=''):
    """
    自定义接口 for model_factory
    指定arch=官方模型，即可从map中调用官方模型的名称name，配置cfg，以及预训练参数ckpt
    指定arch=自定义模型，即可使用传入的模型名称name，配置cfg，以及预训练参数ckpt
    """
    model_name_map = {
        'mbvdl': 'MobileV3D-large',
        'mbvds': 'MobileV3D-small',
        'mbvxl': 'MobileV3X-large',
        'mbvxs': 'MobileV3X-small',
        'mbvyl': 'MobileV3Y-large',
        'mbvys': 'MobileV3Y-small',
    }
    model_cfg_map = {
        'mbvdl': 'large',
        'mbvds': 'small',
        'mbvxl': 'large',
        'mbvxs': 'small',
        'mbvyl': 'large',
        'mbvys': 'small',
    }
    model_ckpt_map = {
        'mbvdl': 'mobilev3_d_large.pth.tar',
        'mbvds': 'mobilev3_d_small.pth.tar',
        'mbvxl': 'mobilev3_x_large.pth.tar',
        'mbvxs': 'mobilev3_x_small.pth.tar',
        'mbvyl': 'mobilev3_y_large.pth.tar',
        'mbvys': 'mobilev3_y_small.pth.tar',
    }

    try:
        # 调用官方模型
        name = model_name_map[arch]
    except:
        # 使用自定义模型，如mbvd33, mbvd22
        name = arch

    if cfg == '':
        # 调用官方配置
        cfg = model_cfg_map[arch]
    else:
        # 使用自定义配置
        assert isinstance(cfg, dict)
        raise NotImplementedError

    if name.startswith('MobileV3D'):
        model = MobileV3D(cfg)
    elif name.startswith('MobileV3X'):
        model = MobileV3X(cfg)
    elif name.startswith('MobileV3Y'):
        model = MobileV3Y(cfg)
    elif name.startswith('MobileV3Z'):
        model = MobileV3Z(cfg)
    else:
        raise NotImplementedError

    model_dir = xtils.get_pretrained_models()

    if os.path.isfile(model_path):
        model = xtils.load_ckpt_weights(model, model_path, device='cpu', mgpus_to_sxpu='auto')
    elif model_path == 'local':
        model_path = os.path.join(model_dir, model_ckpt_map[arch])
        model = xtils.load_ckpt_weights(model, model_path, device='cpu', mgpus_to_sxpu='auto')
    elif model_path == 'download':
        # model_url_map = {}
        # import torch.utils.model_zoo as model_zoo
        # model.load_state_dict(model_zoo.load_url(model_url_map['arch'], model_dir))
        raise NotImplementedError
    else:
        assert model_path == '', '<model_path> must refer to "local" or "download" or "" or "model.ckpt".'

    return model


if __name__ == '__main__':
    # model = MobileV3(arch_cfg='d_large')
    model = MobileV3(**mbvdl)
    print('\n', model, '\n')

    # utils.tensorboard_add_model(model, x)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_FLOPs_scale(model, use_gpu=False, input_size=224, multiply_adds=False)
    xtils.calculate_layers_num(model, layers=('conv2d', 'deconv2d', 'fc'))
    xtils.calculate_time_cost(model, insize=224, use_gpu=False)
