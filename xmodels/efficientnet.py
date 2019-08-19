import torch
from torch import nn
from torch.nn import functional as F
import xtils, os

from xmodules.eff_utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    Conv2dSamePadding,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2dSamePadding(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2dSamePadding(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2dSamePadding(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2dSamePadding(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2dSamePadding(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')
    """

    def __init__(self, blocks_args=None, global_params=None, model_name=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args
        self._model_name = model_name

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2dSamePadding(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2dSamePadding(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x)  # , drop_connect_rate) # see https://github.com/tensorflow/tpu/issues/381

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.extract_features(inputs)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

    @classmethod
    def get_model_params(cls, model_name, override_params=None):
        blocks_args, global_params = get_model_params(model_name, override_params)
        return blocks_args, global_params

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name):
        model = EfficientNet.from_name(model_name)
        load_pretrained_weights(model, model_name)
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b' + str(i) for i in range(num_models)]
        if model_name.replace('-', '_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


# 自定义接口 for model_factory  ##################################
# model  depth  param   GFLOPs   224x224 crop @ 256x256 img    320x320 crop @ 350x350 img
# effb0  82l    5.28M   0.778G   prec@1-75.880 prec@5-92.758   prec@1-78.286 prec@5-94.108
# effb1  116L   7.79M   1.148G   prec@1-76.622 prec@5-93.232   prec@1-79.736 prec@5-94.930
# effb2  116L   9.11M   1.327G   prec@1-76.484 prec@5-92.932   prec@1-80.110 prec@5-95.038
# effb3  131L   12.23M  1.938G   prec@1-75.964 prec@5-92.888   prec@1-80.700 prec@5-95.204
# effb4  161L   19.34M  3.024G

# http://studyai.com/article/1f9d25de
# model  260@260-BICUBIC    372@372-BICUBIC
# effb2  79.336 / 94.650    81.132 / 95.540
# model  300@300-BICUBIC    428@428-BICUBIC
# effb3  80.782 / 95.246    82.270 / 96.014

# 一、 官方模型配置
effb0 = {'arch': 'effb0', 'cfg': '', 'model_path': ['local', 'download', '', '/model.pth'][2], 'override_params': None}
effb1 = {'arch': 'effb1', 'cfg': '', 'model_path': ['local', 'download', '', '/model.pth'][2], 'override_params': None}
effb2 = {'arch': 'effb2', 'cfg': '', 'model_path': ['local', 'download', '', '/model.pth'][2], 'override_params': None}
effb3 = {'arch': 'effb3', 'cfg': '', 'model_path': ['local', 'download', '', '/model.pth'][2], 'override_params': None}
effb4 = {'arch': 'effb4', 'cfg': '', 'model_path': ['local', 'download', '', '/model.pth'][2], 'override_params': None}
effb5 = {'arch': 'effb5', 'cfg': '', 'model_path': ['local', 'download', '', '/model.pth'][2], 'override_params': None}
effb6 = {'arch': 'effb6', 'cfg': '', 'model_path': ['local', 'download', '', '/model.pth'][2], 'override_params': None}
effb7 = {'arch': 'effb7', 'cfg': '', 'model_path': ['local', 'download', '', '/model.pth'][2], 'override_params': None}


def EFFNets(arch='effb0', cfg='', model_path='', override_params=None):
    """
    自定义接口 for model_factory
    指定arch=官方模型，即可从map中调用官方模型的名称name，配置cfg，以及预训练参数ckpt
    指定arch=自定义模型，即可使用传入的模型名称name，配置cfg，以及预训练参数ckpt
    """
    model_name_map = {
        'effb0': 'efficientnet-b0',
        'effb1': 'efficientnet-b1',
        'effb2': 'efficientnet-b2',
        'effb3': 'efficientnet-b3',
        'effb4': 'efficientnet-b4',
        'effb5': 'efficientnet-b5',
        'effb6': 'efficientnet-b6',
        'effb7': 'efficientnet-b7',
    }
    model_cfg_map = {
        """no-use"""
    }
    model_ckpt_map = {
        'effb0': 'efficientnet-b0-08094119.pth',
        'effb1': 'efficientnet-b1-dbc7070a.pth',
        'effb2': 'efficientnet-b2-27687264.pth',
        'effb3': 'efficientnet-b3-c8376fa2.pth',
        'effb4': '',
        'effb5': '',
        'effb6': '',
        'effb7': '',
    }

    try:
        # 调用官方模型
        model_name = model_name_map[arch]
    except:
        # 使用自定义模型，如effb11, effb888
        model_name = arch

    if cfg == '':
        # 调用官方配置
        # cfg = model_cfg_map[arch]
        pass
    else:
        # 使用自定义配置
        assert isinstance(cfg, dict)
        # cfg = cfg

    blocks_args, global_params = get_model_params(model_name, override_params)
    model = EfficientNet(blocks_args, global_params, model_name)

    model_dir = xtils.get_pretrained_models()

    if os.path.isfile(model_path):
        model = xtils.load_ckpt_weights(model, model_path, device='cpu', mgpus_to_sxpu='auto')
    elif model_path == 'local':
        model_path = os.path.join(model_dir, model_ckpt_map[arch])
        model = xtils.load_ckpt_weights(model, model_path, device='cpu', mgpus_to_sxpu='auto')
    elif model_path == 'download':
        model_url_map = {
            'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth',
            'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth',
            'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth',
            'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth',
        }
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(model_zoo.load_url(model_url_map[model_name], model_dir))
    else:
        assert model_path == '', '<model_path> must refer to "local" or "download" or "" or "model.ckpt".'

    return model


if __name__ == '__main__':
    # model_name = 'efficientnet-b0'
    # model = EfficientNet.from_name(model_name)
    # blocks_args, global_params = get_model_params(model_name)
    # print(blocks_args, '\n', global_params)

    model = EFFNets(**effb0)

    print(model)
    xtils.calculate_layers_num(model, layers=('conv2d', 'linear', 'deconv2d'))
    xtils.calculate_FLOPs_scale(model, input_size=224, use_gpu=False, multiply_adds=True)
    xtils.calculate_params_scale(model, format='million')
    xtils.calculate_time_cost(model, insize=224, toc=1, use_gpu=False, pritout=True)
