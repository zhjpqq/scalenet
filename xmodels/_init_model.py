# import your model to here

from .tvm_vggs import VGGNets
from .tvm_resnet import RESNets
from .tvm_densenet import DENSENets
from .fishnet import FISHNets
from .mobilev3 import MobileV3
from .hrnet import HRNets
from .efficientnet import EFFNets

from .scalenet import ScaleNet
from .vivonet import VivoNet
from .richnet import RichNet
from .wavenet import WaveNet
from .nameinet import NameiNet
from .xresnet import XResNet
from .dxnet import DxNet
from .msnet import MultiScaleNet
from .scale_resnet import ScaleResNet
from .activeresnet import ActiveRes


# 支持小写导入
vgg = VGGNets
resnet = RESNets
densenet = DENSENets
fishnet = FISHNets
mobilev3 = MobileV3
hrnet = HRNets
effnet = EFFNets
msnet = MultiScaleNet
srnet = ScaleResNet

scalenet = ScaleNet
vivonet = VivoNet
richnet = RichNet
wavenet = WaveNet
nameinet = NameiNet
xresnet = XResNet
dxnet = DxNet
actres = ActiveRes