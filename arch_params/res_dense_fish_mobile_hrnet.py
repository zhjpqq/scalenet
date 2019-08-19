# res18   11.69M  3.63G   prec@1-69.758  prec@5-89.076  514M   0.00276s
# res34   21.80M  7.34G   prec@1-73.314  prec@5-91.420  573M   0.00466s
# res50   25.56M  8.20G   prec@1-76.130  prec@5-92.862  695M   0.00778s
# res101  44.54M  15.63   prec@1-77.374  prec@5-93.546  846M   0.01459s
# res152  60.19M  23.07G  prec@1-78.312  prec@5-94.046  1005M  0.02109s

# resnet
from xmodels.tvm_resnet import res18, res34, res50, res101, res152

# densenet
from xmodels.tvm_densenet import dense121, dense201, dense161, dense169, dense264

# hrnet
from xmodels.hrnet import hrw18, hrw32, hrw30, hrw40, hrw44, hrw48

# MobileNet-V3
from xmodels.mobilev3 import mbvdl, mbvds, mbvyl, mbvys, mbvxs, mbvxl

# FishNet
from xmodels.fishnet import fish99, fish150, fish201

# EfficientNet
from xmodels.efficientnet import effb0, effb1, effb2, effb3, effb4, effb5, effb6, effb7


