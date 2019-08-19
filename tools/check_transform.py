# -*- coding: utf-8 -*-
__author__ = 'ooo'
__date__ = '2018/12/15 12:17'

import time
import argparse
import random
import os
from datetime import datetime
import warnings
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from factory.data_factory_bp import MinMaxResize, ColorAugmentation, ReturnOrigin

"""
对数据增强效果进行可视化
"""

# 一、加载图片  #################################
# image
imgdir = 'C://Users//xo//Pictures'
imgname = ['Penguins.jpg', 'xiongmao.jpg', 'cat1.jpg', 'cat2.jpg', 'namei.jpg'][-1]

image = Image.open(os.path.join(imgdir, imgname))
# image.show()
# image = image.resize((224, 224))
# image.show()

# 二、全局变量、全局函数

insize, xscale = 300, [360, 480]

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=mean, std=std)
coloraug = ColorAugmentation()
totensor = transforms.ToTensor()
toimage = transforms.ToPILImage()


# 三、 组合各种形变增强： Rotation + Scale + Ratio  + Shear +  Crop ######################

resize_crop1 = [transforms.RandomChoice([transforms.Resize(x) for x in xscale]),
                transforms.RandomCrop(insize),
                transforms.RandomHorizontalFlip()]  # 0.14899

resize_crop2 = [transforms.RandomChoice([transforms.Resize(x) for x in xscale]),
                transforms.RandomAffine(degrees=(-15, 15), translate=None, scale=None, shear=(-5, 5)),
                transforms.RandomCrop(insize),
                transforms.RandomHorizontalFlip()]  # 0.16606  0.1628

resize_crop3 = [transforms.RandomChoice([transforms.Resize(x) for x in xscale]),
                transforms.RandomApply([transforms.RandomAffine(degrees=(-15, 15), shear=(-5, 5))]),
                transforms.RandomCrop(insize),
                transforms.RandomHorizontalFlip()]  # 0.15647

resize_crop4 = [transforms.RandomChoice([transforms.Resize(x) for x in xscale]),
                transforms.RandomChoice([
                    transforms.RandomAffine(degrees=(-15, 15)),
                    transforms.RandomAffine(degrees=(-0, 0), shear=(-5, 5)),
                ]),
                transforms.RandomCrop(insize)]  # 0.1719

resize_crop7 = [MinMaxResize(400, 600),
                transforms.RandomChoice([
                    ReturnOrigin(),
                    transforms.RandomAffine(degrees=(-10, 10)),
                    transforms.RandomAffine(degrees=(-0, 0), shear=(-5, 5)),
                ])] + [transforms.RandomCrop(insize)]

resize_crop8 = [transforms.RandomResizedCrop(insize, scale=(0.08, 1), ratio=(3. / 4., 4. / 3.)),
                transforms.RandomChoice([
                    ReturnOrigin(),
                    transforms.RandomAffine(degrees=(-15, 15)),
                ])]

resize_crop5 = [MinMaxResize(480, 560)] + [transforms.RandomCrop(insize)]

resize_crop6 = [transforms.RandomResizedCrop(insize, scale=(0.8, 1.0), ratio=(3. / 4., 4. / 3.))]

resize_crop9 = [transforms.RandomRotation(degrees=(-15, 15), expand=False),
                transforms.RandomResizedCrop(insize, (1./5, 4./5), (3./4, 4./3))]

color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)


# 四、组合各种数值增强  color, normalize ~~~ ##############################################################

resize_crop = resize_crop6
resize_crop.insert(1, color_jitter)

isnorm = False
if isnorm:
    trans_list = resize_crop + [totensor, coloraug, normalize, toimage]
else:
    trans_list = resize_crop + [totensor, coloraug, toimage]


# 五、图像处理正式开始 ~~~ ##############################################################

image_trans = transforms.Compose(trans_list)
# print(image_trans)
print('\nYour Image Transform is -->' + '\n' + repr(image_trans))

toc, issave, nums = 0, True, 18

save_dir = os.path.join(imgdir, 'trans-imgs')
if issave and not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print('\n保存路径：%s ' % save_dir)

for i in range(nums):
    tic = time.time()
    ximg = image_trans(image)
    toc += time.time() - tic
    if issave:
        ximg.save(os.path.join(save_dir, str(time.time()) + '.png'))
        # image.show()

print('\n消耗时间：%.6f 分.' % (toc / 60))
