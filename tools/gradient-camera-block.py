import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils as tvutils
import cv2
import sys
from collections import OrderedDict
import torch.nn as nn
import numpy as np
import argparse
import torch.nn.functional as F

"""
original version:

https://github.com/jacobgil/pytorch-grad-cam

https://github.com/utkuozbulak/pytorch-cnn-visualizations

new features +++:

重写特征抓取模型，XModelOutput， 无需封装原模型为<features+classifier>.

可对模型中1~3级的特征图进行可视化，支持 target_layers 以 moduleX.stageY.blcokZ 的方式工作.

支持 GPU, CPU，一般只需要CPU, 支持多Gpu保存的模型的加载(nn.DataParallel->cpu/1gpu)

支持 VGG, ResNet, DenseNet, ScaleNet, VivoNet.
"""


class XModelOutput(object):
    """
    手动遍历模型的各级模块，这些模块必须是串联的！
    具有内部并联结构的Block必须当做整体，无法手动进入!
        eg. ResNet 的 BottleBlock, ScaleNe 的 DoubelCouple, 当做整体.
    元模块：Convo, ReLu, MaxPool, Squential, 自定义Block等等，。。
    """

    def __init__(self, model, target_layer):
        super(XModelOutput, self).__init__()
        self.model = model
        self.target_layer = target_layer
        self.arch_name = model.arch_name
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients

    def __call__(self, x):
        target_features = []
        self.gradients = []
        # x = self.model.features(x)
        # x = self.model.classifier(x.view(x.size(0), -1))

        dot_nums = len(self.target_layer.split('.'))

        if dot_nums == 1:
            for name, module in self.model._modules.items():
                first_name = name
                print('->first_name ----->', first_name)
                if 'classifier' in name or 'fc' in name:
                    if x.size(-1) != 1 and xtils.startwithxyz(self.arch_name, ['dense', 'res', 'vivo']):
                        x = F.avg_pool2d(x, kernel_size=x.size()[-2:], stride=1)  # pool() in forward(), like densent
                    x = x.view(x.size(0), -1)
                x = module(x)
                if name in self.target_layer:
                    print('\t-> -----> FIND Target OK', first_name)
                    if not isinstance(x, (list, tuple)):
                        x.register_hook(self.save_gradient)
                        target_features += [x]
                    else:
                        x[0].register_hook(self.save_gradient)
                        target_features += [x[0]]
            assert len(target_features) >= 1, 'Target-Feature canot be Empty!'
            predicts = x
            return target_features, predicts

        elif dot_nums == 2:
            for name, module in self.model._modules.items():
                first_name = name
                print('->first_name ----->', first_name)
                if 'classifier' in name or 'fc' in name:
                    if x.size(-1) != 1 and xtils.startwithxyz(self.arch_name, ['dense', 'res', 'vivo']):
                        x = F.avg_pool2d(x, kernel_size=x.size()[-2:], stride=1)
                    x = x.view(x.size(0), -1)
                if len(module._modules) > 1:
                    for nm, mo in module._modules.items():
                        second_name = first_name + '.' + nm
                        print('\t->second_name ----->', second_name)
                        x = mo(x)
                        if second_name == self.target_layer:
                            print('\t\t-> -----> FIND Target OK', second_name)
                            if not isinstance(x, (list, tuple)):
                                x.register_hook(self.save_gradient)
                                target_features += [x]
                            else:
                                x[0].register_hook(self.save_gradient)
                                target_features += [x[0]]
                else:
                    x = module(x)
            assert len(target_features) >= 1, 'Target-Feature canot be Empty!'
            predicts = x
            return target_features, predicts

        elif dot_nums == 3:
            for name, module in self.model._modules.items():
                first_name = name
                print('->first_name ----->', first_name)
                if 'classifier' in name or 'fc' in name:
                    if x.size(-1) != 1 and xtils.startwithxyz(self.arch_name, ['dense', 'res', 'vivo']):
                        x = F.avg_pool2d(x, kernel_size=x.size()[-2:], stride=1)
                    x = x.view(x.size(0), -1)
                if len(module._modules) > 1:
                    for nm, mo in module._modules.items():
                        second_name = first_name + '.' + nm
                        print('\t->second_name ----->', second_name)
                        if len(mo._modules) > 1:
                            for n, m in mo._modules.items():
                                third_name = second_name + '.' + n
                                print('\t\t->third_name ----->', third_name)
                                x = m(x)
                                if third_name == self.target_layer:
                                    print('\t\t\t-> -----> FIND Target OK !', third_name)
                                    if not isinstance(x, (list, tuple)):
                                        x.register_hook(self.save_gradient)
                                        target_features += [x]
                                    else:
                                        x[0].register_hook(self.save_gradient)
                                        target_features += [x[0]]
                        else:
                            x = mo(x)
                else:
                    x = module(x)
            assert len(target_features) >= 1, 'Target-Feature canot be Empty!'
            predicts = x
            return target_features, predicts


class GradCam(object):
    def __init__(self, model, target_layer, imgsize=(224, 224), device=torch.device('cpu')):
        assert isinstance(imgsize, (tuple, list)) and len(imgsize) == 2
        self.model = model
        self.model.eval()
        self.device = device
        self.imgsize = imgsize

        self.modelout = XModelOutput(self.model, target_layer)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None, extinfo='noinfo'):
        # input: an input image, index: the expected category-Index of this image
        input = input.to(self.device)
        features, predicts = self.modelout(input)

        if isinstance(predicts, (tuple, list)):
            predicts = predicts[-1]

        pred_logits = F.softmax(predicts, dim=1)
        pred_score = round(pred_logits.max(dim=1)[0].item(), 5)
        pred_index = pred_logits.argmax(dim=1).item()
        pred_topk = pred_logits.topk(k=5, dim=1)

        print('\npredict-Score: %s, predict-Index: %s, target-Index: %s' % (pred_score, pred_index, index))
        print('\npredict-Top5: ', pred_topk)

        if extinfo != 'noinfo':
            extinfo += '-top%s-pid%s-tid%s' % \
                       (round(pred_score, 3), predicts.argmax(dim=1).item(), index)
            # print(extinfo)

        # index: category index.
        # index指定: http://www.wliang.me/2018/10/05/20181005_ImageNet1000分类名称和编号/
        # 也可先用某一类的简单样本，试出该类的输出端口(index)，然后在复杂样本上使用该index计算损失.
        if index == None:
            index = np.argmax(predicts.cpu().data.numpy())

        # 计算 one-hot 损失:
        # (1)使用指定的类别编号 index=xxid; (2)使用预测出（最大激活）的类别编号 index=None
        one_hot = np.zeros((1, predicts.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        one_hot = one_hot.to(self.device)
        one_hot = torch.sum(one_hot * predicts)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        self.model.zero_grad()
        one_hot.backward()

        grads_val = self.modelout.get_gradients()
        grads_val = grads_val[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, self.imgsize)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, extinfo


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = Variable(preprocessed_img, requires_grad=True)
    return input


def show_cam_on_image(img, mask, imgdir, imgname, savefmt='jpg', extinfo='noinfo'):
    assert savefmt in ['jpg', 'png', 'bmp']
    if extinfo == 'noinfo':
        extinfo = ''
    else:
        extinfo = '--' + extinfo
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    imgname = imgname.split('.')[0] + '_cam%s.%s' % (extinfo, savefmt)
    cam_path = os.path.join(imgdir, imgname)
    cv2.imwrite(cam_path, np.uint8(255 * cam))
    print('\nIimage has been saved at %s' % cam_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args


if __name__ == '__main__':

    import xtils, os
    from factory.model_factory import model_factory

    imgdir = 'E:\ClassifyTask\ClassifyNeXt\images'
    # imgname = ['n01531178_26526.JPEG', 'n01531178_4522.JPEG', 'n01531178_12015.JPEG', 'nacat.jpg', 'bird',
    #            'n03838899_1712.JPEG', 'music1.png', 'namei.png', 'waniao.jpg', 'catbirdfish.png',
    #            'n01531178_30004.JPEG', 'n01531178_3567.JPEG', 'xybirds.png', 'catbirdfish.png',
    #            'catbirdfishoars1.png', 'catbirdfishoars2.bmp', 'snowbird-goldfinch.png'][2]
    imgname = 'cat.jpg'
    args = get_args()
    args.use_cuda = False
    args.image_path = os.path.join(imgdir, imgname)
    args.image_size = [16, 32, 64, 128, 224, 256, 320, 512, 610, (192, 128)][-4]  # according to mode, most is 224
    args.image_size = (args.image_size, args.image_size) if isinstance(args.image_size, int) else args.image_size
    args.gpu_ids = [0, 1] if args.use_cuda else []
    args.device = torch.device('cuda:{}'.format(args.gpu_ids[0])) if len(args.gpu_ids) > 1 else torch.device('cpu')

    model_dir = 'E:\PreTrainedModels'                                     # 预训练模型存放仓库
    model_arch = ['vgg11', 'vgg19', 'resnet18', 'resnet50', 'resnet101',  # -7
                  'densenet121', 'densenet201', 'densenet161',            # -4
                  'vivonet', 'fishnet', 'scalenet'][-3]
    classId = [11, 1, 13, 285, 282, 356, 693, 2, 688, 693, 920, 682, 780, 423,  # -11
               850, 278, 775, 593, 13, 15, 20, 16, 920, None][-1]
    target_layer, target_class = {'vgg': ('features.36', classId),
                                  'res': ('layer4.2', classId),
                                  'den': ('features.denseblock4.denselayer24', classId),
                                  'sca': ('stage1.24', classId),
                                  'fis': ('fish.0', classId),
                                  'viv': ('features.denseblock4.denselayer16', classId)}[model_arch[:3]]

    if xtils.startwithxyz(model_arch, ('vgg', 'resnet', 'densenet')):
        model_path = xtils.get_pretrained_path(model_dir, arch_name=model_arch)
        model = model_factory(arch_name=model_arch, dataset='imagenet',
                              arch_kwargs={'pretrained': True, 'model_path': model_path})
        model = model.to(args.device)

    elif xtils.startwithxyz(model_arch, ('scalenet', 'vivonet', 'fishnet')):
        import arch_params
        arch_kwargs = getattr(arch_params, ['vo19', 'vo53', 'vo56', 'vo69', 'vo37', 'vo38', 'vc121v1111', 'fish99', 'ax16'][-3])
        arch_ckpt_vo19 = 'imagenet-scalenet48-ep99-it500499-acc73.97-best73.97-train76.9231-par20.09M-best-uxp.vo19.ckpt'
        arch_ckpt_vo53 = 'imagenet-scalenet103-ep97-it490489-acc69.53-best69.54-train68.5315-par5.08M-norm-uxp.vo53.ckpt'
        arch_ckpt_vo56 = 'imagenet-scalenet78-ep99-it500499-acc69.26-best69.27-train62.9371-par4.58M-norm-uxp.vo56.ckpt'
        arch_ckpt_vo69 = 'imagenet-scalenet103-ep128-it645644-acc71.63-best71.63-train77.6224-par5.09M-best-uxp.vo69.ckpt'
        arch_ckpt_vo37 = 'imagenet-scalenet68-ep99-it500499-acc73.08-best73.08-train69.9301-par10.11M-best-uxp.vo37.ckpt'
        arch_ckpt_vo38 = 'imagenet-scalenet68-ep95-it480479-acc71.18-best71.18-train71.3287-par6.71M-best-uxp.vo38.ckpt'
        arch_ckpt_ax16 = 'cifar10-scalenet54-ep300-it117690-acc94.75-best94.75-train100.0000-par2.07M-best-uxp.ax16.ckpt'
        arch_ckpt_vc12 = 'imagenet-vivonet295-ep70-it355354-acc71.62-best71.62-train70.6294-par7.87M-best-uxp.vc121v1111.ckpt'
        arch_ckpt = os.path.join(model_dir, arch_ckpt_vc12)
        model = model_factory(arch_name=model_arch, dataset='imagenet', arch_kwargs=arch_kwargs)
        model = xtils.load_ckpt_weights(model, arch_ckpt, args.device, mgpus_to_sxpu='auto', noload=False)
        # torch.save({'model': model.state_dict()}, f=os.path.join(model_dir, 'xxmodel-cpu.ckpt'))
    else:
        raise NotImplementedError('Unkown Model Name %s' % model_arch)
    model.arch_name = model_arch
    print('\noriginal model-->\n', model)

    assert os.path.isfile(args.image_path)
    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, args.image_size))
    img /= 255
    image = preprocess_image(img)

    grad_cam = GradCam(model=model, target_layer=target_layer, imgsize=args.image_size, device=args.device)

    # If index is None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    extinfo = '%s-%s' % (model_arch, args.image_size)

    mask, extinfo = grad_cam(image, index=target_class, extinfo=extinfo)

    show_cam_on_image(img, mask, imgdir, imgname, savefmt='bmp', extinfo=extinfo)
