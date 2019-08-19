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

"""
https://github.com/jacobgil/pytorch-grad-cam

https://github.com/utkuozbulak/pytorch-cnn-visualizations

增加 XModel 包装函数，将Model分割为features + classifier

缺点：只能对一级结构的特征图进行可视化，only work by stage!
"""


class XModel(nn.Module):
    """
    将一个模型分割为两部分，features + ( view ) + classifier
    need_view=True.   某些模型的Squzee()操作以F.view()的形式在forward()函数中进行.
    need_view=False.  某些模型的Squzee()操作被合并到了其classifier模块中.
    """

    def __init__(self, need_view=False):
        super(XModel, self).__init__()
        self.need_view = need_view
        self.features = nn.Sequential()
        self.classifier = nn.Sequential()

    def foward(self, x):
        x = self.features(x)
        if self.need_view:
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def x_model(self, model, features=(), classifier=()):
        # common Interface
        for name in features:
            module = model._modules.get(name, None)
            if module is not None:
                self.features.add_module(name, module)
        for name in classifier:
            module = model._modules.get(name, None)
            self.classifier.add_module(name, module)

    def x_vgg(self, vgg):
        self.features = vgg.features
        self.classifier = vgg.classifier

    def x_resnet(self, resnet):
        features = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1',
                    'layer2', 'layer3', 'layer4', 'avgpool']
        classifier = ['fc']
        for name in features:
            self.features.add_module(name, getattr(resnet, name))
        for name in classifier:
            self.classifier.add_module(name, getattr(resnet, name))

    def x_scalenet(self, scalenet):
        features = ['pyramid', 'stage1', 'stage2', 'stage3', 'stage4']
        classifier = ['summary', 'boost']
        for name in features:
            module = scalenet._modules.get(name, None)
            if module is not None:
                self.features.add_module(name, module)
        for name in classifier:
            self.classifier.add_module(name, getattr(scalenet, name))


class FeatureExtractor(object):
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __callx__(self, x):
        outputs = []
        self.gradients = []
        tlayers = [layer.split('.') for layer in self.target_layers][0]
        for name, module in self.model._modules.items():
            print(self.target_layers, '->', name)
            x = module(x)
            if name in tlayers[0]:
                tmodule = self.model._modules
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            print('target_layer: %s ---> current_name: %s'% (self.target_layers, name))
            x = module(x)
            if name in self.target_layers:
                if not isinstance(x, (list, tuple)):
                    x.register_hook(self.save_gradient)
                    outputs += [x]
                else:
                    x[0].register_hook(self.save_gradient)
                    outputs += [x[0]]
        return outputs, x


class ModelOutputs(object):
    """ Class for making a forward pass, and getting:
    1. The network final output/predict.
    2. Features from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers, target_part='features'):
        self.model = model
        if '.' not in target_part:
            # self.feature_extractor = FeatureExtractor(self.model._modules[target_part], target_layers)
            self.feature_extractor = FeatureExtractor(getattr(self.model, target_part), target_layers)
        else:
            express = 'self.model'
            target_part = target_part.split('.')
            for part in target_part:
                express += '._modules["%s"]' % part
            target_part = eval(express)
            self.feature_extractor = FeatureExtractor(target_part, target_layers)
            print('xxxxx')

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_features, final_features = self.feature_extractor(x)
        if not isinstance(final_features, (tuple, list)):
            output = final_features.view(final_features.size(0), -1)
            predicts = self.model.classifier(output)
            return target_features, predicts
        else:
            # output = final_features.view(final_features.size(0), -1)
            predicts = self.model.classifier(final_features)
            return target_features, predicts


class GradCam(object):
    # one method
    def __init__(self, model, target_layer_names, target_part_name='features', use_cuda=False):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.modelout = ModelOutputs(self.model, target_layer_names, target_part_name)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, predicts = self.modelout(input.cuda())
        else:
            features, predicts = self.modelout(input)

        if isinstance(predicts, (tuple, list)):
            predicts = predicts[-1]

        if index == None:
            index = np.argmax(predicts.cpu().data.numpy())

        one_hot = np.zeros((1, predicts.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * predicts)
        else:
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
        cam = cv2.resize(cam, (224, 224))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


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


def show_cam_on_image(img, mask, imgdir, imgname):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cam_path = os.path.join(imgdir, imgname.split('.')[0] + '_cam.jpg')
    cv2.imwrite(cam_path, np.uint8(255 * cam))


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
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    import xtils, os
    from factory.model_factory import model_factory

    imgdir = './images'
    imgname = ['both.png', 'cat.jpg'][-1]
    args = get_args()
    args.use_cuda = False
    args.image_path = os.path.join(imgdir, imgname)

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    model_dir = '/data/zhangjp/PreTrainedModels'
    model_arch = ['vgg11', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121', 'scalenet'][3]
    target_part, target_layer, target_index = ('features', 'layer4', None)

    if not model_arch.startswith('scalenet'):
        model_path = xtils.get_pretrained_path(model_dir, arch_name=model_arch)
        model = model_factory(arch_name=model_arch, dataset='imagenet',
                              arch_kwargs={'pretrained': True, 'model_path': model_path})
    else:
        import arch_params

        arch_name = 'vo30'
        arch_ckpt = ''
        gpu_ids = [0, 1]
        device = torch.device('cuda:{}'.format(gpu_ids[0]))
        model = model_factory(arch_name=model_arch, dataset='imagenet', arch_kwargs=getattr(arch_params, arch_name))
        # model.to(device)
        # if len(gpu_ids) > 1:
        #     model = nn.DataParallel(model, device_ids=gpu_ids)
        # ckptf = torch.load(f=arch_ckpt, map_location=device)  # multi-gpu saved ckpt, must let gpu_ids > 1
        # model.load_state_dict(torch.load(f=arch_ckpt, map_location=device)['model'])
    # print('\noriginal xmodel-->\n', model)

    xmodel = XModel()

    # print('\nold xmodel-->\n', xmodel)
    if model_arch.startswith('vgg'):
        xmodel.x_vgg(vgg=model)
    elif model_arch.startswith('resnet'):
        xmodel.x_resnet(resnet=model)
    elif model_arch.startswith('scalenet'):
        xmodel.x_scalenet(scalenet=model)
    print('\nnew xmodel-->\n', xmodel)

    model = xmodel

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224)))
    img /= 255
    input = preprocess_image(img)

    grad_cam = GradCam(model=model, target_part_name=target_part,
                       target_layer_names=target_layer, use_cuda=args.use_cuda)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask, imgdir, imgname)
