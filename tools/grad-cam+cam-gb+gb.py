import torch
from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models
from torchvision import utils as tvutils
import cv2
import sys
from collections import OrderedDict
import numpy as np
import argparse

"""
https://github.com/jacobgil/pytorch-grad-cam

https://github.com/utkuozbulak/pytorch-cnn-visualizations

gitub中的原始代码，运行ok

"""


class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers, target_part='features'):
        self.model = model
        self.feature_extractor = FeatureExtractor(getattr(self.model, target_part), target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


class GradCam:
    # one method
    def __init__(self, model, target_layer_names, use_cuda, target_part_name='features'):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names, target_part_name)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        self.model.zero_grad()
        one_hot.backward()

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

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


class GuidedBackpropReLU(Function):
    # another method
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    # another method
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        # # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            print('module-name', module.__class__.__name__)
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU()
                print('Replace:', module, self.model.features._modules[idx])

        # for idx, module in self.model._modules.items():
        #     print('module-name', module.__class__.__name__)
        #     if module.__class__.__name__ == 'ReLU':
        #         self.model._modules[idx] = GuidedBackpropReLU()
        #         print('xxxxx', module, self.model._modules[idx])

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad=True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.zero_grad()
        one_hot.backward()

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


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
    from xmodels import tvm_vggs, tvm_resnet, tvm_densenet

    imgdir = './images'
    imgname = 'both.png'
    args = get_args()
    args.use_cuda = False
    args.image_path = os.path.join(imgdir, imgname)

    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.

    model_arch = ['vgg11', 'vgg19', 'resnet18', 'resnet34', 'resnet50'][1]
    model_dir = '/data/zhangjp/PreTrainedModels'
    model_path = xtils.get_pretrained_path(model_dir, arch_name=model_arch)
    if model_arch.startswith('vgg'):
        model = getattr(tvm_vggs, model_arch)(pretrained=True, model_path=model_path)
    elif model_arch.startswith('resnet'):
        model = getattr(tvm_resnet, model_arch)(pretrained=True, model_path=model_path)
    elif model_arch.startswith('dense'):
        model = getattr(tvm_densenet, model_path)(pretrained=True, model_path=model_path)
    else:
        raise NotImplementedError
    model.arch_name = model_arch
    print(model)

    img = cv2.imread(args.image_path, 1)
    img = np.float32(cv2.resize(img, (224, 224)))
    img /= 255
    input = preprocess_image(img)

    grad_cam = GradCam(model=model, target_layer_names=["35"], use_cuda=args.use_cuda, target_part_name='features')

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask, imgdir, imgname)

    # another metod 1
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
    gb = gb_model(input, index=target_index)
    gb_path = os.path.join(imgdir, imgname.split('.')[0] + '_gb.jpg')
    tvutils.save_image(torch.from_numpy(gb), gb_path)

    # another metod 2
    cam_mask = np.zeros(gb.shape)
    for i in range(0, gb.shape[0]):
        cam_mask[i, :, :] = mask

    cam_gb = np.multiply(cam_mask, gb)
    cam_gb_path = os.path.join(imgdir, imgname.split('.')[0] + '_cam_gb.jpg')
    tvutils.save_image(torch.from_numpy(cam_gb), cam_gb_path)
