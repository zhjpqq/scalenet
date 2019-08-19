# ScaleNet  Source Code

#ScaleNet Architecture
![ScaleNet Architecture](images/scalenet-architecture.jpg)

#the CAM Camparision between ScaleNet and DenseNet
![Multi-Scale Input](images/multi-scale-show-5.jpg)
top: scalenet  bottom:densenet

#How to Train ScaleNet

## Train by commands in Terminal
'run_main.py' is the entry function.

It can be used as following:

```
cd /to/your/project/root

# 
python run_main.py -name 'scalenet' -arch 'net1' -cfg 'cfgnet1' -exp 'exp.net1' -gpu 1 3

# also can run by nohup to save logs into a file
nohup python run_main.py -name 'scalenet' -arch 'net1' -cfg 'cfgnet1' -exp 'exp.net1' -gpu 1 3 1>printlog/net1.out 2>&1 &

```

## Train by client in Pycharm
Find the following lines in 'run_main.py', and remove the comments on these lines:
```
args.arch_name = 'scalenet'
args.arch_list = ['net1']
args.cfg_dict = 'cfgnet1'
args.exp_version = 'exp.net1'
args.gpu_ids = [0, 1, 2, 3, 5, 6]
print('\n=> Your Args is :', args, '\n')
```

# Other Features

- Unified Training Framework for Classification.
-
- Unified Data Factory, including CIFAR, IMAGENET, SVHN, LSV etc..
-
- Unified Model Factory, including Pytorch-official models and New Models in 2019.  
-
- New Models in 2019: ScaleNet, EfficientNet, MobieNet-V3, HighResolutionNet etc..
-
- Many Usefull tools in xtils.py and /tools, eg CAM, FLOPS, Params etc ..
-

# Pre-trained ScaleNet models on ImageNet
preparing ... coming soon!