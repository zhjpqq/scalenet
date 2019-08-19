"""
 训练github模型，可复制其架构设计到此处，并用新名称代替原来的架构名称

 统一格式：

 xxnet = {'arch':       'xx',
          'cfg':        '' || { custom-cfg },
          'model_path': ['local' || 'download' || 'path/to/xxnet.pth' || '']}
"""

# mobile v3
mb1 = {'arch': 'mbvdl', 'cfg': '', 'model_path': ''}

mbs = {'arch': 'mbvds', 'cfg': '', 'model_path': ''}

# hrnet
hr1 = {'MODEL': {'NAME': 'hrnet_w18', 'IMAGE_SIZE': [224, 224], 'NUM_CLASSES': 1000,
                 'PRETRAINED': '',
                 'EXTRA':
                     {'STAGE2': {'NUM_MODULES': 1, 'NUM_BRANCHES': 2, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4],
                                 'NUM_CHANNELS': [18, 36], 'FUSE_METHOD': 'SUM'},
                      'STAGE3': {'NUM_MODULES': 4, 'NUM_BRANCHES': 3, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4],
                                 'NUM_CHANNELS': [18, 36, 72], 'FUSE_METHOD': 'SUM'},
                      'STAGE4': {'NUM_MODULES': 3, 'NUM_BRANCHES': 4, 'BLOCK': 'BASIC', 'NUM_BLOCKS': [4, 4, 4, 4],
                                 'NUM_CHANNELS': [18, 36, 72, 144], 'FUSE_METHOD': 'SUM'}}}}

# resnet
res1 = {'arch': 'resnet50', 'cfg': '', 'model_path': ''}   # 标准resnet50

res2 = res1

resx = {'arch': 'resx', 'cfg': {'block': 'Bottleneck', 'layers': [2, 2, 2, 2], 'num_classes': 1000},
        'model_path': ''}   # 自定义resx

# fishnet
fi1 = {'arch': 'fish99', 'cfg': '', 'model_path': ''}

# effnet
ef1 = {'arch': 'effb0', 'cfg': '', 'model_path': '', 'override_params': None}
