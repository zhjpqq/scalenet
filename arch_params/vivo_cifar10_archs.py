vv1 = dict(depth=40, growthRate=18, reduction=0.5, nClasses=10,
           kinds=(0, 0, 0), dividers=((), (), ()), branchs=('-', '-', '-'))  # 0.39M

# vivonet-121
vc121v1111 = {'num_init_features': 64, 'growth_rate': 32, 'num_classes': 1000, 'drop_rate': 0, 'bn_size': 4,
              'block_config': (('v1', 6), ('v1', 12), ('v1', 24), ('v1', 16)), }