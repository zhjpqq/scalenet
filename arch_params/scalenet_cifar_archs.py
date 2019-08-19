# cifar10 #####################

ci7 = {'stages': 3, 'depth': 22, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (3, 3, 3), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
       'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 22, 2 * 22),
       'dfunc': ('O', 'O'), 'dstyle': 'maxpool', 'fcboost': 'none', 'nclass': 10,
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 32,
       'summer': 'split', 'afisok': False, 'version': 2}  # 1M  35L  0.22s  88fc

# NET Width
ao1 = {'stages': 1, 'depth': 64, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 2.06M  0.96G  44L  0.05s  94.57%   => ax16@titan but-only-lastfc

ao2 = {'stages': 1, 'depth': 56, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.57M  0.74G  44L  0.05s

ao3 = {'stages': 1, 'depth': 48, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 1.16M  0.54G  44L  0.05s

ao4 = {'stages': 1, 'depth': 40, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 0.81M  0.38G  44L  0.04s

ao5 = {'stages': 1, 'depth': 36, 'branch': 3, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 10, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 2}  # 0.65M  0.30G  44L  0.03s


# cifar100 #####################

# NET WIDTH
bo1 = {'stages': 1, 'depth': 100, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 3.62M  1.44G  42L  0.22s

bo2 = {'stages': 1, 'depth': 90, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 2.93M  1.18G  42L  0.22s

bo3 = {'stages': 1, 'depth': 80, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 2.32M  0.93G  42L  0.22s

bo4 = {'stages': 1, 'depth': 70, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 1.78M  0.71G  42L  0.17s

bo5 = {'stages': 1, 'depth': 60, 'branch': 1, 'rock': 'U', 'kldloss': False,
       'layers': (10,), 'blocks': ('D',), 'slink': ('A',),
       'growth': (0,), 'classify': (0,), 'expand': (), 'dfunc': (),
       'fcboost': 'none', 'nclass': 100, 'summer': 'split',
       'last_branch': 1, 'last_down': False, 'last_dfuc': 'D', 'last_expand': 36,
       'afisok': False, 'version': 3}  # 1.30M  0.52G  42L  0.17s