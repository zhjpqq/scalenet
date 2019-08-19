vo69 = {'stages': 1, 'branch': 1, 'rock': 'N', 'depth': 64, 'kldloss': False,
        'layers': (25,), 'blocks': ('D',), 'slink': ('A',), 'growth': (0,),
        'classify': (0,), 'expand': (), 'dfunc': (), 'afisok': False,
        'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1100 - 64, 'version': 3}  # 5.08M  9.53G  103L  0.65s  520fc

vo72 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
        'layers': (4, 5, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (-8, -20, -50), 'classify': (0, 0, 0), 'expand': (1 * 120, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1700 - 440, 'version': 3}  # 30.51M  10.83G  59L  0.51s  1700fc

vo76 = {'stages': 3, 'branch': 1, 'rock': 'N', 'depth': 80, 'kldloss': False,
        'layers': (4, 5, 6), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (-8, -8, -8), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 80), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1730 - 320, 'version': 3}  # 20.02M  7.62G  57L  0.25s  1730fc

vo21 = {'stages': 3, 'branch': 1, 'rock': 'U', 'depth': 80, 'kldloss': False,
        'layers': (4, 5, 5), 'blocks': ('D', 'D', 'S'), 'slink': ('A', 'A', 'A'),
        'growth': (0, 0, 0), 'classify': (0, 0, 0), 'expand': (1 * 80, 2 * 120), 'afisok': False,
        'dfunc': ('O', 'O'), 'dstyle': ('maxpool', 'convk2m', 'convk2'), 'fcboost': 'none',
        'nclass': 1000, 'summer': 'split', 'last_branch': 1, 'last_down': True, 'last_dfuc': 'E',
        'last_expand': 1746 - 400, 'version': 3}  # 25.01M  4.64G  54L  0.46s  1746fc

