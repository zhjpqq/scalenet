from collections import OrderedDict

ms5_hkc = {
    '20-32-32': dict(blocks=0, indepth=530, growth=5, pre_ind_grow=250,
                     expand=(), exksp=(), exisse='ft-1:3', pre_ksp_half=False,
                     groups='auto', skgroups='gcu', conv_drop='all-0', conv_active='relu',
                     fc_middep=0, fc_drop=(0, 0), fc_active='relu', seactive='hsig', with_fc=True)
}  # indepth = 4*60 + 290 = 530 ; pre_ind_grow = 4*60+10=250 ;
ms5 = OrderedDict(depth=60, growth=10, blocks=20, expand=((2, 60), (5, 60), (10, 60), (15, 290)),
                  exksp=((2, '3.2.1'), (5, '3.2.1'), (10, '3.2.1'), (15, '3.2.1')), exisse='ft-1:20',
                  groups='auto', skgroups='gcu', prestride='1/2-2', conv_drop='all-0', conv_act='relu',
                  summar='independ', sfc_with=False, sfc_poll=('avg', 'minmax'), sfc_indep=1770, sfc_middep=2014,
                  sfc_drop=(0, 0), sfc_active='relu', seactive='hsig', head_key_cfg=ms5_hkc)
# 118L 5.80M 1.06G 0.89s  ==> ms2 without dropout


# cifar10
ms2_hkc = {
    '3-1-1': dict(blocks=3, indepth=16, growth=10, pre_ind_grow=32,
                  expand=((1, 5), (2, 5), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                  groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                  fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                  active_fc=False, active_me=True, main_aux='aux'),
    '3-1-1@3': dict(blocks=3, indepth=16, growth=10, pre_ind_grow=32,
                    expand=((1, 5), (2, 5), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                    groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                    fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                    active_fc=False, active_me=True, main_aux='aux'),
    '8-2-2': dict(blocks=3, indepth=32, growth=10, pre_ind_grow=48,
                  expand=((1, 10), (2, 10), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                  groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                  fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                  active_fc=False, active_me=True, main_aux='aux'),
    '8-2-2@6': dict(blocks=3, indepth=32, growth=10, pre_ind_grow=48,
                    expand=((1, 10), (2, 10), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                    groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                    fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                    active_fc=False, active_me=True, main_aux='aux'),
    '15-4-4': dict(blocks=0, indepth=96, growth=0, pre_ind_grow=0,
                   expand=(), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                   groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                   fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                   active_fc=False, active_me=True, main_aux='main'),
}  # 132L 0.259M  0.051G  292fc  0.11s
ms2 = OrderedDict(depth=16, growth=16, blocks=15, expand=((4, 16), (9, 32), (9, 64)),
                  exksp=((4, '3.2.1'), (9, '3.2.1')), exisse='ft-0:0',
                  groups='auto', skgroups='gcu', prestride='1/1-1', conv_drop='all-0.0', conv_act='relu',
                  summar='concat', sfc_with=True, sfc_poll=('avg', 'minmax'), sfc_indep=292, sfc_middep=0,
                  sfc_drop=(0.0, 0.0), sfc_active='relu', seactive='hsig', head_key_cfg=ms2_hkc, dataset='cifar10')
ms4 = ms2

# cifar10
ms3_hkc = OrderedDict({
    '8-2-2': dict(blocks=3, indepth=32, growth=10, pre_ind_grow=48,
                  expand=((1, 10), (2, 10), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                  groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                  fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                  active_fc=False, active_me=True, main_aux='aux'),
    '15-4-4': dict(blocks=0, indepth=96, growth=0, pre_ind_grow=0,
                   expand=(), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                   groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                   fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                   active_fc=False, active_me=True, main_aux='main'),
})  # 75L 0.203M  0.0269G  158fc  0.066s
ms3 = OrderedDict(depth=16, growth=16, blocks=15, expand=((4, 16), (9, 32), (9, 64)),
                  exksp=((4, '3.2.1'), (9, '3.2.1')), exisse='ft-0:0',
                  groups='auto', skgroups='gcu', prestride='1/1-1', conv_drop='all-0.0', conv_act='relu',
                  summar='concat', sfc_with=True, sfc_poll=('avg', 'minmax'), sfc_indep=158, sfc_middep=0,
                  sfc_drop=(0.0, 0.0), sfc_active='relu', seactive='hsig', head_key_cfg=ms3_hkc, dataset='cifar10')

# cifar10
ms6_hkc = {
    '8-2-2': dict(blocks=3, indepth=32, growth=10, pre_ind_grow=48,
                  expand=((1, 10), (2, 10), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                  groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                  fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                  active_fc=False, active_me=True, main_aux='aux'),
    '8-2-2@6': dict(blocks=3, indepth=32, growth=10, pre_ind_grow=48,
                    expand=((1, 10), (2, 10), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                    groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                    fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                    active_fc=False, active_me=True, main_aux='aux'),
    '15-4-4': dict(blocks=0, indepth=96, growth=0, pre_ind_grow=0,
                   expand=(), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                   groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                   fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                   active_fc=False, active_me=True, main_aux='main'),
}  # 94L  0.232M  0.031G  220fc  0.08s
ms6 = OrderedDict(depth=16, growth=16, blocks=15, expand=((4, 16), (9, 32), (9, 64)),
                  exksp=((4, '3.2.1'), (9, '3.2.1')), exisse='ft-0:0',
                  groups='auto', skgroups='gcu', prestride='1/1-1', conv_drop='all-0.0', conv_act='relu',
                  summar='concat', sfc_with=True, sfc_poll=('avg', 'minmax'), sfc_indep=220, sfc_middep=0,
                  sfc_drop=(0.0, 0.0), sfc_active='relu', seactive='hsig', head_key_cfg=ms6_hkc, dataset='cifar10')
# cifar10
ms7_hkc = {
    '3-1-1': dict(blocks=3, indepth=16, growth=10, pre_ind_grow=32,
                  expand=((1, 0), (2, 0), (3, 46)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                  groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                  fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                  active_fc=False, active_me=True, main_aux='aux'),
    '3-1-1@3': dict(blocks=3, indepth=16, growth=10, pre_ind_grow=32,
                    expand=((1, 0), (2, 0), (3, 46)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                    groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                    fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                    active_fc=False, active_me=True, main_aux='aux'),
    '8-2-2': dict(blocks=3, indepth=32, growth=10, pre_ind_grow=48,
                  expand=((1, 0), (2, 0), (3, 30)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                  groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                  fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                  active_fc=False, active_me=True, main_aux='aux'),
    '8-2-2@6': dict(blocks=3, indepth=32, growth=10, pre_ind_grow=48,
                    expand=((1, 0), (2, 0), (3, 30)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                    groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                    fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                    active_fc=False, active_me=True, main_aux='aux'),
    '15-4-4': dict(blocks=0, indepth=224, growth=0, pre_ind_grow=0,
                   expand=(), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                   groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                   fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                   active_fc=False, active_me=True, main_aux='main'),
}  # 109L 0.236M  0.036G  292fc  0.11s
ms7 = OrderedDict(depth=16, growth=16, blocks=15, expand=((4, 16), (9, 32), (9, 64), (15, 128)),
                  exksp=((4, '3.2.1'), (9, '3.2.1')), exisse='ft-0:0',
                  groups='auto', skgroups='gcu', prestride='1/1-1', conv_drop='all-0.0', conv_act='relu',
                  summar='concat', sfc_with=True, sfc_poll=('avg', 'minmax'), sfc_indep=472, sfc_middep=0,
                  sfc_drop=(0.0, 0.0), sfc_active='relu', seactive='hsig', head_key_cfg=ms7_hkc, dataset='cifar10')

# cifar10
ms8_hkc = {
    '8-2-2': dict(blocks=3, indepth=32, growth=10, pre_ind_grow=48,
                  expand=((1, 10), (2, 10), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                  groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                  fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                  active_fc=False, active_me=True, main_aux='aux',
                  squeeze='conv', sq_outdep=62 * 2, sq_ksize=16, sq_groups='gcu', sq_active='relu', sq_isview=True),
    '8-2-2@6': dict(blocks=3, indepth=32, growth=10, pre_ind_grow=48,
                    expand=((1, 10), (2, 10), (3, 10)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                    groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                    fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                    active_fc=False, active_me=True, main_aux='aux',
                    squeeze='conv', sq_outdep=62 * 2, sq_ksize=12, sq_groups='gcu', sq_active='relu', sq_isview=True),
    '15-4-4': dict(blocks=0, indepth=96, growth=0, pre_ind_grow=0,
                   expand=(), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                   groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                   fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                   active_fc=False, active_me=True, main_aux='main',
                   squeeze='conv', sq_outdep=96 * 2, sq_ksize=8, sq_groups='gcu', sq_active='relu', sq_isview=True),
}  # 97L  0.298M  0.031G  440fc  0.08s
ms8 = OrderedDict(depth=16, growth=16, blocks=15, expand=((4, 16), (9, 32), (9, 64)),
                  exksp=((4, '3.2.1'), (9, '3.2.1')), exisse='ft-0:0',
                  groups='auto', skgroups='gcu', prestride='1/1-1', conv_drop='all-0.0', conv_act='relu',
                  summar='concat', sfc_with=True, sfc_poll=('avg', 'minmax'), sfc_indep=440, sfc_middep=0,
                  sfc_drop=(0.0, 0.0), sfc_active='relu', seactive='hsig', head_key_cfg=ms8_hkc, dataset='cifar10')

# cifar10
ms9_hkc = {
    '8-2-2': dict(blocks=3, indepth=32, growth=10, pre_ind_grow=48,
                  expand=((1, 0), (2, 0), (3, 0)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                  groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                  fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                  active_fc=False, active_me=True, main_aux='aux',
                  squeeze='conv', sq_outdep=32 * 2, sq_ksize=16, sq_groups='gcu', sq_active='relu', sq_isview=True),
    '8-2-2@6': dict(blocks=3, indepth=32, growth=10, pre_ind_grow=48,
                    expand=((1, 0), (2, 0), (3, 0)), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                    groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                    fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                    active_fc=False, active_me=True, main_aux='aux',
                    squeeze='conv', sq_outdep=32 * 2, sq_ksize=12, sq_groups='gcu', sq_active='relu', sq_isview=True),
    '15-4-4': dict(blocks=0, indepth=96, growth=0, pre_ind_grow=0,
                   expand=(), exksp=(), exisse='ft-0:0', pre_ksp_half=False,
                   groups='auto', skgroups='gcu', conv_drop='all-0.', conv_active='relu',
                   fc_middep=0, fc_drop=(0.0, 0.0), fc_active='relu', with_fc=True,
                   active_fc=False, active_me=True, main_aux='main',
                   squeeze='pool', sq_outdep=96 * 2, sq_ksize=8, sq_groups='gcu', sq_active='relu', sq_isview=True),
}  # 82L  0.223M  0.024G  224fc  0.08s
ms9 = OrderedDict(depth=16, growth=16, blocks=15, expand=((4, 16), (9, 32), (9, 64)),
                  exksp=((4, '3.2.1'), (9, '3.2.1')), exisse='ft-0:0',
                  groups='auto', skgroups='gcu', prestride='1/1-1', conv_drop='all-0.0', conv_act='relu',
                  summar='concat', sfc_with=True, sfc_poll=('avg', 'minmax'), sfc_indep=224, sfc_middep=0,
                  sfc_drop=(0.0, 0.0), sfc_active='relu', seactive='hsig', head_key_cfg=ms9_hkc, dataset='cifar10')
