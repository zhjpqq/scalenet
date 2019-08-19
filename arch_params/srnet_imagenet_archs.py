# ScaleResNet

from collections import OrderedDict

# imagenet
sr1_cfg = {
    '3-4-4': dict(indepth=64, bnums=1, btype='bottle', main_aux='aux',
                  fc_middep=0, fc_drop=(0, 0), fc_actfuc='relu', with_fc=True, active_fc=False, active_me=True,
                  squeeze='pool', sq_outdep=256, sq_ksize=7, sq_groups='gcu', sq_active='relu', sq_isview=True),
    '7-8-8': dict(indepth=128, bnums=1, btype='bottle', main_aux='aux',
                  fc_middep=0, fc_drop=(0, 0), fc_actfuc='relu', with_fc=True, active_fc=False, active_me=True,
                  squeeze='pool', sq_outdep=512, sq_ksize=7, sq_groups='gcu', sq_active='relu', sq_isview=True),
    '13-16-16': dict(indepth=256, bnums=1, btype='bottle', main_aux='aux',
                     fc_middep=0, fc_drop=(0, 0), fc_actfuc='relu', with_fc=True, active_fc=False, active_me=True,
                     squeeze='pool', sq_outdep=1024, sq_ksize=7, sq_groups='gcu', sq_active='relu', sq_isview=True),
    '16-32-32': dict(indepth=512, bnums=0, btype='bottle', main_aux='main',
                     fc_middep=0, fc_drop=(0, 0), fc_actfuc='relu', with_fc=True, active_fc=False, active_me=True,
                     squeeze='pool', sq_outdep=2048, sq_ksize=7, sq_groups='gcu', sq_active='relu', sq_isview=True)
}
sr1 = OrderedDict(depth=64, btype='bottle', layers=[3, 4, 6, 3], dataset='imagenet',
                  summar='concat', sum_active=True, sfc_with=True, sfc_indep=3840,
                  sfc_middep=0, sfc_drop=(0, 0), sfc_active='relu', head_key_cfg=sr1_cfg)
