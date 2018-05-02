from __future__ import division

import os
import shutil

src_dir = 'datasets/Zalando/Img/img_zalando_256/'
tar_dir1 = 'datasets/Zalando/Img/img_zalando_person/'
tar_dir2 = 'datasets/Zalando/Img/img_zalando_cloth/'

fn_list = os.listdir(src_dir)

for idx, fn in enumerate(fn_list):
    print(idx)
    if fn.endswith('_1.jpg'):
        fn1 = fn[0:6]+'.jpg'
        shutil.copyfile(src_dir + fn, tar_dir2 + fn1)
    else:
        shutil.copyfile(src_dir + fn, tar_dir1 + fn)

