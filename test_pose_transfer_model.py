from __future__ import division, print_function

import torch
from data.data_loader import CreateDataLoader
from options.pose_transfer_options import TestPoseTransferOptions
from misc.visualizer import GANVisualizer_V3
from misc.loss_buffer import LossBuffer

import util.io as io
import os
import sys
import time
import numpy as np
from collections import OrderedDict

opt = TestPoseTransferOptions().parse()
train_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'batch_size', 'is_train'}
for k, v in train_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)
# create model
if opt.which_model_T in {'unet', 'resnet'}:
    from models.supervised_pose_transfer_model import SupervisedPoseTransferModel
    model = SupervisedPoseTransferModel()
elif opt.which_model_T == 'vunet':
    from models.vunet_pose_transfer_model import VUnetPoseTransferModel
    model = VUnetPoseTransferModel()
else:
    raise NotImplementedError()

model.initialize(opt)
# create data loader
# train_loader = CreateDataLoader(opt, split = 'train')
val_loader = CreateDataLoader(opt, split = 'test')
# create visualizer
visualizer = GANVisualizer_V3(opt)

pavi_upper_list = ['PSNR', 'SSIM']
pavi_lower_list = ['loss_L1', 'loss_content', 'loss_style', 'loss_G', 'loss_D', 'loss_pose', 'loss_kl']

loss_buffer = LossBuffer(size=len(val_loader))

for i, data in enumerate(val_loader):
    model.set_input(data)
    model.test(compute_loss=False)
    loss_buffer.add(model.get_current_errors())
    print('\rTesting %d/%d (%.2f%%)' % (i, len(val_loader), 100.*i/len(val_loader)), end = '')
    sys.stdout.flush()

print('\n')

test_error = loss_buffer.get_errors()
visualizer.print_error(test_error)

   