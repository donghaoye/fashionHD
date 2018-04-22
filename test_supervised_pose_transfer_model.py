from __future__ import division, print_function

import torch
from models.supervised_pose_transfer_model import SupervisedPoseTransferModel
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
# create model
model = SupervisedPoseTransferModel()
model.initialize(opt)
# create data loader
# train_loader = CreateDataLoader(opt, split = 'train')
val_loader = CreateDataLoader(opt, split = 'test')
# create visualizer
visualizer = GANVisualizer_V3(opt)

pavi_upper_list = ['PSNR', 'SSIM']
pavi_lower_list = ['loss_L1', 'loss_content', 'loss_style', 'loss_G', 'loss_D', 'loss_pose']

   
loss_buffer = LossBuffer(size=len(val_loader))

for i, data in enumerate(val_loader):
    model.set_input(data)
    model.test(compute_loss=True)
    loss_buffer.add(model.get_current_errors())
    print('\rTesting %d/%d (%.2f%%)' % (i, len(val_loader), 100.*i/len(val_loader)), end = '')
    sys.stdout.flush()
print('\n')

test_error = loss_buffer.get_errors()
visualizer.print_error(test_error)

   