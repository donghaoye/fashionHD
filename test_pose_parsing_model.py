from __future__ import division, print_function

import torch
from data.data_loader import CreateDataLoader
from options.pose_parsing_options import TestPoseParsingOptions
from models.pose_parsing_model import PoseParsingModel
from misc.visualizer import GANVisualizer_V3
from misc.loss_buffer import LossBuffer

import util.io as io
import os
import sys
import time
import numpy as np
from collections import OrderedDict
import tqdm

############################################
# Initialize
############################################
opt = TestPoseParsingOptions().parse()
train_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'is_train'}
for k, v in train_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)
# create model
model = PoseParsingModel()
model.initialize(opt)
# save terminal order line
io.save_str_list([' '.join(sys.argv)], os.path.join(model.save_dir, 'order_line.txt'))
# create data loader
val_loader = CreateDataLoader(opt, split='test')
# create visualizer
visualizer = GANVisualizer_V3(opt)

############################################
# Visualize
############################################
if opt.nvis > 0:
    print('visualizing first %d samples' % opt.nvis)
    num_vis_batch = int(np.ceil(1.0*opt.nvis/opt.batch_size))
    visuals = None
    for i, data in enumerate(val_loader):
        if i == num_vis_batch:
            break
        model.set_input(data)
        model.test(compute_loss=False)
        v = model.get_current_visuals()
        if visuals is None:
            visuals = v
        else:
            for name, item in v.iteritems():
                visuals[name][0] = torch.cat((visuals[name][0], item[0]), dim=0)
        visualizer.visualize_image(epoch=opt.which_epoch, subset='test', visuals=visuals)
if opt.vis_only:
    exit()

############################################
# Test
############################################
loss_buffer = LossBuffer(size=len(val_loader))
for i, data in enumerate(tqdm.tqdm(val_loader)):
    if opt.nbatch>0 and i == opt.nbatch:
        break
    model.set_input(data)
    model.test(compute_loss=False)
    loss_buffer.add(model.get_current_errors())
test_error = loss_buffer.get_errors()
visualizer.print_error(test_error)