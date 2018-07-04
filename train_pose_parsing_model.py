from __future__ import division, print_function

import torch
from data.data_loader import CreateDataLoader
from options.pose_parsing_options import TrainPoseParsingOptions
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
opt = TrainPoseParsingOptions().parse()
# create model
model = PoseParsingModel()
model.initialize(opt)
# save terminal order line
io.save_str_list([' '.join(sys.argv)], os.path.join(model.save_dir, 'order_line.txt'))
# create data loader
train_loader = CreateDataLoader(opt, split='train')
val_loader = CreateDataLoader(opt, split='test')
# create visualizer
visualizer = GANVisualizer_V3(opt)

############################################
# Train Loop
############################################
if not opt.continue_train:
    total_steps = 0
    epoch_count = 1
else:
    epoch_count = 1 + int(opt.which_epoch)
    total_steps = len(train_loader)*int(opt.which_epoch)

for epoch in range(epoch_count, opt.niter + opt.niter_decay + 1):
    model.update_learning_rate()
    for i, data in enumerate(train_loader):
        total_steps += 1
        model.set_input(data)
        model.forward()
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            train_error = model.get_current_errors()
            visualizer.print_train_error(
                iter_num = total_steps,
                epoch = epoch, 
                num_batch = len(train_loader), 
                lr = model.optimizers[0].param_groups[0]['lr'], 
                errors = train_error)

    if epoch % opt.test_epoch_freq == 0:
        _ = model.get_current_errors()
        loss_buffer = LossBuffer(size=len(val_loader))
        for i, data in enumerate(tqdm.tqdm(val_loader)):
            model.set_input(data)
            model.test(compute_loss=True)
            loss_buffer.add(model.get_current_errors())

        test_error = loss_buffer.get_errors()
        visualizer.print_test_error(iter_num=total_steps, epoch=epoch, errors=test_error)

    if epoch % opt.vis_epoch_freq == 0:
        # visualize training set
        num_vis_batch = int(np.ceil(1.0*opt.nvis/opt.batch_size))
        visuals = None
        for i, data in enumerate(train_loader):
            if i == num_vis_batch:
                break
            model.set_input(data)
            model.test()
            v = model.get_current_visuals()
            if visuals is None:
                visuals = v
            else:
                for name, item in v.iteritems():
                    visuals[name][0] = torch.cat((visuals[name][0], item[0]), dim=0)
        visualizer.visualize_image(epoch=epoch, subset='train', visuals=visuals)

        # visualize test set
        visuals = None
        for i, data in enumerate(val_loader):
            if i == num_vis_batch:
                break
            model.set_input(data)
            model.test()
            v = model.get_current_visuals()
            if visuals is None:
                visuals = v
            else:
                for name, item in v.iteritems():
                    visuals[name][0] = torch.cat((visuals[name][0], item[0]), dim=0)
        visualizer.visualize_image(epoch=epoch, subset='test', visuals=visuals)

    if epoch % opt.save_epoch_freq == 0:
        model.save(epoch)
    model.save('latest')


