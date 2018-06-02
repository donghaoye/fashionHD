from __future__ import division, print_function

import torch
from data.data_loader import CreateDataLoader
from options.pose_transfer_options import TrainPoseTransferOptions
from misc.visualizer import GANVisualizer_V3
from misc.loss_buffer import LossBuffer

import util.io as io
import os
import sys
import time
import numpy as np
from collections import OrderedDict

opt = TrainPoseTransferOptions().parse()
# create model
if opt.which_model_T in {'unet', 'resnet'}:
    from models.supervised_pose_transfer_model import SupervisedPoseTransferModel
    model = SupervisedPoseTransferModel()
elif opt.which_model_T == 'vunet':
    from models.vunet_pose_transfer_model import VUnetPoseTransferModel
    model = VUnetPoseTransferModel()
elif opt.which_model_T == '2stage':
    from models.two_stage_pose_transfer_model import TwoStagePoseTransferModel
    model = TwoStagePoseTransferModel()
else:
    raise NotImplementedError()

model.initialize(opt)
# create data loader
train_loader = CreateDataLoader(opt, split = 'train')
val_loader = CreateDataLoader(opt, split = 'test')
# create visualizer
visualizer = GANVisualizer_V3(opt)

pavi_upper_list = ['PSNR', 'SSIM']
pavi_lower_list = ['loss_L1', 'loss_content', 'loss_style', 'loss_G', 'loss_D', 'loss_pose', 'loss_kl']

total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    model.update_learning_rate()
    for i, data in enumerate(train_loader):
        total_steps += 1
        model.set_input(data)
        model.forward()
        model.optimize_parameters(check_grad=(total_steps%opt.check_grad_freq==0))

        if total_steps % opt.display_freq == 0:
            train_error = model.get_current_errors()
            visualizer.print_train_error(
                iter_num = total_steps,
                epoch = epoch, 
                num_batch = len(train_loader), 
                lr = model.optimizers[0].param_groups[0]['lr'], 
                errors = train_error)
            if opt.pavi:
                visualizer.pavi_log(phase = 'train', iter_num = total_steps, outputs = train_error, upper_list = pavi_upper_list, lower_list = pavi_lower_list)

    if epoch % opt.test_epoch_freq == 0:
        _ = model.get_current_errors()

        loss_buffer = LossBuffer(size=len(val_loader))

        for i, data in enumerate(val_loader):
            model.set_input(data)
            model.test(compute_loss=True)
            loss_buffer.add(model.get_current_errors())
            print('\rTesting %d/%d (%.2f%%)' % (i, len(val_loader), 100.*i/len(val_loader)), end = '')
            sys.stdout.flush()
        print('\n')

        test_error = loss_buffer.get_errors()
        visualizer.print_test_error(iter_num = total_steps, epoch=epoch, errors = test_error)
        if opt.pavi:
            visualizer.pavi_log(phase = 'test', iter_num = total_steps, outputs = test_error, upper_list = pavi_upper_list, lower_list = pavi_lower_list)


    if epoch % opt.vis_epoch_freq == 0:
        num_vis_batch = int(np.ceil(1.0*opt.nvis/opt.batch_size))
        train_visuals = None
        val_visuals = None

        for i, data in enumerate(train_loader):
            if i == num_vis_batch:
                break
            model.set_input(data)
            # model.test(compute_loss=False)
            model.forward()
            visuals = model.get_current_visuals()
            if train_visuals is None:
                train_visuals = visuals
            else:
                for name, v in visuals.iteritems():
                    train_visuals[name][0] = torch.cat((train_visuals[name][0], v[0]),dim=0)
            
        visualizer.visualize_image(epoch = epoch, subset = 'train', visuals = train_visuals)

        
        for i, data in enumerate(val_loader):
            if i == num_vis_batch:
                break
            model.set_input(data)
            model.test(compute_loss=False)
            visuals = model.get_current_visuals()
            if val_visuals is None:
                val_visuals = visuals
            else:
                for name, v in visuals.iteritems():
                    val_visuals[name][0] = torch.cat((val_visuals[name][0], v[0]),dim=0)
            
        visualizer.visualize_image(epoch = epoch, subset = 'test', visuals = val_visuals)
    
    if epoch % opt.save_epoch_freq == 0:
        model.save(epoch)
    model.save('latest')
