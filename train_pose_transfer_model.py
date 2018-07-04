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
# save terminal order line
io.save_str_list([' '.join(sys.argv)], os.path.join(model.save_dir, 'order_line.txt'))
# create data loader
train_loader = CreateDataLoader(opt, split = 'train')
val_loader = CreateDataLoader(opt, split = 'test')
# create visualizer
visualizer = GANVisualizer_V3(opt)

pavi_upper_list = ['PSNR', 'SSIM']
pavi_lower_list = ['loss_L1', 'loss_content', 'loss_style', 'loss_G', 'loss_D', 'loss_pose', 'loss_kl']

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
        visuals = None

        for i, data in enumerate(train_loader):
            if i == num_vis_batch:
                break
            model.set_input(data)
            # model.test(compute_loss=False)
            with torch.no_grad():
                model.forward()
            v = model.get_current_visuals()
            if visuals is None:
                visuals = v
            else:
                for name, item in v.iteritems():
                    visuals[name][0] = torch.cat((visuals[name][0], item[0]),dim=0)
            
        visualizer.visualize_image(epoch = epoch, subset = 'train', visuals = visuals)

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
                    visuals[name][0] = torch.cat((visuals[name][0], item[0]),dim=0)
            
        visualizer.visualize_image(epoch = epoch, subset = 'test', visuals = visuals)
        visuals = None
    
    if epoch % opt.save_epoch_freq == 0:
        model.save(epoch)
    model.save('latest')
