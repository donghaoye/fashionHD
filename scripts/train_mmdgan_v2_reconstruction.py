from __future__ import division, print_function

import torch
# from models.attribute_encoder import AttributeEncoder
from models.multimodal_designer_gan_model_v2 import MultimodalDesignerGAN_V2
from data.data_loader import CreateDataLoader
from options.multimodal_gan_options_v2 import TrainMMGANOptions_V2
from models.networks import MeanAP
from misc.visualizer import GANVisualizer_V2

import util.io as io
import os
import sys
import time
import numpy as np
from collections import OrderedDict

opt = TrainMMGANOptions_V2().parse()

# create model
model = MultimodalDesignerGAN_V2()
model.initialize(opt)
# always set model at training phase
model.netG.train()
model.netD.train()
# create data loader
train_loader = CreateDataLoader(opt, split = 'train')
val_loader = CreateDataLoader(opt, split = 'test')

# create visualizer
visualizer = GANVisualizer_V2(opt)

total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    model.update_learning_rate()
    for i, data in enumerate(train_loader):
        total_steps += 1
        model.set_input(data)

        if total_steps <= opt.D_pretrain:
            train_D = True
            train_G = False
        else:
            train_D = (total_steps % opt.D_train_freq == 0)
            train_G = (total_steps % opt.G_train_freq == 0)

        check_grad = (total_steps % opt.check_grad_freq == 0) and total_steps > opt.D_pretrain

        model.optimize_parameters(train_D = train_D, train_G = train_G, check_grad = check_grad)

        if total_steps % opt.display_freq == 0:
            train_error = model.get_current_errors()
            
            visualizer.print_train_error(
                iter_num = total_steps,
                epoch = epoch, 
                num_batch = len(train_loader), 
                lr = model.optimizers[0].param_groups[0]['lr'], 
                errors = train_error)

            if opt.pavi:
                visualizer.pavi_log(phase = 'train', iter_num = total_steps, outputs = train_error)

    if epoch % opt.vis_epoch_freq == 0:
        # visualize training samples
        train_visuals = model.get_current_visuals()
        visualizer.visualize_image(epoch = epoch, subset = 'train', visuals = train_visuals)
        # visualize test samples
        val_data = iter(val_loader).next()
        model.set_input(val_data)
        fwd_mode = 'normal' if model.opt.ftn_model == 'none' else 'dual'
        model.test(fwd_mode)
        val_visuals = model.get_current_visuals()
        visualizer.visualize_image(epoch = epoch, subset = 'test', visuals = val_visuals)
    
    if epoch % opt.save_epoch_freq == 0:
        model.save(epoch)
    model.save('latest')

