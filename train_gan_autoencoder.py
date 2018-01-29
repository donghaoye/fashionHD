from __future__ import division, print_function

import torch
# from models.attribute_encoder import AttributeEncoder
from models.designer_gan_model import DesignerGAN
from data.data_loader import CreateDataLoader
from options.gan_options import TrainGANOptions
from models.networks import MeanAP
from misc.visualizer import GANVisualizer

import util.io as io
import os
import sys
import time
import numpy as np
from collections import OrderedDict


opt = TrainGANOptions().parse()

# create model
model = DesignerGAN()
model.initialize(opt)
# always set model at training phase
model.netG.train()
model.netD.train()
# create data loader
train_loader = CreateDataLoader(opt, split = 'train')
val_loader = CreateDataLoader(opt, split = 'test')

# create visualizer
visualizer = GANVisualizer(opt)

total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    model.update_learning_rate()
    for i, data in enumerate(train_loader):
        total_steps += 1
        model.set_input(data)
        model.optimize_parameters()

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

        train_visuals = model.get_current_visuals()
        visualizer.visualize_image(epoch = epoch, subset = 'train', visuals = train_visuals)

        val_data = iter(val_loader).next()
        model.set_input(val_data)
        model.test()
        val_visuals = model.get_current_visuals()
        visualizer.visualize_image(epoch = epoch, subset = 'test', visuals = val_visuals)
    
    if epoch % opt.save_epoch_freq == 0:
        model.save(epoch)
        model.save('latest')

