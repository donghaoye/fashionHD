from __future__ import division, print_function

import torch
from models.feature_transform_model import FeatureSpatialTransformer
from data.data_loader import CreateDataLoader
from options.feature_spatial_transformer_options import TrainFeatureSpatialTransformerOptions
from misc.visualizer import BaseVisualizer

import util.io as io
import os
import sys
import time
import numpy as np
from collections import OrderedDict


opt = TrainFeatureSpatialTransformerOptions().parse()

# create model
model = FeatureSpatialTransformer()
model.initialize(opt)
# create data loader
train_loader = CreateDataLoader(opt, split = 'train')
val_loader = CreateDataLoader(opt, split = 'test')
# create visualizer
visualizer = BaseVisualizer(opt)

total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    model.update_learning_rate()
    model.train()
    
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
    
    if epoch % opt.test_epoch_freq == 0:
        _ = model.get_current_errors()# clean loss buffer

        model.eval()
        for i, data in enumerate(val_loader):
            model.set_input(data)
            model.test()
            print('\rTesting %d/%d (%.2f%%)' % (i, len(val_loader), 100.*i/len(val_loader)), end = '')
            sys.stdout.flush()
        print('\n')

        test_error = model.get_current_errors()
        visualizer.print_test_error(iter_num = total_steps, epoch = epoch, errors = test_error)
        if opt.pavi:
            visualizer.pavi_log(phase = 'test', iter_num = total_steps, outputs = test_error)

    # save model
    if epoch % opt.save_epoch_freq == 0:
        model.save(epoch)
    model.save('latest')
        

