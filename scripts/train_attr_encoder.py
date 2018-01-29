from __future__ import division, print_function

import torch
from models.attribute_encoder import AttributeEncoder
from data.data_loader import CreateDataLoader
from options.attribute_options import TrainAttributeOptions
from models.networks import MeanAP, ClassificationAccuracy
from misc.visualizer import AttributeVisualizer

import util.io as io
import os
import sys
import time
import numpy as np
from collections import OrderedDict


opt = TrainAttributeOptions().parse()

# create model
model = AttributeEncoder()
model.initialize(opt)

# create data loader
train_loader = CreateDataLoader(opt, split = 'train')
val_loader = CreateDataLoader(opt, split = 'test')

# create visualizer
visualizer = AttributeVisualizer(opt)

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
                pavi_outputs = {
                    'loss_attr': train_error['loss_attr'],
                }
                visualizer.pavi_log(phase = 'train', iter_num = total_steps, outputs = pavi_outputs)



    if epoch % opt.test_epoch_freq == 0:
        crit_ap = MeanAP()
        crit_cat = ClassificationAccuracy()

        _ = model.get_current_errors()# clean loss buffer

        model.eval()
        for i, data in enumerate(val_loader):
            model.set_input(data)
            model.test()
            crit_ap.add(model.output['prob'], model.input['label'])

            if opt.joint_cat:
                crit_cat.add(model.output['cat_pred'], model.input['cat_label'])

            print('\rTesting %d/%d (%.2f%%)' % (i, len(val_loader), 100.*i/len(val_loader)), end = '')
            sys.stdout.flush()
        print('\n')

        test_error = model.get_current_errors()
        mean_ap, ap_list = crit_ap.compute_mean_ap()
        mean_bp, bp_list = crit_ap.compute_balanced_precision()
        rec3_class_avg, _, rec3_overall = crit_ap.compute_recall(k=3)
        rec5_class_avg, _, rec5_overall = crit_ap.compute_recall(k=5)
        crit_ap.clear()

        test_result = OrderedDict([
            ('loss_attr', test_error['loss_attr']),
            ('mAP', mean_ap),
            ('mBP', mean_bp),
            ('rec3_class_avg', rec3_class_avg),
            ('rec5_class_avg', rec5_class_avg),
            ('rec3_overall', rec3_overall),
            ('rec5_overall', rec5_overall)
            ])

        if opt.joint_cat:
            test_result['cat_acc3'] = crit_cat.compute_accuracy(k=3)
            test_result['cat_acc5'] = crit_cat.compute_accuracy(k=5)

        visualizer.print_test_error(iter_num = total_steps, epoch = epoch, errors = test_result)

        if opt.pavi:
            pavi_outputs = {
                'loss_attr': test_error['loss_attr'],
                'mAP_upper': float(mean_ap),
                'mBP_upper': float(mean_bp),
            }
            visualizer.pavi_log(phase = 'test', iter_num = total_steps, outputs = pavi_outputs)

        # save model
        if epoch % opt.save_epoch_freq == 0:
            model.save(epoch)
            model.save('latest')
