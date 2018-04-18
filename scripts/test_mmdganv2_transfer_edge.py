from __future__ import division, print_function

import torch
# from models.attribute_encoder import AttributeEncoder
from models.multimodal_designer_gan_model_v2 import MultimodalDesignerGAN_V2
from data.data_loader import CreateDataLoader
from options.multimodal_gan_options_v2 import TestMMGANOptions_v2
from models.networks import MeanAP
from misc.visualizer import GANVisualizer_V2
from models import network_loader

import util.io as io
import os
import sys
import time
import numpy as np
from collections import OrderedDict

# config
batch_size = 6
num_batch = 5

# load options
opt = TestMMGANOptions_v2().parse()
opt.batch_size = batch_size
train_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'batch_size', 'is_train'}
for k, v in train_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)
# create model
assert opt.is_train == False
model = MultimodalDesignerGAN_V2()
model.initialize(opt)
# create data loader
val_loader_iter = iter(CreateDataLoader(opt, split = 'test'))
# create visualizer
visualizer = GANVisualizer_V2(opt)

use_ftn = model.opt.ftn_model != 'none'

for i in range(num_batch):
    print('[%s] test edge transfer: %d / %d' % (opt.id, i+1, num_batch))

    data = val_loader_iter.next()
    imgs_title = data['img'].clone()
    imgs_generated = []
    imgs_generated_trans = []

    edge_map = data['edge_map']


    for j in range(batch_size):
        if j == 0:
            data['edge_map'] = edge_map.clone()
        else:
            data['edge_map'] = torch.cat((edge_map[j::], edge_map[0:j]), 0)
        
        model.set_input(data)
        model.test(mode='dual' if use_ftn else 'normal')
        imgs_generated.append(model.output['img_fake'].data.cpu().clone())
        if use_ftn:
            imgs_generated_trans.append(model.output['img_fake_trans'].data.cpu().clone())
    
    
    imgs_generated = torch.stack(imgs_generated, 0)
    if use_ftn:
        imgs_generated_trans = torch.stack(imgs_generated_trans, 0)
    for j in range(1, batch_size):
        img_col = imgs_generated[:,j]
        imgs_generated[:,j] = torch.cat((img_col[-j::], img_col[0:-j]), 0)
        if use_ftn:
            img_col = imgs_generated_trans[:,j]
            imgs_generated_trans[:,j] = torch.cat((img_col[-j::], img_col[0:-j]), 0)

    
    vis_dir = 'vis_trans_edge'
    visualizer.visualize_image_matrix(imgs=imgs_generated, imgs_title=imgs_title, vis_dir=vis_dir, label='trans_edge_%d' % i)
    if use_ftn:
        visualizer.visualize_image_matrix(imgs=imgs_generated_trans, imgs_title=imgs_title, vis_dir=vis_dir, label='trans_edge_ftn_%d' % i)
