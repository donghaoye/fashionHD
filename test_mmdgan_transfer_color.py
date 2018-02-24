from __future__ import division, print_function

import torch
# from models.attribute_encoder import AttributeEncoder
from models.multimodal_designer_gan_model import MultimodalDesignerGAN
from data.data_loader import CreateDataLoader
from options.multimodal_gan_options import TestMMGANOptions
from models.networks import MeanAP
from misc.visualizer import GANVisualizer
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
opt = TestMMGANOptions().parse()
opt.batch_size = batch_size
train_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'batch_size', 'is_train'}
for k, v in train_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)
# create model
model = MultimodalDesignerGAN()
model.initialize(opt)
# create data loader
val_loader_iter = iter(CreateDataLoader(opt, split = 'test'))
# create visualizer
visualizer = GANVisualizer(opt)


for i in range(num_batch):
    print('[%s] test attribute transfer: %d / %d' % (opt.id, i+1, num_batch))

    data = val_loader_iter.next()
    imgs_title = data['img'].clone()
    imgs_generated = []

    if opt.affine_aug:
        color_map = data['color_map_aug']
    else:
        color_map = data['color_map']


    for j in range(batch_size):
        if opt.affine_aug:
            if j == 0:
                data['color_map_aug'] = color_map.clone()
            else:
                data['color_map_aug'] = torch.cat((color_map[j::], color_map[0:j]), 0)
        else:
            if j == 0:
                data['color_map'] = color_map.clone()
            else:
                data['color_map'] = torch.cat((color_map[j::], color_map[0:j]), 0)
        
        model.set_input(data)
        model.test()
        imgs_generated.append(model.output['img_fake'].data.cpu().clone())
    
    imgs_generated = torch.stack(imgs_generated, 0)
    for j in range(1, batch_size):
        img_col = imgs_generated[:,j]
        imgs_generated[:,j] = torch.cat((img_col[-j::], img_col[0:-j]), 0)
    
    vis_dir = 'vis_trans_color'
    visualizer.visualize_image_matrix(imgs=imgs_generated, imgs_title=imgs_title, vis_dir=vis_dir, label='trans_color_%d' % i)
