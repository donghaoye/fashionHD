from __future__ import division, print_function

import torch
# from models.attribute_encoder import AttributeEncoder
from models.encoder_decoder_framework_DFN import EncoderDecoderFramework_DFN
from data.data_loader import CreateDataLoader
from options.encoder_decoder_options_v2 import TestEncoderDecoderOptions_V2
from misc.visualizer import GANVisualizer_V3
from models.networks import MeanAP

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
opt = TestEncoderDecoderOptions_V2().parse()
opt.batch_size = batch_size
train_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'batch_size', 'is_train'}
for k, v in train_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)

# create model
model = EncoderDecoderFramework_DFN()
model.initialize(opt)
# create data loader
val_loader_iter = iter(CreateDataLoader(opt, split='test'))
# create viusalizer
visualizer = GANVisualizer_V3(opt)

for i in range(num_batch):
    print('[%s] test edge transfer: %d / %d' % (opt.id, i+1, num_batch))
    data = val_loader_iter.next()
    imgs_title = data['img'].expand(batch_size, 3, opt.fine_size, opt.fine_size).cpu()

    for name in ['img', 'seg_map', 'seg_mask', 'pose_map', 'edge_map', 'color_map']:
        # print('%s: %s' % (name, data[name].size()))
        new_1 = torch.cat([data[name]] * batch_size, dim=0).contiguous()
        new_2 = new_1.view(batch_size, batch_size, -1).transpose(0, 1).contiguous().view(new_1.size())
        new_3 = torch.cat([data[name+'_def']] * batch_size, dim=0).contiguous()
        new_3 = new_3.view(batch_size, batch_size, -1).transpose(0, 1).contiguous().view(new_1.size())
        data[name] = new_2
        data[name+'_def'] = new_1
        # data[name+'_def'] = new_3

    model.set_input(data)
    model.test()

    result = model.output['output_trans'].cpu().contiguous()
    result = result.view(batch_size, batch_size, 1, opt.fine_size, opt.fine_size)
    result = result.expand(batch_size, batch_size, 3, opt.fine_size, opt.fine_size)

    vis_dir = 'vis_trans'
    visualizer.visualize_image_matrix(imgs=result, imgs_title_top=imgs_title, imgs_title_left=imgs_title, vis_dir=vis_dir, label='trans_%d'%i)

