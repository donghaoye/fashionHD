from __future__ import division, print_function

import torch
from models.multimodal_designer_gan_model_v3 import MultimodalDesignerGAN_V3
from data.data_loader import CreateDataLoader
from options.multimodal_gan_options_v3 import TestMMGANOptions_V3
from misc.visualizer import GANVisualizer_V3, seg_to_rgb

import util.io as io
import os
import sys
import time
import numpy as np
from collections import OrderedDict

# config
batch_size = 49 # (1+6)edge * (1+6)color
num_batch = 20

# load options
opt = TestMMGANOptions_V3().parse()
opt.batch_size = batch_size
org_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'batch_size', 'is_train', 'dataset_mode'}
for k, v in org_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)
# create model
model = MultimodalDesignerGAN_V3()
model.initialize(opt)
for name, net in model.modules.iteritems():
    for p in net.parameters():
        p.requires_grad = False
# create data loader
val_loader_iter = iter(CreateDataLoader(opt, split='vis'))
# create visualizer
visualizer = GANVisualizer_V3(opt)

for i in range(num_batch):
    print('[%s] test generation: %d / %d' % (opt.id, i+1, num_batch))

    data = val_loader_iter.next()
    imgs_edge = data['img_edge'][0::7].cpu().clone()
    imgs_color = data['img_color'][0:7].cpu().clone()
    s_id = data['id'][0][0]

    model.set_input(data)
    model.test('normal')

    vis_dir = 'vis_gen'

    imgs_gen = model.output['img_gen']
    imgs_gen = imgs_gen.view([7,7] + list(imgs_gen.size()[1::])).cpu()
    visualizer.visualize_image_matrix(imgs=imgs_gen, imgs_title_top=imgs_color, imgs_title_left=imgs_edge, vis_dir=vis_dir, label='gen_%d_%s' % (i, s_id))

    if opt.G_output_seg:
        segs_gen = model.output['seg_pred_gen'].cpu()
        segs_gen = seg_to_rgb(segs_gen)
        segs_gen = segs_gen.view([7,7] + list(segs_gen.size()[1::])).cpu()
        visualizer.visualize_image_matrix(imgs=segs_gen, imgs_title_top=imgs_color, imgs_title_left=imgs_edge, vis_dir=vis_dir, label='seg_%d_%s' % (i, s_id))




