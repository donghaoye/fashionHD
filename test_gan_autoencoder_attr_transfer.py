from __future__ import division, print_function

import util.io as io
import torch
# from models.attribute_encoder import AttributeEncoder
from models.designer_gan_model import DesignerGAN
from data.data_loader import CreateDataLoader
from options.gan_options import TestGANOptions
from models.networks import MeanAP
from misc.visualizer import GANVisualizer

import os
import sys
import time
import numpy as np
from collections import OrderedDict

# config
batch_size = 6 # after attribute transfer, samples in a batch will create bsz*bsz output iamges
num_batch = 5

opt = TestGANOptions().parse()
opt.batch_size = batch_size
# create model
model = DesignerGAN()
model.initialize(opt)
# always set model at training phase
model.netG.train()
# create data loader
val_loader_iter = iter(CreateDataLoader(opt, split = 'test'))

# create visualizer
visualizer = GANVisualizer(opt)

for i in range(num_batch):

    print('[%s] attribute transfer test: %d / %d' % (opt.id, i+1, num_batch))

    data = val_loader_iter.next()
    model.set_input(data)
    img_real = model.input['img']
    shape_code = model.encode_shape(model.input['lm_map'], model.input['seg_mask'])
    attr_code = model.encode_attribute(model.input['img'], model.input['lm_map'])

    img_title = data['img']
    imgs_generated = []
    for j in range(img_real.size(0)):
        if j == 0:
            anchor_attr_code = attr_code.clone()
        else:
            anchor_attr_code = torch.cat((attr_code[j::], attr_code[0:j]), 0)
        img_out_raw = model.netG(shape_code, anchor_attr_code)
        img_out = model.mask_image(img_out_raw, model.input['seg_map'], model.input['img'].clone())
        imgs_generated.append(img_out.data.cpu())

    imgs_generated = torch.stack(imgs_generated, 0)

    for j in range(1, img_real.size(0)):
        imgs_col = imgs_generated[:,j]
        imgs_generated[:,j] = torch.cat((imgs_col[-j::], imgs_col[0:-j]),0)

    visualizer.visualize_image_matrix(imgs = imgs_generated, imgs_title = img_title, vis_dir = 'vis_attr_trans', label = 'attr_trans_%d' % i)



