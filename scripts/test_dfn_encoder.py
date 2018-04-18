from __future__ import division, print_function

import torch
import torch.nn.functional as F
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

    # visualize coef
    local_size = opt.dfn_local_size
    half_size = (local_size-1)//2
    feat_size = opt.feat_size

    coef = model.output['coef_trans'].data.cpu()
    for j in range(batch_size):
        img = model.input['img'][j]
        img_def = model.input['img_def'][j]
        cmap_tar = torch.zeros(feat_size, feat_size, feat_size, feat_size)
        cmap_src_pad = torch.zeros(feat_size, feat_size, feat_size+local_size-1, feat_size+local_size-1)
        for y in range(feat_size):
            for x in range(feat_size):
                cmap_tar[y,x,y,x] = 1
                cmap_src_pad[y,x,y:(y+local_size), x:(x+local_size)] = coef[j,:,y,x].contiguous().view(local_size, local_size)

        cmap_src = cmap_src_pad[:,:,half_size:(-half_size), half_size:(-half_size)]

        cmap_tar = cmap_tar.unsqueeze(dim=2).expand(feat_size, feat_size, 3, feat_size, feat_size)
        cmap_src = cmap_src.unsqueeze(dim=2).expand(feat_size, feat_size, 3, feat_size, feat_size)
        
        cmap_mix = torch.cat((cmap_tar, cmap_src), dim=4)
        cmap_mix = cmap_mix.view(feat_size*feat_size, 3, feat_size, feat_size*2)
        cmap_mix = F.upsample(cmap_mix, size=(opt.fine_size, opt.fine_size*2), mode='bilinear')
        cmap_mix = cmap_mix.view(feat_size, feat_size, 3, opt.fine_size, opt.fine_size*2)

        img_mix = torch.cat((model.input['img'][j], model.input['img_def'][j]), dim=2).data.cpu()
        img_title = torch.stack([img_mix] * feat_size, dim=0)
        cmap_mix = cmap_mix * (img_mix.view(1, 1, 3, opt.fine_size, opt.fine_size*2))
        # img_tar = torch.stack([model.input['img'][j]]*feat_size, dim=0).data.cpu()
        # img_src = torch.stack([model.input['img_def'][j]]*feat_size, dim=0).data.cpu()
        # img_tar = torch.cat((img_tar, img_tar), dim=3)
        # img_src = torch.cat((img_src, img_src), dim=3)
        visualizer.visualize_image_matrix(imgs=cmap_mix, imgs_title_top=img_title, imgs_title_left=img_title, vis_dir=vis_dir, label='coef_%d_%d-%d'%(i, j%batch_size, j//batch_size))


