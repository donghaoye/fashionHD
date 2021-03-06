from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import networks
from torch.autograd import Variable
from misc.image_pool import ImagePool
from base_model import BaseModel

import os
import sys
import numpy as np
import time
from collections import OrderedDict
import argparse
import util.io as io

from misc.visualizer import seg_to_rgb

class EncoderDecoderFramework_V2(BaseModel):
    def name(self):
        return 'EncoderDecoderFramework_V2'

    def initialize(self, opt):
        super(EncoderDecoderFramework_V2, self).initialize(opt)
        ###################################
        # define encoder
        ###################################
        self.encoder = networks.define_encoder_v2(opt)
        ###################################
        # define decoder
        ###################################
        self.decoder = networks.define_decoder_v2(opt)
        ###################################
        # guide encoder
        ###################################
        if opt.use_guide_encoder:
            self.guide_encoder = networks.load_encoder_v2(opt, opt.which_model_guide)
            self.guide_encoder.eval()
            for p in self.guide_encoder.parameters():
                p.requires_grad = False
        ###################################
        # loss functions
        ###################################
        self.loss_functions = []
        self.schedulers = []
        self.crit_image = networks.SmoothLoss(nn.L1Loss())
        self.crit_seg = networks.SmoothLoss(nn.CrossEntropyLoss())
        self.crit_edge = networks.SmoothLoss(nn.BCELoss())
        self.loss_functions += [self.crit_image, self.crit_seg, self.crit_edge]

        self.optim = torch.optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}], lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizers = [self.optim]
        for optim in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optim, opt))

    def set_input(self, data):
        input_list = [
            'img',
            'seg_map',
            'seg_mask',
            'pose_map',
            'edge_map',
            'color_map',

            'img_def',
            'seg_map_def',
            'seg_mask_def',
            'pose_map_def',
            'edge_map_def',
            'color_map_def',
        ]

        for name in input_list:
            self.input[name] = Variable(self.Tensor(data[name].size()).copy_(data[name]))

        self.input['id'] = data['id']


    def get_encoder_input(self, input_type = 'image', deformation=False):
        if not deformation:
            if input_type == 'image':
                input = self.input['img']
            elif input_type == 'seg':
                input = self.input['seg_mask']
            elif self.opt.input_type == 'edge':
                input = self.input['edge_map']
            elif self.opt.input_type == 'shape':
                input = torch.cat((self.input['seg_mask'], self.input['pose_map']), dim=1)
        else:
            if input_type == 'image':
                input = self.input['img_def']
            elif input_type == 'seg':
                input = self.input['seg_mask_def']
            elif self.opt.input_type == 'edge':
                input = self.input['edge_map_def']
            elif self.opt.input_type == 'shape':
                input = torch.cat((self.input['seg_mask_def'], self.input['pose_map_def']), dim=1)
        return input

    def get_decoder_target(self, output_type = 'image'):
        if output_type == 'image':
            output = self.input['img']
        elif output_type == 'seg':
            output = self.input['seg_map']
        elif output_type == 'edge':
            output = self.input['edge_map']

        return output

    def mask_image(self, img, seg_map, img_ref):
        if self.opt.post_mask_mode == 'none':
            return img
        elif self.opt.post_mask_mode == 'fuse_face':
            # mask = ((seg_map == 0) | (seg_map > 2)).float()
            mask = Variable(((seg_map.data == 0) | (seg_map.data > 2))).float()
            return img * mask + img_ref * (1-mask)
        elif self.opt.post_mask_mode == 'fuse_face+bg':
            mask = (seg_map>2).float()
            return img * mask + img_ref * (1-mask)
        else:
            raise ValueError('post_mask_mode invalid value: %s' % self.opt.post_mask_mode)

    def forward(self, mode='normal'):
        '''
        mode:
            - normal: only use normal input
        '''
        # set input
        self.output['input'] = self.get_encoder_input(self.opt.input_type, deformation = False)
        # encode
        feat = self.encoder(self.output['input'])
        if list(feat.size())[2:4] != [self.opt.feat_size, self.opt.feat_size]:
            feat = F.upsample(feat, size=self.opt.feat_size, mode='bilinear')
        self.output['feat'] = feat

        # guide
        if self.opt.decode_guide:
            feat_guide = self.guide_encoder(self.get_encoder_input(input_type = 'seg', deformation=False))
            if list(feat_guide.size())[2:4] != [self.opt.feat_size, self.opt.feat_size]:
                feat_guide = F.upsample(feat_guide, size=self.opt.feat_size, mode='bilinear')
            self.output['feat_guide'] = feat_guide
            self.output['feat'] = torch.cat((self.output['feat'], self.output['feat_guide']), dim=1)

        # decode
        self.output['output'] = self.decoder(self.output['feat'])
        # set target
        self.output['tar'] = self.get_decoder_target(self.opt.output_type)

        self.output['loss'] = 0
        if self.opt.output_type == 'image':
            self.output['loss_decode'] = self.crit_image(self.output['output'], self.output['tar'])
        elif self.opt.output_type == 'seg':
            out_flat = self.output['output'].transpose(1,3).contiguous().view(-1, 7)
            tar_flat = self.output['tar'].transpose(1,3).contiguous().view(-1).long()
            self.output['loss_decode'] = self.crit_seg(out_flat, tar_flat)
        elif self.opt.output_type == 'edge':
            self.output['loss_decode'] = self.crit_edge(self.output['output'], self.output['tar'])
        self.output['loss'] += self.output['loss_decode'] * self.opt.loss_weight_decode

    def test(self, mode='normal'):
        if float(torch.__version__[0:3]) >= 0.4:
            with torch.no_grad():
                self.forward(mode)
        else:
            for k,v in self.input.iteritems():
                if isinstance(v, Variable):
                    v.volatile = True
            self.forward(mode)
    
    def optimize_parameters(self):
        self.optim.zero_grad()
        self.forward()
        self.output['loss'].backward()
        self.optim.step()

    def get_current_errors(self):
        errors = OrderedDict([
            ('loss_image', self.crit_image.smooth_loss(clear=True)),
            ('loss_seg', self.crit_seg.smooth_loss(clear=True)),
            ('loss_edge', self.crit_edge.smooth_loss(clear=True))
            ])

        return errors

    def get_current_visuals(self):        
        # output visual type
        if self.opt.output_type == 'image':
            output_vtype = 'rgb'
        elif self.opt.output_type == 'seg':
            output_vtype = 'seg'
        elif self.opt.output_type == 'edge':
            output_vtype = 'edge'

        visuals = OrderedDict([
            ('image', (self.input['img'].data.cpu(), 'rgb')),
            ('pose', (self.input['pose_map'].data.cpu(), 'pose')),
            ('seg', (self.input['seg_mask'].data.cpu(), 'seg')),
            ('color', (self.input['color_map'].data.cpu(), 'color')),
            ('output', (self.output['output'].data.cpu(), output_vtype)),
            ('target', (self.output['tar'].data.cpu(), output_vtype))
            ])

        return visuals

    def save(self, label):
        self.save_network(self.encoder, 'encoder', label, self.gpu_ids)
        self.save_network(self.decoder, 'decoder', label, self.gpu_ids)
