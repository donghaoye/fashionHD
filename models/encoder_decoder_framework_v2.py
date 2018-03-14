from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import networks
from torch.autograd import Variable
from misc.image_pool import ImagePool
from base_model import BaseModel
import network_loader

import os
import sys
import numpy as np
import time
from collections import OrderedDict
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
        # Todo

        ###################################
        # loss functions
        ###################################
        self.loss_functions = []
        self.schedulers = []
        self.crit_L1 = networks.SmoothLoss(nn.L1Loss())
        self.crit_CE = networks.SmoothLoss(nn.CrossEntropyLoss())
        self.loss_functions += [self.crit_L1, self.crit_CE]

        self.optim = torch.optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}], lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizers = [self.optim]
        for optim in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optim, opt))

    def set_input(self, data):
        self.input['img'] = self.Tensor(data['img'].size()).copy_(data['img'])
        self.input['seg_mask'] = self.Tensor(data['seg_mask'].size()).copy_(data['seg_mask'])
        self.input['seg_map'] = self.Tensor(data['seg_map'].size()).copy_(data['seg_map'])
        self.input['edge_map'] = self.Tensor(data['edge_map'].size()).copy_(data['edge_map'])
        self.input['color_map'] = self.Tensor(data['color_map'].size()).copy_(data['color_map'])
        self.input['id'] = data['id']

        self.input['img'] = self.Tensor(data['img'].size()).copy_(data['img_aug'])
        self.input['seg_mask_aug'] = self.Tensor(data['seg_mask_aug'].size()).copy_(data['seg_mask_aug'])
        self.input['edge_map_aug'] = self.Tensor(data['edge_map_aug'].size()).copy_(data['edge_map_aug'])
        self.input['color_map_aug'] = self.Tensor(data['color_map_aug'].size()).copy_(data['color_map_aug'])

        # create input variable
        for k, v in self.input.iteritems():
            if isinstance(v, torch.Tensor):
                self.input[k] = Variable(v)


    def get_encoder_input(self, input_type = 'image', aug=False):
        if not aug:
            if input_type == 'image':
                input = self.input['img']
            elif input_type == 'seg':
                input = self.input['seg_mask']
            elif self.opt.input_type == 'edge':
                input = self.input['edge_map']
        else:
            if input_type == 'image':
                input = self.input['img_aug']
            elif input_type == 'seg':
                input = self.input['seg_mask_aug']
            elif self.opt.input_type == 'edge':
                input = self.input['edge_map_aug']

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
            - aug: use normal & augment(geometircally transformed) input, and enforce the coincidence
        '''
        # set input
        self.output['input'] = self.get_encoder_input(self.opt.input_type, aug = False)
        # encode
        feat = self.encoder(self.output['input'])
        if list(feat.size())[2:4] != [self.opt.feat_size, self.opt.feat_size]:
            feat = F.upsample(feat, size=self.opt.feat_size, mode='bilinear')
        self.output['feat'] = feat
        # decode
        self.output['output'] = self.decoder(self.output['feat'])
        # set target
        self.output['tar'] = self.get_decoder_target(self.opt.output_type)

        self.output['loss'] = 0
        if self.opt.output_type == 'seg':
            out_flat = self.output['output'].transpose(1,3).contiguous().view(-1, 7)
            tar_flat = self.output['tar'].transpose(1,3).contiguous().view(-1).long()
            self.output['loss_decode'] = self.crit_CE(out_flat, tar_flat)
        else:
            self.output['loss_decode'] = self.crit_L1(self.output['output'], self.output['tar'])
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
            ('loss_L1', self.crit_L1.smooth_loss(clear=True)),
            ('loss_CE', self.crit_CE.smooth_loss(clear=True))
            ])

        return errors

    def get_current_visuals(self):
        # input visual type
        if self.opt.input_type == 'image':
            input_vtype = 'rgb'
        elif self.opt.input_type == 'seg':
            input_vtype = 'seg'
        elif self.opt.input_type == 'edge':
            input_vtype = 'edge'
        # output visual type
        if self.opt.output_type == 'image':
            output_vtype = 'rgb'
        elif self.opt.output_type == 'seg':
            output_vtype = 'seg'
        elif self.opt.output_type == 'edge':
            output_vtype = 'edge'

        visuals = OrderedDict([
            ('image', (self.input['img'].data.cpu(), 'rgb')),
            ('input', (self.output['input'].data.cpu(), input_vtype)),
            ('output', (self.output['output'].data.cpu(), output_vtype)),
            ('target', (self.output['tar'].data.cpu(), output_vtype))
            ])

        return visuals

    def save(self, label):
        self.save_network(self.encoder, 'encoder', label, self.gpu_ids)
        self.save_network(self.decoder, 'decoder', label, self.gpu_ids)
