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

class EncoderDecoderFramework(BaseModel):
    def name(self):
        return 'EncoderDecoderFramework'

    def initialize(self, opt):
        super(EncoderDecoderFramework, self).initialize(opt)
        ###################################
        # load/define networks
        ###################################

        if opt.use_edge:
            self.encoder_type = 'edge'
        elif opt.use_color:
            self.encoder_type = 'color'
        else:
            raise ValueError('either use_edge, use_color should be set')

        # encoder
        self.encoder = networks.define_image_encoder(opt, self.encoder_type)

        # decoder
        if self.encoder_type == 'edge':
            self.decoder = networks.define_image_decoder_from_params(input_nc = opt.edge_nof + opt.shape_nc, num_ups=opt.edge_ndowns, norm=opt.norm, output_activation=None, gpu_ids=opt.gpu_ids, init_type=opt.init_type)
        elif self.encoder_type == 'color':
            self.decoder = networks.define_image_decoder_from_params(input_nc = opt.color_nof + opt.shape_nc, num_ups=opt.color_ndowns, norm=opt.norm, output_activation=nn.Tanh, gpu_ids=opt.gpu_ids, init_type=opt.init_type)

        if not self.is_train or (self.is_train and self.continue_train):
            self.load_network(self.encoder, 'E', opt.which_opoch)
            self.load_network(self.decoder, 'D', opt.which_opoch)

        # loss functions
        self.crit_L1 = networks.SmoothLoss(nn.L1Loss())
        self.loss_functions = [self.crit_L1]

        self.optim = torch.optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}], lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizers = [self.optim]
        for optim in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optim, opt))

    def set_input(self, data):
        self.input['img'] = self.Tensor(data['img'].size()).copy_(data['img'])
        self.input['attr_label'] = self.Tensor(data['attr_label'].size()).copy_(data['attr_label'])
        self.input['lm_map'] = self.Tensor(data['lm_map'].size()).copy_(data['lm_map'])
        self.input['seg_mask'] = self.Tensor(data['seg_mask'].size()).copy_(data['seg_mask'])
        self.input['seg_map'] = self.Tensor(data['seg_map'].size()).copy_(data['seg_map'])
        self.input['edge_map'] = self.Tensor(data['edge_map'].size()).copy_(data['edge_map'])
        self.input['color_map'] = self.Tensor(data['color_map'].size()).copy_(data['color_map'])
        self.input['id'] = data['id']

        if self.opt.affine_aug:
            self.input['seg_mask_aug'] = self.Tensor(data['seg_mask_aug'].size()).copy_(data['seg_mask_aug'])
            self.input['edge_map_aug'] = self.Tensor(data['edge_map_aug'].size()).copy_(data['edge_map_aug'])
            self.input['color_map_aug'] = self.Tensor(data['color_map_aug'].size()).copy_(data['color_map_aug'])
            self.input['lm_map_aug'] = self.Tensor(data['lm_map_aug'].size()).copy_(data['lm_map_aug'])

        # create input variables
        for k, v in self.input.iteritems():
            if isinstance(v, torch.tensor._TensorBase):
                self.input[k] = Variable(v)

    def encode_edge(self, img, shape_repr, guide=None):
        if self.opt.edge_shape_guided:
            input = torch.cat((img, shape_repr), 1)
        else:
            input = img
        
        if self.opt.encoder_type == 'st':
            return self.edge_encoder(input, guide)
        else:
            return self.edge_encoder(input)
    
    def encode_color(self, img, shape_repr, guide=None):
        if self.opt.color_shape_guided:
            input = torch.cat((img, shape_repr), 1)
        else:
            input = img

        if self.opt.encoder_type == 'st':
            return self.color_encoder(input, guide)
        else:
            return self.color_encoder(input)

    def encode_shape(self, lm_map, seg_mask, edge_map):
        if self.opt.shape_encode == 'lm':
            shape_repr = lm_map
        elif self.opt.shape_encode == 'seg':
            shape_repr = seg_mask
        elif self.opt.shape_encode == 'lm+seg':
            shape_repr = torch.cat((lm_map, seg_mask), dim = 1)
        elif self.opt.shape_encode == 'seg+e':
            shape_repr = torch.cat((seg_mask, edge_map), dim = 1)
        elif self.opt.shape_encode == 'lm+seg+e':
            shape_repr = torch.cat((lm_map, seg_mask, edge_map), dim = 1)
        elif self.opt.shape_encode == 'e':
            shape_repr = edge_map


    def forward(self):
        pass

    def test(self):
        if float(torch.__version__[0:3]) >= 0.4:
            with torch.no_grad():
                self.forward()
        else:
            for k,v in self.input.iteritems():
                if isinstance(v, Variable):
                    v.volatile = True
            self.forward()

    def optimize_parameters():
        pass
        
    def save(self, label):
        self.save_network(self.encoder, 'E', label, self.gpu_ids)
        self.save_network(self.decoder, 'D', label, self.gpu_ids)


