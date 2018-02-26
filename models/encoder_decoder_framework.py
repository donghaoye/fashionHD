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
            self.encoder_name = 'edge_encoder'
            self.decoder_name = 'decoder'
        elif opt.use_color:
            self.encoder_type = 'color'
            self.encoder_name = 'color_encoder'
            self.decoder_name = 'decoder'
        else:
            raise ValueError('either use_edge, use_color should be set')

        # encoder
        self.encoder = networks.define_image_encoder(opt, self.encoder_type)

        # decoder
        if self.encoder_type == 'edge':
            input_nc = opt.edge_nof+opt.shape_nc if opt.decode_guided else opt.edge_nof
            self.decoder = networks.define_image_decoder_from_params(input_nc = input_nc, output_nc=1, nf = opt.edge_nf, num_ups=opt.edge_ndowns, norm=opt.norm, output_activation=None, gpu_ids=opt.gpu_ids, init_type=opt.init_type)
        elif self.encoder_type == 'color':
            input_nc = opt.color_nof+opt.shape_nc if opt.decode_guided else opt.color_nof
            self.decoder = networks.define_image_decoder_from_params(input_nc = input_nc, output_nc=3, nf = opt.color_nf, num_ups=opt.color_ndowns, norm=opt.norm, output_activation=nn.Tanh, gpu_ids=opt.gpu_ids, init_type=opt.init_type)

        if not self.is_train or (self.is_train and self.opt.continue_train):
            self.load_network(self.encoder, self.encoder_name, opt.which_opoch)
            self.load_network(self.decoder, self.decoder_name, opt.which_opoch)

        # loss functions
        self.loss_functions = []
        self.schedulers = []
        self.crit_L1 = networks.SmoothLoss(nn.L1Loss())
        self.loss_functions.append(self.crit_L1)

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
        self.input['color_map_full'] = self.Tensor(data['color_map_full'].size()).copy_(data['color_map_full'])
        self.input['id'] = data['id']

        if self.opt.affine_aug:
            self.input['seg_mask_aug'] = self.Tensor(data['seg_mask_aug'].size()).copy_(data['seg_mask_aug'])
            self.input['edge_map_aug'] = self.Tensor(data['edge_map_aug'].size()).copy_(data['edge_map_aug'])
            self.input['color_map_aug'] = self.Tensor(data['color_map_aug'].size()).copy_(data['color_map_aug'])
            self.input['lm_map_aug'] = self.Tensor(data['lm_map_aug'].size()).copy_(data['lm_map_aug'])

        # create input variable
        for k, v in self.input.iteritems():
            if isinstance(v, torch.tensor._TensorBase):
                self.input[k] = Variable(v)
    
    def encode(self, img, shape_repr, guide=None):
        if (self.encoder_type=='edge' and self.opt.edge_shape_guided) or (self.encoder_type=='color' and self.opt.color_shape_guided):
            input = torch.cat((img, shape_repr), 1)
        else:
            input = img

        if self.opt.encoder_type == 'st' and self.opt.tar_guided:
            return self.encoder(input, guide)
        else:
            return self.encoder(input)
    
    def decode(self, feat, guide):
        if self.opt.decode_guided:
            if not (guide.size(2)==feat.size(2) and guide.size(3)==feat.size(3)):
                guide = F.upsample(guide, feat.size()[2:4], mode = 'bilinear')
            input = torch.cat((feat, guide), 1)
        else:
            input = feat
        return self.decoder(input)
        

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
        return shape_repr

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

    def forward(self):
        if self.opt.affine_aug:
            lm_map = self.input['lm_map_aug']
            edge_map = self.input['edge_map_aug']
            color_map = self.input['color_map_aug']
            seg_mask = self.input['seg_mask_aug']
        else:
            lm_map = self.input['lm_map']
            edge_map = self.input['edge_map']
            color_map = self.input['color_map']
            seg_mask = self.input['seg_mask']

        if self.encoder_type == 'edge':
            input = edge_map
            self.output['tar'] = self.input['edge_map']
        elif self.encoder_type == 'color':
            input = color_map
            self.output['tar'] = self.input['color_map_full']
        
        shape_repr = self.encode_shape(lm_map, seg_mask, edge_map)
        shape_repr_tar = self.encode_shape(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'])
        
        self.output['feat'] = self.encode(input, shape_repr, shape_repr_tar)
        self.output['img_raw'] = self.decode(self.output['feat'], shape_repr_tar)
        self.output['img'] = self.mask_image(self.output['img_raw'], self.input['seg_map'], self.output['tar'])
        self.output['loss'] = self.crit_L1(self.output['img'], self.output['tar'])

    def test(self):
        if float(torch.__version__[0:3]) >= 0.4:
            with torch.no_grad():
                self.forward()
        else:
            for k,v in self.input.iteritems():
                if isinstance(v, Variable):
                    v.volatile = True
            self.forward()

    def optimize_parameters(self):
        self.optim.zero_grad()
        self.forward()
        loss = self.output['loss']
        loss.backward()
        self.optim.step()
        
    def save(self, label):
        self.save_network(self.encoder, self.encoder_name, label, self.gpu_ids)
        self.save_network(self.decoder, self.decoder_name, label, self.gpu_ids)

    def get_current_errors(self, clear=True):
        errors = OrderedDict([
            ('loss_L1', self.crit_L1.smooth_loss(clear))
            ])
        return errors

    def get_current_visuals(self):
        visuals = OrderedDict([
            ('img_real', self.input['img'].data.clone()),
            ('img_fake', self.output['img'].data.clone()),
            ('img_fake_raw', self.output['img_raw'].data.clone()),
            ('img_real_raw', self.output['tar'].data.clone())
            ])

        if self.encoder_type == 'edge':
            visuals['edge_map'] = self.input['edge_map'].data.clone()
        elif self.encoder_type == 'color':
            visuals['color_map'] = self.input['color_map'].data.clone()


        for k, v in visuals.iteritems():
            if v.size(1)==1:
                visuals[k] = v.expand(v.size(0), 3, v.size(2), v.size(3))
        return visuals


