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
        if opt.use_shape:
            self.encoder_type = 'shape'
            self.encoder_name = 'shape_encoder'
            self.decoder_name = 'decoder'
        elif opt.use_edge:
            self.encoder_type = 'edge'
            self.encoder_name = 'edge_encoder'
            self.decoder_name = 'decoder'
        elif opt.use_color:
            self.encoder_type = 'color'
            self.encoder_name = 'color_encoder'
            self.decoder_name = 'decoder'
        else:
            raise ValueError('either use_shape, use_edge, use_color should be set')

        # encoder
        self.encoder = networks.define_image_encoder(opt, self.encoder_type)

        # decoder
        if self.encoder_type == 'shape':
            ndowns = opt.shape_ndowns
            nf = opt.shape_nf
            nof = opt.shape_nof
            output_nc = 7
            output_activation = None
            assert opt.decode_guided == False
        elif self.encoder_type == 'edge':
            ndowns = opt.edge_ndowns
            nf = opt.edge_nf
            nof = opt.edge_nof
            output_nc = 1
            output_activation = None
        elif self.encoder_type == 'color':
            ndowns = opt.color_ndowns
            nf = opt.color_nf
            nof = opt.color_nof
            output_nc = 3
            output_activation = nn.Tanh

        if opt.encoder_type in {'normal', 'st'}:
                self.feat_size = 256 // 2**(opt.edge_ndowns)
                self.mid_feat_size = self.feat_size
        else:
            self.feat_size = 1
            self.mid_feat_size = 8

        self.use_concat_net = False
        if opt.decode_guided:
            if self.feat_size > 1:
                self.decoder = networks.define_image_decoder_from_params(input_nc=nof+opt.shape_nc, output_nc=output_nc, nf=nf, num_ups=ndowns, norm=opt.norm, output_activation=output_activation, gpu_ids=opt.gpu_ids, init_type=opt.init_type)
            else:
                self.decoder = networks.define_image_decoder_from_params(input_nc=nof, output_nc=output_nc, nf=nf, num_ups=5, norm=opt.norm, output_activation=output_activation, gpu_ids=opt.gpu_ids, init_type=opt.init_type)
                self.concat_net = networks.FeatureConcatNetwork(feat_nc=nof, guide_nc=opt.shape_nc, nblocks=3, norm=opt.norm, gpu_ids=opt.gpu_ids)
                if len(self.gpu_ids) > 0:
                    self.concat_net.cuda()
                networks.init_weights(self.concat_net, opt.init_type)
                self.use_concat_net = True
                print('encoder_decoder contains a feature_concat_network!')
        else:
            if self.feat_size > 1:
                self.decoder = networks.define_image_decoder_from_params(input_nc=nof, output_nc=output_nc, nf=nf, num_ups=ndowns, norm=opt.norm, output_activation=output_activation, gpu_ids=opt.gpu_ids, init_type=opt.init_type)
            else:
                self.decoder = networks.define_image_decoder_from_params(input_nc=nof, output_nc=output_nc, nf=nf, num_ups=8, norm=opt.norm, output_activation=output_activation, gpu_ids=opt.gpu_ids, init_type=opt.init_type)

        if not self.is_train or (self.is_train and self.opt.continue_train):
            self.load_network(self.encoder, self.encoder_name, opt.which_opoch)
            self.load_network(self.decoder, self.decoder_name, opt.which_opoch)
            if self.use_concat_net:
                self.load_network(self.concat_net, 'concat_net', opt.which_opoch)

        # loss functions
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
        if not (guide.size(2)==self.mid_feat_size and guide.size(3) == mid_feat_size):
            guide = F.upsample(guide, (self.mid_feat_size, self.mid_feat_size), mode='bilinear')

        if self.opt.decode_guided:
            if self.use_concat_net:
                feat = self.concat_net(feat, guide)
            else:
                feat = torch.cat((feat, guide), dim=1)
        return self.decoder(feat)

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

        shape_repr = self.encode_shape(lm_map, seg_mask, edge_map)
        shape_repr_tar = self.encode_shape(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'])

        if self.encoder_type == 'edge':
            input = edge_map
            self.output['tar'] = self.input['edge_map']
        elif self.encoder_type == 'color':
            input = color_map
            self.output['tar'] = self.input['color_map_full']
        elif self.encoder_type == 'shape':
            input = shape_repr
            self.output['tar'] = self.input['seg_map']
        
        self.output['feat'] = self.encode(input, shape_repr, shape_repr_tar)
        self.output['img_raw'] = self.decode(self.output['feat'], shape_repr_tar)

        if self.encoder_type == 'shape':
            self.output['img'] = self.output['img_raw']
            img_flatten = self.output['img'].transpose(1,3).contiguous().view(-1, 7)
            tar_flatten = self.output['tar'].transpose(1,3).contiguous().view(-1).long()
            self.output['loss'] = self.crit_CE(img_flatten, tar_flatten)

        else:
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
        if self.use_concat_net:
            self.save_network(self.concat_net, 'concat_net', label, self.gpu_ids)

    def get_current_errors(self, clear=True):
        errors = OrderedDict([
            ('loss_L1', self.crit_L1.smooth_loss(clear)),
            ('loss_CE', self.crit_CE.smooth_loss(clear))
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
        elif self.encoder_type == 'shape':
            visuals['img_fake'] = networks.seg_to_rgb(self.output['img'].data.clone())
            visuals['img_fake_raw'] = networks.seg_to_rgb(self.output['img_raw'].data.clone())
            visuals['img_real_raw'] = networks.seg_to_rgb(self.output['tar'].data.clone())

        for k, v in visuals.iteritems():
            v = v.cpu()
            if v.size(1)==1:
                v = v.expand(v.size(0), 3, v.size(2), v.size(3))
            visuals[k] = v
        return visuals



