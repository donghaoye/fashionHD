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

class EncoderDecoderFramework_DFN(BaseModel):
    def name(self):
        return 'EncoderDecoderFramework_DFN'

    def initialize(self, opt):
        super(EncoderDecoderFramework_DFN, self).initialize(opt)
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
        self.guide_encoder, self.opt_guide = networks.load_encoder_v2(opt, opt.which_model_guide)
        self.guide_encoder.eval()
        for p in self.guide_encoder.parameters():
            p.requires_grad = False
        ###################################
        # DFN Modules
        ###################################
        if self.opt.use_dfn:
            self.dfn = networks.define_DFN_from_params(nf=opt.nof, ng=self.opt_guide.nof, nmid=opt.dfn_nmid, feat_size=opt.feat_size, local_size=opt.dfn_local_size, nblocks=opt.dfn_nblocks, norm=opt.norm, gpu_ids=opt.gpu_ids, init_type=opt.init_type)
        else:
            self.dfn = None
        ###################################
        # Discriminator
        ###################################
        self.use_GAN = opt.loss_weight_gan > 0
        if self.use_GAN > 0:
            if not self.opt.D_cond:
                input_nc = self.decoder.output_nc
            else:
                input_nc = self.decoder.output_nc + self.encoder.input_nc
            self.netD = networks.define_D_from_params(input_nc=input_nc, ndf=64, which_model_netD='n_layers', n_layers_D=3, norm=opt.norm, which_gan='dcgan', init_type=opt.init_type, gpu_ids=opt.gpu_ids)
        else:
            self.netD = None
        ###################################
        # loss functions
        ###################################
        self.loss_functions = []
        self.schedulers = []
        self.crit_image = nn.L1Loss()
        self.crit_seg = nn.CrossEntropyLoss()
        self.crit_edge = nn.BCELoss()
        self.loss_functions += [self.crit_image, self.crit_seg, self.crit_edge]
        if self.opt.use_dfn:
            self.optim = torch.optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}, {'params': self.dfn.parameters()}], lr=opt.lr, betas=(opt.beta1, opt.beta2))
        else:
            self.optim = torch.optim.Adam([{'params': self.encoder.parameters()}, {'params': self.decoder.parameters()}], lr=opt.lr, betas=(opt.beta1, opt.beta2))
        self.optimizers = [self.optim]
        # GAN loss and optimizers
        if self.use_GAN > 0:
            self.crit_GAN = networks.GANLoss(use_lsgan=False, tensor=self.Tensor)
            self.optim_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(0.5, 0.999))
            self.loss_functions += [self.crit_GAN]
            self.optimizers += [self.optim_D]

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
            elif input_type == 'edge':
                input = self.input['edge_map']
            elif input_type == 'shape':
                input = torch.cat((self.input['seg_mask'], self.input['pose_map']), dim=1)
        else:
            if input_type == 'image':
                input = self.input['img_def']
            elif input_type == 'seg':
                input = self.input['seg_mask_def']
            elif input_type == 'edge':
                input = self.input['edge_map_def']
            elif input_type == 'shape':
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

    def compute_loss(self, output, target, output_type):
        if output_type == 'image':
            return self.crit_image(output, target)
        elif output_type == 'seg':
            out_flat = output.transpose(1,3).contiguous().view(-1,7)
            tar_flat = target.transpose(1,3).contiguous().view(-1).long()
            return self.crit_seg(out_flat, tar_flat)
        elif output_type == 'edge':
            return self.crit_edge(output, target)
        else:
            raise NotImplementedError()

    def decode(self, feat, guide=None):
        if self.opt.decode_guide:
            assert guide is not None
            return self.decoder(torch.cat((feat, guide), dim=1))
        else:
            return self.decoder(feat)


    def forward(self):
        self.output['input'] = self.get_encoder_input(self.opt.input_type, deformation = False)
        self.output['input_def'] = self.get_encoder_input(self.opt.input_type, deformation = True)
        # encode
        feat_A = self.encoder(self.output['input'])
        feat_B = self.encoder(self.output['input_def'])
        # guide
        guide_A = self.guide_encoder(self.get_encoder_input(input_type=self.opt_guide.input_type, deformation=False))
        guide_B = self.guide_encoder(self.get_encoder_input(input_type=self.opt_guide.input_type, deformation=True))
        # DFN
        if self.opt.use_dfn:
            if self.opt.dfn_detach:
                feat_B2A, _ = self.dfn(feat_B.detach(), guide_B, guide_A)
                feat_A2B, _ = self.dfn(feat_A.detach(), guide_A, guide_B)
                feat_A2A, _ = self.dfn(feat_A2B, guide_B, guide_A)
            else:
                feat_B2A, _ = self.dfn(feat_B, guide_B, guide_A)
                feat_A2B, _ = self.dfn(feat_A, guide_A, guide_B)
                feat_A2A, _ = self.dfn(feat_A2B, guide_B, guide_A)
            # decode
            self.output['output'] = self.decode(feat_A)
            self.output['output_trans'] = self.decode(feat_B2A)
            self.output['output_cycle'] = self.decode(feat_A2A)
        else:
            # use shape guided decoder, instead of DFN module
            feat_A = F.upsample(feat_A, size=self.opt.feat_size, mode='bilinear')
            feat_B = F.upsample(feat_B, size=self.opt.feat_size, mode='bilinear')
            guide_A = F.upsample(guide_A, size=self.opt.feat_size, mode='bilinear')
            guide_B = F.upsample(guide_B, size=self.opt.feat_size, mode='bilinear')
            self.output['output'] = self.decode(feat_A, guide_A)
            self.output['output_trans'] = self.decode(feat_B, guide_A)
            self.output['output_cycle'] = self.output['output'] # a fake output_cycle. this output will not affect the loss value.
        # set target
        self.output['tar'] = self.get_decoder_target(self.opt.output_type)
        # compute loss


    def backward_D(self):
        if self.opt.D_cond:
            D_input_fake = torch.cat((self.output['output_trans'].detach(), self.output['input']), dim=1)
            D_input_real = torch.cat((self.output['tar'], self.output['input']), dim=1)
        else:
            D_input_fake = self.output['output_trans'].detach()
            D_input_real = self.output['tar']

        loss_D_fake = self.crit_GAN(self.netD(D_input_fake), False)
        loss_D_real = self.crit_GAN(self.netD(D_input_real), True)
        self.output['loss_D'] = 0.5 * (loss_D_fake + loss_D_real)
        (self.output['loss_D'] * self.opt.loss_weight_gan).backward()

    def backward(self):            
        self.output['loss_decode'] = self.compute_loss(self.output['output'], self.output['tar'], self.opt.output_type)
        self.output['loss_trans'] = self.compute_loss(self.output['output_trans'], self.output['tar'], self.opt.output_type)
        self.output['loss_cycle'] = self.compute_loss(self.output['output_cycle'], self.output['tar'], self.opt.output_type)
        self.output['loss'] = self.output['loss_decode'] * self.opt.loss_weight_decode + self.output['loss_trans'] * self.opt.loss_weight_trans + self.output['loss_cycle'] * self.opt.loss_weight_cycle
        # gan loss
        if self.use_GAN:
            if self.opt.D_cond:
                D_input = torch.cat((self.output['output_trans'], self.output['input']), dim=1)
            else:
                D_input = self.output['output_trans']
            self.output['loss_G'] = self.crit_GAN(self.netD(D_input), True)
            self.output['loss'] += self.output['loss_G'] * self.opt.loss_weight_gan
        self.output['loss'].backward()


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
        self.forward()
        if self.use_GAN:
            self.optim_D.zero_grad()
            self.backward_D()
            self.optim_D.step()

        self.optim.zero_grad()
        self.backward()
        self.optim.step()

    def get_current_errors(self):
        errors = OrderedDict([
            ('loss_decode', self.output['loss_decode'].data.item()),
            ('loss_trans', self.output['loss_trans'].data.item()),
            ('loss_cycle', self.output['loss_cycle'].data.item())
            ])
        if self.use_GAN:
            errors['loss_G'] = self.output['loss_G'].data.item()
            errors['loss_D'] = self.output['loss_D'].data.item()

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
            ('image_def', (self.input['img_def'].data.cpu(), 'rgb')),
            ('target', (self.output['tar'].data.cpu(), output_vtype)),
            ('output', (self.output['output'].data.cpu(), output_vtype)),
            ('output_trans', (self.output['output_trans'].data.cpu(), output_vtype)),
            ('output_cycle', (self.output['output_cycle'].data.cpu(), output_vtype))
            ])

        return visuals



    def save(self, label):
        self.save_network(self.encoder, 'encoder', label, self.gpu_ids)
        self.save_network(self.decoder, 'decoder', label, self.gpu_ids)
        if self.opt.use_dfn:
            self.save_network(self.dfn, 'dfn', label, self.gpu_ids)
