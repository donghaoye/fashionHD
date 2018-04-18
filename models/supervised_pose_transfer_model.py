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

class SupervisedPoseTransferModel(BaseModel):
    def name(self):
        return 'SupervisedPoseTransferModel'

    def initialize(self, opt):
        super(SupervisedPoseTransferModel, self).initialize(opt)
        ###################################
        # define transformer
        ###################################
        if opt.which_model_T == 'resnet':
            self.netT = networks.ResnetGenerator(
                input_nc=3+self.get_pose_dim(opt.pose_type),
                output_nc=3,
                ngf=opt.T_nf, 
                norm_layer=networks.get_norm_layer(opt.norm),
                use_dropout=not opt.no_dropout,
                n_blocks=9,
                gpu_ids=opt.gpu_ids)
        elif opt.which_model_T == 'unet':
            self.netT = networks.UnetGenerator_v2(
                input_nc=3+self.get_pose_dim(opt.pose_type),
                output_nc=3,
                num_downs=8,
                ngf=opt.T_nf,
                norm_layer=networks.get_norm_layer(opt.norm),
                use_dropout=not opt.no_dropout,
                gpu_ids=opt.gpu_ids)
        else:
            raise NotImplementedError()

        if opt.gpu_ids:
            self.netT.cuda()
        networks.init_weights(self.netT, init_type=opt.init_type)
        ###################################
        # define discriminator
        ###################################
        self.use_GAN = self.is_train and opt.loss_weight_gan > 0
        if self.use_GAN > 0:
            self.netD = networks.define_D_from_params(
                input_nc=3+self.get_pose_dim(opt.pose_type) if opt.D_cond else 3,
                ndf=opt.D_nf,
                which_model_netD='n_layers',
                n_layers_D=3,
                norm=opt.norm,
                which_gan=opt.which_gan,
                init_type=opt.init_type,
                gpu_ids=opt.gpu_ids)
        else:
            self.netD = None
        ###################################
        # loss functions
        ###################################
        if self.is_train:
            self.loss_functions = []
            self.schedulers = []
            self.optimizers =[]
            
            self.crit_L1 = nn.L1Loss()
            self.crit_vgg = networks.VGGLoss(self.gpu_ids)
            self.crit_psnr = networks.PSNR()
            self.loss_functions += [self.crit_L1, self.crit_vgg]
            self.optim = torch.optim.Adam(self.netT.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers += [self.optim]

            if self.use_GAN:
                self.crit_GAN = networks.GANLoss(use_lsgan=opt.which_gan=='lsgan', tensor=self.Tensor)
                self.optim_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
                self.loss_functions.append(self.use_GAN)
                self.optimizers.append(self.optim_D)
            # todo: add pose loss
            for optim in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optim, opt))

            self.fake_pool = ImagePool(opt.pool_size)

        ###################################
        # load trained model
        ###################################
        if not self.is_train:
            self.load_network(self.netT, 'netT', opt.which_model)

    def set_input(self, data):
        input_list = [
            'img_1',
            'pose_1',
            'seg_1',
            'seg_mask_1',

            'img_2',
            'pose_2',
            'seg_2',
            'seg_mask_2',
        ]

        for name in input_list:
            self.input[name] = Variable(self.Tensor(data[name].size()).copy_(data[name]))

        self.input['id'] = zip(data['id_1'], data['id_2'])
        # torch.save(data, 'data.pth')
        # exit(0)

    def get_pose_dim(self, pose_type):
        if pose_type == 'joint':
            dim = 18
        elif pose_type == 'joint+seg':
            dim = 18 + 7

        return dim

    def get_pose(self, pose_type, index='1'):
        assert index in {'1', '2'}
        pose = self.input['pose_%s' % index]
        seg_mask = self.input['seg_mask_%s' % index]

        if pose_type == 'joint':
            return pose
        elif pose_type == 'joint+seg':
            return torch.cat((pose, seg_mask), dim=1)

    def forward(self):
        img_ref = self.input['img_1']
        pose_tar = self.get_pose(self.opt.pose_type, index='2')
        self.output['img_out'] = self.netT(torch.cat((img_ref, pose_tar), dim=1))

        self.output['img_tar'] = self.input['img_2']
        self.output['pose_tar'] = pose_tar
        self.output['PSNR'] = self.crit_psnr(self.output['img_out'], self.output['img_tar'])

    def test(self, compute_loss=False):
        if float(torch.__version__[0:3]) >= 0.4:
            with torch.no_grad():
                self.forward()
        else:
            for k,v in self.input.iteritems():
                if isinstance(v, Variable):
                    v.volatile = True
            self.forward()

        # compute loss
        if compute_loss:
            if self.use_GAN:
                # D loss
                if self.opt.D_cond:
                    D_input_fake = torch.cat((self.output['img_out'].detach(), self.output['pose_tar']), dim=1)
                    D_input_real = torch.cat((self.output['img_tar'], self.output['pose_tar']), dim=1)
                else:
                    D_input_fake = self.output['img_out'].detach()
                    D_input_real = self.output['img_tar']
                
                D_input_fake = self.fake_pool.query(D_input_fake.data)
                loss_D_fake = self.crit_GAN(self.netD(D_input_fake), False)
                loss_D_real = self.crit_GAN(self.netD(D_input_real), True)
                self.output['loss_D'] = 0.5*(loss_D_fake + loss_D_real)
                # G loss
                if self.opt.D_cond:
                    D_input = torch.cat((self.output['img_out'], self.output['pose_tar']), dim=1)
                else:
                    D_input = self.output['img_out']
                self.output['loss_G'] = self.crit_GAN(self.netD(D_input), True)
            # L1 loss
            self.output['loss_L1'] = self.crit_L1(self.output['img_out'], self.output['img_tar'])
            # vgg loss
            self.output['loss_vgg'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'])
            

    def backward_D(self):
        if self.opt.D_cond:
            D_input_fake = torch.cat((self.output['img_out'].detach(), self.output['pose_tar']), dim=1)
            D_input_real = torch.cat((self.output['img_tar'], self.output['pose_tar']), dim=1)
        else:
            D_input_fake = self.output['img_out'].detach()
            D_input_real = self.output['img_tar']

        D_input_fake = self.fake_pool.query(D_input_fake.data)

        loss_D_fake = self.crit_GAN(self.netD(D_input_fake), False)
        loss_D_real = self.crit_GAN(self.netD(D_input_real), True)
        self.output['loss_D'] = 0.5*(loss_D_fake + loss_D_real)
        (self.output['loss_D'] * self.opt.loss_weight_gan).backward()

    def backward(self):
        loss = 0
        # L1
        self.output['loss_L1'] = self.crit_L1(self.output['img_out'], self.output['img_tar'])
        loss += self.output['loss_L1'] * self.opt.loss_weight_L1
        # VGG
        self.output['loss_vgg'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'])
        loss += self.output['loss_vgg'] * self.opt.loss_weight_vgg
        # GAN
        if self.use_GAN:
            if self.opt.D_cond:
                D_input = torch.cat((self.output['img_out'], self.output['pose_tar']), dim=1)
            else:
                D_input = self.output['img_out']
            self.output['loss_G'] = self.crit_GAN(D_input_real, True)
            loss  += self.output['loss_G'] * self.opt.loss_weight_gan

        loss.backward()

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
            ('PSNR', self.output['PSNR'].data.item()),
            ('loss_L1', self.output['loss_L1'].data.item()),
            ('loss_vgg', self.output['loss_vgg'].data.item()),
            ])
        if self.use_GAN:
            errors['loss_G'] = self.output['loss_G'].data.item()
            errors['loss_D'] = self.output['loss_D'].data.item()
        return errors

    def get_current_visuals(self):
        visuals = OrderedDict([
            ('img_ref', (self.input['img_1'].data.cpu(), 'rgb')),
            # ('poes_tar', (self.output['pose_tar'].data.cpu(), 'pose')),
            ('poes_tar', (self.input['pose_2'].data.cpu(), 'pose')),
            ('seg_tar', (self.input['seg_mask_2'].data.cpu(), 'seg')),
            ('img_tar', (self.output['img_tar'].data.cpu(), 'rgb')),
            ('img_out', (self.output['img_out'].data.cpu(), 'rgb'))
            ])
        return visuals

    def save(self, label):
        self.save_network(self.netT, 'netT', label, self.gpu_ids)
        if self.use_GAN:
            self.save_network(self.netD, 'netD', label, self.gpu_ids)