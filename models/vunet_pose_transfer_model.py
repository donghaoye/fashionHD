from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import networks
from torch.autograd import Variable
from misc.image_pool import ImagePool
from base_model import BaseModel
from misc import pose_util

import os
import sys
import numpy as np
import time
from collections import OrderedDict
import argparse
import util.io as io

class VUnetPoseTransferModel(BaseModel):
    def name(self):
        return 'VUnetPoseTransferModel'

    def initialize(self, opt):
        super(VUnetPoseTransferModel, self).initialize(opt)
        ###################################
        # define transformer
        ###################################
        self.netT = networks.VariationalUnet(
            input_nc_dec = self.get_pose_dim(opt.pose_type),
            input_nc_enc = 3,
            output_nc = 3,
            nf = opt.vunet_nf,
            max_nf = opt.vunet_max_nf,
            input_size = opt.fine_size,
            n_latent_scales = opt.vunet_n_latent_scales,
            bottleneck_factor = opt.vunet_bottleneck_factor,
            box_factor = opt.vunet_box_factor,
            n_residual_blocks = 2,
            norm_layer = networks.get_norm_layer(opt.norm),
            activation = nn.ReLU(False),
            use_dropout = False,
            gpu_ids = opt.gpu_ids,
            )
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
        self.crit_psnr = networks.PSNR()
        self.crit_ssim = networks.SSIM()

        if self.is_train:
            self.loss_functions = []
            self.schedulers = []
            self.optimizers =[]
            self.crit_L1 = nn.L1Loss()
            self.crit_vgg = networks.VGGLoss_v2(self.gpu_ids)
            # self.crit_vgg_old = networks.VGGLoss(self.gpu_ids)
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
            self.load_network(self.netT, 'netT', opt.which_epoch)

    def set_input(self, data):
        input_list = [
            'img_1',
            'joint_1',
            'stickman_1',
            'seg_1',
            'seg_mask_1',

            'img_2',
            'joint_2',
            'stickman_2',
            'seg_2',
            'seg_mask_2',
        ]
        for name in input_list:
            self.input[name] = self.Tensor(data[name].size()).copy_(data[name])

        self.input['id'] = zip(data['id_1'], data['id_2'])
        self.input['joint_c_1'] = data['joint_c_1']
        self.input['joint_c_2'] = data['joint_c_2']

    def compute_kl_loss(self, ps, qs):
        assert len(ps) == len(qs)
        kl_loss = 0
        for p, q in zip(ps, qs):
            kl_loss += self.netT.latent_kl(p, q)
        return kl_loss

    def forward(self, mode='train'):
        ''' mode in {'train', 'transfer'} '''
        if self.opt.supervised or mode == 'transfer':
            img_ref = self.input['img_1']
            pose_ref = self.get_pose(self.opt.pose_type, index='1')
            img_tar = self.input['img_2']
            pose_tar = self.get_pose(self.opt.pose_type, index='2')
            # for visualization
            self.output['joint_tar'] = self.input['joint_2']
            self.output['stickman_tar'] = self.input['stickman_2']
        else:
            img_ref = img_tar = self.input['img_1']
            pose_ref = pose_tar = self.get_pose(self.opt.pose_type, index='1')
            # for visualization
            self.output['joint_tar'] = self.input['joint_1']
            self.output['stickman_tar'] = self.input['stickman_1']

        self.output['img_out'], self.output['ps'], self.output['qs'] = self.netT(img_ref, pose_ref, pose_tar, mode)
        self.output['img_tar'] = img_tar
        self.output['pose_tar'] = pose_tar
        self.output['PSNR'] = self.crit_psnr(self.output['img_out'], self.output['img_tar'])
        self.output['SSIM'] = Variable(self.Tensor(1).fill_(0)) # to save time, do not compute ssim during training

    def test(self, compute_loss=False):
        with torch.no_grad():
            self.forward(mode='transfer')
        # compute ssim
        self.output['SSIM'] = self.crit_ssim(self.output['img_out'], self.output['img_tar'])
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
            # KL
            self.output['loss_kl'] = self.compute_kl_loss(self.output['ps'], self.output['qs'])
            # L1
            self.output['loss_L1'] = self.crit_L1(self.output['img_out'], self.output['img_tar'])
            # content
            self.output['loss_content'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], 'content')
            # style
            if self.opt.loss_weight_style > 0:
                self.output['loss_style'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], 'style')

            # patch style
            if self.opt.loss_weight_patch_style > 0:
                self.output['loss_patch_style'] = self.compute_patch_style_loss(self.output['img_out'], self.input['joint_c_2'], self.output['img_tar'], self.input['joint_c_2'], self.opt.patch_size)
        

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
        # KL
        self.output['loss_kl'] = self.compute_kl_loss(self.output['ps'], self.output['qs'])
        loss += self.output['loss_kl'] * self.opt.loss_weight_kl
        # L1
        self.output['loss_L1'] = self.crit_L1(self.output['img_out'], self.output['img_tar'])
        loss += self.output['loss_L1'] * self.opt.loss_weight_L1
        # content
        self.output['loss_content'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], 'content')
        loss += self.output['loss_content'] * self.opt.loss_weight_content
        # style
        if self.opt.loss_weight_style > 0:
            self.output['loss_style'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], 'style')
            loss += self.output['loss_style'] * self.opt.loss_weight_style
        # patch style
        if self.opt.loss_weight_patch_style > 0:
            self.output['loss_patch_style'] = self.compute_patch_style_loss(self.output['img_out'], self.input['joint_c_2'], self.output['img_tar'], self.input['joint_c_2'], self.opt.patch_size)
            loss += self.output['loss_patch_style'] * self.opt.loss_weight_patch_style
        # GAN
        if self.use_GAN:
            if self.opt.D_cond:
                D_input = torch.cat((self.output['img_out'], self.output['pose_tar']), dim=1)
            else:
                D_input = self.output['img_out']
            self.output['loss_G'] = self.crit_GAN(self.netD(D_input), True)
            loss  += self.output['loss_G'] * self.opt.loss_weight_gan
        loss.backward()

    def backward_checkgrad(self):
        self.output['img_out'].retain_grad()
        loss = 0
        # L1
        self.output['loss_L1'] = self.crit_L1(self.output['img_out'], self.output['img_tar'])
        (self.output['loss_L1'] * self.opt.loss_weight_L1).backward(retain_graph=True)
        self.output['grad_L1'] = self.output['img_out'].grad.norm()
        grad = self.output['img_out'].grad.clone()
        # content 
        self.output['loss_content'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], 'content')
        (self.output['loss_content'] * self.opt.loss_weight_content).backward(retain_graph=True)
        self.output['grad_content'] = (self.output['img_out'].grad - grad).norm()
        grad = self.output['img_out'].grad.clone()
        # style
        if self.opt.loss_weight_style > 0:
            self.output['loss_style'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], 'style')
            (self.output['loss_style'] * self.opt.loss_weight_style).backward(retain_graph=True)
            self.output['grad_style'] = (self.output['img_out'].grad - grad).norm()
            grad = self.output['img_out'].grad.clone()
        # patch style 
        if self.opt.loss_weight_patch_style > 0:
            self.output['loss_patch_style'] = self.compute_patch_style_loss(self.output['img_out'], self.input['joint_c_2'], self.output['img_tar'], self.input['joint_c_2'], self.opt.patch_size)
            (self.output['loss_patch_style'] * self.opt.loss_weight_patch_style).backward(retain_graph=True)
            self.output['grad_patch_style'] = (self.output['img_out'].grad - grad).norm()
            grad = self.output['img_out'].grad.clone()
        # gan 
        if self.use_GAN:
            if self.opt.D_cond:
                D_input = torch.cat((self.output['img_out'], self.output['pose_tar']), dim=1)
            else:
                D_input = self.output['img_out']
            self.output['loss_G'] = self.crit_GAN(self.netD(D_input), True)
            (self.output['loss_G'] * self.opt.loss_weight_gan).backward(retain_graph=True)
            self.output['grad_gan'] = (self.output['img_out'].grad - grad).norm()
        # KL 
        self.output['loss_kl'] = self.compute_kl_loss(self.output['ps'], self.output['qs'])
        (self.output['loss_kl'] * self.opt.loss_weight_kl).backward()

    def optimize_parameters(self, check_grad=False):
        # clear previous output
        self.output = {}
        self.forward()
        if self.use_GAN:
            self.optim_D.zero_grad()
            self.backward_D()
            self.optim_D.step()
        self.optim.zero_grad()
        if check_grad:
            self.backward_checkgrad()
        else:
            self.backward()
        self.optim.step()


    def get_pose_dim(self, pose_type):
        dim = 0
        pose_items = pose_type.split('+')
        pose_items.sort()
        for item in pose_items:
            if item == 'joint':
                dim += 18
            elif item == 'seg':
                dim += 7
            elif item == 'stickman':
                dim += 3
            else:
                raise Exception('invalid pose representation type %s' % item)
        return dim

    def get_pose(self, pose_type, index='1'):
        assert index in {'1', '2'}

        pose = []
        pose_items = pose_type.split('+')
        pose_items.sort()
        for item in pose_items:
            if item == 'joint':
                pose.append(self.input['joint_%s'%index])
            elif item == 'seg':
                pose.append(self.input['seg_mask_%s'%index])
            elif item == 'stickman':
                pose.append(self.input['stickman_%s'%index])
            else:
                raise Exception('invalid pose representation type %s' % item)

        assert len(pose) > 0
        pose = torch.cat(pose, dim=1)
        return pose

    def get_patch(self, images, coords, patch_size=32):
        '''
        image_batch: images (bsz, c, h, w)
        coord: coordinates of joint points (bsz, 18, 2)
        '''
        bsz, c, h, w = images.size()

        # use 0-None for face area, ignore [14-REye, 15-LEye, 16-REar, 17-LEar]
        joint_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        patches = []

        for i in joint_index:
            patch = []
            for j in range(bsz):
                img = images[j]
                x = int(coords[j, i, 0].item())
                y = int(coords[j, i, 1].item())
                if x < 0 or y < 0:
                    p = img.new(1, c, patch_size, patch_size).fill_(0)
                else:
                    left    = x-(patch_size//2)
                    right   = x-(patch_size//2)+patch_size
                    top     = y-(patch_size//2)
                    bottom  = y-(patch_size//2)+patch_size

                    left, p_l   = (left, 0) if left >= 0 else (0, -left)
                    right, p_r  = (right, 0) if right <= w else (w, right-w)
                    top, p_t    = (top, 0) if top >= 0 else (0, -top)
                    bottom, p_b = (bottom, 0) if bottom <= h else (h, bottom-h)

                    p = img[:, top:bottom, left:right].unsqueeze(dim=0)
                    if not (p_l == p_r == p_t == p_b == 0):
                        p = F.pad(p, pad=(p_l, p_r, p_t, p_b), mode='reflect')

                patch.append(p)
            patch = torch.cat(patch, dim=0)
            patches.append(patch)
        return patches

    def get_body_limb(self, images, c):
        '''
        crop 6 patches:
            0: lshoulder,lhip, rhip, rshoulder
            1: lshoulder, rshoulder, nose
            2: lshoulder, lelbow
            3: lelbow, lwritst
            4: rshoulder, relbow
            5: relbow, rwritst
            6: lhip, lknee
            7: rhip, rknee
        '''
        bpart = [
            ['lshoulder', 'lhip', 'rhip', 'rshoulder'],
            ['lshoulder', 'rshoulder', 'nose'],
            ['lshoulder', 'lelbow'],
            ['lelbow', 'lwritst'],
            ['rshoulder', 'relbow'],
            ['relbow', 'rwritst'],
            ['lhip', 'lknee'],
            ['rhip', 'rknee']]

    
    def compute_patch_style_loss(self, images_1, c_1, images_2, c_2, patch_size=32):
        '''
        images_1: (bsz, h, w, h)
        images_2: (bsz, h, w, h)
        c_1: (bsz, 18, 2) # patch center coordinates of images_1
        c_2: (bsz, 18, 2) # patch center coordinates of images_2
        '''
        bsz = images_1.size(0)
        # remove invalid joint point
        c_invalid = (c_1 < 0) | (c_2 < 0)
        vc_1 = c_1.clone()
        vc_2 = c_2.clone()
        vc_1[c_invalid] = -1
        vc_2[c_invalid] = -1
        # get patches
        patches_1 = self.get_patch(images_1, vc_1, patch_size) # list: [patch_c1, patch_c2, ...]
        patches_2 = self.get_patch(images_2, vc_2, patch_size)
        # compute style loss
        patches_1 = torch.cat(patches_1, dim=0)
        patches_2 = torch.cat(patches_2, dim=0)
        loss_patch_style = self.crit_vgg(patches_1, patches_2, 'style')
        return loss_patch_style

    def get_current_errors(self):
        error_list = ['PSNR', 'SSIM', 'loss_L1', 'loss_content', 'loss_style', 'loss_patch_style', 'loss_kl', 'loss_G', 'loss_D', 'grad_L1', 'grad_content', 'grad_style', 'grad_patch_style', 'grad_gan']
        errors = OrderedDict()
        for item in error_list:
            if item in self.output:
                errors[item] = self.output[item].data.item()
        return errors

    def get_current_visuals(self):
        visuals = OrderedDict([
            ('img_ref', (self.input['img_1'].data.cpu(), 'rgb')),
            ('joint_tar', (self.output['joint_tar'].data.cpu(), 'pose')),
            ('stickman_tar', (self.output['stickman_tar'].data.cpu(), 'rgb')),
            ('img_tar', (self.output['img_tar'].data.cpu(), 'rgb')),
            ('img_out', (self.output['img_out'].data.cpu(), 'rgb')),
            ])
        return visuals

    def save(self, label):
        self.save_network(self.netT, 'netT', label, self.gpu_ids)
        if self.use_GAN:
            self.save_network(self.netD, 'netD', label, self.gpu_ids)


