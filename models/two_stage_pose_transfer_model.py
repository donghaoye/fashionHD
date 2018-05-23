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

class TwoStagePoseTransferModel(BaseModel):
    def name(self):
        return 'TwoStagePoseTransferModel'

    def _create_stage_1_net(self, opt):
        '''
        stage-1 network should be a pretrained pose transfer model.
        assume it is a vunet for now
        '''
        # load options
        opt_s1 = argparse.Namespace()
        dict_opt_s1 = io.load_json(os.path.join('checkpoints', opt.which_model_stage_1, 'train_opt.json'))
        opt_s1.__dict__.update(dict_opt_s1)
        self.opt_s1 = opt_s1
        # create model
        if opt_s1.which_model_T == 'vunet':
            self.netT_s1 = networks.VariationalUnet(
                input_nc_dec = self.get_pose_dim(opt_s1.pose_type),
                input_nc_enc = self.get_appearance_dim(opt_s1.appearance_type),
                output_nc = self.get_output_dim(opt_s1.output_type),
                nf = opt_s1.vunet_nf,
                max_nf = opt_s1.vunet_max_nf,
                input_size = opt_s1.fine_size,
                n_latent_scales = opt_s1.vunet_n_latent_scales,
                bottleneck_factor = opt_s1.vunet_bottleneck_factor,
                box_factor = opt_s1.vunet_box_factor,
                n_residual_blocks = 2,
                norm_layer = networks.get_norm_layer(opt_s1.norm),
                activation = nn.ReLU(False),
                use_dropout = False,
                gpu_ids = opt.gpu_ids,
                output_tanh = False,
                )
            if opt.gpu_ids:
                self.netT_s1.cuda()
        else:
            raise NotImplementedError()

    def initialize(self, opt):
        super(TwoStagePoseTransferModel, self).initialize(opt)
        ###################################
        # load pretrained stage-1 (coarse) network
        ###################################
        self._create_stage_1_net(opt)
        ###################################
        # define stage-2 (refine) network
        ###################################
        # local patch encoder
        if opt.which_model_s2e == 'patch_embed':
            self.netT_s2e = networks.LocalPatchEncoder(
                n_patch = len(opt.patch_indices),
                input_nc = 3,
                output_nc = opt.s2e_nof,
                nf = opt.s2e_nf,
                max_nf = opt.s2e_max_nf,
                input_size = opt.patch_size,
                bottleneck_factor = opt.s2e_bottleneck_factor,
                n_residual_blocks = 2,
                norm_layer = networks.get_norm_layer(opt.norm),
                activation = nn.ReLU(False),
                use_dropout = False,
                gpu_ids = opt.gpu_ids,
                )
            s2e_nof = opt.s2e_nof
        elif opt.which_model_s2e == 'patch':
            self.netT_s2e = networks.LocalPatchRearranger(
                n_patch = len(opt.patch_indices),
                image_size = opt.fine_size,
                )
            s2e_nof = 3
        elif opt.which_model_s2e == 'seg_embed':
            self.netT_s2e = networks.SegmentRegionEncoder(
                seg_nc = 7,
                input_nc = 3,
                output_nc = opt.s2e_nof,
                nf = opt.s2d_nf,
                input_size = opt.fine_size,
                n_blocks = 3,
                norm_layer = networks.get_norm_layer(opt.norm),
                activation = nn.ReLU,
                use_dropout = False,
                gpu_ids = opt.gpu_ids,
                )
            s2e_nof = opt.s2e_nof
        else:
            raise NotImplementedError()
        if opt.gpu_ids:
            self.netT_s2e.cuda()
        
        # decoder
        if self.opt.which_model_s2d == 'resnet':
            self.netT_s2d = networks.ResnetGenerator(
                input_nc = 3 + s2e_nof,
                output_nc = 3,
                ngf = opt.s2d_nf,
                norm_layer = networks.get_norm_layer(opt.norm),
                activation = nn.ReLU,
                use_dropout = False,
                n_blocks = opt.s2d_nblocks,
                gpu_ids = opt.gpu_ids,
                output_tanh = False,
                )
        elif self.opt.which_model_s2d == 'unet':
            self.netT_s2d = networks.UnetGenerator_v2(
                input_nc = 3 + s2e_nof,
                output_nc = 3,
                num_downs = 8,
                ngf = opt.s2d_nf,
                max_nf = opt.s2d_nf*2**3,
                norm_layer = networks.get_norm_layer(opt.norm),
                use_dropout = False,
                gpu_ids = opt.gpu_ids,
                output_tanh = False,
                )
        else:
            raise NotImplementedError()
        if opt.gpu_ids:
            self.netT_s2d.cuda()
        ###################################
        # define discriminator
        ###################################
        self.use_GAN = self.is_train and opt.loss_weight_gan > 0
        if self.use_GAN:
            self.netD = networks.define_D_from_params(
                input_nc = 3 + self.get_pose_dim(opt.pose_type) if opt.D_cond else 3,
                ndf = opt.D_nf,
                which_model_netD = 'n_layers',
                n_layers_D = 3,
                norm = opt.norm,
                which_gan = opt.which_gan,
                init_type = opt.init_type,
                gpu_ids = opt.gpu_ids
                )
        else:
            self.netD = None
        ###################################
        # init/load model
        ###################################
        if self.is_train:
            self.load_network(self.netT_s1, 'netT', 'latest', self.opt_s1.id)
            networks.init_weights(self.netT_s2e, init_type=opt.init_type)
            networks.init_weights(self.netT_s2d, init_type=opt.init_type)
        else:
            self.load_network(self.netT_s1, 'netT_s1', opt.which_epoch)
            self.load_network(self.netT_s2e, 'netT_s2e', opt.which_epoch)
            self.load_network(self.netT_s2d, 'netT_s2d', opt.which_epoch)
        ###################################
        # loss functions
        ###################################
        self.crit_psnr = networks.PSNR()
        self.crit_ssim = networks.SSIM()

        if self.is_train:
            self.schedulers = []
            self.optimizers = []
            self.crit_vgg = networks.VGGLoss_v2(self.gpu_ids)

            self.optim = torch.optim.Adam([
                    {'params': self.netT_s2e.parameters()},
                    {'params': self.netT_s2d.parameters()}
                ], lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optim)

            if opt.train_s1:
                self.optim_s1 = torch.optim.Adam(self.netT_s1.parameters(), lr=opt.lr_s1, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optim_s1)

            if self.use_GAN:
                self.crit_GAN = networks.GANLoss(use_lsgan=opt.which_gan=='lsgan', tensor=self.Tensor)
                self.optim_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optim_D)
                self.fake_pool = ImagePool(opt.pool_size)

            for optim in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optim, opt))

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

            # optional
            'limb_1',
            'limb_2',
        ]
        for name in input_list:
            if name in data:
                self.input[name] = self.Tensor(data[name].size()).copy_(data[name])

        self.input['id'] = zip(data['id_1'], data['id_2'])
        self.input['joint_c_1'] = data['joint_c_1']
        self.input['joint_c_2'] = data['joint_c_2']

    def forward(self, mode='train'):
        ''' mode in {'train', 'transfer'} '''
        ######################################
        # set reference/target index
        ######################################
        if self.opt.supervised or mode == 'transfer':
            ref_idx = '1'
            tar_idx = '2'
        else:
            ref_idx = '1'
            tar_idx = '1'

        ######################################
        # stage-1
        ######################################
        appr_ref_s1 = self.get_appearance(self.opt_s1.appearance_type, index=ref_idx)
        pose_ref_s1 = self.get_pose(self.opt_s1.pose_type, index=ref_idx)
        pose_tar_s1 = self.get_pose(self.opt_s1.pose_type, index=tar_idx)

        if self.is_train and self.opt.train_s1:
            output_s1, self.output['ps_s1'], self.output['qs_s1'] = self.netT_s1(appr_ref_s1, pose_ref_s1, pose_tar_s1, mode=mode)
        else:
            with torch.no_grad():
                output_s1, self.output['ps_s1'], self.output['qs_s1'] = self.netT_s1(appr_ref_s1, pose_ref_s1, pose_tar_s1, mode=mode)

        output_s1 = self.parse_output(output_s1, self.opt_s1.output_type)
        self.output['img_out_s1'] = F.tanh(output_s1['image'])
        ######################################
        # stage-2
        ######################################
        # encoder
        if self.opt.which_model_s2e == 'patch_embed':
            img_ref = self.input['img_%s'%ref_idx]
            joint_c_ref = self.input['joint_c_%s'%ref_idx]
            patch_ref = self.get_patch(img_ref, joint_c_ref, self.opt.patch_size, self.opt.patch_indices)
            joint_tar = self.get_pose(pose_type='joint_ext', index=tar_idx)[:,self.opt.patch_indices]
            s2e_out = self.netT_s2e(patch_ref, joint_tar)
            # data = {
            #     'img_1': self.input['img_1'].cpu(),
            #     'img_2': self.input['img_2'].cpu(),
            #     'patch_ref': patch_ref.cpu(),
            #     'joint_tar': joint_tar.cpu(),
            # }
            # torch.save(data, 'data.pth')
            # exit(0)
        elif self.opt.which_model_s2e == 'patch':
            img_ref = self.input['img_%s'%ref_idx]
            joint_c_ref = self.input['joint_c_%s'%ref_idx]
            patch_ref = self.get_patch(img_ref, joint_c_ref, self.opt.patch_size, self.opt.patch_indices)
            joint_c_tar = self.input['joint_c_%s'%tar_idx][:,self.opt.patch_indices]
            s2e_out = self.netT_s2e(patch_ref, joint_c_tar)
        elif self.opt.which_model_s2e == 'seg_embed':
            img_ref = self.input['img_%s'%ref_idx]
            seg_ref = self.input['seg_mask_%s'%ref_idx]
            if self.opt.s2e_seg_src == 'gt':
                seg_gen = self.input['seg_mask_%s'%tar_idx]
            else:
                seg_fake = output_s1['seg'].detach()
                if (not self.is_train) or self.opt.s2e_seg_src == 'fake':
                    seg_gen = seg_fake
                else:
                    bsz, c = seg_fake.size()[0:2]
                    mask = seg_fake.new(bsz, c, 1, 1).bernoulli_()
                    seg_gen = seg_fake * mask + self.input['seg_mask_%s'%tar_idx] * (1-mask)
            s2e_out = self.netT_s2e(img_ref, seg_ref, seg_gen)
            self.output['seg_ref'] = seg_ref
            self.output['seg_gen'] = seg_gen
            self.output['seg_tar'] = self.input['seg_mask_%s'%tar_idx]
        else:
            raise NotImplementedError()
        # decoder
        dec_input = torch.cat((self.output['img_out_s1'], s2e_out), dim=1)
        s2d_out = self.netT_s2d(dec_input)
        if self.opt.which_model_s2d == 'unet':
            self.output['img_out'] = F.tanh(output_s1['image'] + s2d_out)
        else:
            self.output['img_out'] = F.tanh(s2d_out)
        self.output['img_out_res'] = self.output['img_out'] - self.output['img_out_s1']
        ######################################
        # other
        ######################################
        self.output['img_tar'] = self.input['img_%s'%tar_idx]
        self.output['pose_tar'] = self.get_pose(self.opt.pose_type, index=tar_idx)
        self.output['joint_tar'] = self.input['joint_%s'%tar_idx]
        self.output['joint_c_tar'] = self.input['joint_c_%s'%tar_idx]
        self.output['stickman_tar'] = self.input['stickman_%s'%tar_idx]
        self.output['stickman_ref'] = self.input['stickman_%s'%ref_idx]
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
            # KL: Add if using VAE model
            # L1
            self.output['loss_L1'] = F.l1_loss(self.output['img_out'], self.output['img_tar'])
            # content
            if self.opt.loss_weight_content > 0:
                self.output['loss_content'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], 'content')
            # style
            if self.opt.loss_weight_style > 0:
                self.output['loss_style'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], 'style')
            # patch style
            if self.opt.loss_weight_patch_style > 0:
                self.output['loss_patch_style'] = self.compute_patch_style_loss(self.output['img_out'], self.output['joint_c_tar'], self.output['img_tar'], self.output['joint_c_tar'], self.opt.patch_size, self.opt.patch_indices)
            # patch l1
            if self.opt.loss_weight_patch_l1 > 0:
                self.output['loss_patch_l1'] = self.compute_patch_l1_loss(self.output['img_out'], self.output['joint_c_tar'], self.output['img_tar'], self.output['joint_c_tar'], self.opt.patch_size, self.opt.patch_indices)

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
        # L1
        self.output['loss_L1'] = F.l1_loss(self.output['img_out'], self.output['img_tar'])
        loss += self.output['loss_L1'] * self.opt.loss_weight_L1
        # content
        if self.opt.loss_weight_content > 0:
            self.output['loss_content'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], 'content')
            loss += self.output['loss_content'] * self.opt.loss_weight_content
        # style
        if self.opt.loss_weight_style > 0:
            self.output['loss_style'] = self.crit_vgg(self.output['img_out'], self.output['img_tar'], 'style')
            loss += self.output['loss_style'] * self.opt.loss_weight_style
        # local style
        if self.opt.loss_weight_patch_style > 0:
            self.output['loss_patch_style'] = self.compute_patch_style_loss(self.output['img_out'], self.output['joint_c_tar'], self.output['img_tar'], self.output['joint_c_tar'], self.opt.patch_size, self.opt.patch_indices)
            loss += self.output['loss_patch_style'] * self.opt.loss_weight_patch_style
        # local l1
        if self.opt.loss_weight_patch_l1 > 0:
            self.output['loss_patch_l1'] = self.compute_patch_l1_loss(self.output['img_out'], self.output['joint_c_tar'], self.output['img_tar'], self.output['joint_c_tar'], self.opt.patch_size, self.opt.patch_indices)
            loss += self.output['loss_patch_l1'] * self.opt.loss_weight_patch_l1
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
        self.output['loss_L1'] = F.l1_loss(self.output['img_out'], self.output['img_tar'])
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
            self.output['loss_patch_style'] = self.compute_patch_style_loss(self.output['img_out'], self.output['joint_c_tar'], self.output['img_tar'], self.output['joint_c_tar'], self.opt.patch_size, self.opt.patch_indices)
            (self.output['loss_patch_style'] * self.opt.loss_weight_patch_style).backward(retain_graph=True)
            self.output['grad_patch_style'] = (self.output['img_out'].grad - grad).norm()
            grad = self.output['img_out'].grad.clone()
        # patch l1
        if self.opt.loss_weight_patch_l1 > 0:
            self.output['loss_patch_l1'] = self.compute_patch_l1_loss(self.output['img_out'], self.output['joint_c_tar'], self.output['img_tar'], self.output['joint_c_tar'], self.opt.patch_size, self.opt.patch_indices)
            (self.output['loss_patch_l1'] * self.opt.loss_weight_patch_l1).backward(retain_graph=True)
            self.output['grad_patch_l1'] = (self.output['img_out'].grad - grad).norm()
            grad = self.output['img_out'].grad.clone()
        # gan 
        if self.use_GAN:
            if self.opt.D_cond:
                D_input = torch.cat((self.output['img_out'], self.output['pose_tar']), dim=1)
            else:
                D_input = self.output['img_out']
            self.output['loss_G'] = self.crit_GAN(self.netD(D_input), True)
            (self.output['loss_G'] * self.opt.loss_weight_gan).backward()
            self.output['grad_gan'] = (self.output['img_out'].grad - grad).norm()
        
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


    def get_output_dim(self, output_type):
        dim = 0
        output_items = output_type.split('+')
        for item in output_items:
            if item == 'image':
                dim += 3
            elif item == 'seg':
                dim += 7
            else:
                raise Exception('invalid output type %s'%item)
        return dim

    def parse_output(self, output, output_type):
        assert output.size(1) == self.get_output_dim(output_type)
        output_items = output_type.split('+')
        output_items.sort()
        i = 0
        rst = {}
        for item in output_items:
            if item == 'image':
                rst['image'] = output[:,i:(i+3)]
                i += 3
            elif item == 'seg':
                rst['seg'] = output[:,i:(i+7)]
                i += 7
            else:
                raise Exception('invalid output type %s'%item)
        return rst

    def get_pose_dim(self, pose_type):
        dim = 0
        pose_items = pose_type.split('+')
        pose_items.sort()
        for item in pose_items:
            if item == 'joint':
                dim += 18
            elif item == 'joint_ext':
                dim += 23
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
                # joint 0-17 are for pose
                joint_for_pose = self.input['joint_%s'%index][:,0:18]
                pose.append(joint_for_pose)
            elif item == 'joint_ext':
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

    def get_appearance_dim(self, appearance_type):
        dim = 0
        appr_items = appearance_type.split('+')
        for item in appr_items:
            if item == 'image':
                dim += 3
            elif item == 'limb':
                dim += 24 # (3channel x 8limbs)
            else:
                raise Exception('invalid appearance prepresentation type %s'%item)
        return dim
    
    def get_appearance(self, appearance_type, index='1'):
        assert index in {'1', '2'}
        appr = []
        appr_items = appearance_type.split('+')
        for item in appr_items:
            if item == 'image':
                appr.append(self.input['img_%s'%index])
            elif item == 'limb':
                appr.append(self.input['limb_%s'%index])
            else:
                raise Exception('invalid appearance representation type %s' % item)
        assert len(appr) > 0
        appr = torch.cat(appr, dim=1)
        return appr

    def get_patch(self, images, coords, patch_size=32, patch_indices=None):
        '''
        Input:
            image_batch: images (bsz, c, h, w)
            coord: coordinates of joint points (bsz, 18, 2)
        Output:
            patches: (bsz, npatch, c, hp, ww)
        '''
        bsz, c, h, w = images.size()

        # use 0-None for face area, ignore [14-REye, 15-LEye, 16-REar, 17-LEar]
        if patch_indices is None:
            patch_indices = self.opt.patch_indices

        patches = []
        for i in range(bsz):
            patch = []
            img = images[i]
            for j in patch_indices:
                x = int(coords[i,j,0].item())
                y = int(coords[i,j,1].item())

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
                        p = F.pad(p, pad=(p_l, p_r, p_t, p_b), mode='constant')
                patch.append(p)
            patch = torch.cat(patch, dim=0)#[npatch, c, hp, wp]
            patches.append(patch)
        patches = torch.stack(patches)#[bsz, npatch, c, hp, wp]

        return patches


    def compute_patch_style_loss(self, images_1, c_1, images_2, c_2, patch_size=32, patch_indices=None):
        '''
        images_1: (bsz, h, w, h)
        images_2: (bsz, h, w, h)
        c_1: (bsz, 18, 2) # patch center coordinates of images_1
        c_2: (bsz, 18, 2) # patch center coordinates of images_2
        '''
        # remove invalid joint point
        c_invalid = (c_1 < 0) | (c_2 < 0)
        vc_1 = c_1.clone()
        vc_2 = c_2.clone()
        vc_1[c_invalid] = -1
        vc_2[c_invalid] = -1
        # get patches
        patches_1 = self.get_patch(images_1, vc_1, patch_size, patch_indices) # list: [patch_c1, patch_c2, ...]
        patches_2 = self.get_patch(images_2, vc_2, patch_size, patch_indices)
        # compute style loss
        bsz, npatch, c, h, w = patches_1.size()
        patches_1 = patches_1.view(bsz*npatch, c, h, w)
        patches_2 = patches_2.view(bsz*npatch, c, h, w)
        loss_patch_style = self.crit_vgg(patches_1, patches_2, 'style')

        # output = {
        #     'images_1': images_1.cpu(),
        #     'images_2': images_2.cpu(),
        #     'c_1': c_1.cpu(),
        #     'c_2': c_2.cpu(),
        #     'patches_1': patches_1.cpu(),
        #     'patches_2': patches_2.cpu(),
        #     'npatch': npatch,
        #     'id': self.input['id']
        # }

        # torch.save(output, 'data.pth')
        # exit()
        return loss_patch_style

    def compute_patch_l1_loss(self, images_1, c_1, images_2, c_2, patch_size=32, patch_indices=None):
        '''
        images_1: (bsz, h, w, h)
        images_2: (bsz, h, w, h)
        c_1: (bsz, 18, 2) # patch center coordinates of images_1
        c_2: (bsz, 18, 2) # patch center coordinates of images_2
        '''

        # remove invalid joint point
        c_invalid = (c_1 < 0) | (c_2 < 0)
        vc_1 = c_1.clone()
        vc_2 = c_2.clone()
        vc_1[c_invalid] = -1
        vc_2[c_invalid] = -1
        # get patches
        patches_1 = self.get_patch(images_1, vc_1, patch_size, patch_indices) # list: [patch_c1, patch_c2, ...]
        patches_2 = self.get_patch(images_2, vc_2, patch_size, patch_indices)
        # compute style loss
        loss_patch_l1 = F.l1_loss(patches_1, patches_2)
        return loss_patch_l1

    def get_current_errors(self):
        error_list = ['PSNR', 'SSIM', 'loss_L1', 'loss_content', 'loss_style', 'loss_patch_style', 'loss_patch_l1', 'loss_kl', 'loss_G', 'loss_D', 'grad_L1', 'grad_content', 'grad_style', 'grad_patch_style', 'grad_patch_l1', 'grad_gan']
        errors = OrderedDict()
        for item in error_list:
            if item in self.output:
                errors[item] = self.output[item].data.item()
        return errors

    def get_current_visuals(self):
        visuals = OrderedDict([
            ('img_ref', [self.input['img_1'].data.cpu(), 'rgb']),
            ('stickman_ref', [self.output['stickman_ref'].data.cpu(), 'rgb']),
            ('stickman_tar', [self.output['stickman_tar'].data.cpu(), 'rgb']),
            ('img_tar', [self.output['img_tar'].data.cpu(), 'rgb']),
            ('img_out_s1', [self.output['img_out_s1'].data.cpu(), 'rgb']),
            ('img_out', [self.output['img_out'].data.cpu(), 'rgb']),
            ('img_out_res', [self.output['img_out_res'].data.cpu(), 'rgb']),
            ])
        if 'seg_ref' in self.output:
            visuals['seg_ref'] = [self.output['seg_ref'].data.cpu(), 'seg']
        if 'seg_tar' in self.output:
            visuals['seg_tar'] = [self.output['seg_tar'].data.cpu(), 'seg']
        return visuals

    def save(self, label):
        self.save_network(self.netT_s1, 'netT_s1', label, self.gpu_ids)
        self.save_network(self.netT_s2e, 'netT_s2e', label, self.gpu_ids)
        self.save_network(self.netT_s2d, 'netT_s2d', label, self.gpu_ids)
        if self.use_GAN:
            self.save_network(self.netD, 'netD', label, self.gpu_ids)


###############################
# other functions
###############################
def show_patch():
    from options.pose_transfer_options import TrainPoseTransferOptions
    from data.data_loader import CreateDataLoader
    import imageio
    import cv2
    
    opt = TrainPoseTransferOptions().parse()

    # output_dir = 'temp/visualize_patch/%d/' % opt.patch_size
    output_dir = 'temp/visualize_patch/test/'
    io.mkdir_if_missing(output_dir)
    
    loader = iter(CreateDataLoader(opt, 'test'))
    model = TwoStagePoseTransferModel()
    model.initialize(opt)
    data = next(loader)
    model.set_input(data)
    
    patch_1 = model.get_patch(model.input['img_1'], model.input['joint_c_1'], opt.patch_size, opt.patch_indices)
    patch_2 = model.get_patch(model.input['img_2'], model.input['joint_c_2'], opt.patch_size, opt.patch_indices)
    images = [model.input['img_1'], model.input['img_2']]
    patches = [patch_1, patch_2]
    joints = [model.input['joint_c_1'], model.input['joint_c_2']]
    id_list = model.input['id']


    def _tensor_to_numpy(img_t):
        img = img_t.cpu().detach().numpy().transpose(1,2,0)
        img = (img*127.5 + 127.5).clip(0,255).astype(np.uint8)
        return img

    for idx in range(opt.batch_size):
        for i in range(2):
            fn_i = output_dir + '%d_%d.jpg'%(idx,i)
            img = _tensor_to_numpy(images[i][idx])
            for j, joint_idx in enumerate(opt.patch_indices):
                fn_p = output_dir + '%d_%d_%d.jpg' % (idx, i, joint_idx)
                p = _tensor_to_numpy(patches[i][idx,j])
                imageio.imwrite(fn_p, p)

                x_c, y_c = joints[i][idx, joint_idx]
                if x_c > 0 and y_c > 0:
                    x_1, y_1 = x_c - opt.patch_size//2, y_c - opt.patch_size//2
                    x_2, y_2 = x_c + opt.patch_size//2, y_c + opt.patch_size//2
                    cv2.rectangle(img, (x_1, y_1), (x_2, y_2), color=(0,255,0))
                    cv2.putText(img, str(joint_idx).encode('ascii', 'replace'), (x_c, y_c), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0))

            imageio.imwrite(fn_i, img)


if __name__ == '__main__':
    show_patch()
    