from __future__ import division, print_function

import torch
import torch.nn as nn
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


class DesignerGAN(BaseModel):
    def name(self):
        return 'DesignerGAN'

    def initialize(self, opt):
        super(DesignerGAN, self).initialize(opt)
        ###################################
        # define data tensors
        ###################################
        # self.input['img'] = self.Tensor()
        # self.input['img_attr'] = self.Tensor()
        # self.input['lm_map'] = self.Tensor()
        # self.input['seg_mask'] = self.Tensor()
        # self.input['attr_label'] = self.Tensor()
        # self.input['id'] = []

        ###################################
        # load/define networks
        ###################################

        # Todo modify networks.define_G
        # 1. add specified generator networks

        self.netG = networks.define_G(opt)
        self.netAE, self.opt_AE = network_loader.load_attribute_encoder_net(id = opt.which_model_AE, gpu_ids = opt.gpu_ids)
        if opt.which_model_FeatST != 'none':
            self.netFeatST, self.opt_FeatST = network_loader.load_feature_spatial_transformer_net(id=opt.which_model_FeatST, gpu_ids = opt.gpu_ids)
            self.use_FeatST = True
            assert self.opt_FeatST.shape_encode == self.opt.shape_encode, 'GAN model and FeatST model has different shape encode mode'
            assert self.opt_FeatST.input_mask_mode == self.opt.input_mask_mode, 'GAN model and FeatST model has different segmentation input mode'
        else:
            self.use_FeatST = False

        if self.is_train:
            self.netD = networks.define_D(opt)
            if opt.which_model_init_netG != 'none' and not opt.continue_train:
                self.load_network(self.netG, 'G', 'latest', opt.which_model_init_netG)

        if not self.is_train or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.is_train:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.is_train:
            self.fake_pool = ImagePool(opt.pool_size)

            ###################################
            # define loss functions and loss buffers
            ###################################
            if opt.which_gan in {'dcgan', 'lsgan'}:
                self.crit_GAN = networks.GANLoss(use_lsgan = opt.which_gan == 'lsgan', tensor = self.Tensor)
            else:
                # WGAN loss will be calculated in self.backward_D_wgangp and self.backward_G
                self.crit_GAN = None
            self.crit_L1 = nn.L1Loss()
            self.crit_attr = nn.BCELoss()

            self.loss_functions = []
            self.loss_functions.append(self.crit_GAN)
            self.loss_functions.append(self.crit_L1)
            self.loss_functions.append(self.crit_attr)

            if self.opt.loss_weight_vgg > 0:
                self.crit_vgg = networks.VGGLoss(self.gpu_ids)
                self.loss_functions.append(self.crit_vgg)

            ###################################
            # create optimizers
            ###################################
            self.schedulers = []
            self.optimizers = []

            self.optim_G = torch.optim.Adam(self.netG.parameters(),
                lr = opt.lr, betas = (opt.beta1, opt.beta2))
            self.optim_D = torch.optim.Adam(self.netD.parameters(),
                lr = opt.lr, betas = (opt.beta1, opt.beta2))
            self.optimizers.append(self.optim_G)
            self.optimizers.append(self.optim_D)
            
            for optim in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optim, opt))

        # color transformation from std to imagenet
        # img_imagenet = img_std * a + b
        self.trans_std_to_imagenet = {
            'a': Variable(self.Tensor([0.5/0.229, 0.5/0.224, 0.5/0.225]), requires_grad = False).view(3,1,1),
            'b': Variable(self.Tensor([(0.5-0.485)/0.229, (0.5-0.456)/0.224, (0.5-0.406)/0.225]), requires_grad = False).view(3,1,1)
        }

    def _std_to_imagenet(self, img):
        return img * self.trans_std_to_imagenet['a'] + self.trans_std_to_imagenet['b']

    def set_input(self, data):
        self.input['img'] = self.Tensor(data['img'].size()).copy_(data['img'])
        self.input['attr_label'] = self.Tensor(data['attr_label'].size()).copy_(data['attr_label'])
        self.input['lm_map'] = self.Tensor(data['lm_map'].size()).copy_(data['lm_map'])
        self.input['seg_mask'] = self.Tensor(data['seg_mask'].size()).copy_(data['seg_mask'])
        self.input['seg_map'] = self.Tensor(data['seg_map'].size()).copy_(data['seg_map'])
        self.input['edge_map'] = self.Tensor(data['edge_map'].size()).copy_(data['edge_map'])
        self.input['id'] = data['id']


        # create input variables
        for k, v in self.input.iteritems():
            if isinstance(v, torch.tensor._TensorBase):
                self.input[k] = Variable(v)

    def forward(self):
        # Todo: consider adding "extra_code" generated by a CNN jointly trained with GAN
        shape_code = self.encode_shape(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'])
        if not self.opt.no_attr_cond:
            attr_code = self.encode_attribute(self.input['img'], self.input['lm_map'])
            if self.use_FeatST:
                attr_code = self.feat_spatial_transform(attr_code, self.input['seg_mask'], self.input['seg_mask'])
            self.output['img_fake_raw'] = self.netG(shape_code, attr_code)
        else:
            attr_code = None
            self.output['img_fake_raw'] = self.netG(shape_code)

        self.output['img_real_raw'] = self.input['img']
        self.output['img_fake'] = self.mask_image(self.output['img_fake_raw'], self.input['seg_map'], self.output['img_real_raw'])
        self.output['img_real'] = self.mask_image(self.output['img_real_raw'], self.input['seg_map'], self.output['img_real_raw'])
        self.output['shape_code'] = shape_code
        self.output['attr_code'] = attr_code

    def test(self):
        if float(torch.__version__[0:3]) >= 0.4:
            with torch.no_grad():
                self.forward()
        else:
            self.input['img'].volatile = True
            self.input['lm_map'].volatile = True
            self.input['seg_mask'].volatile = True
            self.forward()


    def backward_D(self):
        # fake
        # here we use masked images
        repr_fake = self.encode_shape(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'], self.output['img_fake'].detach())
        repr_fake = self.fake_pool.query(repr_fake.data)
        pred_fake = self.netD(repr_fake)
        self.output['loss_D_fake'] = self.crit_GAN(pred_fake, False)
        
        # real
        repr_real = self.encode_shape(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'], self.output['img_real'])
        pred_real = self.netD(repr_real)
        self.output['loss_D_real'] = self.crit_GAN(pred_real, True)

        # combined loss
        self.output['loss_D'] = (self.output['loss_D_real'] + self.output['loss_D_fake']) * 0.5
        self.output['loss_D'].backward()

    def backward_D_wgangp(self):
        # optimize netD using wasserstein gan loss with gradient penalty. 
        # when using wgan, loss_D_fake(real) means critic output for fake(real) data, instead of loss
        bsz = self.output['img_fake'].size(0)
        # fake
        repr_fake = self.encode_shape(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'], self.output['img_fake'].detach())
        disc_fake = self.netD(repr_fake)
        self.output['loss_D_fake'] = disc_fake.mean()
        # real
        repr_real = self.encode_shape(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'], self.output['img_real'])
        disc_real = self.netD(repr_real)
        self.output['loss_D_real'] = disc_real.mean()

        loss_D = self.output['loss_D_fake'] - self.output['loss_D_real']
        loss_D.backward()
        self.output['loss_D'] = -loss_D # wasserstein distance, not real loss

        # gradient penalty
        alpha_sz = [bsz] + [1]*(repr_fake.ndimension()-1)
        alpha = torch.rand(alpha_sz).expand(repr_fake.size())
        alpha = repr_fake.data.new(alpha.size()).copy_(alpha)

        repr_interp = alpha * repr_real.data + (1 - alpha) * repr_fake.data
        repr_interp = Variable(repr_interp, requires_grad = True)
        
        disc_interp = self.netD(repr_interp).view(bsz,-1).mean(1)
        # grad = torch.autograd.grad(outputs = disc_interp, inputs = repr_interp,
        #     grad_outputs = disc_interp.data.new(disc_interp.size()).fill_(1),
        #     create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad = torch.autograd.grad(outputs = disc_interp.sum(), inputs = repr_interp,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_penalty = ((grad.view(bsz,-1).norm(2,dim=1)-1)**2).mean()
        self.output['loss_gp'] = grad_penalty * self.opt.loss_weight_gp * self.opt.loss_weight_GAN
        self.output['loss_gp'].backward()
        # print('D_fake: %f, D_real: %f, gp: %f' %(self.output['loss_D_fake'].data[0], self.output['loss_D_real'].data[0], self.output['loss_gp'].data[0]))

    def backward_G(self):
        repr_fake = self.encode_shape(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'], self.output['img_fake'])
        self.output['loss_G'] = 0
        # GAN Loss
        if self.opt.which_gan == 'wgan':
            disc_fake = self.netD(repr_fake)
            self.output['loss_G_GAN'] = -disc_fake.mean()
        else:
            pred_fake = self.netD(repr_fake)
            self.output['loss_G_GAN'] = self.crit_GAN(pred_fake, True)
        self.output['loss_G'] += self.output['loss_G_GAN'] * self.opt.loss_weight_GAN
        # L1 Loss
        self.output['loss_G_L1'] = self.crit_L1(self.output['img_fake'], self.output['img_real'])
        self.output['loss_G'] += self.output['loss_G_L1'] * self.opt.loss_weight_L1
        # Attribute Loss
        attr_prob = self.encode_attribute(self.output['img_fake'], self.input['lm_map'], output_type = 'prob')
        self.output['loss_G_attr'] = self.crit_attr(attr_prob, self.input['attr_label'])
        self.output['loss_G'] += self.output['loss_G_attr'] * self.opt.loss_weight_attr
        # VGG Loss
        if self.opt.loss_weight_vgg > 0:
            self.output['loss_G_VGG'] = self.crit_vgg(self.output['img_fake'], self.output['img_real'])
            self.output['loss_G'] += self.output['loss_G_VGG'] * self.opt.loss_weight_vgg
        # backward
        self.output['loss_G'].backward()


    def backward_G_shape_adaptive(self):
        # create unmatched shape and attribute code
        shape_code, attr_code = self.output['shape_code'], self.output['attr_code']
        shuffle_idx = Variable(torch.randperm(attr_code.size(0)))
        if attr_code.is_cuda:
            shuffle_idx = shuffle_idx.cuda(attr_code.get_device())
        attr_code_shuffle = attr_code.index_select(dim=0, index=shuffle_idx)

        # generate images
        self.output['img_sa_raw'] = self.netG(shape_code, attr_code_shuffle)
        self.output['img_sa'] = self.mask_image(self.output['img_sa_raw'], self.input['seg_map'], self.output['img_real_raw'])
        
        # compute GAN loss
        repr_sa = self.encode_shape(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'], self.output['img_sa'])
        if self.opt.which_gan == 'wgan':
            disc_sa = self.netD(repr_sa)
            self.output['loss_G_sa'] = -disc_sa.mean()
        else:
            pred_sa = self.netD(repr_sa)
            self.output['loss_G_sa'] = self.crit_GAN(pred_sa, True)

        (self.output['loss_G_sa'] * self.opt.loss_weight_GAN).backward()


    def backward_G_grad_check(self):
        self.output['img_fake'].retain_grad()
        repr_fake = self.encode_shape(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'], self.output['img_fake'])
        self.output['loss_G'] = 0
        # GAN Loss
        if self.opt.which_gan == 'wgan':
            disc_fake = self.netD(repr_fake)
            self.output['loss_G_GAN'] = -disc_fake.mean()
        else:
            pred_fake = self.netD(repr_fake)
            self.output['loss_G_GAN'] = self.crit_GAN(pred_fake, True)

        (self.output['loss_G_GAN'] * self.opt.loss_weight_GAN).backward(retain_graph=True)
        self.output['loss_G'] += self.output['loss_G_GAN'] * self.opt.loss_weight_GAN
        self.output['grad_G_GAN'] = (self.output['img_fake'].grad).norm()
        grad = self.output['img_fake'].grad.clone()
        # L1 Loss
        self.output['loss_G_L1'] = self.crit_L1(self.output['img_fake'], self.output['img_real'])
        (self.output['loss_G_L1'] * self.opt.loss_weight_L1).backward(retain_graph=True)
        self.output['loss_G'] += self.output['loss_G_L1'] * self.opt.loss_weight_L1
        self.output['grad_G_L1'] = (self.output['img_fake'].grad - grad).norm()
        grad = self.output['img_fake'].grad.clone()
        # Attribute Loss
        attr_prob = self.encode_attribute(self.output['img_fake'], self.input['lm_map'], output_type = 'prob')
        self.output['loss_G_attr'] = self.crit_attr(attr_prob, self.input['attr_label'])
        (self.output['loss_G_attr'] * self.opt.loss_weight_attr).backward(retain_graph=True)
        self.output['loss_G'] += self.output['loss_G_attr'] * self.opt.loss_weight_attr
        self.output['grad_G_attr'] = (self.output['img_fake'].grad - grad).norm()
        grad = self.output['img_fake'].grad.clone()
        # VGG Loss
        if self.opt.loss_weight_vgg > 0:
            self.output['loss_G_VGG'] = self.crit_vgg(self.output['img_fake'], self.output['img_real'])
            (self.output['loss_G_VGG'] * self.opt.loss_weight_vgg).backward()
            self.output['loss_G'] += self.output['loss_G_VGG'] * self.opt.loss_weight_vgg
            self.output['grad_G_VGG'] = (self.output['img_fake'].grad - grad).norm()


    def optimize_parameters(self, train_D = True, train_G = True, check_grad = False):
        # clear previous output
        self.output = {}

        self.forward()
        # optimize D
        self.optim_D.zero_grad()
        if self.opt.which_gan == 'wgan':
            self.backward_D_wgangp()
        else:
            self.backward_D()

        if train_D:
            self.optim_D.step()
        # optimize G
        self.optim_G.zero_grad()
        if check_grad:
            self.backward_G_grad_check()
        else:
            self.backward_G()

        if self.opt.shape_adaptive:
            self.backward_G_shape_adaptive()
        if train_G:
            self.optim_G.step()

    def get_current_errors(self):
        errors = OrderedDict([
            ('D_GAN', self.output['loss_D'].data[0]),
            ('G_GAN', self.output['loss_G_GAN'].data[0]),
            ('G_L1', self.output['loss_G_L1'].data[0]),
            ('G_attr', self.output['loss_G_attr'].data[0])
            ])

        if 'loss_G_VGG' in self.output:
            errors['G_VGG'] = self.output['loss_G_VGG'].data[0]
        if 'loss_gp' in self.output:
            errors['D_GP'] = self.output['loss_gp'].data[0]
        if 'loss_G_sa' in self.output:
            errors['G_sa'] = self.output['loss_G_sa'].data[0]

        # gradients
        grad_list = ['grad_G_GAN', 'grad_G_L1', 'grad_G_VGG', 'grad_G_attr']
        for grad_name in grad_list:
            if grad_name in self.output:
                errors[grad_name] = self.output[grad_name].data[0]

        return errors

    def get_current_visuals(self):
        visuals = OrderedDict([
            ('img_real', self.output['img_real'].data.clone()),
            ('img_fake', self.output['img_fake'].data.clone()),
            ('img_real_raw', self.output['img_real_raw'].data.clone()),
            ('img_fake_raw', self.output['img_fake_raw'].data.clone()),
            ('seg_map', self.input['seg_map'].data.clone()),
            ('landmark_heatmap', self.input['lm_map'].data.clone()),
            ('edge_map', self.input['edge_map'].data.clone())
            ])
        return visuals

    def encode_attribute(self, img, lm_map = None, output_type = None):
        if output_type is None:
            output_type = self.opt.attr_cond_type
        v_img = img if isinstance(img, Variable) else Variable(img)

        if self.opt_AE.image_normalize == 'imagenet':
            v_img = self._std_to_imagenet(v_img)

        if self.opt_AE.input_lm:
            v_lm_map = lm_map if isinstance(lm_map, Variable) else Variable(lm_map)
            # prob, prob_map = self.netAE(v_img, v_lm_map)
            input = (v_img, v_lm_map)
        else:
            input = (v_img,)

        if output_type == 'feat':
            feat, _ = self.netAE.extract_feat(*input)
            return feat
        elif output_type == 'feat_map':
            _, feat_map = self.netAE.extract_feat(*input)
            return feat_map
        elif output_type == 'prob':
            prob, _ = self.netAE(*input)
            return prob
        elif output_type == 'feat_map':
            _, prob_map = self.netAE(*input)
            return prob_map

    def feat_spatial_transform(self, feat_input, shape_src, shape_tar):
        return self.netFeatST(feat_input, shape_src, shape_tar)

    def encode_shape(self, lm_map, seg_mask, edge_map, img = None):
        if self.opt.shape_encode == 'lm':
            shape_code = lm_map
        elif self.opt.shape_encode == 'seg':
            shape_code = seg_mask
        elif self.opt.shape_encode == 'lm+seg':
            shape_code = torch.cat((lm_map, seg_mask), dim = 1)
        elif self.opt.shape_encode == 'seg+e':
            shape_code = torch.cat((seg_mask, edge_map), dim = 1)
        elif self.opt.shape_encode == 'lm+seg+e':
            shape_code = torch.cat((lm_map, seg_mask, edge_map), dim = 1)

        if img is not None:
            shape_code = torch.cat((img, shape_code), dim = 1)
        return shape_code

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
       

    def save(self, label):
        # Todo: if self.netAE is jointly trained, also save its parameter
        # Todo: if att_fuse module is added, save its parameters
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

