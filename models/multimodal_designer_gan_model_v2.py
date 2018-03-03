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

class MultimodalDesignerGAN_V2(BaseModel):
    '''
    modules:
        encoders: Es, Ee and Ec
        feat_transfer_networks
        netG: generator (decoder)
        netD: discriminator
        netD_feat: feature level discriminator?
    '''
    def name(self):
        return 'MultimodalDesignerGAN_V2'

    def initialize(self, opt):
        super(MultimodalDesignerGAN_V2, self).initialize(opt)
        ###################################
        # define networks
        ###################################
        self.modules = {}
        # shape branch
        if opt.which_model_netG != 'unet':
            self.shape_encoder = networks.define_image_encoder(opt, 'shape')
            self.modules['shape_encoder'] = self.shape_encoder
        else:
            self.shape_encoder = None
        # edge branch
        if opt.use_edge:
            self.edge_encoder = networks.define_image_encoder(opt, 'edge')
            self.modules['edge_encoder'] = self.edge_encoder
        else:
            self.encoder_edge = None
        # color branch
        if opt.use_color:
            self.color_encoder = networks.define_image_encoder(opt, 'color')
            self.modules['color_encoder'] = self.color_encoder
        else:
            self.color_encoder = None

        # fusion model
        if opt.ftn_model == 'none':
            # shape_feat, edge_feat and color_feat will be simply upmpled to same size (size of shape_feat) and concatenated
            pass
        elif opt.ftn_model == 'concat':
            assert opt.use_edge or opt.use_color
            if opt.use_edge:
                self.edge_trans_net = networks.define_feature_fusion_network(name='FeatureConcatNetwork', feat_nc=opt.edge_nof, guide_nc=opt.shape_nof, nblocks=opt.ftn_nblocks, 
                    norm=opt.norm, gpu_ids=self.gpu_ids, init_type=opt.init_type)
                self.modules['edge_trans_net'] = self.edge_trans_net
            if opt.use_color:
                self.color_trans_net = networks.define_feature_fusion_network(name='FeatureConcatNetwork', feat_nc=opt.color_nof, guide_nc=opt.shape_nof, nblocks=opt.ftn_nblocks, 
                    norm=opt.norm, gpu_ids=self.gpu_ids, init_type=opt.init_type)
                self.modules['color_trans_net'] = self.color_trans_net
        elif opt.ftn_model == 'reduce':
            assert opt.use_edge or opt.use_color
            if opt.use_edge:
                self.edge_trans_net = networks.define_feature_fusion_network(name='FeatureReduceNetwork', feat_nc=opt.edge_nof, guide_nc=opt.shape_nof, nblocks=opt.ftn_nblocks, 
                    ndowns=opt.ftn_ndowns, norm=opt.norm, gpu_ids=self.gpu_ids, init_type=opt.init_type)
                self.modules['edge_trans_net'] = self.edge_trans_net
            if opt.use_color:
                self.color_trans_net = networks.define_feature_fusion_network(name='FeatureReduceNetwork', feat_nc=opt.color_nof, guide_nc=opt.shape_nof, nblocks=opt.ftn_nblocks,
                    ndowns=opt.ftn_ndowns, norm=opt.norm, gpu_ids=self.gpu_ids, init_type=opt.init_type)
                self.modules['color_trans_net'] = self.color_trans_net

        elif opt.ftn_model == 'trans':
            assert opt.use_edge or opt.use_color
            if opt.use_edge:
                self.edge_trans_net = networks.define_feature_fusion_network(name='FeatureTransformNetwork', feat_nc=opt.edge_nof, guide_nc=opt.shape_nof, nblocks=opt.ftn_nblocks, 
                    feat_size=opt.feat_size_lr, norm=opt.norm, gpu_ids=self.gpu_ids, init_type=opt.init_type)
                self.modules['edge_trans_net'] = self.edge_trans_net
            if opt.use_color:
                self.color_trans_net = networks.define_feature_fusion_network(name='FeatureTransformNetwork', feat_nc=opt.color_nof, guide_nc=opt.shape_nof, nblocks=opt.ftn_nblocks,
                    feat_size=opt.feat_size_lr, norm=opt.norm, gpu_ids=self.gpu_ids, init_type=opt.init_type)
                self.modules['color_trans_net'] = self.color_trans_net

        # netG
        self.netG = networks.define_generator(opt)
        self.modules['netG'] = self.netG

        # netD
        if self.is_train:
            self.netD = networks.define_D(opt)
            self.modules['netD'] = self.netD

        ###################################
        # load weights
        ###################################
        if self.is_train:
            if opt.continue_train:
                for label, net in self.modules.iteritems():
                    self.load_network(net, label, opt.which_epoch)
            else:
                if opt.which_model_init != 'none':
                    # load pretrained entire model
                    for label, net in self.modules.iteritems():
                        self.load_network(net, label, 'latest', opt.which_model_init, forced=False)
                else:
                    # load pretrained encoder
                    if opt.which_model_netG != 'unet' and opt.pretrain_shape:
                        self.load_network(self.shape_encoder, 'shape_encoder', 'latest', opt.which_model_init_shape_encoder)
                    if opt.use_edge and opt.pretrain_edge:
                        self.load_network(self.edge_encoder, 'edge_encoder', 'latest', opt.which_model_init_edge_encoder)
                    if opt.use_color and opt.pretrain_color:
                        self.load_network(self.color_encoder, 'color_encoder', 'latest', opt.which_model_init_color_encoder)
        else:
            for label, net in self.modules.iteritems():
                if label != 'netD':
                    self.load_network(net, label, opt.which_epoch)

        ###################################
        # prepare for training
        ###################################
        if self.is_train:
            self.fake_pool = ImagePool(opt.pool_size)
            ###################################
            # define loss functions
            ###################################
            self.loss_functions = []
            if opt.which_gan in {'dcgan', 'lsgan'}:
                self.crit_GAN = networks.GANLoss(use_lsgan = opt.which_gan == 'lsgan', tensor = self.Tensor)
                self.loss_functions.append(self.crit_GAN)
            else:
                # WGAN loss will be calculated in self.backward_D_wgangp and self.backward_G
                self.crit_GAN = None

            self.crit_L1 = nn.L1Loss()
            self.loss_functions.append(self.crit_L1)

            if self.opt.loss_weight_vgg > 0:
                self.crit_vgg = networks.VGGLoss(self.gpu_ids)
                self.loss_functions.append(self.crit_vgg)

            if self.opt.G_output_seg:
                self.crit_CE = nn.CrossEntropyLoss()
                self.loss_functions.append(self.crit_CE)

            self.crit_psnr = networks.SmoothLoss(networks.PSNR())
            self.loss_functions.append(self.crit_psnr)
            ###################################
            # create optimizers
            ###################################
            self.schedulers = []
            self.optimizers = []

            # G optimizer
            G_module_list = ['shape_encoder', 'edge_encoder', 'color_encoder', 'netG']
            G_param_groups = [{'params': self.modules[m].parameters()} for m in G_module_list if m in self.modules]
            self.optim_G = torch.optim.Adam(G_param_groups, lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optim_G)
            # D optimizer
            self.optim_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optim_D)
            # feature transfer network optimizer
            FTN_module_list = ['edge_trans_net', 'color_trans_net']
            FTN_param_groups = [{'params': self.modules[m].parameters()} for m in FTN_module_list if m in self.modules]
            if len(FTN_param_groups) > 0:
                self.optim_FTN = torch.optim.Adam(FTN_param_groups, lr=opt.lr_FTN, betas=(0.9, 0.999))
                self.optimizers.append(self.optim_FTN)
            else:
                self.optim_FTN = None
            # schedulers
            for optim in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optim, opt))

    def set_input(self, data):
        self.input['img'] = self.Tensor(data['img'].size()).copy_(data['img'])
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
            if isinstance(v, torch.Tensor):
                self.input[k] = Variable(v)

    def forward(self, mode='normal', check_grad=False):
        '''
        mode:
            - normal: encoder -> netG
            - trans: encoder -> feat_trans_net -> netG
            - dual: both above
        '''
        ###################################
        # prepare for checking grad
        ###################################
        if check_grad:
            self.input['seg_mask'].requires_grad=True
            self.input['lm_map'].requires_grad=True
            self.input['edge_map'].requires_grad=True
            self.input['color_map'].requires_grad=True
        ###################################
        # encode shape, edge and color
        ###################################
        # shape repr and shape feat
        self.output['shape_repr'] = self.get_shape_repr(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'])
        # edge feat
        if self.opt.use_edge:
            self.output['edge_feat'] = self.encode_edge(self.input['edge_map'], self.output['shape_repr'])
        # color feat
        if self.opt.use_color:
            self.output['color_feat'] = self.encode_color(self.input['color_map'], self.output['shape_repr'])

        if self.opt.which_model_netG == 'decoder':
            # extract shape feat -> feat concat -> generate image
            self.output['shape_feat'] = self.encode_shape(self.output['shape_repr'])
            feat = [self.output[k] for k in ['shape_feat', 'edge_feat', 'color_feat'] if k in self.output]
            self.output['feat'] = self.align_and_concat(feat, self.opt.feat_size_lr)
        elif self.opt.which_model_netG == 'unet':
            # generate image & shape feat at the same time
            feat = [self.output[k] for k in ['edge_feat', 'color_feat'] if k in self.output]
            if len(feat) > 0:
                feat = self.align_and_concat(feat, self.opt.feat_size_lr)
            else:
                feat = None
            self.output['img_fake_raw'], self.output['seg_pred'], self.output['shape_feat'] = self.generate_image(self.output['shape_repr'], feat)
            self.output['feat'] = feat
        ###################################
        # feature fusion
        ###################################
        if self.opt.ftn_model != 'none':
            self.transfer_feature(detach=True)
            if self.opt.which_model_netG == 'decoder':
                feat_trans = [self.output[k] for k in ['shape_feat_trans', 'edge_feat_trans', 'color_feat_trans'] if k in self.output]
            elif self.opt.which_model_netG == 'unet':
                feat_trans = [self.output[k] for k in ['edge_feat_trans', 'color_feat_trans'] if k in self.output]
            self.output['feat_trans'] = self.align_and_concat(feat_trans, self.opt.feat_size_lr)
        ###################################
        # netG
        ###################################
        if mode in {'normal', 'dual'}:
            if self.opt.which_model_netG == 'decoder':
                if self.opt.G_shape_guided:
                    shape_guide = F.adaptive_avg_pool(self.output['shape_repr'], self.opt.feat_size_hr)
                    self.output['img_fake_raw'], self.output['seg_pred'], _ = self.generate_image(self.output['feat'], shape_guide)
                else:
                    self.output['img_fake_raw'], self.output['seg_pred'], _ = self.generate_image(self.output['feat'], None)
            elif self.opt.which_model_netG == 'unet':
                pass
            self.output['img_fake'] = self.mask_image(self.output['img_fake_raw'], self.input['seg_map'], self.input['img'])
        if mode in {'trans', 'dual'}:
            if self.opt.which_model_netG == 'decoder':
                if self.opt.G_shape_guided:
                    shape_guide = F.adaptive_avg_pool(self.output['shape_repr'], self.opt.feat_size_hr)
                    self.output['img_fake_trans_raw'], self.output['seg_pred_trans'], _ = self.generate_image(self.output['feat_trans'], shape_guide)
                else:
                    self.output['img_fake_trans_raw'], self.output['seg_pred_trans'], _ = self.generate_image(self.output['feat_trans'], None)
            elif self.opt.which_model_netG == 'unet':
                self.output['img_fake_trans_raw'], self.output['seg_pred_trans'], _ = self.generate_image(self.output['shape_repr'], self.output['feat_trans'])

            self.output['img_fake_trans'] = self.mask_image(self.output['img_fake_trans_raw'], self.input['seg_map'], self.input['img'])
        self.output['img_real'] = self.output['img_real_raw'] = self.input['img']

    def test(self, mode='normal'):
        if float(torch.__version__[0:3]) >= 0.4:
            with torch.no_grad():
                self.forward(mode=mode, check_grad=False)
        else:
            for k,v in self.input.iteritems():
                if isinstance(v, Variable):
                    v.volatile = True
            self.forward(mode=mode, check_grad=False)

    def backward_D(self):
        # PSNR
        self.output['PSNR'] = self.crit_psnr(self.output['img_fake'], self.output['img_real'])
        # fake
        repr_fake = self.get_sample_repr_for_D('fake', detach_image=True)
        repr_fake = self.fake_pool.query(repr_fake.data)
        pred_fake = self.netD(repr_fake)
        self.output['loss_D_fake'] = self.crit_GAN(pred_fake, False)
        # real
        repr_real = self.get_sample_repr_for_D('real')
        pred_real = self.netD(repr_real)
        self.output['loss_D_real'] = self.crit_GAN(pred_real, True)
        # combine loss
        self.output['loss_D'] = (self.output['loss_D_real'] + self.output['loss_D_fake']) * 0.5 * self.opt.loss_weight_GAN
        self.output['loss_D'].backward()

    def backward_D_wgangp(self):
        raise NotImplementedError('WGAN-GP not supported!')

    def backward_G(self, mode='normal'):
        # check forward mode
        if mode in {'normal', 'dual'}:
            repr_fake = self.get_sample_repr_for_D('fake', detach_image=False)
            img_fake = self.output['img_fake']
        elif mode == 'trans':
            repr_fake = self.get_sample_repr_for_D('trans', detach_image=False)
            img_fake = self.output['img_fake_trans']
        self.output['loss_G'] = 0
        # GAN loss
        pred_fake = self.netD(repr_fake)
        self.output['loss_G_GAN'] = self.crit_GAN(pred_fake, True)
        self.output['loss_G'] += self.output['loss_G_GAN'] * self.opt.loss_weight_GAN
        # L1 loss
        self.output['loss_G_L1'] = self.crit_L1(img_fake, self.output['img_real'])
        self.output['loss_G'] += self.output['loss_G_L1'] * self.opt.loss_weight_L1
        # VGG Loss
        if self.opt.loss_weight_vgg > 0:
            self.output['loss_G_VGG'] = self.crit_vgg(img_fake, self.output['img_real'])
            self.output['loss_G'] += self.output['loss_G_VGG'] * self.opt.loss_weight_vgg
        # segmentation prediction loss
        if self.opt.G_output_seg:
            assert self.output['seg_pred'] is not None
            self.output['seg_ref'] = self.get_shape_repr(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'], shape_encode='seg')
            self.output['loss_G_seg'] = self.calc_seg_loss(self.output['seg_pred'], self.output['seg_ref'])
            self.output['loss_G'] += self.output['loss_G_seg'] * self.opt.loss_weight_seg
        # backward
        self.output['loss_G'].backward()

    def backward_G_grad_check(self, mode='normal'):
        # check forward mode
        if mode in {'normal', 'dual'}:
            repr_fake = self.get_sample_repr_for_D('fake', detach_image=False)
            img_fake = self.output['img_fake']
        elif mode == 'trans':
            repr_fake = self.get_sample_repr_for_D('trans', detach_image=False)
            img_fake = self.output['img_fake_trans']

        img_fake.retain_grad()
        self.output['loss_G'] = 0
        # GAN loss
        pred_fake = self.netD(repr_fake)
        self.output['loss_G_GAN'] = self.crit_GAN(pred_fake, True)
        (self.output['loss_G_GAN'] * self.opt.loss_weight_GAN).backward(retain_graph=True)
        self.output['loss_G'] += self.output['loss_G_GAN'] * self.opt.loss_weight_GAN
        self.output['grad_G_GAN'] = (img_fake.grad).norm()
        grad = img_fake.grad.clone()
        # L1 loss
        self.output['loss_G_L1'] = self.crit_L1(img_fake, self.output['img_real'])
        (self.output['loss_G_L1'] * self.opt.loss_weight_L1).backward(retain_graph=True)
        self.output['loss_G'] += self.output['loss_G_L1'] * self.opt.loss_weight_L1
        self.output['grad_G_L1'] = (img_fake.grad - grad).norm()
        grad = img_fake.grad.clone()
        # segmentation prediction loss
        if self.opt.G_output_seg:
            assert self.output['seg_pred'] is not None
            self.output['seg_ref'] = self.get_shape_repr(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'], shape_encode='seg')
            self.output['loss_G_seg'] = self.calc_seg_loss(self.output['seg_pred'], self.output['seg_ref'])
            (self.output['loss_G_seg'] * self.opt.loss_weight_seg).backward(retain_graph=True)
            self.output['loss_G'] += self.output['loss_G_seg'] * self.opt.loss_weight_seg
        # VGG Loss
        if self.opt.loss_weight_vgg > 0:
            self.output['loss_G_VGG'] = self.crit_vgg(img_fake, self.output['img_real'])
            (self.output['loss_G_VGG'] * self.opt.loss_weight_vgg).backward()
            self.output['loss_G'] += self.output['loss_G_VGG'] * self.opt.loss_weight_vgg
            self.output['grad_G_VGG'] = (img_fake.grad - grad).norm()
        # gradient of input channels
        self.output['grad_seg'] = self.input['seg_mask'].grad.norm() if self.input['seg_mask'].grad is not None else Variable(torch.zeros(1))
        self.output['grad_edge'] = self.input['edge_map'].grad.norm() if self.input['edge_map'].grad is not None else Variable(torch.zeros(1))
        self.output['grad_color'] = self.input['color_map'].grad.norm() if self.input['color_map'].grad is not None else Variable(torch.zeros(1))

    def backward_trans(self):
        '''
        compute loss and back propagate gradient for feat_trans_net
        '''
        # check output
        for output in ['feat_trans', 'img_fake_trans', 'feat', 'img_fake']:
            assert output in self.output
        self.output['loss_trans'] = 0
        # feature level distance loss
        self.output['loss_trans_feat'] = self.crit_L1(self.output['feat_trans'], self.output['feat'].detach())
        self.output['loss_trans'] += self.output['loss_trans_feat'] * self.opt.loss_weight_trans_feat
        # image level distance loss
        self.output['loss_trans_img'] = self.crit_L1(self.output['img_fake_trans'], self.output['img_fake'].detach())
        self.output['loss_trans'] += self.output['loss_trans_img'] * self.opt.loss_weight_trans_img
        # backward
        self.output['loss_trans'].backward()

    def optimize_parameters(self, train_D = True, train_G = True, check_grad = False):
        # forward
        if self.opt.ftn_model == 'none':
            fwd_mode = 'normal'
        else:
            fwd_mode = 'dual'
        self.output = {} # clear previous output
        self.forward(fwd_mode, check_grad)
        # optimize D
        self.optim_D.zero_grad()
        self.backward_D()
        if train_D:
            self.optim_D.step()
        # optimize G
        self.optim_G.zero_grad()
        if check_grad:
            self.backward_G_grad_check(fwd_mode)
        else:
            self.backward_G(fwd_mode)
        if train_G:
            self.optim_G.step()
        # optimize feature transfer network
        if self.opt.ftn_model != 'none':
            self.optim_FTN.zero_grad()
            self.backward_trans()
            self.optim_FTN.step()

    def get_shape_repr(self, lm_map, seg_mask, edge_map, shape_encode=None):
        if shape_encode is None:
            shape_encode = self.opt.shape_encode
        if shape_encode == 'lm':
            shape_repr = lm_map
        elif shape_encode == 'seg':
            shape_repr = seg_mask
        elif shape_encode == 'lm+seg':
            shape_repr = torch.cat((lm_map, seg_mask), dim = 1)
        elif shape_encode == 'seg+e':
            shape_repr = torch.cat((seg_mask, edge_map), dim = 1)
        elif shape_encode == 'lm+seg+e':
            shape_repr = torch.cat((lm_map, seg_mask, edge_map), dim = 1)
        elif shape_encode == 'e':
            shape_repr = edge_map
        elif shape_encode == 'reduce_seg':
            shape_repr = torch.cat((seg_mask[:,0:3], seg_mask[:,3::].max(dim=1, keepdim=True)[0]), dim=1)
        return shape_repr

    def encode_shape(self, shape_repr):
        if self.opt.which_model_netG != 'unet':
            return self.shape_encoder(shape_repr)
        else:
            return self.netG(shape_repr, None, 'encode')

    def encode_edge(self, input, shape_repr):
        if self.opt.edge_shape_guided:
            input = torch.cat((input, shape_repr), dim=1)
        return self.edge_encoder(input)

    def encode_color(self, input, shape_repr):
        if self.opt.color_shape_guided:
            input = torch.cat((input, shape_repr), dim=1)
        return self.color_encoder(input)
    
    def calc_seg_loss(self, seg_pred, seg_ref):
        seg_pred_f = seg_pred.transpose(1,3).contiguous().view(-1,7)
        seg_ref_f = seg_ref.transpose(1,3).contiguous().view(-1,7)
        return self.crit_CE(seg_pred_f, seg_ref_f)

    def generate_image(self, input_1, input_2):
        '''
        for unet generator
            # input_1: shape_repr
            # input_2: edge/color feature
        for decoder generator
            # input_1: feature
            # input_2: HR shape guide
        '''
        if self.opt.which_model_netG == 'unet':
            out, shape_feat = self.netG(input_1, input_2)
        elif self.opt.which_model_netG == 'decoder':
            out = self.netG(input_1, input_2)
            shape_feat = None
        if self.opt.G_output_seg:
            assert out.size(1) == 10
            img_out = F.tanh(out[:,0:3])
            seg_out = out[:,3::]
        else:
            img_out = F.tanh(out)
            seg_out = None
        return img_out, seg_out, shape_feat
    
    def align_and_concat(self, inputs, size):
        # print(size)
        if isinstance(size, int):
            size = (size, size)
        for i, x in enumerate(inputs):
            if x.size(2)!=size[0] or x.size(3)!=size[1]:
                inputs[i] = F.upsample(x, size, mode='bilinear')
        output = torch.cat(inputs, dim=1)
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

    def transfer_feature(self, detach=True):
        # prepare shape feat as guide
        if self.is_train and self.opt.affine_aug:
            self.output['shape_repr_aug'] = self.get_shape_repr(self.input['lm_map_aug'], self.input['seg_mask_aug'], self.input['edge_map_aug'])
            self.output['shape_feat_aug'] = self.encode_shape(self.output['shape_repr_aug'])
        self.output['shape_feat_trans'] = self.output['shape_feat'].detach() if detach else self.output['shape_feat']
        # transfer edge feature
        if self.opt.use_edge:
            if self.is_train and self.opt.affine_aug:
                self.output['edge_feat_aug'] = self.encode_edge(self.input['edge_map_aug'], self.output['shape_repr_aug'])
                input_feat = self.output['edge_feat_aug']
                input_guide = self.output['shape_feat_aug']
                output_guide = self.output['shape_feat']
            else:
                input_feat = self.output['edge_feat']
                input_guide = output_guide = self.output['shape_feat']
            
            if detach:
                input_feat, input_guide, output_guide = input_feat.detach(), input_guide.detach(), output_guide.detach()

            self.output['edge_feat_trans'] = self.edge_trans_net(input_feat, input_guide, output_guide)
        # transfer color feature
        if self.opt.use_color:
            if self.is_train and self.opt.affine_aug:
                self.output['color_feat_aug'] = self.encode_color(self.input['color_map_aug'], self.output['shape_repr_aug'])
                input_feat = self.output['color_feat_aug']
                input_guide = self.output['shape_feat_aug']
                output_guide = self.output['shape_feat']
            else:
                input_feat = self.output['color_feat']
                input_guide = output_guide = self.output['shape_feat']
            
            if detach:
                input_feat, input_guide, output_guide = input_feat.detach(), input_guide.detach(), output_guide.detach()

            self.output['color_feat_trans'] = self.color_trans_net(input_feat, input_guide, output_guide)

    def get_sample_repr_for_D(self, sample_type='real', detach_image=False):
        '''
        concat image (real/fake) and conditions to form representation of a sample
            - sample_type: 'real', 'fake' or 'trans'
        '''
        if sample_type == 'real':
            img = self.output['img_real']
        elif sample_type == 'fake':
            img = self.output['img_fake']
        elif sample_type == 'trans':
            img = self.output['img_fake_trans']

        if detach_image:
            img = img.detach()

        if self.opt.D_no_cond:
            repr = img
        else:
            repr = [img, self.output['shape_repr']]
            if self.opt.use_edge:
                repr.append(self.input['edge_map'])
            if self.opt.use_color:
                repr.append(self.input['color_map'])
            repr = self.align_and_concat(repr, img.size()[2:4])
        return repr

    def get_current_errors(self):
        # losses
        errors = OrderedDict([
            ('D_GAN', self.output['loss_D'].data.item()),
            ('G_GAN', self.output['loss_G_GAN'].item()),
            ('G_L1', self.output['loss_G_L1'].item())
            ])

        if 'loss_G_VGG' in self.output:
            errors['G_VGG'] = self.output['loss_G_VGG'].item()
        if 'loss_G_seg' in self.output:
            errors['G_seg'] = self.output['loss_G_seg'].item()

        if 'loss_trans_feat' in self.output:
            errors['T_feat'] = self.output['loss_trans_feat'].item()
        if 'loss_trans_img' in self.output:
            errors['T_img'] = self.output['loss_trans_img'].item()
        if 'PSNR' in self.output:
            errors['PSNR'] = self.crit_psnr.smooth_loss(clear=True)

        # gradients
        grad_list = ['grad_G_GAN', 'grad_G_L1', 'grad_G_VGG', 'grad_seg', 'grad_edge', 'grad_color']
        for name in grad_list:
            if name in self.output:
                errors[name] = self.output[name].item()
        
        return errors

    def get_current_visuals(self):
        visuals = OrderedDict([
            ('img_real', self.output['img_real'].data.cpu()),
            ('img_fake', self.output['img_fake'].data.cpu()),
            ('img_real_raw', self.output['img_real_raw'].data.cpu()),
            ('img_fake_raw', self.output['img_fake_raw'].data.cpu()),
            ('seg_map', self.input['seg_map'].data.cpu()),
            ('landmark_heatmap', self.input['lm_map'].data.cpu()),
            ('edge_map', self.input['edge_map'].data.cpu()),
            ('color_map', self.input['color_map'].data.cpu())
            ])

        for name in ['img_fake_trans', 'img_fake_trans_raw']:
            if name in self.output:
                visuals[name] = self.output[name].data.cpu()
        
        for name in ['seg_ref','seg_pred', 'seg_pred_trans']:
            if name in self.output and self.output[name] is not None:
                visuals[name] = self.output[name].data.cpu()
        return visuals

    def save(self, label):
        for name, net in self.modules.iteritems():
            self.save_network(net, name, label, self.gpu_ids)

