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

class MultimodalDesignerGAN_V3(BaseModel):
    '''
    modules:
        encoders: Es, Ee and Ec
        netG: generator (decoder)
        netD: discriminator
        netD_feat: feature level discriminator?
    '''
    def name(self):
        return 'MultimodalDesignerGAN_V3'

    def initialize(self, opt):
        super(MultimodalDesignerGAN_V3, self).initialize(opt)
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
                        self.load_network(net, label, 'latest', opt.which_model_init)
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
            # schedulers
            for optim in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optim, opt))

    def set_input(self, data):
        input_list = [
            # information from image_tar (provide segmentation map)
            'img',          # image_tar
            'seg_map',      # seg map (c=1)
            'seg_mask',     # seg mask (c=7)
            'flx_seg_mask', # flexible seg mask (c=7)
            'edge_map',     # edge_map (c=1)
            'color_map',    # color_map (c=3 or 6)
            # information form image_edge (provide edge)
            'img_edge',     # image_edge
            'edge_map_src', # warped edge
            # information from image_color (provide color)
            'img_color',    # image_color
            'color_map_src',# color map
        ]
        
        for name in input_list:
            self.input[name] = Variable(self.Tensor(data[name].size()).copy_(data[name]))
        # other input
        self.input['id'] = data['id'] # tuple of (tar_id, edge_id, color_id)


    def forward(self, mode='normal'):
        '''
        mode:
            - normal: reconstruction + fusion
        '''
        bsz = self.input['img'].size(0)
        ###################################
        # encode shape, edge and color
        ###################################
        G_input_feat = []
        # shape repr
        self.output['shape_repr_rec'] = self.get_shape_repr(self.input['seg_mask'], self.input['edge_map'], self.input['flx_seg_mask'], self.input['img'])
        shape_repr = torch.cat([self.output['shape_repr_rec'], self.output['shape_repr_rec']], dim=0)
        # shape feat (when use decoder G)
        if self.opt.which_model_netG == 'decoder':
            self.output['shape_feat_rec'] = self.encode_shape(self.output['shape_repr_rec'])
            self.output['shape_feat_gen'] = self.output['shape_feat_rec']
            G_input_feat.append(torch.cat([self.output['shape_feat_rec'], self.output['shape_feat_gen']], dim=0))
        # edge feat
        if self.opt.use_edge:
            self.output['edge_feat_rec'] = self.encode_edge(self.input['edge_map'])
            self.output['edge_feat_gen'] = self.encode_edge(self.input['edge_map_src'])
            G_input_feat.append(torch.cat([self.output['edge_feat_rec'], self.output['edge_feat_gen']], dim=0))
        # color feat
        if self.opt.use_color:
            self.output['color_feat_rec'] = self.encode_color(self.input['color_map'])
            self.output['color_feat_gen'] = self.encode_color(self.input['color_map_src'])
            G_input_feat.append(torch.cat([self.output['color_feat_rec'], self.output['color_feat_gen']], dim=0))
        
        # integrate feature
        if len(G_input_feat) > 0:
            G_input_feat = self.align_and_concat(G_input_feat, self.opt.feat_size_lr)
        else:
            G_input_feat = None
        
        ###################################
        # netG
        ###################################
        if self.opt.shape_encode in {'seg', 'reduced_seg', 'flx_seg'}:
            self.output['shape_input'] = self.output['shape_repr_rec'] # for visualization
        else:
            self.output['shape_input'] = self.input['seg_map']

        self.output['img_real'] = self.output['img_real_raw'] = self.input['img']

        if self.opt.which_model_netG == 'decoder':
            if self.opt.G_shape_guided:
                shape_guide = F.adaptive_avg_pool(shape_repr, self.opt.feat_size_hr)
            else:
                shape_guide = None
            img_out, seg_pred, _ = self.generate_image(G_input_feat, shape_guide)
        elif self.opt.which_model_netG == 'unet':
            img_out, seg_pred, shape_feat = self.generate_image(shape_repr, G_input_feat)
            self.output['shape_feat_rec'] = shape_feat[0:bsz]
            self.output['shape_feat_gen'] = shape_feat[bsz::]
        else:
            raise Exception('invalid netG type: %s' % self.opt.which_model_netG)

        self.output['img_raw_rec'] = img_out[0:bsz]
        self.output['img_raw_gen'] = img_out[bsz::]

        if self.opt.G_output_seg:
            self.output['seg_pred_rec'] = seg_pred[0:bsz]
            self.output['seg_pred_gen'] = seg_pred[bsz::]

        self.output['img_rec'] = self.mask_image(self.output['img_raw_rec'], self.input['seg_map'], self.input['img'])
        self.output['img_gen'] = self.mask_image(self.output['img_raw_gen'], self.input['seg_map'], self.input['img'])


    def test(self, mode='normal'):
        if float(torch.__version__[0:3]) >= 0.4:
            with torch.no_grad():
                self.forward(mode=mode)
        else:
            for k,v in self.input.iteritems():
                if isinstance(v, Variable):
                    v.volatile = True
            self.forward(mode=mode)

    def backward_D(self):
        # PSNR
        self.output['PSNR'] = self.crit_psnr(self.output['img_rec'], self.output['img_real'])
        if self.opt.D_output_type == 'binary':
            # real
            repr_real = self.get_sample_repr_for_D('real')
            pred_real = self.netD(repr_real)
            self.output['loss_D_real'] = self.crit_GAN(pred_real, True)
            # rec
            repr_rec = self.get_sample_repr_for_D('rec', detach=True)
            pred_rec = self.netD(repr_rec)
            self.output['loss_D_rec'] = self.crit_GAN(pred_rec, False)
            # gen
            repr_gen = self.get_sample_repr_for_D('gen', detach=True)
            pred_gen = self.netD(repr_gen)
            self.output['loss_D_gen'] = self.crit_GAN(pred_gen, False)
            # combine loss
            self.output['loss_D'] =(self.output['loss_D_real']*0.5 + self.output['loss_D_rec']*0.25 + self.output['loss_D_gen']*0.25)* self.opt.loss_weight_GAN
        elif self.opt.D_output_type == 'class':
            raise NotImplementedError()
        self.output['loss_D'].backward()

    def backward_D_wgangp(self):
        raise NotImplementedError('WGAN-GP not supported!')

    def backward_G(self, mode='normal'):
        self.output['loss_G'] = 0
        # GAN Loss
        repr_rec = self.get_sample_repr_for_D('rec', detach=False)
        pred_rec = self.netD(repr_rec)
        repr_gen = self.get_sample_repr_for_D('gen', detach=False)
        pred_gen = self.netD(repr_gen)
        if self.opt.D_output_type == 'binary':
            self.output['loss_G_GAN_rec'] = self.crit_GAN(pred_rec, True)
            self.output['loss_G_GAN_gen'] = self.crit_GAN(pred_gen, True)
            self.output['loss_G_GAN'] = (self.output['loss_G_GAN_rec'] + self.output['loss_G_GAN_gen']) * 0.5
            self.output['loss_G'] += self.output['loss_G_GAN'] * self.opt.loss_weight_GAN
        elif self.opt.D_output_type == 'class':
            raise NotImplementedError()
        # L1 loss
        self.output['loss_G_L1'] = self.crit_L1(self.output['img_rec'], self.output['img_real'])
        self.output['loss_G'] += self.output['loss_G_L1'] * self.opt.loss_weight_L1
        # VGG loss
        if self.opt.loss_weight_vgg > 0:
            self.output['loss_G_VGG'] = self.crit_vgg(self.output['img_rec'], self.output['img_real'])
            self.output['loss_G'] += self.output['loss_G_VGG'] * self.opt.loss_weight_vgg
            self.output['loss_G_VGG_gen'] = self.crit_vgg(self.output['img_gen'], self.output['img_real'])
            self.output['loss_G'] += self.output['loss_G_VGG_gen'] * self.opt.loss_weight_vgg_gen
        # segmentation loss
        if self.opt.G_output_seg:
            self.output['loss_G_seg_rec'] = self.calc_seg_loss(self.output['seg_pred_rec'], self.input['seg_map'])
            self.output['loss_G'] += self.output['loss_G_seg_rec'] * self.opt.loss_weight_seg
            self.output['loss_G_seg_gen'] = self.calc_seg_loss(self.output['seg_pred_gen'], self.input['seg_map'])
            self.output['loss_G'] += self.output['loss_G_seg_gen'] * self.opt.loss_weight_seg_gen
        self.output['loss_G'].backward()
       


    def backward_G_grad_check(self, mode='normal'):
        # Not implemented
        self.backward_G(mode)

    def optimize_parameters(self, train_D = True, train_G = True, check_grad = False):
        mode = 'normal'
        # forward
        self.output = {} # clear previous output
        self.forward(mode)
        # optimize D
        self.optim_D.zero_grad()
        self.backward_D()
        if train_D:
            self.optim_D.step()
        # optimize G
        self.optim_G.zero_grad()
        if check_grad:
            self.backward_G_grad_check(mode)
        else:
            self.backward_G(mode)
        if train_G:
            self.optim_G.step()

    def get_shape_repr(self, seg_mask, edge_map, flx_seg_mask, img, shape_encode=None, shape_with_face=None):
        if shape_encode is None:
            shape_encode = self.opt.shape_encode
        if shape_with_face is None:
            shape_with_face = self.opt.shape_with_face

        if shape_encode == 'seg':
            shape_repr = seg_mask
        elif shape_encode == 'seg+e':
            shape_repr = torch.cat((seg_mask, edge_map), dim = 1)
        elif shape_encode == 'e':
            shape_repr = edge_map
        elif shape_encode == 'reduced_seg':
            shape_repr = torch.cat((seg_mask[:,0:3], seg_mask[:,3::].max(dim=1, keepdim=True)[0]), dim=1).detach()
        elif shape_encode == 'flx_seg':
            shape_repr = flx_seg_mask
        else:
            raise Exception('invalid shape_encode mode: %s'%shape_encode)

        if shape_with_face:
            face_mask = seg_mask[:,1:3].sum(dim=1, keepdim=True)
            shape_repr = torch.cat((shape_repr, face_mask * img), dim=1).detach()
        return shape_repr

    def encode_shape(self, shape_repr):
        if self.opt.which_model_netG != 'unet':
            return self.shape_encoder(shape_repr)
        else:
            return self.netG(shape_repr, None, 'encode')

    def encode_edge(self, input):
        if self.opt.edge_shape_guided:
            # input = torch.cat((input, shape_repr), dim=1)
            raise NotImplementedError('V3 model don not support shape guided encoding')
        return self.edge_encoder(input)

    def encode_color(self, input):
        if self.opt.color_shape_guided:
            # input = torch.cat((input, shape_repr), dim=1)
            raise NotImplementedError('V3 model don not support shape guided encoding')
        return self.color_encoder(input)
    
    def calc_seg_loss(self, seg_pred, seg_map):
        seg_pred_f = seg_pred.transpose(1,3).contiguous().view(-1,7)
        seg_map_f = seg_map.transpose(1,3).contiguous().view(-1).long()
        return self.crit_CE(seg_pred_f, seg_map_f)

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
            if self.opt.G_output_region:
                assert out.size(1) == 28 #(7+3*7)
                region_out = out[:,0:21]
                seg_out = out[:,21:28]
                img_out = F.tanh(self.merge_region(region_out, seg_out))
            else:
                assert out.size(1) == 10
                img_out = F.tanh(out[:,0:3])
                seg_out = out[:,3::]
        else:
            img_out = F.tanh(out)
            seg_out = None
        return img_out, seg_out, shape_feat

    def merge_region(self, region_out, seg_out):
        b, c, h, w = region_out.size()
        cs = seg_out.size(1)
        assert c == 3*cs
        region_out = region_out.view(b, 3, cs, h, w)
        seg_out = F.softmax(seg_out, dim=1).view(b,1,cs,h,w)
        img_out = (region_out * seg_out).sum(2)
        return img_out
    
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

    def get_sample_repr_for_D(self, sample_type='real', detach=False):
        '''
        concat image (real/fake) and conditions to form representation of a sample
            - sample_type: 'real', 'fake' or 'trans'
        '''
        if sample_type == 'real':
            img = self.output['img_real']
            shape = self.output['shape_repr_rec']
            edge = self.input['edge_map']
            color = self.input['color_map']
        elif sample_type == 'rec':
            img = self.output['img_rec']
            shape = self.output['shape_repr_rec']
            edge = self.input['edge_map']
            color = self.input['color_map']
        elif sample_type == 'gen':
            img = self.output['img_gen']
            shape = self.output['shape_repr_rec']
            edge = self.input['edge_map_src']
            color = self.input['color_map_src']

        if self.opt.D_cond == 'none':
            repr = img
        elif self.opt.D_cond == 'shape':
            repr = [img, shape]
            repr = self.align_and_concat(repr, img.size()[2:4])
        elif self.opt.D_cond == 'std':
            repr = [img, shape]
            if self.opt.use_edge:
                repr.append(edge)
            if self.opt.use_color:
                repr.append(color)
            repr = self.align_and_concat(repr, img.size()[2:4])

        if detach:
            repr = repr.detach()
        return repr

    def get_current_errors(self):
        # losses
        errors = OrderedDict([
            ('D_GAN', self.output['loss_D'].data.item()),
            ('G_GAN', self.output['loss_G_GAN'].item()),
            ('G_GAN_rec', self.output['loss_G_GAN_rec'].item()),
            ('G_GAN_gen', self.output['loss_G_GAN_gen'].item()),
            ('G_L1', self.output['loss_G_L1'].item())
            ])

        if 'loss_G_VGG' in self.output:
            errors['G_VGG'] = self.output['loss_G_VGG'].item()
        if 'loss_G_seg_rec' in self.output:
            errors['G_seg_rec'] = self.output['loss_G_seg_rec'].item()
        if 'loss_G_seg_gen' in self.output:
            errors['G_seg_gen'] = self.output['loss_G_seg_gen'].item()
        if 'PSNR' in self.output:
            errors['PSNR'] = self.crit_psnr.smooth_loss(clear=True)

        # gradients
        grad_list = ['grad_G_GAN', 'grad_G_L1', 'grad_G_VGG']
        for name in grad_list:
            if name in self.output:
                errors[name] = self.output[name].item()
        
        return errors

    def get_current_visuals(self, mode='normal'):
        '''
        mode:
            - normal: important input and output
            - input: only visualize input (for checking)
        '''

        if mode == 'normal':
            # visual[name] = (data, type)
            # type in {'rgb', 'seg', 'segf', 'edge', 'color'}
            visuals = OrderedDict([
                # input image
                ('img_real', (self.input['img'].data.cpu(), 'rgb')),
                ('img_edge', (self.input['img_edge'].data.cpu(), 'rgb')),
                ('img_color', (self.input['img_color'].data.cpu(), 'rgb')),
                # reconstruction
                ('shape_input', (self.output['shape_input'].data.cpu(), 'segf')),
                ('edge_map', (self.input['edge_map'].data.cpu(), 'edge')),
                ('color_map', (self.input['color_map'].data.cpu(), 'color')),
                ('img_rec', (self.output['img_rec'].data.cpu(), 'rgb')),
                ('seg_pred_rec', (self.output['seg_pred_rec'].data.cpu(), 'seg')),
                ('seg_map', (self.input['seg_map'].data.cpu(), 'seg')),
                # generation
                ('edge_map_src', (self.input['edge_map_src'].data.cpu(), 'edge')),
                ('color_map_src', (self.input['color_map_src'].data.cpu(), 'color')),
                ('img_gen', (self.output['img_gen'].data.cpu(), 'rgb')),
                ('seg_pred_gen', (self.output['seg_pred_gen'].data.cpu(), 'seg'))
                ])
            # if self.opt.G_output_seg:
            #     visuals['seg_map'] = (self.input['seg_map'].data.cpu(), 'seg')
            #     visuals['seg_pred_rec'] = (self.output['seg_pred_rec'].data.cpu(), 'seg')
            #     visuals['seg_pred_gen'] = (self.output['seg_pred_gen'].data.cpu(), 'seg')

        elif mode == 'input':
            visuals = OrderedDict([
                # shape input (target)
                ('img_real', (self.input['img'].data.cpu(), 'rgb')),
                ('seg_mask', (self.input['seg_mask'].data.cpu(), 'seg')),
                ('flx_seg_mask', (self.input['flx_seg_mask'].data.cpu(), 'seg')),
                ('edge_map', (self.input['edge_map'].data.cpu(), 'edge')),
                ('color_map', (self.input['color_map'].data.cpu(), 'color')),
                # edge input
                ('img_edge', (self.input['img_edge'].data.cpu(), 'rgb')),
                ('edge_map_src', (self.input['edge_map_src'].data.cpu(), 'edge')),
                # color input
                ('img_color', (self.input['img_color'].data.cpu(), 'rgb')),
                ('color_map_src', (self.input['color_map_src'].data.cpu(), 'color')),
                ])

        return visuals

    def save(self, label):
        for name, net in self.modules.iteritems():
            self.save_network(net, name, label, self.gpu_ids)

