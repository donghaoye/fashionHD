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


class MultimodalDesignerGAN(BaseModel):
    def name(self):
        return 'MultimodalDesignerGAN'

    def initialize(self, opt):
        super(MultimodalDesignerGAN, self).initialize(opt)
        ###################################
        # load/define networks
        ###################################
        
        # basic G
        self.netG = networks.define_G(opt)

        # encoders
        self.encoders = {}
        if opt.use_edge:
            self.edge_encoder = networks.define_image_encoder(opt, 'edge')
            self.encoders['edge_encoder'] = self.edge_encoder
        if opt.use_color:
            self.color_encoder = networks.define_image_encoder(opt, 'color')
            self.encoders['color_encoder'] = self.color_encoder
        if opt.use_attr:
            self.attr_encoder, self.opt_AE = network_loader.load_attribute_encoder_net(id = opt.which_model_AE, gpu_ids = opt.gpu_ids)
        
        # basic D and auxiliary Ds
        if self.is_train:
            # basic D
            self.netD = networks.define_D(opt)
            # auxiliary Ds
            self.auxiliaryDs = {}
            if opt.use_edge_D:
                assert opt.use_edge
                self.netD_edge = networks.define_D_from_params(input_nc=opt.edge_nof+3, ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                    n_layers_D=opt.n_layers_D, norm=opt.norm, which_gan='dcgan', init_type=opt.init_type, gpu_ids=opt.gpu_ids)
                self.auxiliaryDs['D_edge'] = self.netD_edge
            if opt.use_color_D:
                assert opt.use_color
                self.netD_color = networks.define_D_from_params(input_nc=opt.color_nof+3, ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                    n_layers_D=opt.n_layers_D, norm=opt.norm, which_gan='dcgan', init_type=opt.init_type, gpu_ids=opt.gpu_ids)
                self.auxiliaryDs['D_color'] = self.netD_color
            if opt.use_attr_D:
                assert opt.use_attr
                attr_nof = opt.n_attr_feat if opt.attr_cond_type in {'feat', 'feat_map'} else opt.n_attr
                self.netD_attr = networks.define_D_from_params(input_nc=attr_nof+3, ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                    n_layers_D=opt.n_layers_D, norm=opt.norm, which_gan='dcgan', init_type=opt.init_type, gpu_ids=opt.gpu_ids)
                self.auxiliaryDs['D_attr'] = self.netD_attr
            # load weights
            if not opt.continue_train:
                if opt.which_model_init != 'none':
                    self.load_network(self.netG, 'G', 'latest', opt.which_model_init)
                    self.load_network(self.netD, 'D', 'latest', opt.which_model_init)
                    for l, net in self.encoders.iteritems():
                        self.load_network(net, l, 'latest', opt.which_model_init)
                    for l, net in self.auxiliaryDs.iteritems():
                        self.load_network(net, l, 'latest', opt.which_model_init)
            else:
                self.load_network(self.netG, 'G', opt.which_epoch)
                self.load_network(self.netD, 'D', opt.which_epoch)
                for l, net in self.encoders.iteritems():
                    self.load_network(net, l, opt.which_epoch)
                for l, net in self.auxiliaryDs.iteritems():
                    self.load_network(net, l, opt.which_epoch)
        else:
            self.load_network(self.netG, 'G', opt.which_epoch)
            for l, net in self.encoders.iteritems():
                self.load_network(net, l, opt.which_epoch)

        if self.is_train:
            self.fake_pool = ImagePool(opt.pool_size)
            ###################################
            # define loss functions and loss buffers
            ###################################
            self.loss_functions = []
            if opt.which_gan in {'dcgan', 'lsgan'}:
                self.crit_GAN = networks.GANLoss(use_lsgan = opt.which_gan == 'lsgan', tensor = self.Tensor)
            else:
                # WGAN loss will be calculated in self.backward_D_wgangp and self.backward_G
                self.crit_GAN = None

            self.loss_functions.append(self.crit_GAN)

            self.crit_L1 = nn.L1Loss()
            self.loss_functions.append(self.crit_L1)

            if self.opt.loss_weight_vgg > 0:
                self.crit_vgg = networks.VGGLoss(self.gpu_ids)
                self.loss_functions.append(self.crit_vgg)

            self.crit_psnr = networks.SmoothLoss(networks.PSNR())
            self.loss_functions.append(self.crit_psnr)
            ###################################
            # create optimizers
            ###################################
            self.schedulers = []
            self.optimizers = []

            # optim_G will optimize parameters of netG and all image encoders (except attr_encoder)
            G_param_groups = [{'params': self.netG.parameters()}]
            for l, net in self.encoders.iteritems():
                G_param_groups.append({'params': net.parameters()})
            self.optim_G = torch.optim.Adam(G_param_groups, lr = opt.lr, betas = (opt.beta1, opt.beta2))
            self.optimizers.append(self.optim_G)
            # optim_D will optimize parameters of netD
            self.optim_D = torch.optim.Adam(self.netD.parameters(), lr = opt.lr_D, betas = (opt.beta1, opt.beta2))
            self.optimizers.append(self.optim_D)
            # optim_D_aux will optimize parameters of auxiliaryDs
            if len(self.auxiliaryDs) > 0:
                aux_D_param_groups = [{'params': net.parameters()} for net in self.auxiliaryDs.values()]
                self.optim_D_aux = torch.optim.Adam(aux_D_param_groups, lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optim_D_aux)
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

    def forward(self, check_grad=False):
        # check grad
        if check_grad:
            self.input['seg_mask'].requires_grad=True
            self.input['lm_map'].requires_grad=True
            if self.opt.affine_aug:
                self.input['edge_map_aug'].requires_grad=True
                self.input['color_map_aug'].requires_grad=True
            else:
                self.input['edge_map'].requires_grad=True
                self.input['color_map'].requires_grad=True
        # compute shape representation
        self.output['shape_repr'] = self.encode_shape(self.input['lm_map'], self.input['seg_mask'], self.input['edge_map'])
        if self.opt.affine_aug:
            self.output['shape_repr_aug'] = self.encode_shape(self.input['lm_map_aug'], self.input['seg_mask_aug'], self.input['edge_map_aug'])

        # compute conditions
        if self.opt.G_cond_nc > 0:
            cond_feat = []
            if self.opt.use_edge:
                edge_map = self.input['edge_map'] if not self.opt.affine_aug else self.input['edge_map_aug']
                shape_repr = self.output['shape_repr'] if not self.opt.affine_aug else self.output['shape_repr_aug']
                guide = self.output['shape_repr']
                self.output['edge_feat'] = self.encode_edge(edge_map, shape_repr, guide)
                cond_feat.append(self.output['edge_feat'])
            if self.opt.use_color:
                color_map = self.input['color_map'] if not self.opt.affine_aug else self.input['color_map_aug']
                shape_repr = self.output['shape_repr'] if not self.opt.affine_aug else self.output['shape_repr_aug']
                guide = self.output['shape_repr']
                self.output['color_feat'] = self.encode_color(color_map, shape_repr, guide)
                cond_feat.append(self.output['color_feat'])
            if self.opt.use_attr:
                # Todo: need affine augmentation for attribute encoder?
                if 'img_for_attr' in self.input:
                    # for convenience of attribute transfer test
                    self.output['attr_feat'] = self.encode_attribute(self.input['img_for_attr'], self.opt.attr_cond_type)
                else:
                    self.output['attr_feat'] = self.encode_attribute(self.input['img'], self.opt.attr_cond_type)
                cond_feat.append(self.output['attr_feat'])
            self.output['cond_feat'] = self.align_and_concat(cond_feat, self.opt.G_cond_size)
            self.output['img_fake_raw'] = self.netG(self.output['shape_repr'], self.output['cond_feat'])
        else:
            self.output['img_fake_raw'] = self.netG(self.output['shape_repr'])
        
        self.output['img_real_raw'] = self.input['img']
        self.output['img_fake'] = self.mask_image(self.output['img_fake_raw'], self.input['seg_map'], self.output['img_real_raw'])
        self.output['img_real'] = self.mask_image(self.output['img_real_raw'], self.input['seg_map'], self.output['img_real_raw'])

    def test(self):
        if float(torch.__version__[0:3]) >= 0.4:
            with torch.no_grad():
                self.forward()
        else:
            for k,v in self.input.iteritems():
                if isinstance(v, Variable):
                    v.volatile = True
            self.forward()
    
    def encode_edge(self, img, shape_repr, guide=None):
        if self.opt.edge_shape_guided:
            input = torch.cat((img, shape_repr), 1)
        else:
            input = img
        
        if self.opt.encoder_type == 'st' and self.opt.target_guided:
            return self.edge_encoder(input, guide)
        else:
            return self.edge_encoder(input)
    
    def encode_color(self, img, shape_repr, guide=None):
        if self.opt.color_shape_guided:
            input = torch.cat((img, shape_repr), 1)
        else:
            input = img

        if self.opt.encoder_type == 'st' and self.opt.target_guided:
            return self.color_encoder(input, guide)
        else:
            return self.color_encoder(input)


    def get_sample_repr_for_D(self, sample_type='real', detach_image=False):
        '''
        concat image (real/fake) and conditions to form representation of a sample
        '''
        if sample_type == 'real':
            img = self.output['img_real']
        else:
            if detach_image:
                img = self.output['img_fake'].detach()
            else:
                img = self.output['img_fake']
        
        if self.opt.D_no_cond:
            repr = img
        else:
            repr = [img, self.output['shape_repr']]
            if self.opt.use_edge:
                repr.append(self.input['edge_map'])
            if self.opt.use_color:
                repr.append(self.input['color_map'])
            repr = torch.cat(repr, dim=1)
        
        return repr
        
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

        # combined loss
        self.output['loss_D'] = (self.output['loss_D_real'] + self.output['loss_D_fake']) * 0.5 * self.opt.loss_weight_GAN
        self.output['loss_D'].backward()
    
    def backward_auxiliary_D(self):
        raise NotImplementedError('backward_auxiliary_D not implemented!')


    def backward_D_wgangp(self):
        ''' optimize netD using wasserstein gan loss with gradient penalty.  '''
        # PSNR
        self.output['PSNR'] = self.crit_psnr(self.output['img_fake'], self.output['img_real'])
        # when using wgan, loss_D_fake(real) means critic output for fake(real) data, instead of loss
        bsz = self.output['img_fake'].size(0)
        # fake
        repr_fake = self.get_sample_repr_for_D('fake', detach_image=True)
        disc_fake = self.netD(repr_fake)
        self.output['loss_D_fake'] = disc_fake.mean()
        # real
        repr_real = self.get_sample_repr_for_D('real')
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
        grad = torch.autograd.grad(outputs = disc_interp.sum(), inputs = repr_interp,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_penalty = ((grad.view(bsz,-1).norm(2,dim=1)-1)**2).mean()
        self.output['loss_gp'] = grad_penalty * self.opt.loss_weight_gp * self.opt.loss_weight_GAN
        self.output['loss_gp'].backward()

    def backward_G(self):
        repr_fake = self.get_sample_repr_for_D('fake', detach_image=False)
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
        # VGG Loss
        if self.opt.loss_weight_vgg > 0:
            self.output['loss_G_VGG'] = self.crit_vgg(self.output['img_fake'], self.output['img_real'])
            self.output['loss_G'] += self.output['loss_G_VGG'] * self.opt.loss_weight_vgg
        # backward
        self.output['loss_G'].backward()


    def backward_G_grad_check(self):
        self.output['img_fake'].retain_grad()
        repr_fake = self.get_sample_repr_for_D('fake', detach_image=False)
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
        # VGG Loss
        if self.opt.loss_weight_vgg > 0:
            self.output['loss_G_VGG'] = self.crit_vgg(self.output['img_fake'], self.output['img_real'])
            (self.output['loss_G_VGG'] * self.opt.loss_weight_vgg).backward()
            self.output['loss_G'] += self.output['loss_G_VGG'] * self.opt.loss_weight_vgg
            self.output['grad_G_VGG'] = (self.output['img_fake'].grad - grad).norm()
        # gradient of input channels
        self.output['grad_seg'] = self.input['seg_mask'].grad.norm() if self.input['seg_mask'].grad is not None else Variable(torch.zeros(1))
        if self.opt.affine_aug:
            self.output['grad_edge'] = self.input['edge_map_aug'].grad.norm() if self.input['edge_map_aug'].grad is not None else Variable(torch.zeros(1))
            self.output['grad_color'] = self.input['color_map_aug'].grad.norm() if self.input['color_map_aug'].grad is not None else Variable(torch.zeros(1))
        else:
            self.output['grad_edge'] = self.input['edge_map'].grad.norm() if self.input['edge_map'].grad is not None else Variable(torch.zeros(1))
            self.output['grad_color'] = self.input['color_map'].grad.norm() if self.input['color_map'].grad is not None else Variable(torch.zeros(1))

    def optimize_parameters(self, train_D = True, train_G = True, check_grad = False):
        # clear previous output
        self.output = {}
        self.forward(check_grad)
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
        if train_G:
            self.optim_G.step()

    def get_current_errors(self):
        errors = OrderedDict([
            ('D_GAN', self.output['loss_D'].data[0]),
            ('G_GAN', self.output['loss_G_GAN'].data[0]),
            ('G_L1', self.output['loss_G_L1'].data[0]),
            ])

        if 'loss_G_VGG' in self.output:
            errors['G_VGG'] = self.output['loss_G_VGG'].data[0]
        if 'loss_gp' in self.output:
            errors['D_GP'] = self.output['loss_gp'].data[0]
        if 'PSNR' in self.output:
            errors['PSNR'] = self.crit_psnr.smooth_loss(clear=True)

        # gradients
        grad_list = ['grad_G_GAN', 'grad_G_L1', 'grad_G_VGG', 'grad_seg', 'grad_edge', 'grad_color']
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
            ('edge_map', self.input['edge_map'].data.clone()),
            ('color_map', self.input['color_map'].data.clone())
            ])

        if self.opt.affine_aug:
            visuals['seg_mask_aug'] = self.input['seg_mask_aug'].data.clone()
            visuals['edge_map_aug'] = self.input['edge_map_aug'].data.clone()
            visuals['color_map_aug'] = self.input['color_map_aug'].data.clone()
        return visuals

    
    def align_and_concat(self, inputs, size):
        inputs = [F.upsample(x, size, mode=self.opt.G_cond_interp) for x in inputs]
        output = torch.cat(inputs, dim=1)
        return output

    def encode_attribute(self, img, output_type = None):        
        if output_type is None:
            output_type = self.opt.attr_cond_type
        v_img = img if isinstance(img, Variable) else Variable(img)

        if output_type is not 'feat_map':
            # When computing "prob", pooling layers in attr_encoder network will be used. So the input image must be
            # resize to the standard size of the encoder (224 for resnet).
            # When only computing feat_map, only convolution layers in attr_encoder will be used. So the input can be
            # arbitrary size
            input_size = 224
            v_img = F.upsample(v_img, size=(input_size, input_size), mode='bilinear')
        
        if self.opt_AE.image_normalize == 'imagenet':
            v_img = self._std_to_imagenet(v_img)

        if output_type == 'feat':
            feat, _ = self.attr_encoder.extract_feat(v_img)
            return feat
        elif output_type == 'feat_map':
            _, feat_map = self.attr_encoder.extract_feat(v_img)
            return feat_map
        elif output_type == 'prob':
            prob, _ = self.attr_encoder(v_img)
            return prob
        elif output_type == 'feat_map':
            _, prob_map = self.attr_encoder(v_img)
            return prob_map

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
       
    def save(self, label):
        # Todo: if self.attr_encoder is jointly trained, also save its parameter
        # Todo: if att_fuse module is added, save its parameters
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        for l, net in self.encoders.iteritems():
            self.save_network(net, l, label, self.gpu_ids)
        for l, net in self.auxiliaryDs.iteritems():
            self.save_network(net, l, label, self.gpu_ids)

