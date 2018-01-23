from __future__ import division, print_function

import torch
import networks
from torch.autograd import Variable
from image_pool import ImagePool
from base_model import BaseModel
from attribute_encoder import AttributeEncoder
from options.attribute_options import TestAttributeOptions

import os
import sys
import numpy as np
import time
from collections import OrderedDict
import util.io as io

def load_attribute_encoder_net(id, gpu_ids, is_train, which_epoch = 'latest'):
    '''
    Load pretrained attribute encoder as a module of GAN model.
    All options for attribute encoder will be loaded from its train_opt.json, except:
        - gpu_ids
        - is_train
        - which_epoch

    Input:
        id (str): ID of attribute encoder model
        gpu_ids: set gpu_ids for attribute model
        is_train: set train/test status for attribute model
    Output:
        net (nn.Module): network of attribute encoder
        opt (namespace): updated attribute encoder options
    '''

    if not id.startswith('AE_'):
        id = 'AE_' + id

    # load attribute encoder options
    fn_opt = os.path.join('checkpoints', id, 'train_opt.json')
    if not os.path.isfile(fn_opt):
        raise ValueError('invalid attribute encoder id: %s' % id)
    opt_var = io.load_json(fn_opt)

    # update attribute encoder options
    opt = TestAttributeOptions().parse(save_to_file = False, display = False)
    for k, v in opt_var.iteritems():
        if k in opt:
            opt.__dict__[k] = v

    opt.is_train = is_train
    opt.gpu_ids = gpu_ids
    opt.which_epoch = which_epoch

    model = AttributeEncoder()
    model.initialize(opt)

    return model.net, opt



class DesignerGAN(BaseModel):
    def name(self):
        return 'DesignerGAN'

    def initialize(self, opt):
        super(DesignerGAN, self).initialize(opt)
        ###################################
        # define tensors
        ###################################

        ###################################
        # load/define networks
        ###################################

        # Todo modify networks.define_G
        # 1. input opt, instead of bunch of parameters
        # 2. add specified generator networks

        self.netG = networks.define_G(opt)
        self.netAE, self.opt_AE = load_attribute_encoder_net(id = opt.AE_id, gpu_ids = opt.gpu_ids, is_train = self.is_train)

        if self.is_train:
            self.netD = networks.define_D(opt)

        if not self.is_train or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.is_train:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.is_train:
            self.fake_pool = ImagePool(opt.pool_size)

        ###################################
        # define loss functions
        ###################################
        self.loss_functions = []

        self.crit_GAN = networks.GANLoss(use_lsgan = not opt.no_lsgan, tensor = self.Tensor)
        self.crit_L1 = nn.L1Loss()

        self.loss_functions.append(self.crit_GAN)
        self.loss_functions.append(self.crit_L1)

        ###################################
        # create optimizers
        ###################################
        self.schedulers = []
        self.optimizers = []

        self.optim_G = nn.optim.Adam(self.netG.parameters(),
            lr = opt.lr, betas = (opt.beta1, 0.999))
        self.optim_D = nn.optim.Adam(self.netD.parameters(),
            lr = opt.lr, betas = (opt.beta1, 0.999))
        self.optimizers.append(self.optim_G)
        self.optimizers.append(self.optim_D)

        for optim in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optim, opt))



    def set_input(self, data):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def backward_D(self):
        raise NotImplementedError

    def backward_G(self):
        raise NotImplementedError

    def optimze_parameters(self):
        raise NotImplementedError

    def get_current_errors(self):
        raise NotImplementedError

    def encode_attribute(self, img, lm_heatmap = None):
        v_img = img if isinstance(img, Variable) else Variable(img)

        if self.opt_AE.input_lm:
            v_lm_heatmap = lm_heatmap if isinstance(lm_heatmap, Variable) else Variable(lm_heatmap)
            prob, prob_map = self.netAE(v_img, v_lm_heatmap)
        else:
            prob, prob_map = self.netAE(v_img)

    def save(self, label):
        # Todo: if self.netAE is jointly trained, also save its parameter
        # Todo: if att_fuse module is added, save its parameters
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

