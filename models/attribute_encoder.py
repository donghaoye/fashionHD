from __future__ import division, print_function

import torch
from torch.autograd import Variable
import networks
from base_model import BaseModel

import os
import sys
import numpy as np
import time
from collections import OrderedDict

import util.io as io

class AttributeEncoder(BaseModel):
    def name(self):
        return 'AttributeEncoder'

    def initialize(self, opt):
        super(AttributeEncoder, self).initialize(opt)

        # define tensors
        self.input_img = self.Tensor(opt.batch_size, opt.input_nc, opt.fine_size, opt.fine_size)
        self.input_label = self.Tensor(opt.batch_size, opt.n_attr)

        # load/define networks
        self.net = networks.define_attr_encoder_net(convnet = opt.convnet, input_nc = opt.input_nc,
            output_nc = opt.n_attr, spatial_pool = opt.spatial_pool, init_type = opt.init_type,
            gpu_ids = opt.gpu_ids)

        if not self.is_train or opt.continue_train:
            self.load_network(self.net, network_label = 'AE', epoch_label = opt.which_epoch)

        
        # define loss functions
        # Todo: implement BalanceBCELoss()
        self.crit_attr = networks.Smooth_Loss(torch.nn.BCELoss())

        # initialize optimizers
        self.schedulers = []
        self.optimizers = []
        self.loss_functions = []

        if opt.optim == 'adam':
            self.optim_attr = torch.optim.Adam(self.net.parameters(),
                lr = opt.lr, betas = (opt.beta1, 0.999), weight_decay = 5e-4)
        elif opt.optim == 'sgd':
            self.optim_attr = torch.optim.SGD(self.net.parameters(),
                lr = opt.lr, momentum = 0.9, weight_decay = 5e-4)
        self.optimizers.append(self.optim_attr)
        self.loss_functions.append(self.crit_attr)

        for optim in self.optimizers:
            self.schedulers.append(networks.get_scheduler(optim, opt))


    def set_input(self, data):
        # Todo: add support to spatial info input (seg map, landmark heatmap, etc)
        self.input_img.resize_(data['img'].size()).copy_(data['img'])
        self.input_label.resize_(data['att'].size()).copy_(data['att'])

    
    def forward(self):
        # assert self.net.training
        if not self.net.training:
            self.net.train()

        v_img = Variable(self.input_img)
        v_label = Variable(self.input_label)
        self.output_prob, self.output_map = self.net(v_img)
        self.loss_attr = self.crit_attr(self.output_prob, v_label)


    def test(self):
        # assert not self.net.training
        if self.net.training:
            self.net.eval()

        v_img = Variable(self.input_img, volatile = True)
        v_label = Variable(self.input_label)
        self.output_prob, self.output_map = self.net(v_img)
        self.loss_attr = self.crit_attr(self.output_prob, v_label)


    def optimize_parameters(self):
        self.net.train()
        self.optim_attr.zero_grad()
        self.forward()
        self.loss_attr.backward()
        self.optim_attr.step()

    def get_current_errors(self, clear = True):
        return OrderedDict([
            ('loss_attr', self.crit_attr.smooth_loss(clear)),
            ])


    def save(self, label):
        self.save_network(self.net, 'AE', label, self.gpu_ids)





