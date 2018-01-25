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
        self.input['img'] = self.Tensor(opt.batch_size, opt.input_nc, opt.fine_size, opt.fine_size)
        self.input['label'] = self.Tensor(opt.batch_size, opt.n_attr)

        # load/define networks
        self.net = networks.define_attr_encoder_net(opt)

        if not self.is_train or opt.continue_train:
            self.load_network(self.net, network_label = 'AE', epoch_label = opt.which_epoch)

                
        self.schedulers = []
        self.optimizers = []
        self.loss_functions = []

        # define loss functions
        # attribute
        if opt.loss_type == 'bce':
            self.crit_attr = networks.Smooth_Loss(torch.nn.BCELoss(size_average = not opt.no_size_avg))
        elif opt.loss_type == 'wbce':
            attr_entry = io.load_json(os.path.join(opt.data_root, opt.fn_entry))
            pos_rate = self.Tensor([att['pos_rate'] for att in attr_entry])
            pos_rate.clamp_(min = 0.01, max = 0.99)
            self.crit_attr = networks.Smooth_Loss(networks.WeightedBCELoss(pos_rate = pos_rate, class_norm = opt.wbce_class_norm, size_average = not opt.no_size_avg))
        else:
            raise ValueError('attribute loss type "%s" is not defined' % opt.loss_type)
        self.loss_functions.append(self.crit_attr)
        
        # joint task
        if opt.joint_cat:
            self.crit_cat = networks.Smooth_Loss(torch.nn.CrossEntropyLoss())
            self.loss_functions.append(self.crit_cat)


        # initialize optimizers
        if opt.is_train:
            if opt.optim == 'adam':
                self.optim_attr = torch.optim.Adam(self.net.parameters(),
                    lr = opt.lr, betas = (opt.beta1, 0.999), weight_decay = opt.weight_decay)
            elif opt.optim == 'sgd':
                self.optim_attr = torch.optim.SGD(self.net.parameters(),
                    lr = opt.lr, momentum = 0.9, weight_decay = opt.weight_decay)
            self.optimizers.append(self.optim_attr)
        
            for optim in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optim, opt))


    def set_input(self, data):
        self.input['img'].resize_(data['img'].size()).copy_(data['img'])
        self.input['label'].resize_(data['att'].size()).copy_(data['att'])
        self.input['id'] = data['id']
        
        if self.opt.joint_cat:
            if 'cat_label' not in self.input:
                self.input['cat_label'] = self.Tensor(self.opt.batch_size)
            self.input['cat_label'].resize_(data['cat'].size()).copy_(data['cat']).squeeze_()

        if self.opt.input_lm:
            if 'lm_heatmap' not in self.input:
                self.input['lm_heatmap'] = self.Tensor(self.opt.batch_size, self.opt.lm_input_nc, self.opt.fine_size, self.opt.fine_size)
            self.input['lm_heatmap'].resize_(data['landmark_heatmap'].size()).copy_(data['landmark_heatmap'])

    
    def forward(self):
        # assert self.net.training        

        v_img = Variable(self.input['img'])
        v_label = Variable(self.input['label'])

        if self.opt.joint_cat:
            v_cat_label = Variable(self.input['cat_label'].long())
            output_prob, output_map, output_cat_pred = self.net(v_img)
            self.output['cat_pred'] = output_cat_pred
            self.output['loss_cat'] = self.crit_cat(output_cat_pred, v_cat_label)
        elif self.opt.input_lm:
            v_lm_heatmap = Variable(self.input['lm_heatmap'])
            output_prob, output_map = self.net(v_img, v_lm_heatmap)
        else:
            output_prob, output_map = self.net(v_img)

        self.output['prob'] = output_prob
        self.output['map'] = output_map
        self.output['loss_attr'] = self.crit_attr(output_prob, v_label)


    def test(self):
        # assert not self.net.training

        v_img = Variable(self.input['img'], volatile = True)
        v_label = Variable(self.input['label'])

        if self.opt.joint_cat:
            v_cat_label = Variable(self.input['cat_label'].long())
            output_prob, output_map, output_cat_pred = self.net(v_img)
            self.output['cat_pred'] = output_cat_pred
            self.output['loss_cat'] = self.crit_cat(output_cat_pred, v_cat_label)
        elif self.opt.input_lm:
            v_lm_heatmap = Variable(self.input['lm_heatmap'], volatile = True)
            output_prob, output_map = self.net(v_img, v_lm_heatmap)
        else:
            output_prob, output_map = self.net(v_img)

        self.output['prob'] = output_prob
        self.output['map'] = output_map
        self.output['loss_attr'] = self.crit_attr(output_prob, v_label)

    def extract_feat(self):
        v_img = Variable(self.input['img'], volatile = True)
        v_label = Variable(self.input['label'])

        if self.opt.input_lm:
            v_lm_heatmap = Variable(self.input['lm_heatmap'], volatile = True)
            output_feat, output_feat_map = self.net.extract_feat(v_img, v_lm_heatmap)
        else:
            output_feat, output_feat_map = self.net.extract_feat(v_img)

        self.output['feat'] = output_feat
        self.output['feat_map'] = output_feat_map


    def optimize_parameters(self):
        self.net.train()
        self.optim_attr.zero_grad()
        self.forward()

        loss = self.output['loss_attr']
        if self.opt.joint_cat:
            loss += self.output['loss_cat'] * self.opt.cat_loss_weight
        
        loss.backward()
        self.optim_attr.step()

    def get_current_errors(self, clear = True):
        errors = OrderedDict([
            ('loss_attr', self.crit_attr.smooth_loss(clear)),
            ])
        if self.opt.joint_cat:
            errors['loss_cat'] = self.crit_cat.smooth_loss(clear)
        return errors

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def save(self, label):
        self.save_network(self.net, 'AE', label, self.gpu_ids)


