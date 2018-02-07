from __future__ import division, print_function

import torch
import torch.nn as nn
import torchvision
import networks
from torch.autograd import Variable
from misc.image_pool import ImagePool
from base_model import BaseModel
from network_loader import load_attribute_encoder_net

import os
import sys
import numpy as np
import time
from collections import OrderedDict
import util.io as io

class FeatureSpatialTransformer(BaseModel):
    def name(self):
        return 'FeatureSpatialTransformer'

    def initialize(self, opt):
        super(FeatureSpatialTransformer, self).initialize(opt)
        ###################################
        # load/define networks
        ###################################
        self.net = networks.define_feat_spatial_transformer(opt)
        self.netAE = None

        if opt.continue_train or not self.is_train:
            self.load_network(self.net, 'FeatST', epoch_label = opt.which_epoch)

        if self.is_train:
            ###################################
            # load attribute encoder
            ###################################
            self.netAE, self.opt_AE = load_attribute_encoder_net(id=opt.which_model_AE, gpu_ids=opt.gpu_ids)

            ###################################
            # define loss functions and loss buffers
            ###################################
            self.crit_L1 = networks.SmoothLoss(nn.L1Loss())
            self.crit_attr = networks.SmoothLoss(nn.BCELoss())

            self.loss_functions = []
            self.loss_functions.append(self.crit_L1)
            self.loss_functions.append(self.crit_attr)

            ###################################
            # create optimizers
            ###################################
            self.schedulers = []
            self.optimizers = []

            self.optim = torch.optim.Adam(self.net.parameters(), 
                lr = opt.lr, betas = (opt.beta1, opt.beta2))

            self.optimizers.append(self.optim)

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
        self.input['lm_map'] = self.Tensor(data['lm_map'].size()).copy_(data['lm_map'])
        self.input['seg_mask'] = self.Tensor(data['seg_mask'].size()).copy_(data['seg_mask'])
        self.input['seg_map'] = self.Tensor(data['seg_map'].size()).copy_(data['seg_map'])
        self.input['id'] = data['id']

        # create input variables
        for k, v in self.input.iteritems():
            if isinstance(v, torch.tensor._TensorBase):
                self.input[k] = Variable(v)

    def forward(self):
        shape_code = self.encode_shape(self.input['lm_map'], self.input['seg_mask'])
        feat_input = self.encode_attribute(self.input['img'], self.input['lm_map'], output_type = 'feat_map')
        # feat_output = self.net(feat_input, shape_input, shape_tar)
        feat_output = self.net(feat_input, shape_code, shape_code)
        self.output['feat_input'] = feat_input
        self.output['feat_output'] = feat_output

        # compute loss
        self.output['loss'] = 0

        self.output['loss_L1'] = self.crit_L1(feat_output, feat_input)
        self.output['loss'] += self.output['loss_L1'] * self.opt.loss_weight_L1

        pred_input, _ = self.netAE.predict(feat_input)
        pred_output, _ = self.netAE.predict(feat_output)
        self.output['loss_attr'] = self.crit_attr(pred_output, pred_input)
        self.output['loss'] += self.output['loss_attr'] * self.opt.loss_weight_attr


    def test(self):
        if float(torch.__version__[0:3]) >= 0.4:
            with torch.no_grad():
                self.forward()
        else:
            self.input['img'].volatile = True
            self.input['lm_map'].volatile = True
            self.input['seg_mask'].volatile = True

            self.forward()

    def optimize_parameters(self):
        self.output = {}
        self.forward()
        self.optim.zero_grad()
        self.output['loss'].backward()
        self.optim.step()
        

    def get_current_errors(self):
        errors = OrderedDict([
            ('loss_L1', self.output['loss_L1']),
            ('loss_attr', self.output['loss_attr']),
            ('loss', self.output['loss'])
            ])
        return errors

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def encode_shape(self, lm_map, seg_mask, img = None):
        if self.opt.shape_encode == 'lm':
            shape_code = lm_map
        elif self.opt.shape_encode == 'seg':
            shape_code = seg_mask
        elif self.opt.shape_encode == 'lm+seg':
            shape_code = torch.cat((lm_map, seg_mask), dim = 1)
        if img is not None:
            shape_code = torch.cat((img, shape_code), dim = 1)
        return shape_code

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


    def save(self, label):
        self.save_network(self.net, 'FeatST', label, self.gpu_ids)