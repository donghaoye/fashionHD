from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import networks
from torch.autograd import Variable
from base_model import BaseModel
from misc import pose_util

import os
import sys
import numpy as np
import time
from collections import OrderedDict
import argparse
import util.io as io

class PoseParsingModel(BaseModel):
    def name(self):
        return 'PoseParsingModel'

    def initialize(self, opt):
        super(PoseParsingModel, self).initialize(opt)
        ###################################
        # create model
        ###################################
        if opt.which_model_PP == 'resnet':
            self.netPP = networks.ResnetGenerator(
                input_nc = 3,
                output_nc = self.get_output_dim(opt.pp_pose_type),
                ngf = opt.pp_nf,
                norm_layer = networks.get_norm_layer(opt.norm),
                activation = nn.ReLU,
                use_dropout = False,
                n_blocks = opt.pp_nblocks,
                gpu_ids = opt.gpu_ids,
                output_tanh = False,
                )
        elif opt.which_model_PP == 'unet':
            self.netPP = networks.UnetGenerator_v2(
                input_nc = 3,
                output_nc = self.get_output_dim(opt.pp_pose_type),
                num_downs = 8,
                ngf = opt.pp_nf,
                max_nf = opt.pp_nf*(2**3),
                norm_layer = networks.get_norm_layer(opt.norm),
                use_dropout = False,
                gpu_ids = opt.gpu_ids,
                output_tanh = False,
                )
        else:
            raise NotImplementedError()

        if opt.gpu_ids:
            self.netPP.cuda()
        ###################################
        # init/load model
        ###################################
        if self.is_train and (not opt.continue_train):
            networks.init_weights(self.netPP, init_type=opt.init_type)
        else:
            self.load_network(self.netPP, 'netPP', opt.which_epoch)
        ###################################
        # optimizers and schedulers
        ###################################
        if self.is_train:
            self.optim = torch.optim.Adam(self.netPP.parameters(), lr=opt.lr, betas=(0.9, 0.999))
            self.optimizers = [self.optim]

            self.schedulers = []
            for optim in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optim, opt))

    def set_input(self, data):
        input_list = [
            'img',
            'joint',
            'joint_c',
            'seg',
            'seg_mask',
        ]

        for name in input_list:
            if name in data:
                self.input[name] = self.Tensor(data[name].size()).copy_(data[name])

        self.input['id'] = data['id']

    def get_output_dim(self, output_type):
        dim = 0
        output_items = output_type.split('+')
        for item in output_items:
            if item == 'seg':
                dim += self.opt.seg_nc
            elif item == 'joint':
                dim += self.opt.joint_nc
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
            if item == 'seg':
                rst['seg'] = output[:,i:(i+self.opt.seg_nc)]
                # convert raw output to seg_mask
                max_index = rst['seg'].argmax(dim=1)
                seg_mask = []
                for idx in range(self.opt.seg_nc):
                    seg_mask.append(max_index==idx)
                rst['seg_mask'] = torch.stack(seg_mask, dim=1).float()
                i += self.opt.seg_nc
            elif item == 'joint':
                rst['joint'] = output[:,i:(i+self.opt.joint_nc)]
                i += self.opt.joint_nc
            else:
                raise Exception('invalid output type %s'%item)
        return rst

    def compute_loss(self):
        loss = 0
        if 'seg' in self.opt.pp_pose_type:
            self.output['loss_seg'] = F.cross_entropy(self.output['seg'], self.input['seg'].squeeze(dim=1).long())
            loss += self.output['loss_seg'] * self.opt.loss_weight_seg
        if 'joint' in self.opt.pp_pose_type:
            self.output['loss_joint'] = F.mse_loss(self.output['joint'], self.input['joint'])
            loss += self.output['loss_joint'] * self.opt.loss_weight_joint
        
        return loss

    def forward(self, mode='train'):
        if mode == 'train':
            self.netPP.train()
        else:
            self.netPP.eval()
        output, feat = self.netPP(self.input['img'], output_feature=True)
        output = self.parse_output(output, self.opt.pp_pose_type)

        self.output['feat'] = feat
        if 'seg' in output:
            self.output['seg'] = output['seg']
            self.output['seg_mask'] = output['seg_mask']
        if 'joint' in output:
            self.output['joint'] = F.sigmoid(output['joint']) # normalize to range(0,1)

    def optimize_parameters(self):
        self.output = {}
        self.forward()
        loss = self.compute_loss()
        loss.backward()
        self.optim.step()

    def test(self, compute_loss=False):
        with torch.no_grad():
            self.forward(mode='test')
        if compute_loss:
            self.compute_loss()

    def get_current_errors(self):
        error_list = ['loss_seg', 'loss_joint']
        errors = OrderedDict()
        for item in error_list:
            if item in self.output:
                errors[item] = self.output[item].data.item()
        return errors

    def get_current_visuals(self):
        visuals = OrderedDict([
            ('img', [self.input['img'].data.cpu(), 'rgb'])])
        if 'seg' in self.output:
            visuals['seg_gt'] = [self.input['seg'].data.cpu(), 'seg']
            visuals['seg_gen'] = [self.output['seg'].data.cpu(), 'seg']
        if 'joint' in self.output:
            visuals['joint_gt'] = [self.input['joint'].data.cpu(), 'pose']
            visuals['joint_gen'] = [self.output['joint'].data.cpu(), 'pose']

        return visuals

    def save(self, label):
        self.save_network(self.netPP, 'netPP', label, self.gpu_ids)
        self.save_optim(self.optim, 'optim', label)
