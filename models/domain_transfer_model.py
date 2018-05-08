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
###################################
# domain A: person
# domain B: cloth
# modules: VUnet(Enc_A+Dec_B), Enc_featA, Dec_featA, VAE(Enc_B, Dec_B)

class DomainTransferModel(BaseModel):
    def name(self):
        return 'DomainTransferModel'

    def initialize(self, opt):
        super(DomainTransferModel, self).initialize()
        ###################################
        # define Enc_A and Dec_B (VUnet)
        ###################################
        self.netA = networks.VariationalUnet(
            input_nc_dec = self.get_pose_dim(opt.pose_type),
            input_nc_enc = 3,
            output_nc = 3,
            nf = opt.vunet_nf,
            max_nf = opt.vunet_max_nf,
            input_size = opt.fine_size,
            n_latent_scales = opt.vunet_n_latent_scales,
            bottleneck_factor = opt.vunet_bottleneck_factor,
            box_factor = opt.vunet_box_factor,
            n_residual_blocks = 2,
            norm_layer = networks.get_norm_layer(opt.norm),
            activation = nn.ReLU(False),
            use_dropout = False,
            gpu_ids = opt.gpu_ids,
            )
        if opt.gpu_ids:
            self.netA.cuda()
        networks.init_weights(self.netA, init_type=opt.init_type)
        ###################################
        # define Enc_B and Dec_B (VAE)
        ###################################
        self.netB = networks.VariationalAutoEncoder(
            input_nc = 3,
            output_nc = 3, 
            nf = opt.vae_nf,
            max_nf = opt.vae_max_nf,
            latent_nf = opt.vae_latent_nf,
            input_size = opt.fine_size,
            bottleneck_factor = opt.vae_bottleneck_factor,
            n_residual_blocks = 2,
            norm_layer = networks.get_norm_layer(opt.norm),
            activation = nn.ReLU(False),
            use_dropout = False,
            gpu_ids = opt.gpu_ids,
            )
        if opt.gpu_ids:
            self.netB.cuda()
        network.init_weights(self.netB, init_type=opt.init_type)
        ###################################
        # define feature transfer network 
        ###################################
        self.netFT = networks.VUnetLatentTransformer(
            
            )




