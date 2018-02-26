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
        netG_LR: low resolution generator for feature fusion
        netG_HR: high resolution generator for image generation
        netD: discriminator
        netD_feat: feature level discriminator?
    '''
    def name(self):
        return 'MultimodalDesignerGAN_V2'
    def initialize(self, opt):
        super(MultimodalDesignerGAN_V2, self).initialize(opt)