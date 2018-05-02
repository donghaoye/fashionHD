from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *
from misc.pose_util import get_joint_coord

import cv2
import PIL
import numpy as np
import os
import util.io as io

class DomainTransferDataset(BaseDataset):
    def name(self):
        return 'DomainTransferDataset'

    def initialize(self, opt, split):
        self.opt = opt
        self.root = opt.data_root
        if opt.debug:
            split = 'debug'
        self.split = split
        #############################
        # load data
        #############################
        print('loading data ...')
        

