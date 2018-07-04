from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *

import cv2
import numpy as np
import os
import util.io as io

class PoseParsingDataset(BaseDataset):
    def name(self):
        return 'PoseParsingDataset'

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
        data_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))
        self.img_dir = os.path.join(opt.data_root, opt.img_dir)
        self.seg_dir = os.path.join(opt.data_root, opt.seg_dir)
        self.pose_label = io.load_data(os.path.join(opt.data_root, opt.fn_pose))
        #############################
        # create index list
        #############################
        self.id_list = data_split[split]
        #############################
        # other
        #############################
        self.tensor_normalize_std = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.id_list)

    def to_tensor(self, img):
        return torch.Tensor(img.transpose((2, 0, 1)))

    def read_image(self, s_id):
        fn = os.path.join(self.img_dir, s_id + '.jpg')
        img = cv2.imread(fn).astype(np.float32) / 255.
        img = img[:,:,[2,1,0]]
        return img

    def read_seg(self, s_id):
        fn = os.path.join(self.seg_dir, s_id + '.bmp')
        seg = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float32)[:,:,np.newaxis]
        return seg

    def __getitem__(self, index):
        sid = self.id_list[index]
        ######################
        # load sample
        ######################
        img = self.read_image(sid)
        seg = self.read_seg(sid)
        joint_c = self.pose_label[sid]
        ######################
        # augmentation
        ######################
        if self.split == 'train' and self.opt.is_train:
            coin = np.random.rand()
            img = trans_random_horizontal_flip(img, coin)
            seg = trans_random_horizontal_flip(seg, coin)
            joint_c = trans_random_horizontal_flip_pose_c(joint_c, (img.shape[1], img.shape[0]), coin)
        ######################
        # create pose representation
        ######################
        # follow the settings in LIP: 400*N([x,y], diag([64, 64]))
        joint_input = pose_to_map(img_sz=(img.shape[1], img.shape[0]), label=joint_c, mode='gaussian', radius=8)
        joint_tar = pose_to_map(img_sz=(img.shape[1], img.shape[0]), label=joint_c, mode='gaussian', radius=11.3137)
        ######################
        # output
        ######################
        data = {
            'img': self.tensor_normalize_std(self.to_tensor(img)),
            'joint_input': self.to_tensor(joint_input),
            'joint_tar': self.to_tensor(joint_tar),
            'joint_c': torch.Tensor(joint_c),
            'seg': self.to_tensor(seg),
            'set_mask': self.to_tensor(segmap_to_mask_v2(seg, nc=self.opt.seg_nc, bin_size=1)),
            'id': sid,
        }

        return data
