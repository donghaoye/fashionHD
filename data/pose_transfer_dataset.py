from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *

import cv2
import PIL
import numpy as np
import os
import util.io as io

class PoseTransferDataset(BaseDataset):
    def name(self):
        return 'PoseTransferDataset'

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
        # self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        # self.to_pil_image = transforms.ToPILImage()

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
        sid_1, sid_2 = self.id_list[index]
        ######################
        # load image
        ######################
        img_1 = self.read_image(sid_1)
        img_2 = self.read_image(sid_2)
        pose_c_1 = self.pose_label[sid_1]
        pose_c_2 = self.pose_label[sid_2]
        pose_1 = pose_to_map(img_sz=(img_1.shape[1], img_1.shape[0]), label=pose_c_1, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
        pose_2 = pose_to_map(img_sz=(img_2.shape[1], img_2.shape[0]), label=pose_c_2, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
        seg_1 = self.read_seg(sid_1)
        seg_2 = self.read_seg(sid_2)
        ######################
        # augmentation
        ######################
        if self.split == 'train' and self.opt.is_train:
            # flip img_1
            coin = np.random.rand()
            img_1 = trans_random_horizontal_flip(img_1, coin)
            seg_1 = trans_random_horizontal_flip(seg_1, coin)
            pose_1 = trans_random_horizontal_flip_pose(pose_1, coin)
            pose_c_1 = trans_random_horizontal_flip_pose_c(pose_c_1, (img_1.shape[1], img_1.shape[2]), coin)
            # flip img_2
            coin = np.random.rand()
            img_2 = trans_random_horizontal_flip(img_2, coin)
            seg_2 = trans_random_horizontal_flip(seg_2, coin)
            pose_2 = trans_random_horizontal_flip_pose(pose_2, coin)
            pose_c_1 = trans_random_horizontal_flip_pose_c(pose_c_2, (img_2.shape[1], img_2.shape[2]), coin)
            # swap img_1 and img_2
            coin = np.random.rand()
            if coin > 0.5:
                sid_1, sid_2 = sid_2, sid_1
                img_1, img_2 = img_2, img_1
                pose_c_1, pose_c_2 = pose_c_2, pose_c_1
                pose_1, pose_2 = pose_2, pose_1
                seg_1, seg_2 = seg_2, seg_1

        ######################
        # convert to tensor
        ######################
        t_img_1 = self.tensor_normalize_std(self.to_tensor(img_1))
        t_img_2 = self.tensor_normalize_std(self.to_tensor(img_2))
        t_pose_c_1 = torch.Tensor(pose_c_1)
        t_pose_c_2 = torch.Tensor(pose_c_2)
        t_pose_1 = self.to_tensor(pose_1)
        t_pose_2 = self.to_tensor(pose_2)
        t_seg_1 = self.to_tensor(seg_1)
        t_seg_2 = self.to_tensor(seg_2)
        t_seg_mask_1 = self.to_tensor(segmap_to_mask_v2(seg_1, nc=7, bin_size=self.opt.seg_bin_size))
        t_seg_mask_2 = self.to_tensor(segmap_to_mask_v2(seg_2, nc=7, bin_size=self.opt.seg_bin_size))
        ######################
        # output
        ######################
        data = {
            'img_1': t_img_1,
            'img_2': t_img_2,
            'pose_c_1': t_pose_c_1,
            'pose_c_2': t_pose_c_2,
            'pose_1': t_pose_1,
            'pose_2': t_pose_2,
            'seg_1': t_seg_1,
            'seg_2': t_seg_2,
            'seg_mask_1': t_seg_mask_1,
            'seg_mask_2': t_seg_mask_2,
            'id_1': sid_1,
            'id_2': sid_2
        }

        return data



