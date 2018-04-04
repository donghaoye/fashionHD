from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *

import cv2
import PIL
import numpy as np
import os
import util.io as io

class GANDataset_V2(BaseDataset):

    def name(self):
        return 'GANDataset_V2'

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
        # data split
        data_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))
        self.pose_label = io.load_data(os.path.join(opt.data_root, opt.fn_pose))
        self.img_dir = os.path.join(opt.data_root, opt.img_dir)
        self.seg_dir = os.path.join(opt.data_root, opt.seg_dir)
        self.edge_dir = os.path.join(opt.data_root, opt.edge_dir)
        #############################
        # create index list
        #############################
        self.id_list = data_split[split]
        #############################
        # other
        #############################
        self.tensor_normalize_std = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        self.to_pil_image = transforms.ToPILImage()

    def __len__(self):
        return len(self.id_list)

    def to_tensor(self, img):
        return torch.Tensor(img.transpose((2, 0, 1)))

    def read_seg(self, s_id):
        try:
            fn = os.path.join(self.seg_dir, s_id + '.bmp')
            seg = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float32)[:,:,np.newaxis]
            return seg
        except:
            raise Exception('fail to load image %s' % fn)

    def read_image(self, s_id):
        try:
            fn = os.path.join(self.img_dir, s_id + '.jpg')
            img = cv2.imread(fn).astype(np.float32) / 255.
            img = img[:,:,[2,1,0]]
            return img
        except:
            raise Exception('fail to load image %s' % fn)


    def read_edge(self, s_id):
        try:
            fn = os.path.join(self.edge_dir, '%s.jpg' % s_id)
            edge = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            edge = (edge >= self.opt.edge_threshold) * edge / 255.
            edge = edge[:,:,np.newaxis]
            return edge
        except:
            raise Exception('fail to load image %s' % fn)

    def apply_color_jitter(self, img, seg):
        img_j = self.to_pil_image((img*255).astype(np.uint8))
        img_j = self.color_jitter(img_j)
        img_j = transforms.ToTensor()(img_j).numpy().transpose([1,2,0])
        mask = ((seg==3) | (seg==4)).astype(np.float32)
        return img_j * mask + img * (1-mask)

    def img_to_color(self, img):
        # config
        color  = cv2.GaussianBlur(img, (self.opt.color_gaussian_ksz, self.opt.color_gaussian_ksz), self.opt.color_gaussian_sigma)
        h, w = color.shape[0:2]
        dh, dw = h//self.opt.color_bin_size, w//self.opt.color_bin_size
        color = cv2.resize(color, dsize=(dw, dh), interpolation=cv2.INTER_LINEAR)
        color = cv2.resize(color, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        return color


    def __getitem__(self, index):
        s_id = self.id_list[index]
        ######################
        # load image
        ######################
        img = self.read_image(s_id)
        seg = self.read_seg(s_id)
        edge = self.read_edge(s_id)
        pose = pose_to_heatmap(img_sz=(img.shape[1], img.shape[0]), label=self.pose_label[s_id], size=self.opt.pose_size)
        ######################
        # random flip
        ######################
        if self.split == 'train' and self.opt.is_train:
            coin = np.random.rand()
            img = trans_random_horizontal_flip(img, coin)
            seg = trans_random_horizontal_flip(seg, coin)
            edge = trans_random_horizontal_flip(edge, coin)
            pose = trans_random_horizontal_flip_pose(pose, coin)
        ######################
        # get color map
        ######################
        if self.opt.color_jitter and self.split == 'train' and self.opt.is_train:
            img = self.apply_color_jitter(img, seg)
        color = self.img_to_color(img)
        ######################
        # geometricall transformation
        ######################

        # flip
        if (self.split == 'test') or (self.split == 'train' and self.opt.is_train and self.opt.shape_deformation_flip):
            img_def = trans_random_horizontal_flip(img, coin=1.)
            edge_def = trans_random_horizontal_flip(edge, coin=1.)
            seg_def = trans_random_horizontal_flip(seg, coin=1.)
            color_def = trans_random_horizontal_flip(color, coin=1.)
            pose_def = trans_random_horizontal_flip_pose(pose, coin=1.)
        else:
            img_def, edge_def, seg_def, color_def, pose_def = img, edge, seg, color, pose

        img_def, edge_def, seg_def, color_def, pose_def = trans_random_perspective([img_def, edge_def, seg_def, color_def, pose_def], self.opt.shape_deformation_scale)
        ######################
        # convert to tensor
        ######################
        t_img = self.tensor_normalize_std(self.to_tensor(img))
        t_edge = self.to_tensor(edge)
        t_seg = self.to_tensor(seg)
        t_seg_mask = self.to_tensor(segmap_to_mask_v2(seg, nc=7, bin_size=self.opt.seg_bin_size))
        t_color = self.tensor_normalize_std(self.to_tensor(color))
        t_pose = self.to_tensor(pose)

        t_img_def = self.tensor_normalize_std(self.to_tensor(img_def))
        t_edge_def = self.to_tensor(edge_def)
        t_seg_def = self.to_tensor(seg_def)
        t_seg_mask_def = self.to_tensor(segmap_to_mask_v2(seg_def, nc=7, bin_size=self.opt.seg_bin_size))
        t_color_def = self.tensor_normalize_std(self.to_tensor(color_def))
        t_pose_def = self.to_tensor(pose_def)
        
        ######################
        # output
        ######################
        data = {
            'img': t_img,
            'edge_map': t_edge,
            'seg_map': t_seg,
            'seg_mask': t_seg_mask,
            'color_map': t_color,
            'pose_map': t_pose,

            'img_def': t_img_def,
            'edge_map_def': t_edge_def,
            'seg_map_def': t_seg_def,
            'seg_mask_def': t_seg_mask_def,
            'color_map_def': t_color_def,
            'pose_map_def': t_pose_def,

            'id': s_id
        }

        return data

