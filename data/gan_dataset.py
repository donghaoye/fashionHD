from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *

import cv2
import numpy as np
import os
import util.io as io


class GANDataset(BaseDataset):
    '''
    Dataset for GAN model training and testing
    '''

    def name(self):
        return 'GANDataset'

    def initialize(self, opt, split):
        self.opt = opt
        self.root = opt.data_root
        self.split = split

        print('loading data ...')
        samples = io.load_json(os.path.join(opt.data_root, opt.fn_sample))
        attr_label = io.load_data(os.path.join(opt.data_root, opt.fn_label))
        attr_entry = io.load_json(os.path.join(opt.data_root, opt.fn_entry))
        attr_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))
        lm_label = io.load_data(os.path.join(opt.data_root, opt.fn_landmark))
        seg_paths = io.load_json(os.path.join(opt.data_root, opt.fn_seg_path))

        self.id_list = attr_split[split]
        self.attr_entry = attr_entry
        if opt.max_dataset_size != float('inf'):
            self.id_list = self.id_list[0:opt.max_dataset_size]
        self.sample_list = [samples[s_id] for s_id in self.id_list]
        self.attr_label_list = [attr_label[s_id] for s_id in self.id_list]
        self.lm_list = [lm_label[s_id] for s_id in self.id_list]
        self.seg_path_list = [seg_paths[s_id] for s_id in self.id_list]

        # check data
        assert len(self.attr_entry) == len(self.attr_label_list[0]) == opt.n_attr, 'Attribute number not match!'
        print('dataset created (%d samples)' % len(self))

        # get transform
        self.to_tensor = transforms.ToTensor()

        # use standard normalization, which is different from attribute dataset
        # image will be normalized again (under imagenet distribution) before fed into attribute encoder in GAN model
        self.tensor_normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        s_id = self.id_list[index]

        # load image
        img = cv2.imread(self.sample_list[index]['img_path'])
        if img.ndim == 3:
            # convert BRG to RBG
            img = img[:,:,[2,1,0]]

        # create landmark heatmap
        h, w = img.shape[0:2]
        lm_map = landmark_to_heatmap(
            img_sz = (w, h),
            lm_label = self.lm_list[index],
            cloth_type = self.sample_list[index]['cloth_type']
            )

        # load segmentation map
        seg_map = cv2.imread(self.seg_path_list[index], cv2.IMREAD_GRAYSCALE)[:,:,np.newaxis]

        mix = np.concatenate((img, lm_map, seg_map), axis = 2)

        # transform
        if self.opt.resize_or_crop == 'resize':
            # only resize
            mix = trans_resize(mix, size = (self.opt.fine_size, self.opt.fine_size))
        elif self.opt.resize_or_crop == 'resize_and_crop':
            mix = trans_resize(mix, size = (self.opt.load_size, self.opt.load_size))
            if self.split == 'train':
                mix = trans_random_crop(mix, size = (self.opt.fine_size, self.opt.fine_size))
                mix = trans_random_horizontal_flip(mix)
            else:
                mix = trans_center_crop(mix, size = (self.opt.fine_size, self.opt.fine_size))

        img = mix[:,:,0:3]
        img = self.tensor_normalize(self.to_tensor(img))

        lm_map = mix[:,:,3:-1]
        lm_map = torch.Tensor(lm_map.transpose([2, 0, 1])) # convert to CxHxW

        seg_map = mix[:,:,-1::].round()
        seg_map = torch.Tensor(seg_map.transpose([2, 0, 1]))

        # load label
        att = np.array(self.attr_label_list[index], dtype = np.float32)

        data = {
            'img': img,
            'lm_map': lm_map,
            'seg_map': seg_map,
            'attr_label':att,
            'id': s_id
        }

        return data


