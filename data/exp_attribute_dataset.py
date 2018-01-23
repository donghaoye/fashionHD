from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *

from PIL import Image
import cv2
import numpy as np
import os
import util.io as io


class EXPAttributeDataset(BaseDataset):
    '''
    Attribute dataset with auxiliary data, like landmark heatmap.
    '''
    def name(self):
        return 'EXPAttributeDataset'

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

        self.id_list = attr_split[split]
        if opt.max_dataset_size != float('inf'):
            self.id_list = self.id_list[0:opt.max_dataset_size]
        self.sample_list = [samples[s_id] for s_id in self.id_list]
        self.attr_label_list = [attr_label[s_id] for s_id in self.id_list]
        self.lm_list = [lm_label[s_id] for s_id in self.id_list]
        self.attr_entry = attr_entry

        # check data
        assert len(self.attr_entry) == len(self.attr_label_list[0]) == opt.n_attr, 'Attribute number not match!'
        print('dataset created (%d samples)' % len(self))


        # get transform
        self.to_tensor = transforms.ToTensor()

        if opt.image_normalize == 'imagenet':
            self.tensor_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
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
        lm_heatmap = landmark_to_heatmap(
            img_sz = (w, h),
            lm_label = self.lm_list[index],
            cloth_type = self.sample_list[index]['cloth_type']
            )

        
        mix = np.concatenate((img, lm_heatmap), axis = 2)
        
        # transform
        if self.opt.resize_or_crop == 'resize':
            # only resize
            mix = _trans_resize(mix, size = (self.opt.fine_size, self.opt.fine_size))
        elif self.opt.resize_or_crop == 'resize_and_crop':
            mix = _trans_resize(mix, size = (self.opt.load_size, self.opt.load_size))
            if self.split == 'train':
                mix = _trans_random_crop(mix, size = (self.opt.fine_size, self.opt.fine_size))
                mix = _trans_random_horizontal_flip(mix)
            else:
                mix = _trans_center_crop(mix, size = (self.opt.fine_size, self.opt.fine_size))

        img = mix[:,:,0:3]
        img = self.tensor_normalize(self.to_tensor(img))

        lm_heatmap = mix[:,:,3::]
        lm_heatmap = torch.Tensor(lm_heatmap.transpose([2, 0, 1])) # convert to CxHxW


        # load label
        att = np.array(self.attr_label_list[index], dtype = np.float32)

        data = {
            'img': img,
            'landmark_heatmap': lm_heatmap,
            'att':att,
            'id': s_id
        }

        return data