from __future__ import division

import torchvision.transforms as transforms
from base_dataset import BaseDataset

from PIL import Image
import numpy as np
import os

import util.io as io

class AttributeDataset(BaseDataset):
    
    def name(self):
        return 'AttributeDataset'

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.data_root

        # get transform
        transform_list = []

        if opt.resize_or_crop == 'resize':
            # only resize image
            transform_list.append(transforms.Resize(opt.fine_size, Image.BICUBIC))

        elif opt.resize_or_crop == 'resize_and_crop':
            # scale and crop
            transform_list.append(transforms.Resize(opt.load_size, Image.BICUBIC))
            if opt.is_train:
                transform_list.append(transforms.RandomCrop(opt.fine_size))
            else:
                transform_list.append(transforms.CenterCrop(opt.fine_size))

        if opt.flip == 1 or (opt.flip == 0 and opt.is_train):
            transform_list.append(transforms.RandomHorizontalFlip())

        transform_list.append(transforms.ToTensor())

        if opt.image_normalize == 'imagenet':
            transform_list.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
        else:
            transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))

        self.transform = transforms.Compose(transform_list)


        # load sample list
        samples = io.load_json(os.path.join(opt.data_root, opt.fn_sample))
        attr_label = io.load_json(os.path.join(opt.data_root, opt.fn_label))
        attr_entry = io.load_json(os.path.join(opt.data_root, opt.fn_entry))
        attr_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))

        if opt.is_train == True:
            self.id_list = attr_split['train']
        else:
            self.id_list = attr_split['test']

        self.sample_list = [samples[s_id] for s_id in self.id_list]
        self.attr_label_list = [attr_label[s_id] for s_id in self.id_list]
        self.attr_entry = attr_entry

        # check data
        assert len(self.attr_entry) == len(self.attr_label_list[0]) == opt.n_attr, 'Attribute number not match!'

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):

        s_id = self.id_list[index]
        
        img = Image.open(self.sample_list[index]['img_path']).convert('RGB')
        img = self.transform(img)

        att = np.array(self.attr_label_list[index], dtype = np.float32)

        data = {
            'img': img,
            'att': att,
            'id': s_id
        }

        return data

