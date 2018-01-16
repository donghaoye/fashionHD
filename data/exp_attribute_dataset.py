from __future__ import division

import torchvision.transforms as transforms
from base_dataset import BaseDataset

from PIL import Image
import cv2
import numpy as np
import os
import util.io as io


def landmark_to_heatmap(img_sz, lm_label, cloth_type, delta = 6.):
    '''
    Generate a landmark heatmap from landmark coordinates
    Input:
        img_sz (tuple):     size of heatmap in (width, height)
        lm_label (list):    list of (x,y) coordinates. The length depends on the cloth type: 6 for upperbody
                            4 for lowerbody, 8 for fullbody
        cloth_type(int):    1 for upperbody, 2 for lowerbody, 3 for fullbody
        delta:              parameter to adjuct heat extent of each landmark
    Output:
        lm_heatmap (np.ndarray): landmark heatmap of size H x W x C
    '''

    num_channel = 18
    w, h = img_sz
    heatmap = np.zeros((num_channel, h, w), dtype = np.float32)

    x_grid, y_grid = np.meshgrid(range(w), range(h), indexing = 'xy')

    channels = []
    for x_lm, y_lm in lm_label:
        channel = np.exp(-((x_grid - x_lm)**2 + (y_grid - y_lm)**2)/(delta**2))
        channels.append(channel)

    channels = np.stack(channels).astype(np.float32)

    if cloth_type == 1:
        assert channels.shape[0] == 6, 'upperbody cloth (1) should have 6 landmarks'
        heatmap[0:6] = channels
    elif cloth_type == 2:
        assert channels.shape[0] == 4, 'lowerbody cloth (2) should have 4 landmarks'
        heatmap[6:10] = channels
    elif cloth_type == 3:
        assert channels.shape[0] == 8, 'fullbody cloth (3) should have 8 landmarks'
        heatmap[10:18] = channels

    return heatmap.transpose([1,2,0]) # transpose to HxWxC


###############################################################################
# Image Transforms Based on openCV
###############################################################################

def _trans_resize(img, size):
    '''
    img (np.ndarray): image with arbitrary channels, with size HxWxC
    size (tuple): target size (width, height)
    '''

    return cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)


def _trans_center_crop(img, size):
    '''
    img (np.ndarray): image with arbitrary channels, with size HxWxC
    size (tuple): size of cropped patch (width, height)
    '''
    h, w = img.shape[0:2]
    tw, th = size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))

    return img[i:(i+th), j:(j+tw), :]

def _trans_random_crop(img, size):
    h, w = img.shape[0:2]
    tw, th = size
    i = np.random.randint(0, h-th+1)
    j = np.random.randint(0, w-tw+1)

    return img[i:(i+th), j:(j+tw), :]

def _trans_random_horizontal_flip(img):
    if np.random.rand() >= 0.5:
        return cv2.flip(img, flipCode = 1) # horizontal fhip
    else:
        return img

###############################################################################

class EXPAttributeDataset(BaseDataset):

    def name(self):
        return 'EXPAttributeDataset'

    def initialize(self, opt, spilt):
        self.opt = opt
        self.root = opt.data_root
        self.split = split

        print('loading data ...')
        samples = io.load_json(os.path.join(opt.data_root, opt.fn_sample))
        attr_label = io.load_data(os.path.join(opt.data_root, opt.fn_label))
        attr_entry = io.load_json(os.path.join(opt.data_root, opt.fn_entry))
        attr_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))
        lm_label = iio.load_data(os.path.join(opt.data_root, opt.fn_landmark))

        self.id_list = attr_split[split]
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
        img = cv2.imread(self.sample_list[index]['image_path'])
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

        mix = np.concatenate((img, lm_heatmap))

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
        lm_heatmap = torch.Tensor(lm_heatmap.transpose([2, 0, 1]))


        # load label
        att = np.array(self.attr_label_list[index], dtype = np.float32)

        data = {
            'img': img,
            'landmark_heatmap': lm_heatmap,
            'att':att,
            'id': s_id
        }

        return data