from __future__ import division, print_function
import torch.utils.data as data
import numpy as np
from PIL import Image
import cv2

#####################################
# BaseDataset Class
#####################################

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


#####################################
# Image Transform Modules
#####################################

def landmark_to_heatmap(img_sz, lm_label, cloth_type, delta = 15.):
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
    for x_lm, y_lm, v in lm_label:
        if v == 2:
            channel = np.zeros((h, w))
        else:
            channel = np.exp(-((x_grid - x_lm)**2 + (y_grid - y_lm)**2)/(delta**2))
        channels.append(channel)

    channels = np.stack(channels).astype(np.float32)

    if cloth_type == 1:
        assert channels.shape[0] == 6, 'upperbody cloth (1) should have 6 landmarks'
        heatmap[0:6,:] = channels
    elif cloth_type == 2:
        assert channels.shape[0] == 4, 'lowerbody cloth (2) should have 4 landmarks'
        heatmap[6:10,:] = channels
    elif cloth_type == 3:
        assert channels.shape[0] == 8, 'fullbody cloth (3) should have 8 landmarks'
        heatmap[10:18,:] = channels
    else:
        raise ValueError('invalid cloth type %d' % cloth_type)

    return heatmap.transpose([1,2,0]) # transpose to HxWxC

def segmap_to_mask(seg_map, mask_type, cloth_type):
    '''
    Generate a mask from a segmentation map.
    Input:
        seg_map(np.ndarray): segmentation map of size (HxWx1)
            0-background, 1-hair, 2-head, 3-upperbody, 4-lowerbody, 5-leg, 6-arm
        mask_type(str):
            - 'foreground': body+cloth
            - 'body': arm+leg+cloth (no head, hair, background)
            - 'target': target cloth
            - 'map': output segmentation map in one-hot format
        cloth_type(int):
            - 1: upperdody
            - 2: lowerbody
            - 3: full body
    '''
    if seg_map.ndim == 2:
        seg_map = seg_map[:,:,np.newaxis]

    if mask_type == 'foreground':
        mask = (seg_map != 0).astype(np.float32)
    elif mask_type == 'body':
        mask = (seg_map >=3).astype(np.float32)
    elif mask_type == 'target':
        if cloth_type == 1:
            mask = (seg_map == 3).astype(np.float32)
        elif cloth_type == 2:
            mask = (seg_map == 4).astype(np.float32)
        elif cloth_type == 3:
            mask = np.logical_or(seg_map == 3, seg_map == 4).astype(np.float32)
    elif mask_type == 'map':
        mask = [(seg_map==i) for i in range(7)]
        mask = np.concatenate(mask, axis=2).astype(np.float32)

    return mask


def trans_resize(img, size, interp = cv2.INTER_LINEAR):
    '''
    img (np.ndarray or list): image with arbitrary channels, with size HxWxC
    size (tuple): target size (width, height)
    interps (int or list)
    '''
    return cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)


def trans_center_crop(img, size):
    '''
    img (np.ndarray): image with arbitrary channels, with size HxWxC
    size (tuple): size of cropped patch (width, height)
    '''
    h, w = img.shape[0:2]
    tw, th = size
    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))

    return img[i:(i+th), j:(j+tw), :]

def trans_random_crop(img, size):
    h, w = img.shape[0:2]
    tw, th = size
    i = np.random.randint(0, h-th+1)
    j = np.random.randint(0, w-tw+1)

    return img[i:(i+th), j:(j+tw), :]

def trans_random_horizontal_flip(img):
    if np.random.rand() >= 0.5:
        return cv2.flip(img, flipCode = 1) # horizontal flip
    else:
        return img

###############################################################################