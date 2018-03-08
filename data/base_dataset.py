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
            - 'reduce_map': segmentation map 4-channel: head, hair, body, background
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
    elif mask_type == 'grid_map':
        # encode spatial information into upperbody/lowerbody channels
        # background=0, foreground in range of [-0.5, 0.5]
        mask = []
        for i in range(7):
            m = (seg_map==i).astype(np.float32)[:,:,0]
            if i in {3,4}:
                grid_x, grid_y = np.meshgrid(np.linspace(0,1,m.shape[1]), np.linspace(0,1,m.shape[0]))
                x_max = (grid_x * m).max()
                x_min = 1-((1-grid_x)*m).max()
                grid_x_std = ((grid_x-x_min)/(x_max-x_min+1e-5)-0.5) * m

                y_max = (grid_y * m).max()
                y_min = 1-((1-grid_y)*m).max()
                gird_y_std = ((grid_y-y_min)/(y_max-y_min+1e-5)-0.5) * m

                mask += [grid_x_std, gird_y_std]
            else:
                mask.append(m)
        mask = np.stack(mask, axis=2).astype(np.float32)
    return mask

def segmap_to_mask_v2(seg_map, nc = 7):
    mask = [(seg_map == i) for i in range(nc)]
    return np.concatenate(mask, axis=2).astype(np.float32)

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

def trans_random_horizontal_flip(img, coin=None):
    if coin is None:
        coin = np.random.rand()
    if coin >= 0.5:
        return cv2.flip(img, flipCode = 1) # horizontal flip
    else:
        return img

def trans_random_affine(input, scale=0.05):
    '''
    input: list of ndarray
    scale: float
    '''
    num_input = len(input)
    nc_cum = np.array([0]+[x.shape[2] for x in input], np.int).cumsum()
    input = np.concatenate(input, axis=2)

    w, h, c = input.shape[1], input.shape[0], input.shape[2]
    keypoint_src = np.array([[0,0], [w,0], [0,h]], dtype=np.float32)
    offset = (np.random.rand(3,2)*2-1) * np.array([w, h]) * scale
    keypoint_dst = (keypoint_src + offset).astype(np.float32)
    M = cv2.getAffineTransform(keypoint_src, keypoint_dst)

    if c <= 4:
        output_mix = cv2.warpAffine(input, M, dsize=(w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    else:
        output_mix = []
        for i in range(0, c, 4):
            out = cv2.warpAffine(input[:,:,i:(i+4)], M, dsize=(w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if out.ndim == 2:
                out = out[:,:,np.newaxis]
            output_mix.append(out)
        output_mix = np.concatenate(output_mix, axis=2)

    output = []
    for i in range(num_input):
        output.append(output_mix[:,:,nc_cum[i]:nc_cum[i+1]])
    return output


def get_color_patch(color_map, seg_map, mode):
    patches = []
    w, h = color_map.shape[1], color_map.shape[0]
    grid_x, grid_y = np.meshgrid(range(w), range(h))
    grid = np.stack((grid_x, grid_y), axis=2)
    for r in [3,4]:
        m = (seg_map==r)[:,:,0:1].astype(np.float32)
        area = m.sum()
        if mode == 'single':
            if r == 3:
                patch_size = 64
                m_patch = np.zeros((h,w,1), np.float32)
                m_patch[(h-patch_size)//2:(h+patch_size)//2, (w-patch_size)//2:(w+patch_size)//2] = 1
                patches.append(color_map * m_patch)
        elif mode == 'center':
            patch_size = 64
            if area>0:
                c = (grid*m).sum(axis=(0,1)) / area
                c = c.astype(np.int)
            else:
                c = np.array([w//2, h//2], np.int)
            m_patch = np.zeros((h,w,1), np.float32)
            m_patch[(c[1]-patch_size//2):(c[1]+patch_size//2), (c[0]-patch_size//2):(c[0]+patch_size//2)] = 1
            patches.append(color_map * m_patch * m)
        elif mode == 'crop5':
            patch_size = 32
            if area>0:
                c = (grid*m).sum(axis=(0,1)) / area
                l = (grid*m).max(axis=(0,1)) + ((max(w, h)-grid)*m).max(axis=(0,1)) - max(w,h)
            else:
                c = np.array([w//2, h//2], np.int)
                l = np.array([w//3, h//2], np.int)
            cs = [c, c-l*0.25, c+l*0.25, c+l*0.25*np.array([1,-1]), c+l*0.25*np.array([-1,1])]
            m_patch = np.zeros((h,w,1), np.float32)
            for p in cs:
                p = p.astype(np.int)
                m_patch[(p[1]-patch_size//2):(p[1]+patch_size//2), (p[0]-patch_size//2):(p[0]+patch_size//2)] = 1
            patches.append(color_map * m_patch * m)

    return patches


###############################################################################
