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

def pose_to_heatmap(img_sz, label, size):
    '''
    generate a pose heatmap from pose label
    Input:
        im_sz:      size of image
        label:   list of (x,y) with a lenght of C
        size:       mark size
    Output:
        lm_heatmap: np.ndarray of size HxWxC
    '''

    num_channel = len(label)
    w, h = img_sz
    heat_map = np.zeros((h, w, num_channel), dtype = np.float32)
    radius = int(size)//2

    for i, p in enumerate(label):
        x, y = int(p[0]), int(p[1])
        if [x,y] != [-1,-1]:
            heat_map[(y-radius):(y+radius), (x-radius):(x+radius), i] = 1

    return heat_map

def pose_to_map(img_sz, label, mode='gaussian', radius=5):
    '''
    pose keypoint cordinates to spatial map
    mode: 'gaussian', 'binary'
    '''
    w, h = img_sz
    x_grid, y_grid = np.meshgrid(range(w), range(h), indexing = 'xy')
    m = []
    for x, y in label:
        if x == -1 or y == -1:
            m.append(np.zeros((h, w)).astype(np.float32))
        else:
            if mode == 'gaussian':
                m.append(np.exp(-((x_grid - x)**2 + (y_grid - y)**2)/(radius**2)).astype(np.float32))
            elif mode == 'binary':
                m.append(((x_grid-x)**2 + (y_grid-y)**2 <= radius**2).astype(np.float32))
            else:
                raise NotImplementedError()
    m = np.stack(m, axis=2)
    return m

def pose_to_stickman(img_sz, label):
    w, h = img_sz
    scale_factor = min(w, h)//128
    thickness = int(3*scale_factor)
    m = [np.zeros((h, w), dtype=np.float32) for _ in range(3)]
    # check valid points
    valid = [1 if (x!=-1 and y!=-1) else 0 for x,y in label]
    # body area (channel 2): the polygon with vertext {lhip, 11}-{lshoulder, 5}-{rshoulder, 2}-{rhip, 8}
    body_pt_idx = [11, 5, 2, 8]
    body_pts = np.array([label[i] for i in body_pt_idx if valid[i]], dtype=np.int)
    if body_pts.shape[0] > 2:
        cv2.fillPoly(m[2], [body_pts], color=1.0)
    # left line (channel 0): {lankle, 13}-{lknee, 12}-{lhip, 11}-{lshoulder, 5}-{lelbow, 6}-{lwrist, 7}
    left_pt_idx = [13, 12, 11, 5, 6, 7]
    for i1, i2 in zip(left_pt_idx[0:-1], left_pt_idx[1::]):
        if valid[i1] and valid[i2]:
            p1 = tuple(np.int_(label[i1]))
            p2 = tuple(np.int_(label[i2]))
            cv2.line(m[0], p1, p2, color=1.0, thickness=thickness)
    # right line (channel 1): {rankle, 10}-{rknee, 9}-{rhip, 8}-{rshoulder, 2}-{relbow, 3}-{rwrist, 4}
    right_pt_idx = [10, 9, 8, 2, 3, 4]
    for i1, i2 in zip(right_pt_idx[0:-1], right_pt_idx[1::]):
        if valid[i1] and valid[i2]:
            p1 = tuple(np.int_(label[i1]))
            p2 = tuple(np.int_(label[i2]))
            cv2.line(m[1], p1, p2, color=1.0, thickness=thickness)
    # neck line (channel 0&1): {lshoulder, 5}, {rshoulder, 2}, {nose, 0}, {neck, 1}
    if valid[2] and valid[5] and valid[0]:
        p_nose = tuple(np.int_(label[0]))
        p_neck = tuple(np.int_([label[2][0]+label[5][0], label[2][1]+label[5][1]])//2)
        cv2.line(m[0], p_nose, p_neck, color=0.5, thickness=thickness)
        cv2.line(m[1], p_nose, p_neck, color=0.5, thickness=thickness)
    # eye-nose line (channel 0&1): {nose, 0}, {leye, 15}, {reye, 14}
    if valid[0] and valid[15]:
        p_leye = tuple(np.int_(label[15]))
        p_nose = tuple(np.int_(label[0]))
        cv2.line(m[0], p_nose, p_leye, color=1, thickness=thickness)
    if valid[0] and valid[14]:
        p_reye = tuple(np.int_(label[14]))
        p_nose = tuple(np.int_(label[0]))
        cv2.line(m[1], p_nose, p_reye, color=1, thickness=thickness)

    m = np.stack(m, axis=2)
    return m



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

def segmap_to_mask_v2(seg_map, nc = 7, bin_size=1):
    mask = [(seg_map == i) for i in range(nc)]
    mask = np.concatenate(mask, axis=2).astype(np.float32)
    if bin_size > 1:
        h, w = mask.shape[0:2]
        dh, dw = h//bin_size, w//bin_size
        mask = cv2.resize(mask, dsize=(dw,dh), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
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

def trans_random_horizontal_flip(img, coin=None):
    if coin is None:
        coin = np.random.rand()
    if coin >= 0.5:
        img = cv2.flip(img, flipCode = 1) # horizontal flip
        if len(img.shape)==2:
            img = img[:,:,np.newaxis]
    return img

def trans_random_horizontal_flip_pose(pose, coin=None):
    # pose pair: 2-5, 3-6, 4-7, 8-11, 9-12, 10-13, 14-15 16-17 
    if coin is None:
        coin = np.random.rand()
    if coin >= 0.5:
        pose = cv2.flip(pose, flipCode = 1)
        pose = pose[:,:,[0,1,5,6,7,2,3,4,11,12,13,8,9,10,15,14,17,16]]

    return pose

def trans_random_horizontal_flip_pose_c(pose_c, img_sz, coin=None):
    # pose_c: [[x0, y0], [x1, y1], ... , [x17, y17]]
    if coin is None:
        coin = np.random.rand()
    if coin > 0.5:
        pose_c = [pose_c[i] for i in [0,1,5,6,7,2,3,4,11,12,13,8,9,10,15,14,17,16]]
        w, h = img_sz
        for i, (x, y) in enumerate(pose_c):
            if x > 0 and y > 0:
                x = w - 1 - x
                y = h - 1 - y
                pose_c[i] = [x, y]
    return pose_c

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


def trans_random_perspective(input, scale=0.05):
    '''
    input: list of ndarray
    scale: float
    '''
    num_input = len(input)
    nc_cum = np.array([0]+[x.shape[2] for x in input], np.int).cumsum()
    input = np.concatenate(input, axis=2)

    w, h, c = input.shape[1], input.shape[0], input.shape[2]
    keypoint_src = np.array([[0,0], [w,0], [0,h], [w,h]], dtype=np.float32)
    offset = (np.random.rand(4,2)*2-1) * np.array([w,h])*scale
    keypoint_dst = (keypoint_src + offset).astype(np.float32)
    M = cv2.getPerspectiveTransform(keypoint_src, keypoint_dst)
    output_mix = cv2.warpPerspective(input, M, dsize=(w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
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
