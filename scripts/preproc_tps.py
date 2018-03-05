from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np
from collections import defaultdict

import util.io as io
import util.image as image
import cv2


design_root = 'datasets/DeepFashion/Fashion_design/'

def create_train_pair():
    samples = io.load_json(design_root + 'Label/ca_samples.json')
    split = io.load_json(design_root + 'Split/ca_gan_split_trainval_upper.json')
    cat_label = io.load_data(design_root + 'Label/ca_cat_label.pkl')
    cat_entry = io.load_json(design_root + 'Label/cat_entry.json')
    
    # group samples by category label
    cat_to_ids = defaultdict(lambda :[])
    for s_id in split['train']:
        c = cat_label[s_id]
        cat_to_ids[c].append(s_id)

    n = 0
    pair_list = []
    for c, s_list in cat_to_ids.iteritems():
        print('[%d/%d] %s: %d samples...' % (n, len(cat_to_ids), cat_entry[c]['entry'], len(s_list)))
        n += 1
        s_list_org = [s_id for s_id in s_list]
        for i in range(len(s_list)-1):
            j = np.random.randint(i+1, len(s_list))
            temp = s_list[i]
            s_list[i] = s_list[j]
            s_list[j] = temp
        pair_list += zip(s_list_org, s_list)

    pair_dict = {s_tar:s_src for s_tar, s_src in pair_list}
    io.save_json(pair_dict, design_root + 'Label/ca_tps_pair.json')

    io.save_str_list(pair_dict.keys(), design_root + 'Temp/ca_tps_tar.txt')
    io.save_str_list(pair_dict.values(), design_root + 'Temp/ca_tps_src.txt')


def create_cloth_edge_map():
    '''
    create edge map that only contains cloth edge (inside the cloth mask)
    '''
    # config
    mask_dilate = 5

    seg_dir = design_root + 'Img/seg_ca_syn_256/'
    edge_dir = design_root + 'Img/edge_ca_256/'
    output_dir = design_root + 'Img/edge_ca_256_cloth/'
    io.mkdir_if_missing(output_dir)

    split = io.load_json(design_root + 'Split/ca_gan_split_trainval_upper.json')
    id_list = split['train'] + split['test']

    for i, s_id in enumerate(id_list):
        print('%d/%d' % (i, len(id_list)))
        seg_map = image.imread(seg_dir + s_id + '.bmp', 'grayscale')
        edge_map = image.imread(edge_dir + s_id + '.jpg', 'grayscale')
        assert seg_map.shape == edge_map.shape
        mask = np.logical_or(seg_map==3, seg_map==4).astype(np.uint8)
        mask = cv2.dilate(mask, kernel = np.ones((mask_dilate, mask_dilate)))
        edge_map_cloth = edge_map * mask
        image.imwrite(edge_map_cloth, output_dir + s_id + '.jpg')


def create_uncertain_segmap():
    '''
    reset the region label of a segmap (orignal 7 channel, 0-background, 1-hair, 2-head, 3-upper, 4-lower, 5-arm, 6-leg):
        0-background
        1-head
        2-hair
        3-cloth (both upper and lower)
        4-cloth_uncertain (boundary area of cloth region 3)
        5-arm_uncertain (arm region + sleeve_points region)
        6-leg
    '''

    # config
    cloth_mefilt_size = 15
    cloth_dilate_size = 15
    cloth_dilate_neck_size = 15
    cloth_erode_size = 15
    sleeve_point_dilate_size = 21
    img_size = 256

    # load data
    seg_dir = design_root + 'Img/seg_ca_syn_256/'
    output_dir = design_root + 'Img/seg_ca_syn_256_uncertain/'
    io.mkdir_if_missing(output_dir)

    split = io.load_json(design_root + 'Split/ca_gan_split_trainval_upper.json')
    lm_label = io.load_data(design_root + 'Label/ca_landmark_label_256.pkl')
    id_list = split['train'] + split['test']
    id_list = id_list[0:10]

    # create grid
    gx, gy = np.meshgrid(range(img_size), range(img_size))
    for i, s_id in enumerate(id_list):
        print('%d/%d' % (i, len(id_list)))
        seg_map = image.imread(seg_dir + s_id + '.bmp', 'grayscale')
        # unchanged part: head, hair and leg
        hair_mask = seg_map == 1
        head_mask = seg_map == 2
        leg_mask = seg_map == 6
        # cloth
        cloth_mask = np.logical_or(seg_map==3, seg_map==4).astype(np.uint8)
        cloth_mask = cv2.medianBlur(cloth_mask, ksize=cloth_mefilt_size)
        # cloth uncertain
        cloth_uct_mask_outer = cv2.dilate(cloth_mask, kernel=np.ones((cloth_dilate_size, cloth_dilate_size),np.uint8))
        cloth_uct_mask_inner = cv2.erode(cloth_mask, kernel=np.ones((cloth_erode_size, cloth_erode_size),np.uint8))
        cloth_uct_mask = cloth_uct_mask_outer - cloth_uct_mask_inner
        cloth_mask = cloth_mask.astype(np.bool)
        cloth_uct_mask = cloth_uct_mask.astype(np.bool)
        # arm uncertain
        arm_uct_mask = (seg_map==5)
        slv_points = np.array(lm_label[s_id][2:4])
        for j, p in enumerate(slv_points):
            # if p[2] == 1:
            if 1:
                slv_rg = ((gx-p[0])**2 + (gy-p[1])**2 <= sleeve_point_dilate_size**2)
                slv_rg = np.logical_and(cloth_uct_mask_outer, slv_rg)
                arm_uct_mask = np.logical_or(arm_uct_mask, slv_rg)
        # combine channels
        seg_org = seg_map.copy()
        seg_map[:] = 0
        seg_map[hair_mask] = 1
        seg_map[head_mask] = 2
        seg_map[cloth_mask] = 3
        seg_map[cloth_uct_mask] = 4
        seg_map[arm_uct_mask] = 5
        seg_map[leg_mask] = 6

        # save
        image.imwrite(np.concatenate((seg_org, slv_rg.astype(np.uint8)),axis=0)*20, output_dir + s_id + '.bmp')
        # image.imwrite(np.concatenate((seg_org, seg_map),axis=0)*20, output_dir + s_id + '.bmp')
        # image.imwrite(seg_map, output_dir + s_id + '.bmp')





if __name__ == '__main__':
    # create_train_pair()
    # create_cloth_edge_map()
    create_uncertain_segmap()


