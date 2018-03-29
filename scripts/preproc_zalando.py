from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np
import cv2
from collections import defaultdict

import util.io as io
# import util.image as image

zalando_root = 'datasets/Zalando/'

def create_split():
    '''
    create split file. follow the partition used in VITON paper
    '''
    train_pairs = io.load_str_list(zalando_root + 'Source/viton_train_pairs.txt')
    test_piars = io.load_str_list(zalando_root + 'Source/viton_test_pairs.txt')

    split = {}

    for subset, pairs in [('train', train_pairs), ('test', test_piars)]:
        id_list = [p.split()[0][0:6] for p in pairs]
        split[subset] = id_list

    split['debug'] = split['train'][0:32]

    io.mkdir_if_missing(zalando_root + 'Split')
    io.save_json(split, zalando_root + 'Split/zalando_split.json')
    # io.save_json(split_debug, zalando_root + 'Split/debug_zalando_split.json')


def create_pose_label():
    '''
    create 18-keypoint pose label. follow the setting in VITON
    '''
    pose = io.load_data(zalando_root + 'Source/pose.pkl')
    split = io.load_json(zalando_root + 'Split/zalando_split.json')
    id_list = split['train'] + split['test']

    pose_label = {}
    for idx, s_id in enumerate(id_list):
        print('%d / %d' % (idx, len(id_list)))
        subset = pose[s_id + '_0']['subset'] # [i1, i2, ..., in, totial_score, n]
        candidate = pose[s_id + '_0']['candidate'] # [[x_i, y_i, score_i, id_i]]
        label = []
        for i in subset[0][0:-2]:
            i = int(i)
            if i == -1:
                label.append([-1, -1])
            else:
                x = candidate[i][0]
                y = candidate[i][1]
                label.append([x, y])
        pose_label[s_id] = label

    io.save_data(pose_label, zalando_root + 'Label/zalando_pose_label.pkl')


def visualize_pose():
    '''
    visualize pose label
    '''
    num_vis = 10

    # img_root = zalando_root + 'Img/img_zalando/'
    # pose_label = io.load_data(zalando_root + 'Label/zalando_pose_label.pkl')
    
    img_root = zalando_root + 'Img/img_zalando_256/'
    pose_label = io.load_data(zalando_root + 'Label/zalando_pose_label_256.pkl')

    output_dir = 'temp/zalando_pose_vis/'
    io.mkdir_if_missing(output_dir)

    for i, (s_id, pose) in enumerate(pose_label.items()[0:num_vis]):
        print('%d / %d' % (i, num_vis))
        img = cv2.imread(img_root + s_id + '_0.jpg')
        assert img is not None
        for p in pose:
            if p[0] != -1:
                c = (int(p[0]), int(p[1]))
                cv2.circle(img, center=c, radius=5, color=(0,255,0), thickness=-1)
            cv2.imwrite(output_dir + s_id + '.jpg', img)

def visualize_seg():
    '''
    visualize segmentation
    '''

    num_vis = 100
    img_root = zalando_root + 'Img/img_zalando_256/'
    seg_root = zalando_root + 'Img/seg_zalando_256/'
    output_dir = 'temp/zalando_seg_vis/'
    io.mkdir_if_missing(output_dir)

    split = io.load_json(zalando_root + 'Split/zalando_split.json')
    id_list = split['train'] + split['test']

    for s_id in id_list[0:num_vis]:
        img = cv2.imread(img_root + s_id + '_0.jpg')
        seg = cv2.imread(seg_root + s_id + '_0.bmp')        
        seg = seg * 20
        img = np.concatenate((img, seg), axis = 1)
        cv2.imwrite(output_dir + s_id + '.jpg', img)


def resize_and_pad():
    '''
    resize the image that its longer side equals to new_size. Then pad the image to have the size [new_size, new_size]
    create new pose label at the same time
    '''

    # config
    new_size = 256

    img_root = zalando_root + 'Img/img_zalando/'
    output_dir = zalando_root + 'Img/img_zalando_%d/' % new_size
    split = io.load_json(zalando_root + 'Split/zalando_split.json')
    pose_label = io.load_data(zalando_root + 'Label/zalando_pose_label.pkl')

    io.mkdir_if_missing(output_dir)
    id_list = split['train'] + split['test']
    # id_list = id_list[0:10]
    new_pose_label = {}

    for i, s_id in enumerate(id_list):
        print('%d / %d' % (i, len(id_list)))
        # resize image
        img = cv2.imread(img_root + s_id + '_0.jpg')
        w, h = img.shape[1], img.shape[0]
        if w < h:
            top = 0
            bottom = 0
            left = (h-w)//2
            right = h-w-left
            ratio = new_size/h
        else:
            top = (w-h)//2
            bottom = w-h-top
            left = 0
            right = 0
            ratio = new_size/w

        img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
        img = cv2.resize(img, dsize=(new_size, new_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_dir + s_id + '_0.jpg', img)

        # resize clothing image
        img1 = cv2.imread(img_root + s_id + '_1.jpg')
        if not (img1.shape[0] == h and img1.shape[1] == w):
            img1 = cv2.resize(img1, dsize=(w,h))
        img1 = cv2.copyMakeBorder(img1, top, bottom, left, right, borderType=cv2.BORDER_REPLICATE)
        img1 = cv2.resize(img1, dsize=(new_size, new_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_dir + s_id + '_1.jpg', img1)

        # modify pose label
        pose = pose_label[s_id]
        new_pose = [[(p[0]+left)*ratio, (p[1]+top)*ratio] if p != [-1,-1] else [-1,-1] for p in pose]
        new_pose_label[s_id] = new_pose

    io.save_data(new_pose_label, zalando_root + 'Label/zalando_pose_label_%d.pkl' % new_size)


def rename_files():
    split = io.load_json(zalando_root + 'Split/zalando_split.json')
    id_list = split['train'] + split['test']
    dir_list = ['Img/img_zalando_256/', 'Img/edge_zalando_256/', 'Img/seg_zalando_256/']

    for d in dir_list:
        ext = '.bmp' if 'seg_' in d else '.jpg'
        for s_id in id_list:
            fn_src = zalando_root + d + s_id + '_0' + ext
            fn_tar = zalando_root + d + s_id + ext
            print('%s => %s' % (fn_src, fn_tar))
            shutil.move(fn_src, fn_tar)

def create_cloth_edge_map():
    '''
    extract edges inside the cloth region
    '''
    mask_dilate = 5

    split = io.load_json(zalando_root + 'Split/zalando_split.json')
    id_list = split['train'] + split['test']
    edge_dir = zalando_root + 'Img/edge_zalando_256/'
    seg_dir = zalando_root + 'Img/seg_zalando_256/'
    output_dir = zalando_root + 'Img/edge_zalando_256_cloth/'
    io.mkdir_if_missing(output_dir)

    for i, s_id in enumerate(id_list):
        print('%d/%d' % (i, len(id_list)))
        edge = cv2.imread(edge_dir + s_id + '.jpg', cv2.IMREAD_GRAYSCALE)
        seg = cv2.imread(seg_dir + s_id + '.bmp', cv2.IMREAD_GRAYSCALE)
        assert edge.shape == seg.shape
        mask = ((seg==3) | (seg==4)).astype(np.uint8)
        mask = cv2.dilate(mask, kernel = np.ones((mask_dilate, mask_dilate)))
        edge_cloth = edge * mask
        cv2.imwrite(output_dir + s_id + '.jpg', edge_cloth)




if __name__ == '__main__':
    # create_split()
    # create_pose_label()
    # visualize_pose()
    # resize_and_pad()
    # visualize_seg()
    # rename_files()
    create_cloth_edge_map()

