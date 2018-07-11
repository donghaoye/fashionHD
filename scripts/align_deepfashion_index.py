from __future__ import division, print_function

import os
import numpy as np
import tqdm
import torch
import imageio
import util.io as io
import shutil
import cv2
import pandas as pd
from collections import defaultdict



def similarity_cos(img_1, img_2):
    img_1 = img_1.flatten().astype(np.float)/255.
    img_2 = img_2.flatten().astype(np.float)/255.
    return np.dot(img_1, img_2)/np.sqrt(np.dot(img_1, img_1) * np.dot(img_2, img_2))

def similarity_cos_bat(imgs_1, imgs_2, normalized=False):
    if imgs_1.ndim == 3:
        imgs_1 = imgs_1[np.newaxis,:]
    if imgs_2.ndim == 3:
        imgs_2 = imgs_2[np.newaxis,:]

    n1 = imgs_1.shape[0]
    n2 = imgs_2.shape[0]
    imgs_1 = imgs_1.reshape(n1, -1).astype(np.float)
    imgs_2 = imgs_2.reshape(n2, -1).astype(np.float)
    m1 = np.dot(imgs_1, imgs_2.T) # (n1, n2)
    if normalized:
        return m1
    else:
        m2 = np.repeat(np.dot(imgs_1, imgs_1.T).diagonal()[:,np.newaxis], n2, 1)
        m3 = np.repeat(np.dot(imgs_2, imgs_2.T).diagonal()[np.newaxis,:], n1, 0)
        return m1 / np.sqrt(m2 * m3)
    
def similarity_l1_bat(imgs_1, imgs_2):
    if imgs_1.ndim == 3:
        imgs_1 = imgs_1[np.newaxis,:]
    if imgs_2.ndim == 3:
        imgs_2 = imgs_2[np.newaxis,:]
    n1 = imgs_1.shape[0]
    n2 = imgs_2.shape[0]
    imgs_1 = imgs_1.reshape(n1, -1).astype(np.float)
    imgs_2 = imgs_2.reshape(n2, -1).astype(np.float)
    rst = np.zeros((n1, n2))
    for i in range(n1):
        rst[i,:] = np.abs(imgs_2 - imgs_1[i:(i+1),:]).sum(axis=1)
    return rst

def align_dataset_index():
    '''
    Align the DeepFashion Inshop sample indices between VUnet version and DeformableGAN version
    '''
    # load sample index 1 (VUnet version)
    split = io.load_json('datasets/DF_Pose/Label/split.json')
    img_dir_1 = 'datasets/DF_Pose/Img/img_df/'
    samples_1 = {sid:(img_dir_1 + '%s.jpg'%sid) for sid in split['train']+split['test']}
    # load sample index 2 (deformableGAN version)
    train_list_2 = pd.read_csv('datasets/DF_Pose/DeformableGAN_Version/annotations/fasion-annotation-train.csv', sep=':')['name']
    train_dir_2 = 'datasets/DF_Pose/DeformableGAN_Version/images/train/'
    test_list_2 = pd.read_csv('datasets/DF_Pose/DeformableGAN_Version/annotations/fasion-annotation-test.csv', sep=':')['name']
    test_dir_2 = 'datasets/DF_Pose/DeformableGAN_Version/images/test/'
    samples_2 = [(fn, train_dir_2+fn) for fn in train_list_2] + [(fn, test_dir_2+fn) for fn in test_list_2]
    samples_2 = dict(samples_2)
    
    # debug
    # samples_1 = {k:v for k,v in samples_1.items()[0:100]}
    # samples_2 = samples_1
    # samples_2 = {k:v for k,v in samples_2.items()[0:100]}

    # load images into memory
    print('loading images 1 ...')
    images_1 = {}
    for sid, fn in tqdm.tqdm(samples_1.items()):
        images_1[sid] = imageio.imread(fn)
    images_2 = {}
    print('loading images 2 ...')
    for sid, fn in tqdm.tqdm(samples_2.items()):
        images_2[sid] = imageio.imread(fn)

    # group image by identity. alignment will be applied between identities instead of single images for efficiency.
    persion_ids_1 = defaultdict(lambda :{})
    for sid in samples_1.keys():
        pid = sid.split('_')[0]
        view = sid.split('_')[1][0]
        persion_ids_1[pid][view] = sid
    persion_ids_2 = defaultdict(lambda: {})
    for sid in samples_2.keys():
        pid = sid.split('_')[0]
        view = sid.split('_')[1][0]
        persion_ids_2[pid][view] = sid

    # align index
    print('aligning image index ...')
    num_cand = 3
    func_similarity = similarity_cos
    map_1to2 = {}
    map_1to2_cand = {}
    for pid_1, s_dict_1 in tqdm.tqdm(persion_ids_1.items()):
    # for pid_1, s_dict_1 in persion_ids_1.items():
        cand_list = [(None, -1)]
        s_list_1 = s_dict_1.items()
        s_list_1.sort()
        view_1, sid_1 = s_list_1[0]
        for pid_2, s_dict_2 in tqdm.tqdm(persion_ids_2.items()):
        # for pid_2, s_dict_2 in persion_ids_2.items():
            if view_1 not in s_dict_2:
                continue
            sid_2 = s_dict_2[view_1]
            score = func_similarity(images_1[sid_1], images_2[sid_2])
            i_insert = -1
            for i in range(len(cand_list)-1, -1, -1):
                if score < cand_list[i][1]:
                    break
                i_insert = i
            if i_insert >= 0:
                cand_list.insert(i_insert, (sid_2, score))
                if len(cand_list) > num_cand:
                    cand_list = cand_list[0:num_cand]
        
        map_1to2_cand[pid_1] = cand_list
        pid_2 = cand_list[0][0].split('_')[0]
        s_dict_2 = persion_ids_2[pid_2]
        for view_1, sid_1 in s_dict_1.iteritems():
            if view_1 in s_dict_2:
                map_1to2[sid_1] = s_dict_2[view_1]
            else:
                map_1to2[sid_1] = None

    # output result
    output_dir = 'temp/search_image/DeepFashion_Inshop_VUnetversion_to_DeformableGANversion/'
    io.mkdir_if_missing(output_dir)
    io.save_json(map_1to2, output_dir + 'map.json')
    io.save_json(map_1to2_cand, output_dir + 'map_cand.json')
    # output samples
    io.mkdir_if_missing(output_dir + 'vis')
    print('output visualization ...')
    for sid_1, sid_2 in tqdm.tqdm(map_1to2.iteritems()):
        if sid_2 is not None:
            fn = output_dir + 'vis/%s-%s.jpg'%(sid_1, sid_2)
            img = np.hstack((images_1[sid_1], images_2[sid_2]))
            imageio.imwrite(fn, img)

def align_dataset_index_bat():
    '''
    Align the DeepFashion Inshop sample indices between VUnet version and DeformableGAN version
    '''
    # load sample index 1 (VUnet version)
    split = io.load_json('datasets/DF_Pose/Label/split.json')
    img_dir_1 = 'datasets/DF_Pose/Img/img_df/'
    samples_1 = {sid:(img_dir_1 + '%s.jpg'%sid) for sid in split['train']+split['test']}
    # load sample index 2 (deformableGAN version)
    train_list_2 = pd.read_csv('datasets/DF_Pose/DeformableGAN_Version/annotations/fasion-annotation-train.csv', sep=':')['name']
    train_dir_2 = 'datasets/DF_Pose/DeformableGAN_Version/images/train/'
    test_list_2 = pd.read_csv('datasets/DF_Pose/DeformableGAN_Version/annotations/fasion-annotation-test.csv', sep=':')['name']
    test_dir_2 = 'datasets/DF_Pose/DeformableGAN_Version/images/test/'
    samples_2 = [(fn, train_dir_2+fn) for fn in train_list_2] + [(fn, test_dir_2+fn) for fn in test_list_2]
    samples_2 = dict(samples_2)
    
    # debug
    # samples_1 = {k:v for k,v in samples_1.items()[0:100]}
    # samples_2 = samples_1
    # samples_2 = {k:v for k,v in samples_2.items()[0:200]}

    # load images into memory
    print('loading images 1 ...')
    images_1 = {}
    for sid, fn in tqdm.tqdm(samples_1.items()):
        images_1[sid] = imageio.imread(fn)

    images_2 = {}
    print('loading images 2 ...')
    for sid, fn in tqdm.tqdm(samples_2.items()):
        images_2[sid] = imageio.imread(fn)

    # group image by identity. alignment will be applied between identities instead of single images for efficiency.
    persion_ids_1 = defaultdict(lambda :{})
    for sid in samples_1.keys():
        pid = sid.split('_')[0]
        view = sid.split('_')[1][0]
        persion_ids_1[pid][view] = sid
    persion_ids_2 = defaultdict(lambda: {})
    for sid in samples_2.keys():
        pid = sid.split('_')[0]
        view = sid.split('_')[1][0]
        persion_ids_2[pid][view] = sid

    # align index
    print('aligning image index ...')
    num_cand = 3
    block_size = 1000
    func_similarity = similarity_l1_bat
    map_1to2 = {}
    map_1to2_cand = {}
    for pid_1, s_dict_1 in tqdm.tqdm(persion_ids_1.items()):
    # for pid_1, s_dict_1 in persion_ids_1.items():
        s_list_1 = s_dict_1.items()
        s_list_1.sort()
        view_1, sid_1 = s_list_1[0]
        sid_list_2 = [p[view_1] for p in persion_ids_2.values() if view_1 in p]
        img_1 = images_1[sid_1]
        imgs_2 = np.array([images_2[sid_2] for sid_2 in sid_list_2])
        scores = np.array([])
        for i in range(0, imgs_2.shape[0], block_size):
            imgs_2_block = imgs_2[i:(i+block_size),:]
            scores_block = func_similarity(img_1, imgs_2_block).flatten()
            scores = np.concatenate((scores, scores_block))
        sorted_indices = np.argsort(scores)[::-1]
        cand_list = [(sid_list_2[i], scores[i]) for i in sorted_indices[0:num_cand]]

        map_1to2_cand[pid_1] = cand_list
        pid_2 = cand_list[0][0].split('_')[0]
        s_dict_2 = persion_ids_2[pid_2]
        for view_1, sid_1 in s_dict_1.iteritems():
            if view_1 in s_dict_2:
                map_1to2[sid_1] = s_dict_2[view_1]
            else:
                map_1to2[sid_1] = None

    # output result
    output_dir = 'temp/search_image/DeepFashion_Inshop_VUnetversion_to_DeformableGANversion_bat_l1/'
    io.mkdir_if_missing(output_dir)
    io.save_json(map_1to2, output_dir + 'map.json')
    io.save_json(map_1to2_cand, output_dir + 'map_cand.json')
    # output samples
    io.mkdir_if_missing(output_dir + 'vis')
    print('output visualization ...')
    for sid_1, sid_2 in tqdm.tqdm(map_1to2.iteritems()):
        if sid_2 is not None:
            fn = output_dir + 'vis/%s-%s.jpg'%(sid_1, sid_2)
            img = np.hstack((images_1[sid_1], images_2[sid_2]))
            imageio.imwrite(fn, img)


if __name__ == '__main__':
    #############################
    # Align DeepFashion Inshop index (VUnet version vs. DeformableGAN version)
    #############################
    # align_dataset_index()
    align_dataset_index_bat()