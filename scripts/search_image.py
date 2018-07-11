from __future__ import division, print_function

import os
import numpy as np
import tqdm
import torch
import imageio
import util.io as io
import shutil
import cv2



def search_image(query_fn_list, gallery_fn_list, output_dir, method='cos'):
    num_cand = 20;
    cache_image = True

    if method == 'cos':
        func_similarity = similarity_cos
    if cache_image:
        img_g_dict = dict()

    io.mkdir_if_missing(output_dir)
    result = []
    for idx,fn_q in enumerate(query_fn_list):
        print('searching sample %d/%d' % (idx, len(query_fn_list)))
        io.mkdir_if_missing(output_dir+'/%d/'%idx)
        img_q = imageio.imread(fn_q)
        cand_list = [(None, None, -1)]
        for fn_g in tqdm.tqdm(gallery_fn_list):
            if cache_image:
                if fn_g in img_g_dict:
                    img_g = img_g_dict[fn_g]
                else:
                    img_g = imageio.imread(fn_g)
                    img_g_dict[fn_g] = img_g
            else:
                img_g = imageio.imread(fn_g)
            score = func_similarity(img_q, img_g)
            i_insert = -1
            for i in range(len(cand_list)):
                if score > cand_list[i][2]:
                    i_insert = i
                    break
            if i_insert >= 0:
                cand_list.insert(i_insert, (fn_g, img_g, score))
                if len(cand_list) > num_cand:
                    cand_list = cand_list[0:num_cand]

        imageio.imwrite(output_dir+'/%d/query.jpg'%idx, img_q)
        for i, (fn_g, img_g, score) in enumerate(cand_list):
            if fn_g:
                imageio.imwrite(output_dir + '/%d/cand_%d.jpg'%(idx, i), img_g)
        output_info = [fn_q]
        output_info += ['%d %f %s' % (i, score, fn) for i, (fn, _, score) in enumerate(cand_list)]
        io.save_str_list(output_info, output_dir + '/%d/result.txt'%idx)
        result.append('%d %s %s' % (idx, fn_q, cand_list[0][0]))

    io.save_str_list(result, output_dir+'result.txt')

def similarity_cos(img_1, img_2):
    img_1 = img_1.flatten().astype(np.float)/255.
    img_2 = img_2.flatten().astype(np.float)/255.
    return np.dot(img_1, img_2)/np.sqrt(np.dot(img_1, img_1) * np.dot(img_2, img_2))

def prepro_image(img_dir, output_dir):

    h_out, w_out = [256, 256]
    io.mkdir_if_missing(output_dir)
    for fn in tqdm.tqdm(os.listdir(img_dir)):
        img_org = cv2.imread(img_dir+fn)
        h, w = img_org.shape[0:2]
        if h/h_out > w/w_out:
            w1 = int(w/h*h_out)
            img_rsz = cv2.resize(img_org, (w1, h_out))
            img_out = np.ones((h_out, w_out, 3), dtype=np.uint8)*255
            img_out[:,((w_out-w1)//2):((w_out-w1)//2+w1),:] = img_rsz
        else:
            h1 = int(h/w*w_out)
            img_rsz = cv2.resize(img_org, (w_out, h1))
            img_out = np.ones((h_out, w_out, 3), dtype=np.uint8)*255
            img_out[((h_out-h1)//2):((h_out-h1)//2+h1),:,:] = img_rsz
        cv2.imwrite(output_dir+fn, img_out)

def search_image_in_subset(result, train_pair_list, train_image_list, test_pair_list, test_image_list):
    train_pairs = set([' '.join(p) for p in train_pair_list])
    train_images = set(train_image_list)
    test_pairs = set([' '.join(p) for p in test_pair_list])
    test_images = set(test_image_list)

    sample_list = []
    for n, i in enumerate(range(0,len(result),2)):
        name_1 = os.path.basename(result[i].split()[1])
        sid_1 = os.path.basename(result[i].split()[2])[0:7]
        name_2 = os.path.basename(result[i+1].split()[1])
        sid_2 = os.path.basename(result[i+1].split()[2])[0:7]

        if sid_1 in train_images:
            split_image_1 = 'train'
        elif sid_1 in test_images:
            split_image_1 = 'test'
        else:
            split_image_1 = 'unknown'

        if sid_2 in train_images:
            split_image_2 = 'train'
        elif sid_2 in test_images:
            split_image_2 = 'test'
        else:
            split_image_2 = 'unknown'

        if ' '.join([sid_1, sid_2]) in train_pairs:
            split_pair = 'train'
        elif ' '.join([sid_1, sid_2]) in test_pairs:
            split_pair = 'test'
        elif ' '.join([sid_2, sid_1]) in train_pairs:
            split_pair = 'train'
        elif ' '.join([sid_2, sid_1]) in test_pairs:
            split_pair = 'test'            
        else:
            split_pair = 'unknown'

        print('%d: %s-%s (%s - %s) s1: %-8s s2: %-8s sp: %-8s\n' % (n, name_1, name_2, sid_1, sid_2, split_image_1, split_image_2, split_pair))

if __name__ == '__main__':
    #############################
    # test
    #############################
    # split = io.load_json('datasets/DF_Pose/Label/split.json')
    # id_list = split['train'] + split['test']
    # img_dir = 'datasets/DF_Pose/Img/img_df/'
    # query_fn_list = [img_dir + '%s.jpg'%sid for sid in id_list[0:10]]
    # gallery_fn_list = [img_dir + '%s.jpg'%sid for sid in id_list[0:100]]
    # output_dir = 'temp/search_image/test/'
    # search_image(query_fn_list, gallery_fn_list, output_dir)

    #############################
    # search DeformatbleGAN samples
    #############################
    # preprocessing
    # img_dir = 'temp/search_image/deformableGAN_org/'
    # output_dir = 'temp/search_image/deformableGAN_query/'
    # prepro_image(img_dir, output_dir)
    # search
    # split = io.load_json('datasets/DF_Pose/Label/split.json')
    # id_list = split['train'] + split['test']
    # gallery_dir = 'datasets/DF_Pose/Img/img_df/'
    # gallery_fn_list = [gallery_dir + '%s.jpg'%sid for sid in id_list]
    # query_dir = 'temp/search_image/deformableGAN_query/'
    # query_fn_list = [query_dir+fn for fn in os.listdir(query_dir)]
    # query_fn_list.sort()
    # output_dir = 'temp/search_image/deformableGAN/'
    # search_image(query_fn_list, gallery_fn_list, output_dir)

    # search in subsets
    # result = io.load_str_list('temp/search_image/deformableGAN/result.txt')
    # pair_split = io.load_json('datasets/DF_Pose/Label/pair_split.json')
    # train_pair_list = pair_split['train']
    # test_pair_list = pair_split['test']
    # image_split = io.load_json('datasets/DF_Pose/Label/split.json')
    # train_image_list = image_split['train']
    # test_image_list = image_split['test']

    # search_image_in_subset(result, train_pair_list, train_image_list, test_pair_list, test_image_list)
