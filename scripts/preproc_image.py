from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np

import util.io as io
import util.image as image


design_root = 'datasets/DeepFashion/Fashion_design/'

def align_and_resize_image():
    '''
    Resize images to standard size, and align the clothing region at the center of the image.
    '''

    ###################
    # config
    ###################
    
    # benchmark = 'ca' # Category_and_Attribute
    benchmark = 'inshop' # Inshop

    img_size = 256
    region_rate = 0.8
    num_worker = 16

    ###################

    print('loading data')
    if benchmark == 'ca':
        samples = io.load_json(design_root + 'Label/ca_samples.json')
        bbox_label = io.load_data(design_root + 'Label/ca_bbox_label.pkl')
        lm_label = io.load_data(design_root + 'Label/ca_landmark_label.pkl')

        output_dir = design_root + 'Img/img_ca_%d/' % img_size
        fn_in_sample = design_root + 'Label/ca_samples.json'
        fn_out_bbox_label = design_root + 'Label/ca_bbox_label_%d.pkl' % img_size
        fn_out_lm_label = design_root + 'Label/ca_landmark_label_%d.pkl' % img_size


    elif benchmark == 'inshop':
        samples = io.load_json(design_root + 'Label/inshop_samples.json')
        bbox_label = io.load_data(design_root + 'Label/inshop_bbox_label.pkl')
        lm_label = io.load_data(design_root + 'Label/inshop_landmark_label.pkl')

        output_dir = design_root + 'Img/img_inshop_%d' % img_size
        fn_in_sample = design_root + 'Label/inshop_samples.json'
        fn_out_bbox_label = design_root + 'Label/inshop_bbox_label_%d.pkl' % img_size
        fn_out_lm_label = design_root + 'Label/inshop_landmark_label_%d.pkl' % img_size


    io.mkdir_if_missing(design_root + 'Img')
    io.mkdir_if_missing(output_dir)

    # update sample
    print('updatingg sample index')
    for s_id in samples.keys():
        samples[s_id]['img_path'] = os.path.join(output_dir, s_id + '.jpg')
    io.save_json(samples, fn_in_sample)

    # process images
    from multiprocessing import Process, Manager
    
    id_list = samples.keys()
    block_size = len(id_list) // num_worker + 1

    manager = Manager()
    aligned_bbox_label = manager.dict()
    aligned_lm_label = manager.dict()

    p_list = []
    for worker_idx in range(num_worker):
        id_sublist = id_list[block_size*worker_idx: block_size*(worker_idx+1)]
        p = Process(target = _align_and_resize_image_unit,\
            args = (worker_idx, id_sublist, samples, bbox_label, lm_label, img_size, region_rate, aligned_bbox_label, aligned_lm_label))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

    aligned_bbox_label = dict(aligned_bbox_label)
    aligned_lm_label = dict(aligned_lm_label)
    io.save_data(aligned_bbox_label, fn_out_bbox_label)
    io.save_data(aligned_lm_label, fn_out_lm_label)

def _align_and_resize_image_unit(worker_idx, id_list, samples, bbox_label, lm_label, img_size, region_rate,
    aligned_bbox_label, aligned_lm_label):
    '''
    Parallel helper function of align_and_resize_image()
    '''
    x_c = 0.5 * img_size
    y_c = 0.5 * img_size
    rg_size = region_rate * img_size

    num_sample = len(id_list)
    for idx, s_id in enumerate(id_list):
        s = samples[s_id]
        x1, y1, x2, y2 = bbox_label[s_id]
        w = x2 - x1
        h = y2 - y1

        if w > h:
            t_w = rg_size
            t_h = rg_size * h / w
        else:
            t_w = rg_size * w / h
            t_h = rg_size

        t_x1 = x_c - 0.5 * t_w
        t_x2 = x_c + 0.5 * t_w
        t_y1 = y_c - 0.5 * t_h
        t_y2 = y_c + 0.5 * t_h

        # apply image transform
        p_src = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        p_tar = [(t_x1, t_y1), (t_x1, t_y2), (t_x2, t_y1), (t_x2, t_y2)]

        img = image.imread(s['img_path_org'])
        img_out, trans_mat = image.align_image(img, p_src, p_tar, sz_tar = (img_size, img_size))
        image.imwrite(img_out, s['img_path'])

        # tranform bbox and landmarks
        lm_src = np.array(lm_label[s_id])
        lm_p_src = lm_src[:, 0:2] # landmark coordinates
        lm_v = lm_src[:, 2:3] # visibility
        lm_p_tar = image.transform_coordinate(lm_p_src, trans_mat)
        lm_tar = np.hstack((lm_p_tar, lm_v)).tolist()

        aligned_bbox_label[s_id] = [t_x1, t_y1, t_x2, t_y2]
        aligned_lm_label[s_id] = lm_tar

        print('[align and resize image] worker-%2d: %d / %d' % (worker_idx, idx, num_sample))


def search_unmatched_HR_image():
    '''
    Search high-resolution images which do not match their low-resolution version.
    '''

    # config
    hsz = 256 # histgram size
    threshold = 0.9


    # define matching function
    import cv2

    def _match_image_pair(img1, img2, hsz):
        hists = []
        for img in [img1, img2]:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            h, w = img.shape
            sz = int(max(h, w) * 0.5)
            x1 = int((w - sz)/2)
            y1 = int((h - sz)/2)
            x2 = int(w-(w-sz)/2)
            y2 = int(h-(h-sz)/2)
            img = img[y1:y2, x1:x2]
            
            hist = cv2.calcHist([img], [0], None, [hsz], [0,256])
            hist = hist/np.linalg.norm(hist)
            hists.append(hist)

        return (hists[0] * hists[1]).sum()

    # matching images
    samples = io.load_json('datasets/DeepFashion/Fashion_design/Label/inshop_samples.json')

    unmatch_list = [] # path of LR images which do not match HR images
    missing_list = [] # path of LR images which do not have HR images

    for idx, s in enumerate(samples.values()):
        img_path_lr = s['img_path_org']
        img_path_hr = img_path_lr.replace('/img/', '/img_highres/')
        assert os.path.isfile(img_path_lr)

        if not os.path.isfile(img_path_hr):
            missing_list.append(img_path_lr[img_path_lr.find('/img')::])
        else:
            img_lr = image.imread(img_path_lr)
            img_hr = image.imread(img_path_hr)
            score = _match_image_pair(img_lr, img_hr, hsz)

            if score < threshold:
                unmatch_list.append(img_path_lr[img_path_lr.find('/img')::])
            print('score: %.3f, %d / %d' % (score, idx, len(samples)))

        # print('checking HR and LR images are matched: %d / %d' % (idx, len(samples)))

    unmatch_list.sort()
    missing_list.sort()

    print('')
    print('unmatched images: %d' % len(unmatch_list))
    print('missing images: %d' % len(missing_list))

    output_dir = 'temp/check_HR_LR_matching'
    io.mkdir_if_missing(output_dir)
    io.save_str_list(unmatch_list, os.path.join(output_dir, 'unmatched_images.txt'))
    io.save_str_list(missing_list, os.path.join(output_dir, 'missing_images.txt'))



if __name__ == '__main__':

    # align_and_resize_image()
    search_unmatched_HR_image()