from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np

import util.io as io
import util.image as image
import cv2


design_root = 'datasets/DeepFashion/Fashion_design/'

def align_and_resize_image(benchmark):
    '''
    Resize images to standard size, and align the clothing region at the center of the image.
    '''

    ###################
    # config
    ###################
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
        fn_out_sample = design_root + 'Label/ca_samples.json'
        fn_out_bbox_label = design_root + 'Label/ca_bbox_label_%d.pkl' % img_size
        fn_out_lm_label = design_root + 'Label/ca_landmark_label_%d.pkl' % img_size
        interp_method = cv2.INTER_CUBIC
        update_label = True

    elif benchmark == 'inshop':
        samples = io.load_json(design_root + 'Label/inshop_samples.json')
        bbox_label = io.load_data(design_root + 'Label/inshop_bbox_label.pkl')
        lm_label = io.load_data(design_root + 'Label/inshop_landmark_label.pkl')
        output_dir = design_root + 'Img/img_inshop_%d' % img_size
        fn_out_sample = design_root + 'Label/inshop_samples.json'
        fn_out_bbox_label = design_root + 'Label/inshop_bbox_label_%d.pkl' % img_size
        fn_out_lm_label = design_root + 'Label/inshop_landmark_label_%d.pkl' % img_size
        interp_method = cv2.INTER_CUBIC
        update_label = True

    elif benchmark == 'ca_seg_pad':
        # this setting is to align segmentation map from org+pad size to standard size
        # this will not update landmark and bbox label
        input_dir = design_root + 'Img/seg_ca_pad_%d' % img_size
        bbox_label = io.load_data(design_root + 'Label/ca_bbox_label_pad_%d.pkl' % img_size)
        output_dir = design_root + 'Img/seg_ca_%d/' % img_size
        lm_label = None
        fn_out_sample = fn_out_bbox_label = fn_out_lm_label = None
        interp_method = cv2.INTER_NEAREST
        update_label = False

        # change img_path_org to input segmentation map file (org+pad size)
        # change img_path to output segmentation map file (standard size)
        samples = io.load_json(design_root + 'Label/ca_samples.json')
        split = io.load_json(design_root + 'Split/ca_gan_split_trainval.json')
        samples = {s_id: samples[s_id] for s_id in split['train'] + split['test']}
        for s_id, s in samples.iteritems():
            samples[s_id]['img_path_org'] = os.path.join(input_dir, s_id + '.bmp')
            samples[s_id]['img_path'] = os.path.join(output_dir, s_id + '.bmp')

    elif benchmark == 'ca_seg_syn':
        # this setting is to align segmentation map from org size to standard size
        # this will not update landmark and bbox label
        input_dir = design_root + 'Img/seg_ca_org_syn'
        bbox_label = io.load_data(design_root + 'Label/ca_bbox_label.pkl')
        output_dir = design_root + 'Img/seg_ca_syn_%d_new/' % img_size
        lm_label = None
        fn_out_sample = fn_out_bbox_label = fn_out_lm_label = None
        interp_method = cv2.INTER_NEAREST
        update_label = False

        # change img_path_org to input segmentation map file (org+pad size)
        # change img_path to output segmentation map file (standard size)
        samples = io.load_json(design_root + 'Label/ca_samples.json')
        split = io.load_json(design_root + 'Split/ca_gan_split_trainval.json')
        samples = {s_id: samples[s_id] for s_id in split['train'] + split['test']}

        for s_id, s in samples.iteritems():
            samples[s_id]['img_path_org'] = os.path.join(input_dir, s_id + '.bmp')
            samples[s_id]['img_path'] = os.path.join(output_dir, s_id + '.bmp')
            assert os.path.isfile(samples[s_id]['img_path_org']), 'cannot find image: %s' % samples[s_id]['img_path_org']

    
    io.mkdir_if_missing(design_root + 'Img')
    io.mkdir_if_missing(output_dir)

    # update sample
    if update_label:
        print('updating sample index')
        for s_id in samples.keys():
            samples[s_id]['img_path'] = os.path.join(output_dir, s_id + '.jpg')
        io.save_json(samples, fn_out_sample)

    # process images
    from multiprocessing import Process, Manager
    
    id_list = samples.keys()
    block_size = len(id_list) // num_worker + 1

    if update_label:
        manager = Manager()
        aligned_bbox_label = manager.dict()
        aligned_lm_label = manager.dict()
    else:
        aligned_bbox_label = aligned_lm_label = None

    p_list = []
    for worker_idx in range(num_worker):
        id_sublist = id_list[block_size*worker_idx: block_size*(worker_idx+1)]
        p = Process(target = _align_and_resize_image_unit,\
            args = (worker_idx, id_sublist, samples, bbox_label, lm_label, img_size, region_rate, interp_method, aligned_bbox_label, aligned_lm_label))
        p.start()
        p_list.append(p)

    for p in p_list:
        p.join()

    if update_label:
        aligned_bbox_label = dict(aligned_bbox_label)
        aligned_lm_label = dict(aligned_lm_label)
        io.save_data(aligned_bbox_label, fn_out_bbox_label)
        io.save_data(aligned_lm_label, fn_out_lm_label)

def _align_and_resize_image_unit(worker_idx, id_list, samples, bbox_label, lm_label, img_size, region_rate, interp_method,
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
        img_out, trans_mat = image.align_image(img, p_src, p_tar, sz_tar = (img_size, img_size), flags = interp_method)
        image.imwrite(img_out, s['img_path'])

        if aligned_bbox_label is not None and aligned_lm_label is not None:
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

def merge_seg_map():
    '''
    input seg map:
        0-background, 1-hair, 2-head, 3-upperbody, 4-lowerbody, 5-leg, 6-arm
    '''
    # config
    seg_root = '/data2/ynli/Fashion/ICCV17-fashionGAN/complete_demo/output/img_ca_256/seg_7'
    tar_root = 'datasets/DeepFashion/Fashion_design/Img/seg_ca_256'
    io.mkdir_if_missing(tar_root)

    samples = io.load_json('datasets/DeepFashion/Fashion_design/Label/ca_samples.json')
    seg_map_paths = {}

    for i, (s_id, s) in enumerate(samples.items()):
        seg_org = image.imread(os.path.join(seg_root, s_id + '.bmp'), mode = 'grayscale')
        # assert seg_org
        if s['cloth_type'] == 1:
            seg_mrg = (seg_org == 3).astype(np.uint8)
        elif s['cloth_type'] == 2:
            seg_mrg = (seg_org == 4).astype(np.uint8)
        else:
            seg_mrg = np.logical_or(seg_org == 3, seg_org == 4).astype(np.uint8)

        fn_out = os.path.join(tar_root, s_id + '.bmp')
        image.imwrite(seg_mrg, fn_out)
        seg_map_paths[s_id] = fn_out

        print('\rmerge segmentation map: %d / %d' % (i, len(samples)))
    print('\n')

    io.save_json(seg_map_paths, 'datasets/DeepFashion/Fashion_design/Label/ca_seg_paths.json')

def visualize_seg_map():

    num_sample = 1000
    output_dir = 'temp/seg_map'
    io.mkdir_if_missing(output_dir)
    
    samples = io.load_json(design_root + 'Label/ca_samples.json')
    split = io.load_json(design_root + 'Split/ca_gan_split_trainval.json')
    id_list = split['train'] + split['test']
    id_list = [s_id for s_id in id_list if samples[s_id]['cloth_type'] == 3]

    seg_dir_list = [
        design_root + 'Img/seg_ca_256/',
        design_root + 'Img/seg_ca_syn_256/'
    ]

    for i, s_id in enumerate(id_list[0:num_sample]):
        s = samples[s_id]
        img = image.imread(s['img_path'])
        imgs = [img]
        for seg_dir in seg_dir_list:
            seg = image.imread(seg_dir + s_id + '.bmp') * 20
            mask = img * (seg > 0).astype(np.float)
            imgs += [seg, mask]

        img = image.stitch(imgs, 0)
        image.imwrite(img, os.path.join(output_dir, s_id + '.jpg'))
        print(i)



def pad_image_for_segmentation():
    '''
    resize and padding image for segmentation (using fashionGAN code)
    Todo: add inshop version
    '''

    sz_tar = 256
    output_dir = 'datasets/DeepFashion/Fashion_design/Img/img_ca_pad'

    io.mkdir_if_missing(output_dir)
    samples = io.load_json(design_root + 'Label/ca_samples.json')
    split = io.load_json(design_root + 'Split/ca_gan_split_trainval.json')
    id_list = split['train'] + split['test']

    # update landmark and bbox
    lm_label = io.load_data(design_root + 'Label/ca_landmark_label.pkl')
    bbox_label = io.load_data(design_root + 'Label/ca_bbox_label.pkl')
    lm_label_pad = {}
    bbox_label_pad = {}

    io.save_str_list(id_list, os.path.join(output_dir, 'img_ca_pad.txt'))
    for i, s_id in enumerate(id_list):
        img_org = image.imread(samples[s_id]['img_path_org'])
        h, w = img_org.shape[0:2]

        if h > w:
            img = image.resize(img_org, (-1, sz_tar))
            scale = 1. * sz_tar / h
        else:
            img = image.resize(img_org, (sz_tar, -1))
            scale = 1. * sz_tar / w

        # img = image.pad_square(img, sz_tar, padding_value = 255, mode = 'lefttop')
        # image.imwrite(img, os.path.join(output_dir, s_id + '.jpg'))

        bbox_label_pad[s_id] = [c*scale for c in bbox_label[s_id]]
        lm_label_pad[s_id] = []
        for x, y, v in lm_label[s_id]:
            lm_label_pad[s_id].append([x*scale, y*scale, v])

        print('padding image %d / %d' % (i, len(id_list)))

    io.save_data(lm_label_pad, design_root + 'Label/ca_landmark_label_pad_%d.pkl' % sz_tar)
    io.save_data(bbox_label_pad, design_root + 'Label/ca_bbox_label_pad_%d.pkl' % sz_tar)


def create_synthesis_to_CA_index():
    '''
    create an index map A: img_syn[i] = img_ca[A[i]]
    '''
    import scipy.io

    syn_name_list = io.load_str_list('datasets/DeepFashion/Fashion_synthesis/data_release/benchmark/name_list.txt')
    
    samples = io.load_json('datasets/DeepFashion/Fashion_design/Label/ca_samples.json')
    ca_name2idx = {s['img_path_org'][s['img_path_org'].find('img/')::]:int(s_id[3::]) for s_id, s in samples.iteritems()}
    ca_name2sz = {}
    for i,s in enumerate(samples.values()):
        img = image.imread(s['img_path_org'])
        h, w = img.shape[0:2]
        ca_name2sz[s['img_path_org'][s['img_path_org'].find('img/')::]] = (w, h)
        print('load ca image size: %d/%d' % (i, len(samples)))

    syn_idx_list = [ca_name2idx[name] for name in syn_name_list]
    syn_org_size_list = [ca_name2sz[name] for name in syn_name_list]

    data_out = {
        'syn2ca_index': syn_idx_list,
        'syn2ca_width': [w for w,_ in syn_org_size_list],
        'syn2ca_height': [h for _, h in syn_org_size_list]
    }
    fn_out = 'datasets/DeepFashion/Fashion_synthesis/data_release/benchmark/index_to_Category_and_Attribute.mat'
    scipy.io.savemat(fn_out, data_out)


def create_edge_path():
    
    edge_root = design_root+'Img/edge_ca_256'
    samples = io.load_json(design_root+'Label/ca_samples.json')

    split = io.load_json(design_root+'Split/ca_gan_split_trainval.json')
    edge_path = {s_id:os.path.join(edge_root, s_id+'.jpg') for s_id in split['train']+split['test']}
    io.save_json(edge_path, design_root+'Label/ca_edge_paths.json')

    split = io.load_json(design_root+'Split/debugca_gan_split.json')
    edge_path = {s_id:os.path.join(edge_root, s_id+'.jpg') for s_id in split['train']+split['test']}
    io.save_json(edge_path, design_root+'Label/debugca_edge_paths.json')    

def create_inner_edge_map():
    '''
    extract the edges inside the clothing regions
    '''

    # config
    kernel_size = 7
    threshold = 0

    split = io.load_json(design_root + 'Split/ca_gan_split_trainval.json')
    id_list = split['train'] + split['test']
    edge_root = design_root+'Img/edge_ca_256'
    seg_root = design_root+'Img/seg_ca_256'
    output_dir = design_root + 'Img/edge_ca_256_inner'
    io.mkdir_if_missing(output_dir)

    kernel = np.zeros((kernel_size, kernel_size), np.uint8)
    k = (kernel_size-1)/2
    for i in range(kernel_size):
        for j in range(kernel_size):
            if np.abs(i-k) + np.abs(j-k) <= k:
                kernel[i,j] = 1

    for i, s_id in enumerate(id_list):
        edge = image.imread(os.path.join(edge_root, s_id + '.jpg'), 'grayscale')
        seg = image.imread(os.path.join(seg_root, s_id + '.bmp'), 'grayscale')
        mask_upper = cv2.erode((seg==3).astype(np.uint8), kernel)
        mask_lower = cv2.erode((seg==4).astype(np.uint8), kernel)
        mask = mask_upper | mask_lower
        edge_inner = edge * mask
        edge_inner = (edge_inner >= threshold).astype(np.uint8) * edge_inner
        image.imwrite(edge_inner, os.path.join(output_dir, s_id + '.jpg'))
        print('extracting inner edge %d / %d'%(i, len(id_list)))

    # create labels
    edge_paths = {s_id: os.path.join(output_dir, s_id+'.jpg') for s_id in id_list}
    split_debug = io.load_json(design_root + 'Split/debugca_gan_split.json')
    edge_paths_debug = {s_id:p for s_id, p in edge_paths.iteritems() if s_id in split_debug['train']+split_debug['test']}

    io.save_json(edge_paths, design_root+'Label/ca_edge_inner_paths.json')
    io.save_json(edge_paths_debug, design_root+'Label/debugca_edge_inner_paths.json')

def create_color_map():
    #### test
    img = image.imread('datasets/DeepFashion/Fashion_design/Img/img_ca_256/ca_9.jpg')
    assert img is not None

    output_dir = 'temp/color_map_generation_test'
    io.mkdir_if_missing(output_dir)
    # color map by gaussian blur
    kernel_size = 21
    for sigma in [1,2,5,10,20]:
        img_blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)
        image.imwrite(img_blur, os.path.join(output_dir, 'gaussian_%d_%f.jpg' % (kernel_size, sigma)))
    # color map by downsampling
    for scale in [2,4,8,16,32]:
        w, h = img.shape[1], img.shape[0]
        dw, dh = w//scale, h//scale
        img_blur = cv2.resize(cv2.resize(img, (dw, dh), interpolation=cv2.INTER_LINEAR), (w, h), interpolation=cv2.INTER_LINEAR)
        image.imwrite(img_blur, os.path.join(output_dir, 'downsample_%d.jpg') % scale)
    

def test_affine_augmentation():
    img = image.imread('datasets/DeepFashion/Fashion_design/Img/img_ca_256/ca_9.jpg')
    assert img is not None

    output_dir = 'temp/affine_augmentation'
    io.mkdir_if_missing(output_dir)

    # config
    scale = [0.05, 0.1, 0.15]
    num_per_scale = 10

    w, h  = img.shape[1], img.shape[0]
    keypoint_src = np.array([[0,0], [w,0], [0,h]], dtype=np.float32)
    for s in scale:
        for i in range(num_per_scale):
            offset = (np.random.rand(3,2)*2-1) * np.array([w, h]) * s
            offset = offset.astype(np.float32)
            keypoint_dst = keypoint_src + offset
            m = cv2.getAffineTransform(keypoint_src, keypoint_dst)
            img_trans = cv2.warpAffine(img, m, dsize=(w,h), flags=cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)
            img_trans = img_trans*0.8 + img * 0.2
            image.imwrite(img_trans, os.path.join(output_dir, 'affine_%f_%d.jpg' % (s, i)))



if __name__ == '__main__':
    #################################################
    # align_and_resize_image('ca')
    #
    # align_and_resize_image('inshop')
    # search_unmatched_HR_image()
    # merge_seg_map()

    #################################################
    # get segmentation by fashionGAN code
    #
    # pad_image_for_segmentation()
    # align_and_resize_image('ca_seg_pad')

    
    #################################################
    # get segmentation from Fashon-synthesis data
    #
    # create_synthesis_to_CA_index()
    # align_and_resize_image('ca_seg_syn')


    #################################################
    # edge
    # create_edge_path()
    # create_inner_edge_map()

    #################################################
    # color map
    # create_color_map()

    #################################################
    # geometric transformation
    

    #################################################
    # visualization and test
    #
    # visualize_seg_map()
