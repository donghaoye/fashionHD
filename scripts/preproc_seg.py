from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np

import util.io as io
import util.image as image


root = '/data2/ynli/datasets/DeepFashion/In-shop/'

def create_bbox_label():
    '''
    Create bbox label for inshop images
    '''

    # load samples
    print('loading samples')
    samples = io.load_json(root + 'Label/samples.json')

    # load bbox list
    bbox_list = io.load_str_list(root + 'Anno/list_bbox_inshop.txt')[2::]
    img2bbox = {}
    
    for idx,l in enumerate(bbox_list):
        l = l.split()
        img_path = l[0]
        cloth_type = int(l[1])
        pose_type = int(l[2])
        bbox = [int(x) for x in l[3:7]]

        img2bbox[img_path] = {
            'bbox': bbox,
            'cloth_type': cloth_type,
            'pose_type': pose_type
        }
    
    # align samples to bbox

    bbox_label = {}
    for s_id, s in samples.iteritems():

        img_path = s['img_path_org']
        p = img_path[img_path.find('img')::].replace('//', '/')
        
        bbox_label[s_id] = img2bbox[p]

    # save to file
    print('saving bbox label')
    fn_out = root + 'Label/bbox_label.json'
    io.save_json(bbox_label, fn_out)


def _align_one_image(in_path, out_path, p_src, p_tar, sz_tar):
    '''
    Helper function for parallel image alignment
    '''
    
    img = image.imread(in_path)    
    img_out = image.align_image(img, p_src, p_tar, sz_tar)
    image.imwrite(img_out, out_path)

def align_image_for_seg():
    '''
    Proprecessing for cloth segmentation using FashionGAN-complete-demo
    - Resize images (to 224x224)
    - Place the clothing bbox at the center of the image
    '''
    
    # config
    parallel = False
    img_size = 224 # standard image size
    region_rate = 0.9 # max(item_region_width, item_region_height) / img_size

    dir_out = root + 'Img/img_seg/'

    
    from multiprocessing import Process
    

    # load sample index
    samples = io.load_json(root + 'Label/samples.json')

    # load bounding box
    bbox_label = io.load_json(root + 'Label/bbox_label.json')
    
    # resize and align images
    
    x_c = 0.5 * img_size
    y_c = 0.5 * img_size
    rg_size = region_rate * img_size

    io.mkdir_if_missing(dir_out)
    
    s_idx = 0
    p_list = []
    for s_id, s in samples.iteritems():
        x1, y1, x2, y2 = bbox_label[s_id]['bbox']
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
        
        in_path = s['img_path_org']
        out_path = dir_out + s_id + '.jpg'

        if parallel:
            p = Process(target = _align_one_image, args = (in_path, out_path, p_src, p_tar, (img_size, img_size)))
            p.start()
            p_list.append(p)

        else:

            img = image.imread(in_path)
            img_out = image.align_image(img, p_src, p_tar, sz_tar = (img_size, img_size))
            
            img_path = dir_out + s_id + '.jpg'
            image.imwrite(img_out, out_path)
             
            print('align image %d / %d' % (s_idx, len(samples)))
            s_idx += 1
    
    if parallel:
        for p in p_list:
            p.join()


if __name__ == '__main__':
    
    #create_bbox_label()
    align_image_for_seg()
