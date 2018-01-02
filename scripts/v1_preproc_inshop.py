from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np

import util.io as io


root = '/data2/ynli/datasets/DeepFashion/In-shop/'


def create_index():

    high_res = True
    copy_img = False

    bbox_list = io.load_str_list(root + '/Anno/list_bbox_inshop.txt')[2::]
    samples = {}


    if high_res:
        org_img_root = root + 'Img/img_highres/'
        new_img_root = root + 'Img/img1_highres/'
        fn_out = root + 'Label/samples_highres.json'
    else:
        org_img_root = root + 'Img/img/'
        new_img_root = root + 'Img/img1/'
        fn_out = root + 'Label/samples.json'

    io.mkdir_if_missing(new_img_root)
    num_fail = 0
    for idx, s in enumerate(bbox_list):
        img_id = str(idx)

        s = s.split()
        assert len(s) == 7

        org_path = s[0]
        org_path_split = org_path.split('/')[1:]
        org_path = '/'.join(org_path_split)

        gender = org_path_split[0]
        category = org_path_split[1]
        item_id = org_path_split[2]
        img_path = '/'.join(org_path_split[2:4])
        pose = org_path_split[3].split('_')[2][0:-4]
        assert pose in {'front', 'back', 'side', 'full', 'flat', 'additional'}

        fn_src = org_img_root + '/'+ org_path
        fn_tar = new_img_root + '/' + img_path

        if not os.path.isfile(fn_src):
            num_fail += 1
            print(fn_src)
            continue

        samples[img_id] = {
            'img_id': img_id,
            'item_id': item_id,
            'img_path': img_path,
            'gender': gender,
            'category': category,
            'pose': pose,
            'org_path': org_path,
        }

        if copy_img:
            io.mkdir_if_missing(new_img_root + item_id)
            shutil.copyfile(fn_src, fn_tar)

        # print('%s: %s => %s' % (img_id, fn_src, fn_tar))
        # print('%s' % img_id)

    io.save_json(samples, fn_out)

    print('\n')
    print('save sample index to %s' % fn_out)
    print('%d samples not found!' % num_fail)


def create_seg_input():

    # samples = io.load_json(root + 'Anno/samples_highres.json')
    # in_dir = root + 'Img/img1_highres/'

    samples = io.load_json(root + 'Label/samples.json')
    in_dir = root + 'Img/img1/'

    out_dir = root + 'Img/input/'

    io.mkdir_if_missing(out_dir)

    for img_id, s in samples.iteritems():

        fn_src = in_dir + s['img_path']
        fn_tar = out_dir + '%s.jpg' % img_id

        shutil.copyfile(fn_src, fn_tar)

        print(img_id)


def create_attr_label():

    # attr_list = io.load_str_list(root + 'Anno/list_attr_cloth.txt')[2::]
    # attr_anno = io.load_str_list(root + 'Anno/list_attr_items.txt')[2::]
    # fn_out = root + 'Label/attribute_inshop.json'
    # num_attr = 463
    # n_top = 5

    attr_list = io.load_str_list('/data2/ynli/datasets/DeepFashion/Category_and_Attribute/Anno/list_attr_cloth.txt')[2::]
    attr_list = [' '.join(a.split()[0:-1]) for a in attr_list]
    attr_anno = io.load_str_list('/data2/ynli/datasets/DeepFashion/Category_and_Attribute/Anno/list_attr_img.txt')[2::]
    fn_out = root + 'Label/attribute_ca.json'
    num_attr = 1000
    n_top = 5

    # create label data
    if not os.path.isfile(fn_out):
        attr_data = {}
        for line in attr_anno:
            line = line.split()
            item_id = line[0]
            label = [int(c) for c in line[1::]]
            assert len(label) == num_attr

            attr_data[item_id] = label

        io.save_json(attr_data, fn_out)
    else:
        attr_data = io.load_json(fn_out)

    num_sample = len(attr_data)

    # most frequent attribute in each attribute type
    attr_list_ref = io.load_str_list('/data2/ynli/datasets/DeepFashion/Category_and_Attribute/Anno/list_attr_cloth.txt')[2::]
    attr_type = {' '.join(a.split()[0:-1]): a.split()[-1] for a in attr_list_ref}
    
    
    attr_mat = np.array(attr_data.values(), dtype = float)
    attr_count = np.where(attr_mat > 0, 1, 0).sum(axis = 0)
    

    attr_count_type = {}

    for i, attr_name in enumerate(attr_list):
        t = attr_type[attr_name] if attr_name in attr_type else '-1'
        if t not in attr_count_type:
            attr_count_type[t] = []
        attr_count_type[t].append((attr_name, attr_count[i]))

    for t in {'1', '2', '3', '4', '5', '-1'}:
        if t not in attr_count_type:
            continue
        attr_count_list = attr_count_type[t]
        attr_count_list.sort(key = lambda x: x[1], reverse = True)
        
        print('attribute type: %s' % t)

        for attr_name, count in attr_count_list[0:n_top]:
            print('%s: %d (%.1f %%)' % (attr_name, count, 100. * count / num_sample))
        print('\n')



def extract_attr():
    
    # use attribute entry name defined in category_and_attribute branch (1000)
    # extract attribute label from in-shop branch description annotation

    
    




if __name__ == '__main__':
    # create_index()
    # create_seg_input()
    create_attr_label()


        
