from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np
from collections import defaultdict

import util.io as io

ca_root = 'datasets/DeepFashion/Category_and_Attribute/'
design_root = 'datasets/DeepFashion/Fashion_design/'

def create_sample_index_and_label():
    '''
    Create sample index and label for Category_and_Attribute data
    - sample index
    - landmark label
    - bbox label
    - attribute label
    '''

    # config
    dir_label = design_root + 'Label/'

   
    # create sample index and landmark label

    landmark_list = io.load_str_list(ca_root + 'Anno/list_landmarks.txt')[2::]
    img_root_org = ca_root + 'Img/'

    samples = {}
    landmark_label = {}

    for idx, s in enumerate(landmark_list):
        img_id = 'ca_' + str(idx)

        s = s.split()
        img_path_org = os.path.join(img_root_org, s[0])

        # 1: upper-body, 2: lower-body, 3: full-body
        cloth_type = int(s[1])
        pose_type = -1

        lm_str = s[2::]
        if cloth_type == 1:
            assert len(lm_str) == 18
        elif cloth_type == 2:
            assert len(lm_str) == 12
        elif cloth_type == 3:
            assert len(lm_str) == 24

        # lm is a list: [(x_i, y_i, v_i)]
        lm = [(float(lm_str[i+1]), float(lm_str[i+2]), int(lm_str[i])) for i in range(0,len(lm_str),3)]

        samples[img_id] = {
            'img_id': img_id,
            'cloth_type': cloth_type,
            'pose_type': pose_type,
            'img_path_org': img_path_org
        }

        landmark_label[img_id] = lm

    io.mkdir_if_missing(dir_label)
    io.save_json(samples, os.path.join(dir_label, 'ca_samples.json'))
    io.save_data(landmark_label, os.path.join(dir_label, 'ca_landmark_label.pkl'))

    print('create sample index (%d samples)' % len(samples))
    print('create landmark label')

    img2id = {s['img_path_org'][s['img_path_org'].find('img')::]:s_id for s_id, s in samples.iteritems()}

    # create bbox label
    bbox_list = io.load_str_list(ca_root + 'Anno/list_bbox.txt')[2::]
    bbox_label = {}

    for s in bbox_list:
        s = s.split()
        assert len(s) == 5
        s_id = img2id[s[0]]
        bbox = [float(x) for x in s[1::]]
        bbox_label[s_id] = bbox

    io.save_data(bbox_label, os.path.join(dir_label, 'ca_bbox_label.pkl'))
    print('create bbox label')


    # create attribute label
    attr_list = io.load_str_list(ca_root + 'Anno/list_attr_img.txt')[2::]
    attr_label = {}

    for idx,s in enumerate(attr_list):
        s = s.split()
        s_id = img2id[s[0]]
        att = [1 if c=='1' else 0 for c in s[1::]]
        assert len(att) == 1000

        attr_label[s_id] = att
        print('\rcreating attribute label %d / %d' % (idx, len(attr_list)), end = '')

    io.save_data(attr_label, os.path.join(dir_label, 'ca_attr_label.pkl'))
    print('\ncreate attribute label')

def create_category_label():

    samples = io.load_json(design_root + 'Label/ca_samples.json')
    cat_entry_list = io.load_str_list(ca_root + 'Anno/list_category_cloth.txt')[2::]
    cat_list = io.load_str_list(ca_root + 'Anno/list_category_img.txt')[2::]

    # create category entry
    cat_entry = []
    for cat_str in cat_entry_list:
        cat_name = ' '.join(cat_str.split()[0:-1])
        cat_type = int(cat_str.split()[-1])
        cat_entry.append({'entry': cat_name, 'type': cat_type})

    io.save_json(cat_entry, design_root + 'Label/cat_entry.json')
    print('create category entry')

    # create category label
    img2id = {s['img_path_org'][s['img_path_org'].find('img')::]:s_id for s_id, s in samples.iteritems()}
    cat_label ={}

    for idx, s in enumerate(cat_list):
        s = s.split()
        s_id = img2id[s[0]]
        cat = int(s[1]) - 1
        cat_label[s_id] = cat

    io.save_data(cat_label, design_root + 'Label/ca_cat_label.pkl')
    print('create category label')

def create_split():
    '''
    Create split following the original partition
    '''

    split_list = io.load_str_list(ca_root + 'Eval/list_eval_partition.txt')[2:]
    split = {'train': [], 'val': [], 'test': []}
    samples = io.load_json(design_root + 'Label/ca_samples.json')
    img2id = {s['img_path_org'][s['img_path_org'].find('img')::]:s_id for s_id, s in samples.iteritems()}

    for s in split_list:
        img_path, status = s.split()
        s_id = img2id[img_path]
        split[status].append(s_id)

    io.mkdir_if_missing(design_root + 'Split')
    io.save_json(split, design_root + 'Split/ca_split.json')

    print('create split')
    for status in ['train', 'val', 'test']:
        print('%s: %d' % (status, len(split[status])))

    split_trainval = {
        'train': split['train'] + split['val'],
        'test': split['test']
    }
    io.save_json(split_trainval, design_root + 'Split/ca_split_trainval.json')

def create_attr_entry():
    '''
    Create attribute entry list, which contains original 1000 attributes used in Category_and_Attribute benchmark
    '''

    print('loading data...')
    attr_entry_list = io.load_str_list(ca_root + 'Anno/list_attr_cloth.txt')[2::]
    attr_label = io.load_data(design_root + 'Label/ca_attr_label.pkl')
    split = io.load_json(design_root + 'Split/ca_split.json')
    train_ids = set(split['train'])
    attr_mat = np.array([v for k,v in attr_label.iteritems() if k in train_ids], dtype = np.float32)

    print('computing positive rates')
    num_sample = len(train_ids)
    pos_rate = attr_mat.sum(axis = 0) / num_sample

    attr_entry = []
    for idx, s in enumerate(attr_entry_list):
        s = s.split()
        attr_name = ' '.join(s[0:-1])
        attr_type = int(s[-1])
        attr_entry.append({
            'entry': attr_name,
            'type': attr_type,
            'pos_rate': pos_rate[idx]
            })

    io.save_json(attr_entry, design_root + 'Label/attr_entry.json')

def visualize_samples():

    num_sample = 10
    dir_out = 'temp/attr_example'

    io.mkdir_if_missing(dir_out)
    samples = io.load_json(design_root + 'Label/ca_samples.json')
    attr_label = io.load_data(design_root + 'Label/ca_attr_label.pkl')
    attr_entry = io.load_json(design_root + 'Label/attr_entry.json')

    id_set = set(samples.keys())

    for i, att in enumerate(attr_entry):
        print('attribute %d / %d: %s' % (i, len(attr_entry), att['entry']))
        dir_att = os.path.join(dir_out, att['entry'])
        io.mkdir_if_missing(dir_att)
        pos_id_list = [s_id for s_id, label in attr_label.iteritems() if label[i] == 1]
        np.random.shuffle(pos_id_list)
        for s_id in pos_id_list[0:num_sample]:
            fn_src = samples[s_id]['img_path']
            fn_tar = os.path.join(dir_att, 'pos_' + s_id + '.jpg')
            shutil.copyfile(fn_src, fn_tar)

        neg_id_list = list(id_set - set(pos_id_list))
        np.random.shuffle(neg_id_list)
        for s_id in neg_id_list[0:num_sample]:
            fn_src = samples[s_id]['img_path']
            fn_tar = os.path.join(dir_att, 'neg_' + s_id + '.jpg')
            shutil.copyfile(fn_src, fn_tar)

def create_gan_split():
    '''
    create split for gan training
    ca_gan: containing all frontal images
    ca_gan_upper: containing all frontial, upperbody images
    '''

    samples = io.load_json(design_root + 'Label/ca_samples.json')
    split = io.load_json(design_root + 'Split/ca_split_trainval.json')

    # use same image set as in Zhu Shizhan's ICCV17 FashionGAN paper
    img_list = io.load_str_list('datasets/DeepFashion/Fashion_synthesis/data_release/benchmark/name_list.txt')
    img_set = set(img_list)

    split_gan = {'train':[], 'test':[]}
    split_gan_upper = {'train': [], 'test':[]}

    for set_name in ['train', 'test']:
        for s_id in split[set_name]:
            s = samples[s_id]
            img_name = s['img_path_org']
            img_name = img_name[img_name.find('img/')::]
            if img_name in img_set:
                split_gan[set_name].append(s_id)
                if s['cloth_type'] == 1:
                    split_gan_upper[set_name].append(s_id)

    print('create split "split_gan"')
    print('train: %d, test: %d, total: %d' % (len(split_gan['train']), len(split_gan['test']), len(split_gan['train']) + len(split_gan['test'])))
    print('create split "split_gan_upper"')
    print('train: %d, test: %d, total: %d' % (len(split_gan_upper['train']), len(split_gan_upper['test']), len(split_gan_upper['train']) + len(split_gan['test'])))

    io.save_json(split_gan, design_root + 'Split/ca_gan_split_trainval.json')
    io.save_json(split_gan_upper, design_root + 'Split/ca_gan_split_trainval_upper.json')

def create_color_attribute_label():
    '''
    attributes related to color
    attribute types: 1-texture, 2-fabrix, 3-shape, 4-part, 5-style
    '''

    attr_entry = io.load_json(design_root+'Label/attr_entry.json')
    attr_label = io.load_data(design_root+'Label/ca_attr_label.pkl')
    index = [i for i, entry in enumerate(attr_entry) if entry['type'] in {1,2,5}]
    n = 0
    new_attr_label = {}
    for s_id, label in attr_label.iteritems():
        new_attr_label[s_id] = [label[i] for i in index]
        n+=1
        print('%d/%d' % (n, len(attr_label)))

    new_attr_entry = [attr_entry[i] for i in index]
    print('%d color-related attributes' % len(index))

    io.save_data(new_attr_label, design_root+'Label/ca_color_attr_label.pkl')
    io.save_json(new_attr_entry, design_root+'Label/color_attr_entry.json')


if __name__ == '__main__':
    # create_sample_index_and_label()
    # create_split()
    # create_attr_entry()
    # create_category_label()
    # visualize_samples()
    # create_gan_split()
    create_color_attribute_label()

