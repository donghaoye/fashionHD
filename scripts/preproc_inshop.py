from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np
from collections import defaultdict

import util.io as io


inshop_root = 'datasets/DeepFashion/In-shop/'
design_root = 'datasets/DeepFashion/Fashion_design/'


def create_sample_index_and_label():
    '''
    Create sample index and label for In-shop datasets
    - sample index
    - landmark label
    - bbox label
    '''

    # config
    dir_label = design_root + 'Label/'

    # create sampel index and landmark label
    landmark_list = io.load_str_list(inshop_root + 'Anno/list_landmarks_inshop.txt')[2::]
    img_root_org = inshop_root + 'Img/'

    samples = {}
    landmark_label = {}

    for idx, s in enumerate(landmark_list):
        img_id = 'inshop_' + str(idx)

        s = s.split()
        img_path_org = os.path.join(img_root_org, s[0])

        item_id = img_path_org.split('/')[-2]
        category = img_path_org.split('/')[-3]

        # 1: upper-body, 2: lower-body, 3: full-body
        cloth_type = int(s[1])

        # 1: normal, 2: medium, 3: large, 4: medium zoom-in, 5: larg zoom-in, 6: flat (no person)
        pose_type = int(s[2])

        lm_str = s[3::]
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
            'item_id': item_id,
            'category': category,
            'cloth_type': cloth_type,
            'pose_type': pose_type,
            'img_path_org': img_path_org
        }
        
        landmark_label[img_id] = lm

    io.mkdir_if_missing(dir_label)
    io.save_json(samples, os.path.join(dir_label, 'inshop_samples.json'))
    io.save_data(landmark_label, os.path.join(dir_label, 'inshop_landmark_label.pkl'))

    print('create sample index (%d samples)' % len(samples))
    print('create landmark label')

    img2id = {s['img_path_org'][s['img_path_org'].find('img')::]:s_id for s_id, s in samples.iteritems()}

    # create bbox label
    bbox_list = io.load_str_list(inshop_root + 'Anno/list_bbox_inshop.txt')[2::]
    bbox_label = {}

    for s in bbox_list:
        s = s.split()
        assert len(s) == 7
        s_id = img2id[s[0]]
        bbox = [float(x) for x in s[3::]]
        bbox_label[s_id] = bbox

    io.save_data(bbox_label, os.path.join(dir_label, 'inshop_bbox_label.pkl'))
    print('create bbox label')


def create_split():
    '''
    Split dataset into train/test sets, where ttems are NOT overlapped.
    In original split, train/test sets have equal size. We create our own split with a larger train set.
    '''

    # config
    use_original = False
    train_rate = 0.8

    # load sample
    samples = io.load_json(design_root + 'Label/inshop_samples.json')

    if use_original:
        # load split file
        split_list = io.load_str_list(inshop_root + 'Eval/list_eval_partition.txt')[2::]
        item2split = {}

        for line in split_list:
            line = line.split()
            item_id = line[1]
            status = line[2]

            if status == 'train':
                item2split[item_id] = 'train'
            else:
                item2split[item_id] = 'test'

    else:
        category2item = {}
        for s in samples.values():
            cat = s['category']
            if cat not in category2item:
                category2item[cat] = []
            category2item[cat].append(s['item_id'])

        item2split = {}
        np.random.seed(0)
        cat_list = category2item.keys()
        cat_list.sort()

        for cat in cat_list:
            item_list = list(set(category2item[cat]))
            item_list.sort()
            np.random.shuffle(item_list)
            
            split_point = int(len(item_list) * train_rate)
            for item_id in item_list[0:split_point]:
                item2split[item_id] = 'train'
            for item_id in item_list[split_point::]:
                item2split[item_id] = 'test'
        
        # check overlap
        train_set = set([item_id for item_id, s in item2split.iteritems() if s == 'train'])
        test_set = set([item_id for item_id, s in item2split.iteritems() if s == 'test'])
        assert not set.intersection(train_set, test_set)

    # create split
    split = {'train':[], 'test':[]}

    for s_id, s in samples.iteritems():
        split[item2split[s['item_id']]].append(s_id)


    print('train set: %d items, %d images' % (item2split.values().count('train'), len(split['train'])))
    print('test set:  %d items, %d images' % (item2split.values().count('test'), len(split['test'])))

    fn_out = design_root + 'Split/inshop_split.json'
    io.save_json(split, fn_out)


def create_color_entry_and_label():
    '''
    Create color attribute entries and color labels
    '''

    print('loading data')
    # load description
    desc_list = io.load_json(inshop_root + 'Anno/list_description_inshop.json')
    item2color = {d['item']:d['color'].lower().replace('-', ' ').split() for d in desc_list}
    colors = set([c[0] for c in item2color.values() if len(c) == 1])
    color_entry = [{'entry': c, 'type': 0, 'pos_rate': -1} for c in colors]

    # load sample index
    samples = io.load_json(design_root + 'Label/inshop_samples.json')
    split = io.load_json(design_root + 'Split/inshop_split.json')
    train_ids = set(split['train'])


    print('computing positive rates')
    color_label = {}
    for s_id, s in samples.iteritems():
        color = item2color[s['item_id']]
        label = [1 if c['entry'] in color else 0 for c in color_entry]
        color_label[s_id] = label


    color_mat = np.array([v for k, v in color_label.iteritems() if k in train_ids], dtype = np.float32)
    num_sample = len(train_ids)
    pos_rate = (color_mat.sum(axis = 0)/num_sample).tolist()

    for idx, att in enumerate(color_entry):
        color_entry[idx]['pos_rate'] = pos_rate[idx]

    print('saving data')
    io.save_json(color_entry, design_root + 'Label/color_entry.json')
    io.save_data(color_label, design_root + 'Label/inshop_attr_label.pkl')




def create_attribute_label():
    '''
    Create attribute label using predifined attribute entries
    '''

    # config
    attr_entry = io.load_json(design_root + 'Label/attr_entry.json')

    puncs = u'.,!?"%'
    trans_table = {ord(c): u' ' for c  in puncs}

    # load attribute entry
    num_attr = len(attr_entry)
    item2attr = defaultdict(lambda : [0] * num_attr)

    # load language description
    desc_list = io.load_json(inshop_root + 'Anno/list_description_inshop.json')
    item2desc = {d['item']:d for d in desc_list}

    # attribute matching
    i_item = 0

    for item_id, d in item2desc.iteritems():
        
        color = d['color'].replace('-', ' ')
        d_str = ' ' + ' '.join([color] + d['description']) + ' '
        d_str = d_str.lower().translate(trans_table)
        label = item2attr[item_id]
        
        for idx, att in enumerate(attr_entry):

            if ' '+att['entry']+' ' in d_str:
                label[idx] = 1

        print('extract attribute label: %d / %d' % (i_item, len(item2desc)))
        i_item += 1
        

    samples = io.load_json(design_root + 'Label/inshop_samples.json')
    attr_label = {s_id : item2attr[s['item_id']] for s_id, s in samples.iteritems()}

    io.save_data(attr_label, design_root + 'Label/inshop_attr_label.pkl')
    print('create attribute label')


def clean_attribute_label():
    '''
    Clean attribute labels created by create_attribute_label().
    - Remove attribute entries with only few samples
    - Compute positive rate for each attribute
    '''
    
    # config
    attr_top = 500
    
    # load attribute entry and label
    print('loading attribute label')
    attr_entry = io.load_json(root + 'Label/attribute_entry.json')
    attr_label = io.load_json(root + 'Label/attribute_label.json')


    # count positive sample number of each attribute
    print('select top %d attributes' % attr_top)
    attr_mat = np.array(attr_label.values())
    attr_pos_num = attr_mat.sum(axis = 0)
    
    attr_order = np.argsort(attr_pos_num * -1)[0:attr_top]
    attr_entry_t = [attr_entry[idx] for idx in attr_order]
    
    s_ids = attr_label.keys()
    attr_mat_t = attr_mat[:, attr_order]
    attr_label_t = dict(zip(s_ids, attr_mat_t.tolist()))
    
    attr_pos_num_t = attr_pos_num[attr_order].tolist()

    # compute positive rate for each attribute
    num_sample = len(attr_label)
    for idx, att in enumerate(attr_entry_t):
        attr_entry_t[idx]['pos_rate'] = 1.0 * attr_pos_num_t[idx] / num_sample

    # output
    print('saving cleaned attribute label')
    fn_entry = root + 'Label/attribute_entry_top%d.json' % attr_top
    io.save_json(attr_entry_t, fn_entry)

    fn_label = root + 'Label/attribute_label_top%d.json' % attr_top
    io.save_json(attr_label_t, fn_label)



def create_sample_index_for_attribute_dataset():
    '''
    Simply add "img_path" field for each sample, which is the image path used in training and testing
    '''

    # config
    img_root = root + 'Img/img_aligned/'
    fn_out = root + 'Label/samples_attr.json'

    # update sample
    samples = io.load_json(root + 'Label/samples.json')

    for s_id in samples.keys():
        samples[s_id]['img_path'] = img_root + s_id + '.jpg'

    io.save_json(samples, fn_out)

            
if __name__ == '__main__':

    ##############################################
    # old version, to be deprecated


    # create_sample_index()
    # create_split()
    # create_attribute_entry()
    # create_attribute_label()
    # clean_attribute_label()

    # create_sample_index_for_attribute_dataset()

    ##############################################

    create_sample_index_and_label()
    # create_split()

    # create_color_entry_and_label()
    # create_attribute_label()
    
