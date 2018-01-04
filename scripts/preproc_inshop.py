from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np
from collections import defaultdict

import util.io as io


root = '/data2/ynli/datasets/DeepFashion/In-shop/'


def create_sample_index():
    '''
    Create sample file from original annotation files
    - Assign an unique id for each sample (one item may contain multiple samples)
    '''

    high_res = True

    bbox_list = io.load_str_list(root + '/Anno/list_bbox_inshop.txt')[2::]
    samples = {}


    if high_res:
        img_root = root + 'Img/img_highres/'
        fn_out = root + 'Label/samples_highres.json'
    else:
        img_root = root + 'Img/img/'
        fn_out = root + 'Label/samples.json'

    fail_list = []
    for idx, s in enumerate(bbox_list):
        img_id = str(idx)

        s = s.split()
        assert len(s) == 7

        path = s[0]
        path_split = path.split('/')[1:]
        path = '/'.join(path_split)

        gender = path_split[0]
        category = path_split[1]
        item_id = path_split[2]
        img_path_org = img_root + '/' + path
        pose = path_split[3].split('_')[2][0:-4]
        assert pose in {'front', 'back', 'side', 'full', 'flat', 'additional'}


        if not os.path.isfile(img_path_org):
            fail_list.append(img_path_org)
            continue

        samples[img_id] = {
            'img_id': img_id,
            'item_id': item_id,
            'gender': gender,
            'category': category,
            'pose': pose,
            'img_path_org': img_path_org,
        }


        print('%s' % img_id)

    io.save_json(samples, fn_out)

    print('\n')
    print('save sample index to %s' % fn_out)
    print('%d samples not found!' % len(fail_list))
    for s in fail_list:
        print(s)

def create_split():
    '''
    Split dataset into train/test sets, where ttems are NOT overlapped.
    In original split, train/test sets have equal size. We create our own split with a larger train set.
    '''

    # config
    use_original = False
    train_rate = 0.8

    # load sample
    samples = io.load_json(root + 'Label/samples.json')

    if use_original:
        # load split file
        split_list = io.load_str_list(root + 'Eval/list_eval_partition.txt')[2::]
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

    fn_out = root + 'Label/split_attr.json'
    io.save_json(split, fn_out)


def create_attribute_entry():
    '''
    Create attribute entries
    - Using attribute entries defined in Category_and_Attribute branch
    - Add color entries, which are extracted from the In-shop description annotation
    '''

    # config
    fn_out = root + 'Label/attribute_entry.json'

    # load attribute entry from Category_and_Attribute branch
    attr_list = io.load_str_list('/data2/ynli/datasets/DeepFashion/Category_and_Attribute/Anno/list_attr_cloth.txt')[2::]

    attr_entry = []
    for s in attr_list:
        s = s.split()
        att_name = ' '.join(s[0:-1])
        att_type = int(s[-1])
        attr_entry.append({'entry': att_name, 'type': att_type})

    attr_name = set([att['entry'] for att in attr_entry])


    # add color entries
    description = io.load_json(root + 'Anno/list_description_inshop.json')
    colors = [d['color'].lower().replace('-', ' ').split() for d in description]
    colors = set([c[0] for c in colors if len(c) == 1])

    for c in colors:
        if c in attr_name:
            print('color [%s] already in attribute entry list' % c)
        else:
            attr_entry.append({'entry': c, 'type': 0})

    # manually revise attribute type of some color-attributes
    revise_list = ['pink', 'rose', 'cloud', 'red']
    for idx, att in enumerate(attr_entry):
        if att['entry'] in revise_list:
            revise_list.remove(att['entry'])
            # set type of this attribute to be 0-color
            attr_entry[idx]['type'] = 0

    
    print('attribute entry number: %d' % len(attr_entry))
    print('save to %s' % fn_out)
    io.save_json(attr_entry, fn_out)


def create_attribute_label():
    '''
    Create attribute label by matching attribute entries (defined in create_attribute_entry) in language description annotations.
    '''

    # config
    fn_out = root + 'Label/attribute_label.json'

    puncs = u'.,!?"%'
    trans_table = {ord(c): u' ' for c  in puncs}


    # load attribute entry
    attr_entry = io.load_json(root + 'Label/attribute_entry.json')
    num_attr = len(attr_entry)
    attr_label_item = defaultdict(lambda : [0] * num_attr)

    # load language description
    description = io.load_json(root + 'Anno/list_description_inshop.json')
    description = {d['item']:d for d in description}

    # attribute matching
    
    i_item = 0

    for item_id, d in description.iteritems():
        
        color = d['color'].replace('-', ' ')
        d_str = ' ' + ' '.join([color] + d['description']) + ' '
        d_str = d_str.lower().translate(trans_table)
        label = attr_label_item[item_id]
        
        for idx, att in enumerate(attr_entry):

            if ' '+att['entry']+' ' in d_str:
                label[idx] = 1

        print('extract attribute label: %d / %d' % (i_item, len(description)))
        i_item += 1
        
        #if i_item == 10:
            #break

    samples = io.load_json(root + 'Label/samples.json')
    attr_label = {s_id : attr_label_item[s['item_id']] for s_id, s in samples.iteritems()}

    io.save_json(attr_label, fn_out)


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
    # create_sample_index()
    # create_split()
    # create_attribute_entry()
    # create_attribute_label()
    # clean_attribute_label()

    create_sample_index_for_attribute_dataset()

        
