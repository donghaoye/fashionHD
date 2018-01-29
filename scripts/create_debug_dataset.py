from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np
from collections import defaultdict

import util.io as io

design_root = 'datasets/DeepFashion/Fashion_design/'

def create_debug_ca_dataset():
    '''
    Create a mini subset of Category_and_Attribute data. Assume standard CA index file and label files already exist.
    '''

    num_train = 10
    num_test = 10
    same_train_test = True

    samples = io.load_json(design_root + 'Label/ca_samples.json')
    attr_label = io.load_data(design_root + 'Label/ca_attr_label.pkl')
    bbox_label = io.load_data(design_root + 'Label/ca_bbox_label_256.pkl')
    lm_label = io.load_data(design_root + 'Label/ca_landmark_label_256.pkl')


    if same_train_test:
        id_list = samples.keys()[0:num_train]
        split = {'train': id_list, 'test': id_list}
    else:
        id_list = samples.keys()[0:(num_train + num_test)]
        split = {'train': id_list[0:num_train], 'test': id_list[num_train::]}


    samples = {s_id:samples[s_id] for s_id in id_list}
    attr_label = {s_id:attr_label[s_id] for s_id in id_list}
    bbox_label = {s_id:bbox_label[s_id] for s_id in id_list}
    lm_label = {s_id:lm_label[s_id] for s_id in id_list}
   

    io.save_json(samples, design_root + 'Label/debugca_samples.json')
    io.save_data(attr_label, design_root + 'Label/debugca_attr_label.pkl')
    io.save_data(bbox_label, design_root + 'Label/debugca_bbox_label.pkl')
    io.save_data(lm_label, design_root + 'Label/debugca_landmark_label.pkl')
    io.save_json(split, design_root + 'Split/debugca_split.json')

def create_dug_ca_gan_dataset():
    num_train = 10
    num_test = 10
    same_train_test = True

    samples = io.load_json(design_root + 'Label/ca_samples.json')
    attr_label = io.load_data(design_root + 'Label/ca_attr_label.pkl')
    bbox_label = io.load_data(design_root + 'Label/ca_bbox_label_256.pkl')
    lm_label = io.load_data(design_root + 'Label/ca_landmark_label_256.pkl')
    seg_path_list = io.load_json(design_root + 'Label/ca_seg_paths.json')
    ca_split = io.load_json(design_root + 'Split/ca_gan_split_trainval_upper.json')

    if same_train_test:
        split = {
            'train': ca_split['train'][0:num_train],
            'test': ca_split['train'][0:num_train]
        }
        id_list = split['train']
    else:
        split = {
            'train': ca_split['train'][0:num_train],
            'test': ca_split['test'][0:num_test]
        }
        id_list = split['train'] + split['test']

    samples = {s_id: samples[s_id] for s_id in id_list}
    attr_label = {s_id:attr_label[s_id] for s_id in id_list}
    bbox_label = {s_id:bbox_label[s_id] for s_id in id_list}
    lm_label = {s_id:lm_label[s_id] for s_id in id_list}
    seg_path_list = {s_id: seg_path_list[s_id] for s_id in id_list}

    io.save_json(samples, design_root + 'Label/debugca_gan_samples.json')
    io.save_data(attr_label, design_root + 'Label/debugca_gan_attr_label.pkl')
    io.save_data(bbox_label, design_root + 'Label/debugca_gan_bbox_label.pkl')
    io.save_data(lm_label, design_root + 'Label/debugca_gan_landmark_label.pkl')
    io.save_json(seg_path_list, design_root + 'Label/debugca_seg_paths.json')
    io.save_json(split, design_root + 'Split/debugca_gan_split.json')




if __name__ == '__main__':
    # create_debug_ca_dataset()
    create_dug_ca_gan_dataset()