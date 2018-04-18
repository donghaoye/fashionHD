from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np
import util.io as io

root = 'datasets/DF_Pose/'

def create_image_index():
    ''' create image index, split and pose label from original index.p file '''
    index = io.load_data(root + 'Anno/index.pkl')

    split = {'train': [], 'test': []}
    pose_label = {}

    for img, joints, is_train in zip(index['imgs'], index['joints'], index['train']):
        s_id = img.split('/')[1][0:7]
        if is_train:
            split['train'].append(s_id)
        else:
            split['test'].append(s_id)

        for i in range(len(joints)):
            if not (joints[i][0] == -1 and joints[i][1] == -1):
                joints[i][0] *= 256
                joints[i][1] *= 256
        pose_label[s_id] = joints.tolist()

    io.save_json(split, root + 'Label/split.json')
    io.save_data(pose_label, root + 'Label/pose_label.pkl')


def create_pair_index():
    '''
    create pair index
    '''
    from itertools import combinations
    from collections import defaultdict

    split = io.load_json(root + 'Label/split.json')
    pair_split = {'train': [], 'test': []}
    for subset in ['train', 'test']:
        cloth = defaultdict(lambda : [])
        for s_id in split[subset]:
            cloth[s_id[0:5]].append(s_id)

        for group in cloth.values():
            pair_split[subset] += combinations(group, 2)

    np.random.shuffle(pair_split['train'])
    np.random.shuffle(pair_split['test'])
    pair_split['debug'] = pair_split['train'][0:32]
    io.save_json(pair_split, root + 'Label/pair_split.json')
    print('train pair: %d' % len(pair_split['train']))
    print('test pair: %d' % len(pair_split['test']))



if __name__ == '__main__':
    create_image_index()
    # create_pair_index()
