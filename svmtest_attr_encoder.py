from __future__ import division, print_function

from models.attribute_encoder import AttributeEncoder
from data.data_loader import CreateDataLoader
from options.attribute_options import TestAttributeOptions
from models.networks import MeanAP
from misc.visualizer import AttributeVisualizer

import os
import sys
import time
import numpy as np
import util.io as io
from collections import OrderedDict

liblinear_path = '/data2/ynli/download/liblinear-2.20/python'
sys.path.append(liblinear_path)
assert os.path.isfile(liblinear_path + '/liblinearutil.py')
import liblinearutil as liblinear


def extract_feature(model, save_feat = True):

    opt = model.opt
    fn_feat = os.path.join(os.path.join('checkpoints', opt.id, 'feat', '%s.pkl' % opt.which_epoch))

    if os.path.isfile(fn_feat):
        print('loading feature from %s' % fn_feat)
        feat_data = io.load_data(fn_feat)
        # feat_data['feat_train']
        # feat_data['feat_test']
        # feat_data['id_list_train']
        # feat_data['id_list_test']
        
    else:
        model.eval()
        feat_data = {
            'feat_train': [],
            'feat_test': [],
            'id_list_train': [],
            'id_list_test': []
        }

        for split in ['train', 'test']:
            loader = CreateDataLoader(opt, split = split)

            for i, data in enumerate(loader):
                model.set_input(data)
                model.extract_feat()
                feat_data['feat_%s'%split].append(model.output['feat'].data.cpu().numpy()) # size: batch_size * feat_size
                feat_data['id_list_%s'%split] += data['id'] # list of length batch_size
                print('\r[%s] extract feature from %s samples %d/%d' % (opt.id, split, i, len(loader)), end = '')
            print('\n')
            feat_data['feat_%s'%split] = np.vstack(feat_data['feat_%s'%split])

        if save_feat:
            io.mkdir_if_missing(os.path.join('checkpoints', opt.id, 'feat'))
            io.save_data(feat_data, fn_feat)

    return feat_data

def _svm_test_attr_unit(worker_idx, idx_attr_rng, feat_train, label_train, feat_test, output_pred_list):
    idx_list = range(idx_attr_rng[0], idx_attr_rng[1])

    for idx in idx_list:
        t = time.time()
        label = label_train[:, idx].astype(np.int)
        b = 1.
        c = 512.
        w1 = label.shape[0]/label.sum() - 1
        opt = '-s 0 -B %f -c %f -w1 %f -q' % (b, c, w1)

        svm_model = liblinear.train(label, feat_train, opt)
        svm_out = liblinear.predict([], feat_test, svm_model, '-b 1 -q')
        k = svm_model.get_labels().index(1)
        prob = np.array(svm_out[2])[:,k]
        output_pred_list[idx] = prob
        print('[%d] test attribute %d [%d/%d], cost %.2f sec' % (worker_idx, idx, idx - idx_list[0], len(idx_list), time.time() - t))
        sys.stdout.flush()


if __name__ == '__main__':

    # config
    train_on_val_set = True
    num_worker = 20

    opt = TestAttributeOptions().parse()
    # create model
    model = AttributeEncoder()
    model.initialize(opt)
    
    # extract feature
    feat_data = extract_feature(model)
    print('extract feature done!')

    # load attribute label
    attr_label = io.load_data('datasets/DeepFashion/Fashion_design/' + opt.fn_label)
    print('load label done!')
    attr_entry = io.load_json('datasets/DeepFashion/Fashion_design/' + opt.fn_entry)
    print('load entry done!')


    feat_train = feat_data['feat_train']
    feat_test = feat_data['feat_test']
    label_train = np.array([attr_label[s_id] for s_id in feat_data['id_list_train']])
    label_test = np.array([attr_label[s_id] for s_id in feat_data['id_list_test']])

    if train_on_val_set:
        id_list_val = io.load_json('datasets/DeepFashion/Fashion_design/Split/ca_split.json')['val']
        id2idx = {s_id:idx for idx, s_id in enumerate(feat_data['id_list_train'])}

        idx_list_val = [id2idx[s_id] for s_id in id_list_val]
        feat_train = feat_train[idx_list_val, :]
        label_train = label_train[idx_list_val, :]

    print('organize label done!')

    assert feat_train.shape[1] == feat_test.shape[1]
    assert feat_train.shape[0] == label_train.shape[0]
    assert feat_test.shape[0] == label_test.shape[0]

    num_attr = label_train.shape[1]
    # num_attr = 20
    # label_train = label_train[:,0:20]
    # label_test = label_test[:,0:20]

    print('start to train SVMs!')
    from multiprocessing import Process, Manager
    block_size = num_attr // num_worker + 1
    pred_list = Manager().list(range(num_attr))
    p_list = []

    for worker_idx in range(num_worker):
        idx_attr_rng = [block_size * worker_idx, min(num_attr, block_size * (worker_idx + 1))]
        p = Process(target = _svm_test_attr_unit, 
            args = (worker_idx, idx_attr_rng, feat_train, label_train, feat_test, pred_list))
        p.start()
        p_list.append(p)
        print('worker %d for attribute %d-%d' % (worker_idx, idx_attr_rng[0], idx_attr_rng[1]))

    for p in p_list:
        p.join()

    pred_test = np.stack(pred_list, axis = 1)

    crit_ap = MeanAP()
    crit_ap.add(pred_test, label_test)

    mAP, ap_list = crit_ap.compute_mean_ap()
    mBP, bp_list = crit_ap.compute_balanced_precision()
    rec3_avg, rec3_attr, rec3_overall = crit_ap.compute_recall(k = 3)
    rec5_avg, rec5_attr, rec5_overall = crit_ap.compute_recall(k = 5)

    # display result
    result = OrderedDict([
        ('mAP', mAP),
        ('mBP', mBP),
        ('rec3_avg', rec3_avg),
        ('rec5_avg', rec5_avg),
        ('rec3_overall', rec3_overall),
        ('rec5_overall', rec5_overall),
        ])

    AttributeVisualizer(opt).print_error(result)

    # save result
    rec3_attr = [(attr_entry[i]['entry'], rec3_attr[i]) for i in xrange(num_attr)]
    rec5_attr = [(attr_entry[i]['entry'], rec5_attr[i]) for i in xrange(num_attr)]
    result_output = {
        'rec3_avg': rec3_avg,
        'rec5_avg': rec5_avg,
        'rec3_attr': rec3_attr,
        'rec5_attr': rec5_attr
    }

    io.mkdir_if_missing(os.path.join('checkpoints', opt.id, 'test'))
    io.save_json(result_output, os.path.join('checkpoints', opt.id, 'test', '%s.json'))

