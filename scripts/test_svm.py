from __future__ import division, print_function

import sys
import os
liblinear_path = '/data2/ynli/download/liblinear-2.20/python'
sys.path.append(liblinear_path)
import time
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV

liblinear_path = '/data2/ynli/download/liblinear-2.20/python'
sys.path.append(liblinear_path)
assert os.path.isfile(liblinear_path + '/liblinearutil.py')
import liblinearutil as liblinear
import util.io as io

############################################
# config
############################################
# num_sample = 5000
# feat_sz = 1024
# offset = 0.05

# num_test = 100

# num_pos = 10

############################################
# create data
############################################
# label_train = np.zeros(num_sample, dtype = np.int)
# label_train[0:num_pos] = 1

# feat_train = np.random.rand(num_sample, feat_sz)
# feat_train[0:num_pos,:]+= offset

# label_test = np.zeros(num_test, dtype = np.int)
# label_test[0:(num_test)//2] = 1

# feat_test = np.random.rand(num_test, feat_sz)
# feat_test[0:(num_test)//2,:]+= offset

# label_train = 1 - label_train
# label_test = 1 - label_test

############################################
# test SVC
############################################
# print('\n#######################################')
# print('test SVC')
# t = time.time()
# clf = SVC(kernel = 'linear', probability = True)
# clf.fit(feat_train, label_train)
# prob = clf.predict_proba(feat_test)[:,1]
# t = time.time() -t
# acc = (sum(prob[0:(num_test)//2] > 0.5) + sum(prob[(num_test//2)::] < 0.5)) / num_test

# print('time: %f, acc: %f' % (t, acc))


############################################
# test LinearSVC
############################################
# print('\n#######################################')
# print('test LinearSVC')
# t = time.time()
# clf = CalibratedClassifierCV(LinearSVC())
# clf.fit(feat_train, label_train)
# prob = clf.predict_proba(feat_test)[:,1]
# t = time.time() -t
# acc = (sum(prob[0:(num_test)//2] > 0.5) + sum(prob[(num_test//2)::] < 0.5)) / num_test
# print('time: %f, acc: %f' % (t, acc))


############################################
# test liblinear
############################################
# print('\n#######################################')
# print('test liblinear')

# b = 1.
# w1 = num_sample / num_pos - 1
# for c in [0.0001, 0.01, 1, 100, 10000]:
#     t = time.time()
#     opt = '-s 0 -B %f -c %f -w1 %f -q' % (b, c, w1)
#     # opt = '-s 0 -B %f -C -w1 %f -q' % (b, w1)
#     print(opt)
#     m = liblinear.train(label_train, feat_train, opt)
#     print(m.get_labels())
#     r = liblinear.predict([], feat_test, m, '-b 1 -q')
#     prob = np.array(r[2])[:,0]
#     # print(p)
#     # print(p.shape)

#     t = time.time() -t
#     acc = (sum(prob[0:(num_test)//2] > 0.5) + sum(prob[(num_test//2)::] < 0.5)) / num_test
#     print('time: %f, acc: %f' % (t, acc))


num_sample = 10000

feat_data = io.load_data('checkpoints/AE_1.5/feat/latest.pkl')
attr_label = io.load_data('datasets/DeepFashion/Fashion_design/Label/ca_attr_label.pkl')
feat_train = feat_data['feat_train']
label_train = np.array([attr_label[s_id] for s_id in feat_data['id_list_train']])


feat_train = feat_train[0:num_sample, :]

for idx in range(1000):
    label = label_train[0:num_sample, idx].astype(np.int)
    if label.sum() == 0:
        continue

    w = label.shape[0] / label.sum() - 1

    opt = '-s 0 -B 1 -w1 %f -C -q' % w
    print(opt)
    m = liblinear.train(label, feat_train, opt)
