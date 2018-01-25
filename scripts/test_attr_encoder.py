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

opt = TestAttributeOptions().parse()

# create model
model = AttributeEncoder()
model.initialize(opt)
model.eval()

# create data loader

test_loader = CreateDataLoader(opt, split = 'test')

# create visualizer
visualizer = AttributeVisualizer(opt)

# create metric
crit_ap = MeanAP()

for i, data in enumerate(test_loader):
    model.set_input(data)
    model.test()
    crit_ap.add(model.output['prob'], model.input['label'])
    print('\rTesting %d/%d (%.2f%%)' % (i, len(test_loader), 100.*i/len(test_loader)), end = '')
    sys.stdout.flush()
print('\n')

test_error = model.get_current_errors()
mean_ap, ap_list = crit_ap.compute_mean_ap()
mean_bp, bp_list = crit_ap.compute_balanced_precision()
rec3_avg, rec3_list, rec3_overall = crit_ap.compute_recall(k=3)
rec5_avg, rec5_list, rec5_overall = crit_ap.compute_recall(k=5)

result = OrderedDict([
    ('loss_attr', test_error['loss_attr']),
    ('mAP', mean_ap),
    ('mBP', mean_bp),
    ('rec3_avg', rec3_avg),
    ('rec5_avg', rec5_avg),
    ('rec3_overall', rec3_overall),
    ('rec5_overall', rec5_overall),
    ('AP_list', ap_list),
    ('rec3_list', rec3_list),
    ('rec5_list', rec5_list)
    ])

visualizer.show_attr_pred_statistic(result)

