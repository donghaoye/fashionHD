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


# config
num_img = 100
num_top_attr = 5


opt = TestAttributeOptions().parse()
opt.max_dataset_size = num_img

# create model
model = AttributeEncoder()
model.initialize(opt)
model.eval()

# create data loader

loader = CreateDataLoader(opt, split = 'test')

# create visualizer
visualizer = AttributeVisualizer(opt)

for i, data in enumerate(loader):
    model.set_input(data)
    model.test()

    visualizer.visualize_attr_pred(model, num_top_attr)

    print('\rVisualizing %d/%d (%.2f%%)' % (i, len(loader), 100.*i/len(loader)), end = '')
    sys.stdout.flush()
print('\n')