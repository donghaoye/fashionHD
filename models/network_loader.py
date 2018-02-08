from __future__ import division, print_function

import torch
import torch.nn as nn
import torchvision
import networks
from torch.autograd import Variable


import os
import sys
import numpy as np
import time
from collections import OrderedDict
import util.io as io

def load_attribute_encoder_net(id, gpu_ids, which_epoch = 'latest'):
    '''
    Load pretrained attribute encoder as a module of GAN model.
    All options for attribute encoder will be loaded from its train_opt.json, except:
        - gpu_ids
        - is_train
        - which_epoch

    Input:
        id (str): ID of attribute encoder model
        gpu_ids: set gpu_ids for attribute model
        is_train: set train/test status for attribute model
    Output:
        net (nn.Module): network of attribute encoder
        opt (namespace): updated attribute encoder options
    '''
    from attribute_encoder import AttributeEncoder
    from options.attribute_options import TestAttributeOptions

    if not id.startswith('AE_'):
        id = 'AE_' + id

    # load attribute encoder options
    fn_opt = os.path.join('checkpoints', id, 'train_opt.json')
    if not os.path.isfile(fn_opt):
        raise ValueError('invalid attribute encoder id: %s' % id)
    opt_var = io.load_json(fn_opt)

    # update attribute encoder options
    opt = TestAttributeOptions().parse(ord_str = '', save_to_file = False, display = False, set_gpu = False)
    for k, v in opt_var.iteritems():
        if k in opt:
            opt.__dict__[k] = v

    opt.is_train = False
    opt.continue_train = False
    opt.gpu_ids = gpu_ids
    opt.which_epoch = which_epoch
    # opt.continue_train = False

    model = AttributeEncoder()
    model.initialize(opt)

    # frozen model parameters
    model.eval()
    for p in model.net.parameters():
        p.requires_grad = False

    return model.net, opt

def load_feature_spatial_transformer_net(id, gpu_ids, which_epoch='latest'):
    from feature_transform_model import FeatureSpatialTransformer
    from options.feature_spatial_transformer_options import TestFeatureSpatialTransformerOptions

    if not id.startswith('FeatST_'):
        id = 'FeatST_' + id

    # load options
    fn_opt = os.path.join('checkpoints', id, 'train_opt.json')
    if not os.path.isfile(fn_opt):
        raise ValueError('invalid attribute encoder id: %s' % id)
    opt_var = io.load_json(fn_opt)

    # update attribute encoder options
    opt = TestFeatureSpatialTransformerOptions().parse(ord_str = '', save_to_file = False, display = False, set_gpu = False)
    for k, v in opt_var.iteritems():
        if k in opt:
            opt.__dict__[k] = v

    opt.is_train = False
    opt.continue_train = False
    opt.gpu_ids = gpu_ids
    opt.which_epoch = which_epoch
    # opt.continue_train = False

    model = FeatureSpatialTransformer()
    model.initialize(opt)

    # frozen model parameters
    model.eval()
    for p in model.net.parameters():
        p.requires_grad = False

    return model.net, opt

