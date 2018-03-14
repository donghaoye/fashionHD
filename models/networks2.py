from __future__ import division, print_function

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.optim import lr_scheduler
from resnet_wrapper import create_resnet_conv_layers
import numpy as np
import functools

from networks import *


###############################################################################
# Attribute
###############################################################################

def define_attr_encoder_net(opt):
    if opt.joint_cat:
        if opt.spatial_pool != 'none' or opt.input_lm:
            raise NotImplementedError()
        if opt.spatial_pool == 'none':
            net = JointNoneSpatialAttributeEncoderNet(
                convnet = opt.convnet,
                input_nc = opt.input_nc,
                output_nc = opt.n_attr,
                output_nc1 = opt.n_cat,
                feat_norm = opt.feat_norm,
                gpu_ids = opt.gpu_ids,
                init_type = opt.init_type)
    else:
        if opt.input_lm:
            if opt.spatial_pool == 'none':
                raise NotImplementedError()
            else:
                net = DualSpatialAttributeEncoderNet(
                    convnet = opt.convnet,
                    spatial_pool = opt.spatial_pool,
                    input_nc = opt.input_nc,
                    output_nc = opt.n_attr,
                    lm_input_nc = opt.lm_input_nc,
                    lm_output_nc = opt.lm_output_nc,
                    lm_fusion = opt.lm_fusion,
                    feat_norm = opt.feat_norm,
                    gpu_ids = opt.gpu_ids,
                    init_type = opt.init_type)
        else:
            if opt.spatial_pool == 'none':
                net = NoneSpatialAttributeEncoderNet(
                    convnet = opt.convnet,
                    input_nc = opt.input_nc,
                    output_nc = opt.n_attr,
                    feat_norm = opt.feat_norm,
                    gpu_ids = opt.gpu_ids,
                    init_type = opt.init_type)
            elif opt.spatial_pool in {'max', 'noisyor'}:
                net = SpatialAttributeEncoderNet(
                    convnet = opt.convnet,
                    spatial_pool = opt.spatial_pool,
                    input_nc = opt.input_nc, 
                    output_nc = opt.n_attr,
                    feat_norm = opt.feat_norm,
                    gpu_ids = opt.gpu_ids,
                    init_type = opt.init_type)

    if len(opt.gpu_ids) > 0:
        net.cuda()

    return net



class NoisyOR(nn.Module):
    def __init__(self):
        super(NoisyOR,self).__init__()

    def forward(self, prob_map):
        bsz, nc, w, h = prob_map.size()
        neg_prob_map = 1 - prob_map.view(bsz, nc, -1)
        neg_prob = Variable(prob_map.data.new(bsz, nc).fill_(1))

        for i in xrange(neg_prob_map.size(2)):
            neg_prob = neg_prob * neg_prob_map[:,:,i]

        return 1 - neg_prob

class LandmarkPool(nn.Module):
    def __init__(self, pool = 'max', region_size = (3,3)):
        super(LandmarkPool, self).__init__()

    def forward(feat_map, lm_list):
        raise NotImplementedError('LandmarkPool.forward not implemented')

        

def create_stack_conv_layers(input_nc, feat_nc_s = 64, feat_nc_f = 1024, num_layer = 5, norm = 'batch'):
    
    c_in = input_nc
    c_out = feat_nc_s
    norm_layer = get_norm_layer(norm_type = norm)
    conv_layers = []

    for n in range(num_layer):
        conv_layers.append(nn.Conv2d(c_in, c_out, 4,2,1, bias = False))
        conv_layers.append(norm_layer(c_out))
        conv_layers.append(nn.ReLU())

        c_in = c_out
        c_out = feat_nc_f if n == num_layer-2 else c_out * 2

    conv = nn.Sequential(*conv_layers)

    conv.input_nc = input_nc
    conv.output_nc = feat_nc_f

    return conv


class NoneSpatialAttributeEncoderNet(nn.Module):
    def __init__(self, convnet, input_nc, output_nc, feat_norm, gpu_ids, init_type):
        '''
        Args:
            convnet (str): convnet architecture.
            input_nc (int): number of input channels.
            output_nc (int): number of output channels (number of attribute entries)
        '''
        super(NoneSpatialAttributeEncoderNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.feat_norm = feat_norm

        if convnet == 'stackconv':
            pretrain = False
            self.conv = create_stack_conv_layers(input_nc)
        else:
            pretrain = (input_nc == 3)
            self.conv = create_resnet_conv_layers(convnet, input_nc, pretrain)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(self.conv.output_nc, output_nc)

        # initialize weights
        init_weights(self.fc, init_type = init_type)
        if not pretrain:
            init_weights(self.conv, init_type = init_type)

        if pretrain:
            print('load CNN weight pretrained on ImageNet!')


    def forward(self, input_img):
        
        feat, _ = self.extract_feat(input_img)
        return self.predict(feat)

    def extract_feat(self, input_img):
        bsz = input_img.size(0)

        if self.gpu_ids:
            feat_map = nn.parallel.data_parallel(self.conv, input_img)
        else:
            feat_map = self.conv(input_img)

        if self.feat_norm:
            feat_map = feat_map / feat_map.norm(p=2, dim=1, keepdim=True)

        feat = self.avgpool(feat_map).view(bsz, -1)

        return feat, feat_map

    def predict(self, feat_map):
        '''
        Input:
            feat_map: feature map (bsz, c, H, W) or feat(bsz, c)
        Output:
            prob
            prob_map
        '''
        bsz = feat_map.size(0)
        if feat_map.ndimension() == 4:
            feat = self.avgpool(feat_map).view(bsz, -1)
        else:
            assert feat_map.ndimension() == 2
            feat = feat_map

        prob = F.sigmoid(self.fc(feat))
        return prob, None


class SpatialAttributeEncoderNet(nn.Module):
    def __init__(self, convnet, spatial_pool, input_nc, output_nc, feat_norm, gpu_ids, init_type):
        super(SpatialAttributeEncoderNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.feat_norm = feat_norm

        if convnet == 'stackconv':
            pretrain = False
            self.conv = create_stack_conv_layers(input_nc)
        else:
            pretrain = (input_nc == 3)
            self.conv = create_resnet_conv_layers(convnet, input_nc, pretrain)

        self.cls = nn.Conv2d(self.conv.output_nc, output_nc, kernel_size = 1)

        if spatial_pool == 'max':
            self.pool = nn.MaxPool2d(7, stride=1)
        elif spatial_pool == 'noisyor':
            self.pool = NoisyOR()

        # initialize weights
        init_weights(self.cls, init_type = init_type)
        if spatial_pool == 'noisyor':
            # special initialization
            init.constant(self.cls.bias, -6.58)
            
        if pretrain:
            print('load CNN weight pretrained on ImageNet!')
        else:
            init_weights(self.conv, init_type = init_type)



    def forward(self, input_img):
        _, feat_map = self.extract_feat(input_img)
        return self.predict(feat_map)

    def extract_feat(self, input_img):
        bsz = input_img.size(0)
        if self.gpu_ids:
            feat_map = nn.parallel.data_parallel(self.conv, input_img)
        else:
            feat_map = self.conv(input_img)

        if self.feat_norm:
            feat_map = feat_map / feat_map.norm(p=2, dim=1, keepdim=True)
        feat = F.avg_pool2d(feat_map, kernel_size = 7, stride = 1).view(bsz, -1)

        return feat, feat_map

    def predict(self, feat_map):
        '''
        Input:
            feat_map
        Output:
            prob
            prob_map
        '''
        bsz = feat_map.size(0)
        prob_map = F.sigmoid(self.cls(feat_map))
        prob = self.pool(prob_map).view(bsz, -1)

        return prob, prob_map

class DualSpatialAttributeEncoderNet(nn.Module):
    '''
    Attribute Encoder with 2 branches of ConvNet, for RGB image and Landmark heatmap respectively.
    '''
    def __init__(self, convnet, spatial_pool, input_nc, output_nc, lm_input_nc, lm_output_nc, lm_fusion, feat_norm, gpu_ids, init_type):
        super(DualSpatialAttributeEncoderNet, self).__init__()
        # create RGB channel
        self.gpu_ids = gpu_ids
        self.feat_norm = feat_norm
        self.spatial_pool = spatial_pool
        self.fusion = lm_fusion
        if convnet == 'stackconv':
            pretrain = False
            self.conv = create_stack_conv_layers(input_nc)
        else:
            pretrain = (input_nc == 3)
            self.conv = create_resnet_conv_layers(convnet, input_nc, pretrain)
        

        # create landmark channel
        lm_layer_list = []
        c_in = lm_input_nc
        c_out = lm_output_nc // (2**4)

        for n in range(5):
            lm_layer_list.append(nn.Conv2d(c_in, c_out, 4, 2, 1, bias = False))
            lm_layer_list.append(nn.BatchNorm2d(c_out))
            lm_layer_list.append(nn.ReLU())
            c_in = c_out
            c_out *= 2

        self.conv_lm = nn.Sequential(*lm_layer_list)

        # create fusion layers
        if lm_fusion == 'concat':
            feat_nc = self.conv.output_nc + lm_output_nc
            self.cls = nn.Conv2d(feat_nc, output_nc, kernel_size = 1)
        elif lm_fusion == 'linear':
            feat_nc = self.conv.output_nc + lm_output_nc
            self.fuse_layer = nn.Sequential(
                nn.Conv2d(feat_nc, self.conv.output_nc, kernel_size = 1),
                nn.BatchNorm1d(self.conv.output_nc),
                nn.ReLu()
                )
            self.cls = nn.Conv2d(self.conv.output_nc, output_nc, kernel_size = 1)
        else:
            print(lm_fusion)
            raise NotImplementedError()


        # create pooling layers
        if spatial_pool == 'max':
            self.pool = nn.MaxPool2d(7, stride=1)
        elif spatial_pool == 'noisyor':
            self.pool = NoisyOR()

        # initialize weights
        init_weights(self.cls, init_type = init_type)
        init_weights(self.conv_lm, init_type = init_type)
        if lm_fusion == 'linear':
            init_weights(self.fuse_layer, init_type = init_type)
        if spatial_pool == 'noisyor':
            # special initialization
            init.constant(self.cls.bias, -6.58)
            
        if pretrain:
            print('load CNN weight pretrained on ImageNet!')
        else:
            init_weights(self.conv, init_type = init_type)

    def forward(self, input_img, input_lm_heatmap):
        bsz = input_img.size(0)
        if self.gpu_ids:
            img_feat_map = nn.parallel.data_parallel(self.conv, input_img)
            lm_feat_map = nn.parallel.data_parallel(self.conv_lm, input_lm_heatmap)
        else:
            img_feat_map = self.conv(input_img)
            lm_feat_map = self.conv_lm(input_lm_heatmap)

        feat_map = None
        if self.fusion == 'concat':
            feat_map = torch.cat((img_feat_map, lm_feat_map), dim = 1)
        elif self.fusion == 'linear':
            feat_map = self.fuse_layer(torch.cat((img_feat_map, lm_feat_map), dim = 1))
        else:
            print(self.fusion)
            raise NotImplementedError()

        if self.feat_norm:
            feat_map = feat_map / feat_map.norm(p=2, dim=1, keepdim=True)
        
        prob_map = F.sigmoid(self.cls(feat_map))
        prob = self.pool(prob_map).view(bsz, -1)

        return prob, prob_map

    def extract_feat(self, input_img, input_lm_heatmap):
        bsz = input_img.size(0)
        if self.gpu_ids:
            img_feat_map = nn.parallel.data_parallel(self.conv, input_img)
            lm_feat_map = nn.parallel.data_parallel(self.conv_lm, input_lm_heatmap)
        else:
            img_feat_map = self.conv(input_img)
            lm_feat_map = self.conv_lm(input_lm_heatmap)

        feat_map = None
        if self.fusion == 'cancat':
            feat_map = torch.cat((img_feat_map, lm_feat_map), dim = 1)
        elif self.fusion == 'linear':
            feat_map = self.fuse_layer(torch.cat((img_feat_map, lm_feat_map), dim = 1))

        if self.feat_norm:
            feat_map = feat_map / feat_map.norm(p=2, dim=1, keepdim=True)

        feat = F.avg_pool2d(feat_map, kernel_size = 7, stride = 1).view(bsz, -1)

        return feat, feat_map


class JointNoneSpatialAttributeEncoderNet(nn.Module):
    def __init__(self, convnet, input_nc, output_nc, output_nc1, feat_norm, gpu_ids, init_type):
        '''
        Args:
            convnet (str): convnet architecture.
            input_nc (int): number of input channels.
            output_nc (int): number of output channels (number of attribute entries)
            output_nc1 (int): number of auxiliary output chnnels (number of category entries)
        '''

        super(JointNoneSpatialAttributeEncoderNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.feat_norm = feat_norm

        if convnet == 'stackconv':
            pretrain = False
            self.conv = create_stack_conv_layers(input_nc)
        else:
            pretrain = (input_nc == 3)
            self.conv = create_resnet_conv_layers(convnet, input_nc, pretrain)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(self.conv.output_nc, output_nc)
        self.fc_cat = nn.Linear(self.conv.output_nc, output_nc1)

        # initialize weights
        init_weights(self.fc, init_type = init_type)
        init_weights(self.fc_cat, init_type = init_type)
        if not pretrain:
            init_weights(self.conv, init_type = init_type)

        if pretrain:
            print('load CNN weight pretrained on ImageNet!')

    def forward(self, input_img):
        bsz = input_img.size(0)
        if self.gpu_ids:
            feat_map = nn.parallel.data_parallel(self.conv, input_img)
        else:
            feat_map = self.conv(input_img)

        if self.feat_norm:
            feat_map = feat_map / feat_map.norm(p=2, dim=1, keepdim=True)

        feat = self.avgpool(feat_map).view(bsz, -1)
        prob = F.sigmoid(self.fc(feat))
        pred_cat = self.fc_cat(feat)

        return prob, None, pred_cat

    def extract_feat(self, input_img):
        bsz = input_img.size(0)
        if self.gpu_ids:
            feat_map = nn.parallel.data_parallel(self.conv, input_img)
        else:
            feat_map = self.conv(input_img)

        if self.feat_norm:
            feat_map = feat_map / feat_map.norm(p=2, dim=1, keepdim=True)

        feat = self.avgpool(feat_map).view(bsz, -1)

        return feat, feat_map
