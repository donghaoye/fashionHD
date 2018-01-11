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

###############################################################################
# Functions
###############################################################################

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.startswith('Linear'):
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.startswith('BatchNorm2d'):
        init.normal(m.weight.data, 1.0, 0.02)

    if 'bias' in m._parameters and m.bias is not None:
        init.constant(m.bias.data, 0.0)

def weights_init_normal2(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.normal(m.weight.data, 0.0, 0.001)
    elif classname.startswith('Linear'):
        init.normal(m.weight.data, 0.0, 0.001)
    elif classname.startswith('BatchNorm2d'):
        init.normal(m.weight.data, 1.0, 0.001)

    if 'bias' in m._parameters and m.bias is not None:
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.startswith('Linear'):
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.startswith('BatchNorm2d'):
        init.normal(m.weight.data, 1.0, 0.02)
    
    if 'bias' in m._parameters and m.bias is not None:
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.startswith('Linear'):
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.startswith('BatchNorm2d'):
        init.normal(m.weight.data, 1.0, 0.02)

    if 'bias' in m._parameters and m.bias is not None:
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.orthogonal(m.weight.data, gain=1)
    elif classname.startswith('Linear'):
        init.orthogonal(m.weight.data, gain=1)
    elif classname.startswith('BatchNorm2d'):
        init.normal(m.weight.data, 1.0, 0.02)
    
    if 'bias' in m._parameters and m.bias is not None:
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    # print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'normal2':
        net.apply(weights_init_normal2)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

###############################################################################
# Loss Functions
###############################################################################
class Smooth_Loss():
    '''
    wrapper of pytorch loss layer.
    '''

    def __init__(self, crit):
        self.crit = crit
        self.clear()

    def __call__(self, input_1, input_2, *extra_input):
        loss = self.crit(input_1, input_2, *extra_input)
        self.weight_buffer.append(input_1.size(0))

        if isinstance(loss, Variable):
            self.buffer.append(loss.data[0])
        elif isinstance(loss, torch.tensor._TensorBase):
            self.buffer.append(loss[0])
        else:
            self.buffer.append(loss)

        return loss

    def clear(self):
        self.buffer = []
        self.weight_buffer = []

    def smooth_loss(self, clear = False):
        if len(self.weight_buffer) == 0:
            loss = 0
        else:
            loss = sum([l * w for l, w in zip(self.buffer, self.weight_buffer)]) / sum(self.weight_buffer)
            
        if clear:
            self.clear()
        return loss

class MeanAP():
    '''
    compute meanAP
    '''

    def __init__(self):
        self.clear()

    def clear(self):
        self.score = None
        self.label = None

    def add(self, new_score, new_label):

        inputs = [new_score, new_label]

        for i in range(len(inputs)):

            if isinstance(inputs[i], list):
                inputs[i] = np.array(inputs[i], dtype = np.float32)

            elif isinstance(inputs[i], np.ndarray):
                inputs[i] = inputs[i].astype(np.float32)

            elif isinstance(inputs[i], torch.tensor._TensorBase):
                inputs[i] = inputs[i].cpu().numpy().astype(np.float32)

            elif isinstance(inputs[i], Variable):
                inputs[i] = inputs[i].data.cpu().numpy().astype(np.float32)

        new_score, new_label = inputs
        assert new_score.shape == new_label.shape, 'shape mismatch: %s vs. %s' % (new_score.shape, new_label.shape)

        self.score = np.concatenate((self.score, new_score), axis = 0) if self.score is not None else new_score
        self.label = np.concatenate((self.label, new_label), axis = 0) if self.label is not None else new_label

    def compute_mean_ap(self):

        score, label = self.score, self.label

        assert score is not None and label is not None
        assert score.shape == label.shape, 'shape mismatch: %s vs. %s' % (score.shape, label.shape)
        assert(score.ndim == 2)
        M, N = score.shape[0], score.shape[1]

        # compute tp: column n in tp is the n-th class label in descending order of the sample score.
        index = np.argsort(score, axis = 0)[::-1, :]
        tp = label.copy().astype(np.float)
        for i in xrange(N):
            tp[:, i] = tp[index[:,i], i]
        tp = tp.cumsum(axis = 0)

        m_grid, n_grid = np.meshgrid(range(M), range(N), indexing = 'ij')
        tp_add_fp = m_grid + 1    
        num_truths = np.sum(label, axis = 0)
        # compute recall and precise
        rec = tp / num_truths
        prec = tp / tp_add_fp

        prec = np.append(np.zeros((1,N), dtype = np.float), prec, axis = 0)
        for i in xrange(M-1, -1, -1):
            prec[i, :] = np.max(prec[i:i+2, :], axis = 0)
        rec_1 = np.append(np.zeros((1,N), dtype = np.float), rec, axis = 0)
        rec_2 = np.append(rec, np.ones((1,N), dtype = np.float), axis = 0)
        AP = np.sum(prec * (rec_2 - rec_1), axis = 0)
        AP[np.isnan(AP)] = -1 # avoid error caused by classes that have no positive sample

        assert((AP <= 1).all())

        AP = AP * 100.
        meanAP = AP[AP >= 0].mean()

        return meanAP, AP

    def compute_balance_ap(self):
        '''
        compute the average of true-positive-rate and true-negative-rate
        '''

        score, label = self.score, self.label

        assert score is not None and label is not None
        assert score.shape == label.shape, 'shape mismatch: %s vs. %s' % (score.shape, label.shape)
        assert(score.ndim == 2)

        # compute true-positive and true-negative
        tp = np.where(np.logical_and(score > 0.5, label == 1), 1, 0)
        tn = np.where(np.logical_and(score < 0.5, label == 0), 1, 0)

        # compute average precise
        p_pos = tp.sum(axis = 0) / (label == 1).sum(axis = 0)
        p_neg = tn.sum(axis = 0) / (label == 0).sum(axis = 0)

        ave_p = (p_pos + p_neg) / 2

        ave_p = ave_p * 100.
        ave_ave_p = ave_p.mean()

        return ave_ave_p, ave_p

###############################################################################
# Optimizer and Scheduler
###############################################################################

def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay, gamma=opt.lr_gamma)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

###############################################################################
# Attribute
###############################################################################
def define_attr_encoder_net(convnet, input_nc, output_nc, spatial_pool = 'none', init_type = 'normal', gpu_ids = []):

    if spatial_pool == 'none':
        net = NoneSpatialAttributeEncoderNet(convnet, input_nc, output_nc,
            gpu_ids, init_type)
    elif spatial_pool in {'max', 'noisyor'}:
        net = SpatialAttributeEncoderNet(convnet, spatial_pool, input_nc, output_nc,
            gpu_ids, init_type)

    if len(gpu_ids) > 0:
        net.cuda(gpu_ids[0])

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



class NoneSpatialAttributeEncoderNet(nn.Module):
    def __init__(self, convnet, input_nc, output_nc, gpu_ids, init_type):
        '''
        Args:
            convnet (str): convnet architecture.
            input_nc (int): number of input channels.
            output_nc (int): number of output channels (number of attribute entries)
        '''
        super(NoneSpatialAttributeEncoderNet, self).__init__()
        self.gpu_ids = gpu_ids

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
        bsz = input_img.size(0)

        if self.gpu_ids:
            feat_map = nn.parallel.data_parallel(self.conv, input_img, self.gpu_ids)
        else:
            feat_map = self.conv(input_img)

        feat = self.avgpool(feat_map).view(bsz, -1)
        prob = F.sigmoid(self.fc(feat))

        return prob, None

class SpatialAttributeEncoderNet(nn.Module):
    def __init__(self, convnet, spatial_pool, input_nc, output_nc, gpu_ids, init_type):
        super(SpatialAttributeEncoderNet, self).__init__()
        self.gpu_ids = gpu_ids
        self.spatial_pool = spatial_pool

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
            init_weights(self.conv, init_type = init_type)



    def forward(self, input_img):
        bsz = input_img.size(0)
        if self.gpu_ids and isinstance(input_img, torch.cuda.FloatTensor):
            feat_map = nn.parallel.data_parallel(self.conv, input_img, self.gpu_ids)
        else:
            feat_map = self.conv(input_img)

        prob_map = F.sigmoid(self.cls(feat_map))
        prob = self.pool(prob_map).view(bsz, -1)

        return prob, prob_map
