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

class WeightedBCELoss(nn.Module):
    '''
    Binary Cross Entropy Loss for multilabel classification task. For each class, the positive and negative samples
    have different loss weight according to the positive rates.

    .. math:: loss(o, t) = -1/n \sum_i (t[i] * log(o[i]) * weight_pos[i] + (1-t[i]) * log(1 - o[i]) * weight_neg[i])
    

    Args:
        pos_rate (Tensor): positive rate of each class. This will be used to compute the pos/neg loss weight for each class
        class_norm (bool): normalize loss in each class if true. otherwise normalize loss over all classes

    Shape:
        - Input: (N, *)
        - Target: (N, *)

    '''

    def __init__(self, pos_rate, class_norm = True, size_average = True):
        super(WeightedBCELoss, self).__init__()
        self.class_norm = class_norm
        self.size_average = size_average
        self.register_buffer('w_pos', Variable(0.5 / pos_rate))
        self.register_buffer('w_neg', Variable(0.5 / (1-pos_rate)))

    def forward(self, input, target):
        assert not target.requires_grad, 'criterions do not compute the gradient w.r.t. targets - please'\
        'mark these variables as volatile or not requiring gradients'
        
        # if not (isinstance(self.w_pos, Variable) and isinstance(self.w_neg, Variable)):
        #     self.w_pos = target.data.new(self.w_pos.size()).copy_(self.w_pos)
        #     self.w_neg = target.data.new(self.w_neg.size()).copy_(self.w_net)

        w_mask = target * self.w_pos + (1-target) * self.w_neg
        input = input.clamp(min = 1e-7, max = 1-1e-7)
        loss = -target * input.log() - (1-target) * (1-input).log()
        loss = loss * w_mask

        if self.class_norm:
            loss = loss / w_mask.mean(dim = 0)
        else:
            loss = loss / w_mask.mean()

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
    
###############################################################################
# Metrics
###############################################################################

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

    def compute_recall(self, k = 3):
        score, label = self.score, self.label

        # for each sample, assigned attributes with top-k socre as its tags
        tag = np.where((-score).argsort().argsort() < k, 1, 0)
        tag_rec = tag * label

        rec_overall = tag_rec.sum() / label.sum() * 100.
        rec_class = (tag_rec.sum(axis=0) / label.sum(axis=0))*100.
        rec_class_avg = rec_class.mean()

        return rec_class_avg, rec_class, rec_overall


    def compute_balanced_precision(self):
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

        BP = (p_pos + p_neg) / 2

        BP = BP * 100.
        mBP = BP.mean()

        return mBP, BP


class ClassificationAccuracy():
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
        assert new_score.shape[0] == new_label.shape[0], 'shape mismatch: %s vs. %s' % (new_score.shape, new_label.shape)
        assert new_label.max() < new_score.shape[1], 'invalid label value %f' % new_label.max()

        new_label = new_label.flatten()
        self.score = np.concatenate((self.score, new_score), axis = 0) if self.score is not None else new_score
        self.label = np.concatenate((self.label, new_label), axis = 0) if self.label is not None else new_label

    def compute_accuracy(self, k = 1):
        score = self.score
        label = self.label

        num_sample = score.shape[0]
        label_one_hot = np.zeros(score.shape)
        label_one_hot[np.arange(num_sample), label.astype(np.int)] = 1
        pred_k_hot = np.where((-score).argsort().argsort() < k, 1, 0)

        num_hit = (pred_k_hot * label_one_hot).sum()
        return num_hit / num_sample * 100.


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
                    gpu_ids = opt.gpu_ids,
                    init_type = opt.init_type)
        else:
            if opt.spatial_pool == 'none':
                net = NoneSpatialAttributeEncoderNet(
                    convnet = opt.convnet,
                    input_nc = opt.input_nc,
                    output_nc = opt.n_attr,
                    gpu_ids = opt.gpu_ids,
                    init_type = opt.init_type)
            elif opt.spatial_pool in {'max', 'noisyor'}:
                net = SpatialAttributeEncoderNet(
                    convnet = opt.convnet,
                    spatial_pool = opt.spatial_pool,
                    input_nc = opt.input_nc, 
                    output_nc = opt.n_attr,
                    gpu_ids = opt.gpu_ids,
                    init_type = opt.init_type)

    if len(opt.gpu_ids) > 0:
        net.cuda(opt.gpu_ids[0])

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

    def extract_feat(self, input_img):
        bsz = input_img.size(0)

        if self.gpu_ids:
            feat_map = nn.parallel.data_parallel(self.conv, input_img, self.gpu_ids)
        else:
            feat_map = self.conv(input_img)

        feat = self.avgpool(feat_map).view(bsz, -1)

        return feat, feat_map


class SpatialAttributeEncoderNet(nn.Module):
    def __init__(self, convnet, spatial_pool, input_nc, output_nc, gpu_ids, init_type):
        super(SpatialAttributeEncoderNet, self).__init__()
        self.gpu_ids = gpu_ids

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
        bsz = input_img.size(0)
        if self.gpu_ids:
            feat_map = nn.parallel.data_parallel(self.conv, input_img, self.gpu_ids)
        else:
            feat_map = self.conv(input_img)

        prob_map = F.sigmoid(self.cls(feat_map))
        prob = self.pool(prob_map).view(bsz, -1)

        return prob, prob_map

    def extract_feat(self, input_img):
        bsz = input_img.size(0)
        if self.gpu_ids:
            feat_map = nn.parallel.data_parallel(self.conv, input_img, self.gpu_ids)
        else:
            feat_map = self.conv(input_img)

        feat = F.avg_pool2d(feat_map, kernel_size = 7, stride = 1).view(bsz, -1)

        return feat, feat_map


class DualSpatialAttributeEncoderNet(nn.Module):
    '''
    Attribute Encoder with 2 branches of ConvNet, for RGB image and Landmark heatmap respectively.
    '''
    def __init__(self, convnet, spatial_pool, input_nc, output_nc, lm_input_nc, lm_output_nc, lm_fusion, gpu_ids, init_type):
        super(DualSpatialAttributeEncoderNet, self).__init__()
        # create RGB channel
        self.gpu_ids = gpu_ids
        self.spatial_pool = spatial_pool
        pretrain = (input_nc == 3)
        self.conv = create_resnet_conv_layers(convnet, input_nc, pretrain)
        self.fusion = lm_fusion

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
            self.fuse_layer = nn.Conv2d(feat_nc, self.conv.output_nc, kernel_size = 1)
            self.cls = nn.Conv2d(self.conv.output_nc, output_nc, kernel_size = 1)
        else:
            raise NotImplementedError()


        # create pooling layers
        if spatial_pool == 'max':
            self.pool = nn.MaxPool2d(7, stride=1)
        elif spatial_pool == 'noisyor':
            self.pool = NoisyOR()

        # initialize weights
        init_weights(self.cls, init_type = init_type)
        init_weights(self.conv_lm, init_type = init_type)
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
            img_feat_map = nn.parallel.data_parallel(self.conv, input_img, self.gpu_ids)
            lm_feat_map = nn.parallel.data_parallel(self.conv_lm, input_lm_heatmap, self.gpu_ids)
        else:
            img_feat_map = self.conv(input_img)
            lm_feat_map = self.conv_lm(input_lm_heatmap)

        feat_map = None
        if self.fusion == 'cancat':
            feat_map = torch.cat((img_feat_map, lm_feat_map), dim = 1)
        elif self.fusion == 'linear':
            feat_map = self.fuse_layer(torch.cat((img_feat_map, lm_feat_map), dim = 1))
            feat_map = F.relu(feat_map)
        
        prob_map = F.sigmoid(self.cls(feat_map))
        prob = self.pool(prob_map).view(bsz, -1)

        return prob, prob_map

    def extract_feat(self, input_img, input_lm_heatmap):
        bsz = input_img.size(0)
        if self.gpu_ids:
            img_feat_map = nn.parallel.data_parallel(self.conv, input_img, self.gpu_ids)
            lm_feat_map = nn.parallel.data_parallel(self.conv_lm, input_lm_heatmap, self.gpu_ids)
        else:
            img_feat_map = self.conv(input_img)
            lm_feat_map = self.conv_lm(input_lm_heatmap)

        feat_map = None
        if self.fusion == 'cancat':
            feat_map = torch.cat((img_feat_map, lm_feat_map), dim = 1)
        elif self.fusion == 'linear':
            feat_map = self.fuse_layer(torch.cat((img_feat_map, lm_feat_map), dim = 1))
            feat_map = F.relu(feat_map)

        feat = F.avg_pool2d(feat_map, kernel_size = 7, stride = 1).view(bsz, -1)

        return feat, feat_map




class JointNoneSpatialAttributeEncoderNet(nn.Module):
    def __init__(self, convnet, input_nc, output_nc, output_nc1, gpu_ids, init_type):
        '''
        Args:
            convnet (str): convnet architecture.
            input_nc (int): number of input channels.
            output_nc (int): number of output channels (number of attribute entries)
            output_nc1 (int): number of auxiliary output chnnels (number of category entries)
        '''

        super(JointNoneSpatialAttributeEncoderNet, self).__init__()
        self.gpu_ids = gpu_ids

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
            feat_map = nn.parallel.data_parallel(self.conv, input_img, self.gpu_ids)
        else:
            feat_map = self.conv(input_img)

        feat = self.avgpool(feat_map).view(bsz, -1)
        prob = F.sigmoid(self.fc(feat))
        pred_cat = self.fc_cat(feat)

        return prob, None, pred_cat

    def extract_feat(self, input_img):
        bsz = input_img.size(0)
        if self.gpu_ids:
            feat_map = nn.parallel.data_parallel(self.conv, input_img, self.gpu_ids)
        else:
            feat_map = self.conv(input_img)

        feat = self.avgpool(feat_map).view(bsz, -1)

        return feat, feat_map






