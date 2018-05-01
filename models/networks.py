from __future__ import division, print_function

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.autograd import Variable
from torch.optim import lr_scheduler

from resnet_wrapper import create_resnet_conv_layers

import os
import numpy as np
import functools
from skimage.measure import compare_ssim, compare_psnr

import util.io as io

###############################################################################
# Functions
###############################################################################

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.normal_(m.weight, 0.0, 0.02)
    elif classname.startswith('Linear'):
        init.normal_(m.weight, 0.0, 0.02)
    elif classname.startswith('BatchNorm2d'):
        init.normal_(m.weight, 1.0, 0.02)

    if 'bias' in m._parameters and m.bias is not None:
        init.constant_(m.bias, 0.0)

def weights_init_normal2(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.normal_(m.weight, 0.0, 0.001)
    elif classname.startswith('Linear'):
        init.normal_(m.weight, 0.0, 0.001)
    elif classname.startswith('BatchNorm2d'):
        init.normal_(m.weight, 1.0, 0.001)

    if 'bias' in m._parameters and m.bias is not None:
        init.constant_(m.bias, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.xavier_normal_(m.weight, gain=0.02)
    elif classname.startswith('Linear'):
        init.xavier_normal_(m.weight, gain=0.02)
    elif classname.startswith('BatchNorm2d'):
        init.normal_(m.weight, 1.0, 0.02)
    
    if 'bias' in m._parameters and m.bias is not None:
        init.constant_(m.bias, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif classname.startswith('Linear'):
        init.kaiming_normal_(m.weight, a=0, mode='fan_in')
    elif classname.startswith('BatchNorm2d'):
        init.normal_(m.weight, 1.0, 0.02)

    if 'bias' in m._parameters and m.bias is not None:
        init.constant_(m.bias, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.startswith('Conv'):
        init.orthogonal_(m.weight, gain=1)
    elif classname.startswith('Linear'):
        init.orthogonal_(m.weight, gain=1)
    elif classname.startswith('BatchNorm2d'):
        init.normal_(m.weight, 1.0, 0.02)
    
    if 'bias' in m._parameters and m.bias is not None:
        init.constant_(m.bias, 0.0)

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
class LossBuffer():
    '''
    '''
    def __init__(self, size = 1000):
        self.clear()
        self.size = size
    
    def clear(self):
        self.buffer = []

    def add(self, loss):

        if isinstance(loss, Variable):
            self.buffer.append(loss.data[0])
        elif isinstance(loss, torch.Tensor):
            self.buffer.append(loss[0])
        else:
            self.buffer.append(loss)
        
        if len(self.buffer) > self.size:
            self.buffer = self.buffer[-self.size::]

    def smooth_loss(self, clear = False):
        if len(self.buffer) == 0:
            loss = 0
        else:
            loss = sum(self.buffer) / len(self.buffer)

        if clear:
            self.clear()
            
        return loss

class SmoothLoss():
    '''
    wrapper of pytorch loss layer.
    '''
    def __init__(self, crit):
        self.crit = crit
        self.max_size = 100000
        self.clear()

    def __call__(self, input_1, input_2, *extra_input):
        loss = self.crit(input_1, input_2, *extra_input)
        self.weight_buffer.append(input_1.size(0))

        if isinstance(loss, Variable):
            self.buffer.append(loss.data.item())
        elif isinstance(loss, torch.Tensor):
            self.buffer.append(loss.data.item())
        else:
            self.buffer.append(loss)

        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size::]
            self.weight_buffer = self.weight_buffer[-self.max_size::]

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


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                # real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                real_tensor = input.data.new(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                # fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                fake_tensor = input.data.new(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Vgg19 and VGGLoss is borrowed from pix2pixHD
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.crit = nn.L1Loss(reduce=False)
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, y):
        bsz = x.size(0)
        input = torch.cat((x,y), dim = 0)
        h_relu1 = self.slice1(input)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = 0
        for i, h in enumerate([h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]):
            out += self.weights[i] * self.crit(h[0:bsz], h[bsz::].detach()).view(bsz,-1).mean(dim=1)
        return out

# Perceptual Feature Loss using VGG19 network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.gpu_ids = gpu_ids
        self.vgg = Vgg19()
        if len(gpu_ids) > 0:
            self.vgg.cuda()

    def forward(self, x, y, loss_type='content'):
        if len(self.gpu_ids)>1:
            return nn.parallel.data_parallel(self.vgg, (x, y)).mean()
        else:
            return self.vgg(x, y).mean()

class VGGLoss_v2(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss_v2, self).__init__()
        self.gpu_ids = gpu_ids
        self.content_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.style_weights = [1,1,1,1,1] # use relu-3 layer feature to compure style loss
        # self.style_weights = [0,0,1,0,0] # use relu-3 layer feature to compure style loss
        # define vgg
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False

        if len(gpu_ids) > 0:
            self.cuda()

    def compute_feature(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

    def forward(self, X, Y, loss_type='content', device_mode=None):
        '''
        loss_type: 'all', 'content', 'style'
        device_mode: multi, single, sub
        '''
        bsz = X.size(0)
        if device_mode is None:
            device_mode = 'multi' if len(self.gpu_ids) > 1 else 'single'

        if device_mode == 'multi':
            if loss_type != 'all':
                return nn.parallel.data_parallel(self, (X, Y), module_kwargs={'loss_type': loss_type, 'device_mode': 'sub'}).mean(dim=0)
            else:
                loss_content, loss_style = nn.parallel.data_parallel(self, (X, Y), module_kwargs={'loss_type': loss_type, 'device_mode': 'sub'})
                return loss_content.mean(dim=0), loss_style.mean(dim=0)
        else:
            features_x = self.compute_feature(X)
            features_y = self.compute_feature(Y)
            # compute content loss
            if loss_type in {'all', 'content'}:
                loss_content = 0
                for i, (feat_x, feat_y) in enumerate(zip(features_x, features_y)):
                    loss_content += self.content_weights[i] * F.l1_loss(feat_x, feat_y, reduce=False).view(bsz, -1).mean(dim=1)
                if device_mode == 'single':
                    loss_content = loss_content.mean(dim=0)
            # compute style loss
            if loss_type in {'all', 'style'}:
                loss_style = 0
                for i, (feat_x, feat_y) in enumerate(zip(features_x, features_y)):
                    if self.style_weights[i] > 0:
                        loss_style += self.style_weights[i] * F.mse_loss(self.gram_matrix(feat_x), self.gram_matrix(feat_y), reduce=False).view(bsz, -1).mean(dim=1)
                        # loss_style += self.style_weights[i] * ((self.gram_matrix(feat_x) - self.gram_matrix(feat_y))**2).view(bsz, -1).mean(dim=1)
                if device_mode == 'single':
                    loss_style = loss_style.mean(dim=0)

            if loss_type == 'content':
                return loss_content
            elif loss_type == 'style':
                return loss_style
            elif loss_type == 'all':
                return loss_content, loss_style

    def gram_matrix(self, feat):
        bsz, c, h, w = feat.size()
        feat = feat.view(bsz, c, h*w)
        feat_T = feat.transpose(1,2)
        g = torch.matmul(feat, feat_T) / (c*h*w)
        return g


class TotalVariationLoss(nn.Module):
    def forward(self, x):
        x_grad = x[:,:,:,0:-1] - x[:,:,:,1::]
        y_grad = x[:,:,0:-1,:] - x[:,:,1::,:]
        loss = (x_grad.norm(p=2).pow(2) + y_grad.norm(p=2).pow(2)).pow(0.5) / x.size(0)
        return loss

###############################################################################
# Metrics
###############################################################################
class PSNR_old(nn.Module):
    def __init__(self):
        super(PSNR_old, self).__init__()
        self.Kr = .299
        self.Kg = .587
        self.Kb = .114
        self.lg10 = float(np.log(10))

    def forward(self, images_1, images_2):
        y_1 = self.Kr * images_1[:,0] + self.Kg * images_1[:,1] + self.Kb * images_1[:,2]
        y_2 = self.Kr * images_2[:,0] + self.Kg * images_2[:,1] + self.Kb * images_2[:,2]
        y_d = (y_1 - y_2).view(y_1.size(0), -1)
        rmse = y_d.pow(2).mean(dim = 1).sqrt().clamp(0, 1)
        psnr = 20 / self.lg10 * (1/rmse).log().mean()
        return psnr

class PSNR(nn.Module):
    def forward(self, images_1, images_2):
        numpy_imgs_1 = images_1.data.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_1 = ((numpy_imgs_1 + 1.0) * 127.5).clip(0,255).astype(np.uint8)
        numpy_imgs_2 = images_2.data.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_2 = ((numpy_imgs_2 + 1.0) * 127.5).clip(0,255).astype(np.uint8)

        psnr_score = []
        for img_1, img_2 in zip(numpy_imgs_1, numpy_imgs_2):
            psnr_score.append(compare_psnr(img_2, img_1))

        return Variable(images_1.data.new(1).fill_(np.mean(psnr_score)))


class SSIM(nn.Module):
    def forward(self, images_1, images_2):
        numpy_imgs_1 = images_1.data.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_1 = ((numpy_imgs_1 + 1.0) * 127.5).clip(0,255).astype(np.uint8)
        numpy_imgs_2 = images_2.data.cpu().numpy().transpose(0,2,3,1)
        numpy_imgs_2 = ((numpy_imgs_2 + 1.0) * 127.5).clip(0,255).astype(np.uint8)

        ssim_score = []
        for img_1, img_2 in zip(numpy_imgs_1, numpy_imgs_2):
            ssim_score.append(compare_ssim(img_1, img_2, multichannel=True))

        return Variable(images_1.data.new(1).fill_(np.mean(ssim_score)))

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

            elif isinstance(inputs[i], torch.Tensor):
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

    def compute_recall_sample_avg(self, k = 3):
        '''
        compute recall using method in DeepFashion Paper
        '''
        score, label = self.score, self.label
        tag = np.where((-score).argsort().argsort() < k, 1, 0)
        tag_rec = tag * label

        count_rec = tag_rec.sum(axis = 1)
        count_gt = label.sum(axis = 1)

        # set recall=1 for sample with no positive attribute label
        no_pos_attr = (count_gt == 0).astype(count_gt.dtype)
        count_rec += no_pos_attr
        count_gt += no_pos_attr

        rec = (count_rec / count_gt).mean() * 100.

        return rec

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

            elif isinstance(inputs[i], torch.Tensor):
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
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
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
# GAN
###############################################################################
def define_G(opt):
    netG = None
    use_gpu = len(opt.gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=opt.norm)
    activation  = nn.ReLU
    use_dropout = not opt.no_dropout
    if opt.attr_cond_type in {'feat', 'feat_map'}:
        attr_nc = opt.n_attr_feat
    elif opt.attr_cond_type in {'prob', 'prob_map'}:
        attr_nc = opt.n_attr

    # Todo: add choice of activation function
    if use_gpu:
        assert(torch.cuda.is_available())

    if opt.G_cond_nc > 0:
        if opt.which_model_netG == 'resnet_9blocks':
            netG = ConditionedResnetGenerator(input_nc = opt.G_input_nc, output_nc = opt.G_output_nc, cond_nc = opt.G_cond_nc,
                cond_layer = opt.G_cond_layer, cond_interp = opt.G_cond_interp, ngf = opt.ngf, norm_layer = norm_layer, activation = activation,
                use_dropout = use_dropout, n_blocks = 9, gpu_ids = opt.gpu_ids)
        elif opt.which_model_netG == 'resnet_6blocks':
            netG = ConditionedResnetGenerator(input_nc = opt.G_input_nc, output_nc = opt.G_output_nc, cond_nc = opt.G_cond_nc,
                cond_layer = opt.G_cond_layer, cond_interp = opt.G_cond_interp,  ngf = opt.ngf, norm_layer = norm_layer, activation = activation,
                use_dropout = use_dropout, n_blocks = 6, gpu_ids = opt.gpu_ids)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % opt.which_model_netG)    
    else:
        if opt.which_model_netG == 'resnet_9blocks':
            netG = ResnetGenerator(input_nc = opt.G_input_nc, output_nc = opt.G_output_nc,
                ngf = opt.ngf, norm_layer = norm_layer,
                use_dropout = use_dropout, n_blocks = 9, gpu_ids = opt.gpu_ids)
        elif opt.which_model_netG == 'resnet_6blocks':
            netG = ResnetGenerator(input_nc = opt.G_input_nc, output_nc = opt.G_output_nc,
                ngf = opt.ngf, norm_layer = norm_layer,
                use_dropout = use_dropout, n_blocks = 6, gpu_ids = opt.gpu_ids)
        else:
            raise NotImplementedError('Generator model name [%s] is not recognized' % opt.which_model_netG)    

    # if which_model_netG == 'resnet_9blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    # elif which_model_netG == 'resnet_6blocks':
    #     netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_128':
    #     netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    # elif which_model_netG == 'unet_256':
    #     netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    # else:
    #     raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)

    if len(opt.gpu_ids) > 0:
        netG.cuda()
    init_weights(netG, init_type=opt.init_type)
    return netG


def define_D_from_params(input_nc, ndf, which_model_netD, n_layers_D, norm, which_gan, init_type, gpu_ids):
    netD = None
    use_gpu = len(gpu_ids) > 0
    use_sigmoid = (which_gan == 'dcgan')
    output_bias = (which_gan != 'wgan')
    norm_layer = get_norm_layer(norm)
    # print(which_gan)
    # print(use_sigmoid)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc = input_nc, ndf = ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, output_bias = output_bias, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc = input_nc, ndf = ndf, n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, output_bias = output_bias, gpu_ids=gpu_ids)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc = input_nc, ndf = ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)

    if use_gpu:
        netD.cuda()
    init_weights(netD, init_type=init_type)
    return netD

def define_D(opt):
    netD = define_D_from_params(input_nc=opt.D_input_nc, ndf=opt.ndf, which_model_netD=opt.which_model_netD, n_layers_D=opt.n_layers_D, norm=opt.norm,
        which_gan=opt.which_gan, init_type=opt.init_type, gpu_ids=opt.gpu_ids)
    return netD

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def get_norm_layer(norm_type = 'instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine =False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_bias, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.dim = dim
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

    def print(self):
        print('ResnetBlock: x_dim=%d'%self.dim)

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, activation = nn.ReLU, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 activation()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      activation()]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation = activation(), norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      activation()]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input)
        else:
            return self.model(input)

class ConditionedResnetBlock(nn.Module):
    def __init__(self, x_dim, c_dim, padding_type, norm_layer, use_bias, activation=nn.ReLU(True), use_dropout=False, output_c=False):
        '''
        Args:
            x_dim(int): input feature channel
            c_dim(int): condition feature channel
            output_c(bool): whether concat condition feature to the outout
        Input:
            x(Variable): size of (bsz, x_dim+c_dim, h, w)
        Output:
            y(Variable): size of (bsz, x_dim+c_dim, h, w) if output_c is true, else (bsz, x_dim, h, w)
        '''
        super(ConditionedResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(x_dim, c_dim, padding_type, norm_layer, activation, use_dropout, use_bias)
        self.output_c = output_c
        self.x_dim = x_dim
        self.c_dim = c_dim

    def build_conv_block(self, x_dim, c_dim, padding_type, norm_layer, activation, use_dropout, use_bias):
        conv_block = []

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(x_dim + c_dim, x_dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(x_dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(x_dim, x_dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(x_dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x_and_c):
        # out = x + self.conv_block(x)
        x = x_and_c[:,0:self.x_dim]
        c = x_and_c[:,self.x_dim::]
        x_out = x + self.conv_block(x_and_c)
        if self.output_c:
            return torch.cat((x_out, c), dim = 1)
        else:
            return x_out

    def print(self):
        out_dim = self.x_dim + self.c_dim if self.output_c else self.x_dim
        print('ConditionedResnetBlock: x_dim=%d, c_dim=%d, out_dim=%d'% (self.x_dim, self.c_dim, out_dim))

class ConditionedResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, cond_nc, cond_layer = 'first', ngf=64, norm_layer=nn.BatchNorm2d, activation = nn.ReLU, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect', cond_interp='bilinear'):
        assert(n_blocks >= 0)
        super(ConditionedResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.cond_nc = cond_nc
        self.cond_layer = cond_layer
        self.cond_interp = cond_interp
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        downsample_layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            activation()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            downsample_layers += [
                nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size = 3, stride = 2, padding = 1, bias = use_bias),
                norm_layer(ngf*mult*2),
                activation()
            ]

        res_blocks = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            if (cond_layer == 'first' and i == 0) or cond_layer == 'all':
                output_c = (cond_layer == 'all' and i < n_blocks - 1)
                res_blocks.append(ConditionedResnetBlock(
                    x_dim = ngf*mult,
                    c_dim = cond_nc, 
                    padding_type=padding_type,
                    activation = activation(),
                    norm_layer = norm_layer,
                    use_dropout = use_dropout,
                    output_c = output_c,
                    use_bias = use_bias
                    ))
            else:
                res_blocks.append(ResnetBlock(
                    dim = ngf*mult,
                    padding_type=padding_type,
                    activation = activation(),
                    norm_layer = norm_layer,
                    use_dropout = use_dropout,
                    use_bias = use_bias
                    ))

        upsample_layers = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            upsample_layers += [
                nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size = 3, stride = 2, padding = 1, output_padding = 1, bias = use_bias),
                norm_layer(int(ngf * mult / 2)),
                activation()
            ]
        upsample_layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size = 7, padding = 0),
            nn.Tanh()
        ]


        self.down_sample = nn.Sequential(*downsample_layers)
        self.res_blocks = nn.Sequential(*res_blocks)
        self.up_sample = nn.Sequential(*upsample_layers)

    
    def forward(self, input_x, input_c, single_device = False):
        '''
        Input:
            input_x: size of (bsz, input_nc, h, w)
            input_c: size of (bsz, cond_nc) or (bsz, cond_nc, h_r, w_r)
        '''
        if self.gpu_ids and len(self.gpu_ids) > 1 and isinstance(input_x.data, torch.cuda.FloatTensor) and (not single_device):
            return nn.parallel.data_parallel(self, (input_x, input_c), module_kwargs = {'single_device': True})
        else:
            x = self.down_sample(input_x)
            bsz, _, h_x, w_x = x.size()

            if input_c.dim() == 2:
                c = input_c.view(bsz, self.cond_nc, 1, 1).expand(bsz, self.cond_nc, h_x, w_x)
            elif input_c.dim() == 4:
                if (input_c.size(2) == h_x and input_c.size(3) == w_x):
                    c = input_c
                else:
                    c = F.upsample(input_c, size = (h_x, w_x), mode = self.cond_interp)

            x = self.res_blocks(torch.cat((x, c), dim = 1))
            x = self.up_sample(x)

            return x


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input)
        else:
            return self.model(input)

class UnetGenerator_v2(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator_v2, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 4):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 1, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)

        # self.model = unet_block
        model = [
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(ngf),
            unet_block,
            nn.ReLU(True),
            nn.Conv2d(2*ngf, output_nc, kernel_size=7, padding=3),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input)
        else:
            return self.model(input)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, output_bias = True, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, bias = output_bias)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input)
        else:
            return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(PixelDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
            
        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.net, input)
        else:
            return self.net(input)

###############################################################################
# GAN V2
###############################################################################
def define_feature_fusion_network(name ='FeatureConcatNetwork', feat_nc=128, guide_nc=128, output_nc=-1, ndowns=3, nblocks=3, feat_size=8,norm='batch', init_type='normal', gpu_ids=[]):
    if name == 'FeatureConcatNetwork':
        model = FeatureConcatNetwork(feat_nc, guide_nc, output_nc, nblocks, norm, gpu_ids)
    elif name == 'FeatureReduceNetwork':
        model = FeatureReduceNetwork(feat_nc, guide_nc, output_nc, ndowns, nblocks, norm, gpu_ids)
    elif name == 'FeatureTransformNetwork':
        model = FeatureTransformNetwork(feat_nc, guide_nc, output_nc, feat_size, nblocks, norm, gpu_ids)

    if len(gpu_ids) > 0:
        model.cuda()
    init_weights(model, init_type)
    return model

class FeatureConcatNetwork(nn.Module):
    def __init__(self, feat_nc, guide_nc, output_nc=-1, nblocks=3, norm='batch', gpu_ids=[]):
        super(FeatureConcatNetwork, self).__init__()
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(norm)
        use_bias = (norm_layer.func == nn.InstanceNorm2d)
        activation = nn.ReLU

        if output_nc == -1:
            output_nc = feat_nc

        blocks = []
        for n in range(nblocks):
            c_in = feat_nc + guide_nc if n == 0 else output_nc
            c_out = output_nc
            blocks += [ResidualEncoderBlock(c_in, c_out, norm_layer, activation, use_bias, stride=1)]
        self.model = nn.Sequential(*blocks)

    def forward(self, feat, output_guide, input_guide=None):
        # input_guide is just for unified parameter format
        feat_size = output_guide.size()[2:4] # the size of guide signal is the target feature map size
        if not(feat.size()[2:4] == feat_size):
            feat = F.upsample(feat, feat_size, mode='bilinear')
        feat = torch.cat((feat, output_guide), dim=1)
        if len(self.gpu_ids)>1:
            return nn.parallel.data_parallel(self.model, feat)
        else:
            return self.model(feat)

class FeatureReduceNetwork(nn.Module):
    def __init__(self, feat_nc, guide_nc, output_nc=-1, ndowns=3, n_blocks=3, norm='batch', gpu_ids=[]):
        super(FeatureReduceNetwork, self).__init__()
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(norm)
        use_bias = (norm_layer.func == nn.InstanceNorm2d)
        activation = nn.ReLU

        if output_nc==-1:
            output_nc = feat_nc

        reduce_layer = []
        for n in range(ndowns):
            c_in = feat_nc + guide_nc if n == 0 else feat_nc*2**n
            c_out = feat_nc*2**(n+1)
            reduce_layer += [nn.Conv2d(c_in, c_out, 4, 2, 1, bias=use_bias)]
            if n < ndowns-1:
                reduce_layer += [norm_layer(c_out)]
            reduce_layer += [activation()]
        self.reduce = nn.Sequential(*reduce_layer)

        recover_layer = []
        for n in range(n_blocks):
            c_in = c_out + guide_nc if n == 0 else output_nc
            c_out = output_nc
            recover_layer += [ResidualEncoderBlock(c_in, c_out, norm_layer, activation, use_bias, stride=1)]
        self.recover = nn.Sequential(*recover_layer)
    
    def forward(self, feat, input_guide, output_guide, single_device=False):
        feat_size = feat.size()[2:4]
        if not (input_guide.size()[2:4] == feat_size):
            input_guide = F.upsample(input_guide, feat_size, mode='bilinear')
        if not (output_guide.size()[2:4] == feat_size):
            output_guide = F.upsample(output_guide, feat_size, mode='bilinear')

        if len(self.gpu_ids) > 1 and not single_device:
            return nn.parallel.data_parallel(self, (feat, input_guide, output_guide), module_kwargs={'single_device': True})
        else:
            feat_reduce = self.reduce(torch.cat((feat, input_guide), dim=1))
            feat_reduce_tile = F.upsample(feat_reduce, feat_size, mode='bilinear')
            feat_recover = self.recover(torch.cat((feat_reduce_tile, output_guide), dim=1))
            return feat_recover

class FeatureTransformNetwork(nn.Module):
    def __init__(self, feat_nc, guide_nc, output_nc=-1, feat_size = 8, nblocks=1, norm='batch', gpu_ids=[]):
        super(FeatureTransformNetwork, self).__init__()
        self.gpu_ids = gpu_ids
        self.feat_size = feat_size
        norm_layer = get_norm_layer(norm)
        activation = nn.ReLU
        use_bias = (norm_layer.func == nn.InstanceNorm2d)
        if output_nc == -1:
            output_nc = feat_nc
        self.stn = SpatialTransformerNetwork_V2(feat_nc, guide_nc * 2, feat_size)
        if nblocks > 0:
            blocks = []
            for n in range(nblocks):
                c_in = feat_nc + guide_nc if n == 0 else feat_nc
                c_out = feat_nc
                blocks += [ResidualEncoderBlock(c_in, c_out, norm_layer, activation, use_bias, stride=1)]
            self.res_blocks = nn.Sequential(*blocks)
        else:
            self.res_blocks = None

    def forward(self, feat, input_guide, output_guide, single_device=False):
        if not (input_guide.size(2) == input_guide.size(3) == self.feat_size):
            # input_guide = F.upsample(input_guide, self.feat_size, mode='bilinear')
            input_guide = F.adaptive_avg_pool2d(input_guide, self.feat_size)
        if not (output_guide.size(2) == output_guide.size(3) == self.feat_size):
            output_guide = F.adaptive_avg_pool2d(output_guide, self.feat_size)

        if len(self.gpu_ids) > 1 and not single_device:
            return nn.parallel.data_parallel(self, (feat, input_guide, output_guide), module_kwargs={'single_device': True})
        else:
            feat_trans = self.stn(feat, torch.cat((input_guide, output_guide),dim=1))
            if self.res_blocks is not None:
                feat_trans = self.res_blocks(torch.cat((feat_trans, output_guide),dim=1))
            return feat_trans


class SpatialTransformerNetwork_V2(nn.Module):
    def __init__(self, input_nc, guide_nc, size=8):
        super(SpatialTransformerNetwork_V2, self).__init__()
        self.input_nc = input_nc
        self.size = size

        self.loc_conv = nn.Sequential(
            nn.Conv2d(guide_nc, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True)
            )
        self.loc_fc = nn.Sequential(
            nn.Linear(32*size*size//16, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
            )

        self.loc_fc[2].weight.data.fill_(0)
        self.loc_fc[2].bias.data = torch.FloatTensor([1,0,0,0,1,0])

    def forward(self, x, guide):
        out = self.loc_conv(guide)
        out = out.view(guide.size(0),-1)
        theta = self.loc_fc(out)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid)

def define_generator(opt):
    if opt.which_model_netG == 'decoder':
        input_nc_1 = opt.shape_nof + (opt.edge_nof if opt.use_edge else 0) + (opt.color_nof if opt.use_color else 0)
        input_nc_2 = opt.shape_nc if opt.G_shape_guided else 0
        output_nc = opt.G_output_nc
        nblocks_1 = opt.G_nblocks_1
        nups_1 = opt.G_nups_1
        nblocks_2 = opt.G_nblocks_2
        nups_2 = opt.G_nups_2
        norm = opt.norm
        use_dropout = not opt.no_dropout
        gpu_ids = opt.gpu_ids
        model = DecoderGenerator(input_nc_1, input_nc_2, output_nc, nblocks_1, nups_1, nblocks_2, nups_2, norm, use_dropout, gpu_ids)
    elif opt.which_model_netG == 'unet':
        input_nc_1 = opt.shape_nc
        input_nc_2 = (opt.edge_nof if opt.use_edge else 0) + (opt.color_nof if opt.use_color else 0)
        output_nc = opt.G_output_nc
        nf = opt.shape_nf
        nof = opt.shape_nof
        ndowns = opt.G_ndowns
        nblocks = opt.G_nblocks
        norm = opt.norm
        use_dropout = not opt.no_dropout
        block_type = opt.G_block
        gpu_ids = opt.gpu_ids
        model = UnetResidualGenerator(input_nc_1, input_nc_2, output_nc, nf, nof, ndowns, nblocks, block_type, norm, use_dropout, gpu_ids)

    if len(opt.gpu_ids) > 0:
        model.cuda()
    init_weights(model, opt.init_type)
    return model

class UnetResidualGenerator(nn.Module):
    def __init__(self, input_nc_1, input_nc_2=0, output_nc=3, nf=32, nof=128, ndowns=5, nblocks=5, block_type='normal', norm='batch', use_dropout=False, gpu_ids=[]):
        super(UnetResidualGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.input_nc_1 = input_nc_1
        self.input_nc_2 = input_nc_2
        self.ndowns = ndowns
        self.down_blocks = None
        self.up_blocks = None
        self.bottleneck = None
        norm_layer = get_norm_layer(norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # define downsample blocks
        down_blocks = []
        for n in range(ndowns):
            c_in = input_nc_1 if n==0 else nf * min(2**(n-1), 8)
            c_out = nf * min(2**n, 8) if n < ndowns-1 else nof
            block = self.define_down_block(input_nc=c_in, output_nc=c_out, norm_layer=norm_layer, use_bias=use_bias, block_type=block_type)
            down_blocks.append(block)
        self.down_blocks = nn.ModuleList(down_blocks)

        bottleneck_nc = nof+input_nc_2
        # define bottleneck
        if nblocks > 0:
            self.bottleneck = self.define_bottleneck(nc=bottleneck_nc, nblocks=nblocks, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)
        # define upsample blocks:
        up_blocks = []
        for n in range(ndowns):
            c_in = bottleneck_nc if n==0 else nf * min(2**(ndowns-n), 16)
            c_out = nf * min(2**(ndowns-n-2), 8) if n < ndowns-1 else output_nc
            block = self.define_up_block(input_nc=c_in, output_nc=c_out, norm_layer=norm_layer, use_bias=use_bias, block_type=block_type, outermost=(n==ndowns-1), use_dropout=use_dropout)
            up_blocks.append(block)
        self.up_blocks = nn.ModuleList(up_blocks)

        # for n, block in enumerate(self.down_blocks):
        #     print('down_%d: %d => %d' % (n, block[0].in_channels, block[0].out_channels))
        # for n, block in enumerate(self.up_blocks):
        #     print('up_%d: %d => %d' % (n, block[1].in_channels, block[1].out_channels))


    def forward(self, input_1, input_2=None, mode ='full', single_device=False):
        assert mode in {'full', 'encode'}
        if len(self.gpu_ids) > 1 and (not single_device):
            if input_2 is not None:
                return nn.parallel.data_parallel(self, (input_1, input_2), module_kwargs={'mode':mode, 'single_device':True})
            else:
                assert self.input_nc_2 == 0 or mode == 'encode'
                return nn.parallel.data_parallel(self, input_1, module_kwargs={'input_2':None, 'mode': mode, 'single_device':True})
        else:
            if mode == 'full':
                mid_output = []
                x = input_1
                for block in self.down_blocks:
                    x = block(x)
                    mid_output.append(x)
                encode_rst = x

                if input_2 is not None:
                    x = torch.cat((x, input_2), dim=1)
                if self.bottleneck is not None:
                    x = self.bottleneck(x)

                for n, block in enumerate(self.up_blocks):
                    if n == 0:
                        x = block(x)
                    else:
                        x = block(torch.cat((x, mid_output[-(n+1)]), dim=1))
                return x, encode_rst

            elif mode == 'encode':
                x = input_1
                for block in self.down_blocks:
                    x = block(x)
                return x


    def define_down_block(self, input_nc, output_nc, norm_layer, use_bias, block_type):
        downrelu = nn.LeakyReLU(0.2, True)
        block = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(output_nc),
            downrelu,
            )
        return block
    
    def define_up_block(self, input_nc, output_nc, norm_layer, use_bias, block_type, outermost=False, use_dropout=False):
        uprelu = nn.ReLU(True)
        block = [
            uprelu,
            nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias),
        ]
        if not outermost:
            block += [norm_layer(output_nc)]
            if block_type == 'residual':
                block +=[
                    ResnetBlock(dim=output_nc, padding_type='reflect', norm_layer=norm_layer, use_bias=use_bias, activation=uprelu, use_dropout=use_dropout)
                ]

        block = nn.Sequential(*block)
        return block


    def define_bottleneck(self, nc, nblocks, norm_layer, use_dropout, use_bias):
        bottleneck = []
        for n in range(nblocks):
            bottleneck += [ResnetBlock(dim=nc, padding_type='reflect', norm_layer=norm_layer, use_bias=use_bias, activation=nn.ReLU(True), use_dropout=use_dropout)]
        return nn.Sequential(*bottleneck)

class DecoderGenerator(nn.Module):
    def __init__(self, input_nc_1, input_nc_2=0, output_nc=3, nblocks_1=1, nups_1=3, nblocks_2=5, nups_2=2, norm='batch', use_dropout=False, gpu_ids=[]):
        super(DecoderGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        norm_layer = get_norm_layer(norm)
        use_bias = (norm_layer.func == nn.InstanceNorm2d)
        activation = nn.ReLU
        padding_type = 'reflect'
        # upsample_1 network
        upsample_1_layers = []
        c_in = input_nc_1
        for n in range(nblocks_1):
            upsample_1_layers += [ResnetBlock(c_in, padding_type, norm_layer, use_bias, activation(True), use_dropout)]
            
        for n in range(nups_1):
            c_in = input_nc_1 if n==0 else max(256, input_nc_1//2**n)
            c_out = max(256, input_nc_1//2**(n+1))
            upsample_1_layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(c_out),
                activation()]
        self.upsample_1 = nn.Sequential(*upsample_1_layers)
        # upsample_2 network
        upsample_2_layers = []
        if nblocks_2 > 0:
            c_in = c_out
            c_out = c_out//2
            for n in range(nblocks_2):
                if n == 0 and input_nc_2>0:
                    upsample_2_layers += [ConditionedResnetBlock(c_in, input_nc_2, padding_type, norm_layer, use_bias, activation(True), use_dropout, output_c=False)]
                else:
                    upsample_2_layers += [ResnetBlock(c_in, padding_type, norm_layer, use_bias, activation(True), use_dropout)]
        else:
            c_in = c_out + input_nc_2
            c_out = c_out//2

        for n in range(nups_2):
            upsample_2_layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(c_out),
                activation()]
            c_in = c_out
            c_out = c_in//2
        
        upsample_2_layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c_in, output_nc, kernel_size=7, padding=0),
            # nn.Tanh()
            ]
        self.upsample_2 = nn.Sequential(*upsample_2_layers)
    
    def forward(self, input_1, input_2=None, single_device=False):
        if len(self.gpu_ids)>1 and not single_device:
            if input_2 is not None:
                return nn.parallel.data_parallel(self, (input_1, input_2), module_kwargs={'single_device': True})
            else:
                return nn.parallel.data_parallel(self, input_1, module_kwargs={'input_2': None, 'single_device': True})
        else:
            output_1 = self.upsample_1(input_1)
            if input_2 is not None:
                output_1 = torch.cat((output_1, input_2), dim=1)
            return self.upsample_2(output_1)

###############################################################################
# GAN V3
###############################################################################
class VUnetResidualBlock(nn.Module):
    def __init__(self, dim_1, dim_2, norm_layer, use_bias, activation=nn.ReLU(False), use_dropout=False):
        super(VUnetResidualBlock, self).__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2
        self.use_dropout = use_dropout
        if norm_layer is None:
            use_bias = True
        if dim_2 <= 0:
            self.conv = nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1, bias=use_bias)
            self.norm_layer = norm_layer(dim_1) if norm_layer is not None else None
        else:
            self.conv = nn.Conv2d(2 * dim_1, dim_1, kernel_size=3, padding=1, bias=use_bias)
            self.norm_layer = norm_layer(dim_1) if norm_layer is not None else None
            self.conv_2 = nn.Conv2d(dim_2, dim_1, kernel_size=1, padding=0, bias=use_bias)
            self.norm_layer_2 = norm_layer(2 * dim_1) if norm_layer is not None else None
        self.activation = activation

    def forward(self, x, a=None):
        if a is None:
            residual = x
        else:
            assert self.dim_2 > 0
            a = self.conv_2(self.activation(a))
            residual = torch.cat((x, a), dim=1)
            if self.norm_layer_2 is not None:
                residual = self.norm_layer_2(residual)
        
        residual = self.activation(residual)
        if self.use_dropout:
            residual = F.dropout(residual, p=0.5, training=True)

        residual = self.conv(residual)
        if self.norm_layer is not None:
            residual = self.norm_layer(residual)

        out = x + residual
        return out


class VariationalUnet(nn.Module):
    def __init__(self, input_nc_dec, input_nc_enc, output_nc, nf, max_nf, input_size, n_latent_scales, bottleneck_factor, box_factor, n_residual_blocks, norm_layer, activation, use_dropout, gpu_ids):
        super(VariationalUnet, self).__init__()
        self.gpu_ids = gpu_ids
        self.output_nc = output_nc
        self.input_nc_dec = input_nc_dec
        self.input_size_dec = input_size
        self.n_scales_dec = 1 + int(np.round(np.log2(input_size))) - bottleneck_factor

        self.input_nc_enc = input_nc_enc
        self.input_size_enc = input_size // 2**box_factor
        self.n_scales_enc = self.n_scales_dec - box_factor

        self.n_latent_scales = n_latent_scales
        self.bottleneck_factor = bottleneck_factor
        self.box_factor = box_factor
        self.n_residual_blocks = n_residual_blocks

        if type(norm_layer) == functools.partial:
            self.use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            self.use_bias = norm_layer == nn.InstanceNorm2d
        # define enc_up network
        c_in = min(nf*2**box_factor, max_nf)
        hidden_c = [] # hidden space dims
        self.enc_up_pre_conv = nn.Sequential(
            nn.Conv2d(input_nc_enc, c_in, kernel_size=1),
            norm_layer(c_in))

        for l in range(self.n_scales_enc):
            spatial_shape = self.input_size_enc / 2**l
            nl = None if spatial_shape == 1 else norm_layer
            for i in range(n_residual_blocks):
                self.__setattr__('enc_up_%d_res_%d' % (l, i), VUnetResidualBlock(c_in, 0, nl, self.use_bias, activation, use_dropout))
                hidden_c.append(c_in)
            if l + 1 < self.n_scales_enc:
                c_out = min(2*c_in, max_nf)
                if spatial_shape <= 2:
                    downsample = nn.Sequential(
                        activation,
                        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=True)
                        )
                else:
                    downsample = nn.Sequential(
                        activation,
                        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=self.use_bias),
                        norm_layer(c_out)
                        )
                self.__setattr__('enc_up_%d_downsample'%l, downsample)
                c_in = c_out
        # define enc_down network
        self.enc_down_pre_conv = nn.Conv2d(c_in, c_in, kernel_size=1)
        for l in range(n_latent_scales):
            spatial_shape = self.input_size_enc / 2**(self.n_scales_enc - l - 1)
            nl = None if spatial_shape == 1 else norm_layer
            for i in range(n_residual_blocks//2):
                c_a = hidden_c.pop()
                self.__setattr__('enc_down_%d_res_%d' % (l, i), VUnetResidualBlock(c_in, c_a, nl, self.use_bias, activation, use_dropout))

            self.__setattr__('enc_down_%d_latent'%l, nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, bias=True))

            for i in range(n_residual_blocks//2, n_residual_blocks):
                c_a = c_in + hidden_c.pop()
                self.__setattr__('enc_down_%d_res_%d' % (l, i), VUnetResidualBlock(c_in, c_a, nl, self.use_bias, activation, use_dropout))

            if l + 1 < n_latent_scales:
                c_out = hidden_c[-1]
                upsample = nn.Sequential(
                    activation,
                    nn.Conv2d(c_in, c_out*4, kernel_size=3, padding=1, bias=self.use_bias),
                    nn.PixelShuffle(2),
                    norm_layer(c_out)
                    )
                self.__setattr__('enc_down_%d_upsample'%l, upsample)
                c_in = c_out
        # define dec_up network
        c_in = nf
        hidden_c = []
        self.dec_up_pre_conv = nn.Sequential(
            nn.Conv2d(input_nc_dec, c_in, kernel_size=1),
            norm_layer(c_in))
        for l in range(self.n_scales_dec):
            spatial_shape = self.input_size_dec / 2**l
            nl = None if spatial_shape==1 else norm_layer
            for i in range(n_residual_blocks):
                self.__setattr__('dec_up_%d_res_%d'%(l, i), VUnetResidualBlock(c_in, 0, nl, self.use_bias, activation, use_dropout))
                hidden_c.append(c_in)
            if l + 1 < self.n_scales_dec:
                c_out = min(2*c_in, max_nf)
                if spatial_shape <= 2:
                    downsample = nn.Sequential(
                        activation,
                        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=True)
                        )
                else:
                    downsample = nn.Sequential(
                        activation,
                        nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=self.use_bias),
                        norm_layer(c_out)
                        )
                self.__setattr__('dec_up_%d_downsample'%l, downsample)
                c_in = c_out
        # define dec_down network
        self.dec_down_pre_conv = nn.Conv2d(c_in, c_in, kernel_size=1)
        for l in range(self.n_scales_dec):
            spatial_shape = self.input_size_dec / 2**(self.n_scales_dec - l - 1)
            nl = None if spatial_shape==1 else norm_layer
            for i in range(n_residual_blocks//2):
                c_a = hidden_c.pop()
                self.__setattr__('dec_down_%d_res_%d' % (l, i), VUnetResidualBlock(c_in, c_a, nl, self.use_bias, activation, use_dropout))
            if l < n_latent_scales:
                if spatial_shape == 1:
                    # no spatial correlation
                    self.__setattr__('dec_down_%d_latent'%l, nn.Conv2d(c_in, c_in, kernel_size=3, padding=1, bias=True))
                else:
                    # four autoregressively modeled groups
                    for j in range(4):
                        self.__setattr__('dec_down_%d_latent_%d'%(l,j), nn.Conv2d(c_in*4, c_in, kernel_size=3, padding=1, bias=True))
                        if j + 1 < 4:
                            self.__setattr__('dec_down_%d_ar_%d'%(l,j), VUnetResidualBlock(c_in*4, c_in, None, self.use_bias, activation, use_dropout))
                for i in range(n_residual_blocks//2, n_residual_blocks):
                    if spatial_shape == 1:
                        nin = nn.Conv2d(c_in*2, c_in, kernel_size=1, bias=True)
                    else:
                        nin = nn.Sequential(nn.Conv2d(c_in*2, c_in, kernel_size=1, bias=self.use_bias), norm_layer(c_in))
                    self.__setattr__('dec_down_%d_nin_%d'%(l,i), nin)
                    c_a = hidden_c.pop()
                    self.__setattr__('dec_down_%d_res_%d'%(l,i), VUnetResidualBlock(c_in, c_a, nl, self.use_bias, activation, use_dropout))
            else:
                for i in range(n_residual_blocks//2, n_residual_blocks):
                    c_a = hidden_c.pop()
                    self.__setattr__('dec_down_%d_res_%d'%(l,i), VUnetResidualBlock(c_in, c_a, nl, self.use_bias, activation, use_dropout))

            if l+1 < self.n_scales_dec:
                c_out = hidden_c[-1]
                upsample = nn.Sequential(
                    activation,
                    nn.Conv2d(c_in, c_out*4, kernel_size=3, padding=1, bias=self.use_bias),
                    nn.PixelShuffle(2),
                    norm_layer(c_out)
                    )
                self.__setattr__('dec_down_%d_upsample'%l, upsample)
                c_in = c_out
        # define final decode layer
        self.dec_output = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(c_in, output_nc, kernel_size=7, padding=0, bias=True),
            nn.Tanh()
            )

    def enc_up(self, x, c):
        '''
        Input:
            x: appearance input (rgb image)
            c: pose input (heat map)
        Output:
            hs: hidden units
        '''
        # according to the paper, xc=[x, c]. while in the code they use xc=x.
        # xc = torch.cat((x, c), dim=1)
        xc = x
        assert x.size(1) == self.input_nc_enc
        if not xc.size(2)==xc.size(3)==self.input_size_enc:
            xc = F.adaptive_avg_pool2d(xc, self.input_size_enc)

        hs = []
        h = self.enc_up_pre_conv(xc)
        for l in range(self.n_scales_enc):
            for i in range(self.n_residual_blocks):
                h = self.__getattr__('enc_up_%d_res_%d' % (l, i))(h)
                hs.append(h)
            if l + 1 < self.n_scales_enc:
                h = self.__getattr__('enc_up_%d_downsample'%l)(h)
        return hs

    def enc_down(self, gs):
        '''
        Input:
            gs: input hiddent units
        Output:
            hs: hidden units
            qs: posteriors
            zs: samples from posterior
        '''
        hs = []
        qs = []
        zs = []

        h = self.enc_down_pre_conv(gs[-1])
        for l in range(self.n_latent_scales):
            for i in range(self.n_residual_blocks//2):
                h = self.__getattr__('enc_down_%d_res_%d'%(l, i))(h, gs.pop())
                hs.append(h)
            # posterior
            q = self.__getattr__('enc_down_%d_latent'%l)(h)
            qs.append(q)
            # posterior sample
            z = self.latent_sample(q)
            zs.append(z)
            # sample feedback
            for j in range(self.n_residual_blocks//2, self.n_residual_blocks):
                gz = torch.cat((gs.pop(), z), dim=1)
                h = self.__getattr__('enc_down_%d_res_%d'%(l, j))(h, gz)
                hs.append(h)
            # up sample
            if l + 1 < self.n_latent_scales:
                h = self.__getattr__('enc_down_%d_upsample'%l)(h)

        return hs, qs, zs

    def dec_up(self, c):
        '''
        Input:
            c: pose input
        Output:
            hs: hidden units
        '''
        if not c.size(2)==c.size(3)==self.input_size_dec:
            c = F.adaptive_avg_pool2d(c, self.input_size_dec)
        assert c.size(1) == self.input_nc_dec

        hs = []
        h = self.dec_up_pre_conv(c)
        for l in range(self.n_scales_dec):
            for i in range(self.n_residual_blocks):
                h = self.__getattr__('dec_up_%d_res_%d' % (l, i))(h)
                hs.append(h)
            if l + 1 < self.n_scales_dec:
                h = self.__getattr__('dec_up_%d_downsample'%l)(h)
        return hs

    def dec_down(self, gs, zs_posterior, training):
        '''
        Input:
            gs: input hidden units
            zs_posterior: samples from posterior. from LR layer to HR layer
        Output:
            hs: hidden units
            ps: prior
            zs: samples from prior            
        '''
        hs = []
        ps = []
        zs = []
        h = self.dec_down_pre_conv(gs[-1])
        for l in range(self.n_scales_dec):
            for i in range(self.n_residual_blocks//2):
                h = self.__getattr__('dec_down_%d_res_%d'%(l,i))(h, gs.pop())
                hs.append(h)
            if l < self.n_latent_scales:
                spatial_shape = self.input_size_dec / 2**(self.n_scales_dec - l - 1)
                # n_h_channels = hs[-1].size(1)
                if spatial_shape == 1:
                    p = self.__getattr__('dec_down_%d_latent'%l)(h)
                    ps.append(p)
                    z_prior = self.latent_sample(p)
                    zs.append(z_prior)
                else:
                    # four autoregressively modeled groups
                    if training:
                        z_posterior_groups = self.space_to_depth(zs_posterior[0], scale=2) # the or of zs_posterior is from LR to HR
                        split_size = z_posterior_groups.size(1)//4
                        z_posterior_groups = list(z_posterior_groups.split(split_size, dim=1))
                    p_groups = []
                    z_groups = []
                    p_feat = self.space_to_depth(h, scale=2)
                    for i in range(4):
                        p_group = self.__getattr__('dec_down_%d_latent_%d'%(l,i))(p_feat)
                        p_groups.append(p_group)
                        z_group = self.latent_sample(p_group)
                        z_groups.append(z_group)
                        # ar feedback sampled from
                        if training:
                            feedback = z_posterior_groups.pop(0)
                        else:
                            feedback = z_group
                        if i + 1 < 4:
                            p_feat = self.__getattr__('dec_down_%d_ar_%d'%(l,i))(p_feat, feedback)
                    if training:
                        assert not z_posterior_groups
                    p = self.depth_to_space(torch.cat(p_groups, dim=1), scale=2)
                    ps.append(p)
                    z_prior = self.depth_to_space(torch.cat(z_groups, dim=1), scale=2)
                    zs.append(z_prior)
                # vae feedback sampled from
                if training:
                    # posterior
                    z = zs_posterior.pop(0)
                else:
                    # prior
                    z = z_prior
                for i in range(self.n_residual_blocks//2, self.n_residual_blocks):
                    h = torch.cat((h, z), dim=1)
                    h = self.__getattr__('dec_down_%d_nin_%d'%(l,i))(h)
                    h = self.__getattr__('dec_down_%d_res_%d'%(l,i))(h, gs.pop())
                    hs.append(h)
            else:
                for i in range(self.n_residual_blocks//2, self.n_residual_blocks):
                    h = self.__getattr__('dec_down_%d_res_%d'%(l,i))(h, gs.pop())
                    hs.append(h)

            if l + 1 < self.n_scales_dec:
                h = self.__getattr__('dec_down_%d_upsample'%(l))(h)

        assert not gs
        if training:
            assert not zs_posterior
        return hs, ps, zs

    def dec_to_image(self, h):
        return self.dec_output(h)
    
    def latent_sample(self, p):
        mean = p
        stddev = 1.0
        z = p + stddev * p.new(p.size()).normal_()
        return z

    def latent_kl(self, p, q):
        n = p.size(0)
        kl = 0.5 * (p-q)*(p-q)
        kl = kl.view(n, -1)
        kl = kl.sum(dim=1).mean()
        return kl
    
    def depth_to_space(self, x, scale=2):
        ''' from [n,c*scale^2,h,w] to [n,c,h*scale,w*scale]'''
        return F.pixel_shuffle(x, scale)

    def space_to_depth(self, x, scale=2):
        ''' from [n,c,h*scale,w*scale] to [n,c*scale^2,h,w]'''
        n, c, h, w = x.size()
        assert h%scale==0 and w%scale==0
        nh, nw = h//scale, w//scale
        x = x.unfold(2,scale,scale).unfold(3,scale,scale).contiguous()
        x = x.view(n,c,nh,nw,scale*scale).transpose(3,4).transpose(2,3).contiguous()
        x = x.view(n,c*scale*scale,nh,nw)
        return x

    def train_forward_pass(self, x_ref, c_ref, c_tar):
        # encoder
        hs = self.enc_up(x_ref, c_ref)
        es, qs, zs_posterior = self.enc_down(hs)
        # decoder
        gs = self.dec_up(c_tar)
        ds, ps, zs_prior = self.dec_down(gs, zs_posterior, training=True)
        img = self.dec_to_image(ds[-1])

        return img, qs, ps

    def test_forward_pass(self, c_tar):
        # decoder
        gs = self.dec_up(c_tar)
        ds, ps, zs_prior = self.dec_down(gs, [], training=False)
        img = self.dec_to_image(ds[-1])
        return img

    def transfer_pass(self, x_ref, c_ref, c_tar):
        use_mean = True
        # infer latent code
        hs = self.enc_up(x_ref, c_ref)
        es, qs, zs_posterior = self.enc_down(hs)
        zs_mean = [q.clone() for q in qs]
        gs = self.dec_up(c_tar)

        if use_mean:
            ds, ps, zs_prior = self.dec_down(gs, zs_mean, training=True)
        else:
            ds, ps, zs_prior = self.dec_down(gs, zs_posterior, training=True)
        img = self.dec_to_image(ds[-1])
        return img, qs, ps

    def forward(self, x_ref, c_ref, c_tar, mode='train', single_device=False):
        if len(self.gpu_ids) > 1 and not single_device:
            return nn.parallel.data_parallel(self, (x_ref, c_ref, c_tar), module_kwargs={'mode':mode, 'single_device':True})
        else:
            if mode == 'train':
                # return: img, qs, ps
                return self.train_forward_pass(x_ref, c_ref, c_tar)
            elif mode == 'test':
                # return: img
                return self.test_forward_pass(c_tar)
            elif mode == 'transfer':
                # return: img
                return self.transfer_pass(x_ref, c_ref, c_tar)
            else:
                raise NotImplementedError()


###############################################################################
# Feature Spatial Transformer
###############################################################################

def define_feat_spatial_transformer(opt):
    net = EncoderDecoderFeatureSpatialTransformNet(
            shape_nc = opt.shape_nc,
            feat_nc = opt.feat_nc,
            shape_nf = opt.shape_nf,
            max_nf = opt.max_nf,
            n_shape_downsample = opt.n_shape_downsample,
            reduce_type = opt.reduce_type,
            norm = opt.norm,
            gpu_ids = opt.gpu_ids
        )

    if len(opt.gpu_ids)> 0:
        assert(torch.cuda.is_available())
        net.cuda()

    init_weights(net, init_type = opt.init_type)
    return net

class EncoderDecoderFeatureSpatialTransformNet(nn.Module):
    def __init__(self, shape_nc, feat_nc, shape_nf, max_nf, n_shape_downsample, reduce_type, norm, gpu_ids):
        super(EncoderDecoderFeatureSpatialTransformNet,self).__init__()

        self.gpu_ids = gpu_ids
        self.reduce_type = reduce_type

        norm_layer = get_norm_layer(norm)
        use_bias = norm != 'batch'

        shape_encode_layers = []
        c_in = shape_nc
        c_out = shape_nf

        for n in range(n_shape_downsample):
            shape_encode_layers += [
                nn.Conv2d(c_in, c_out, 4, 2, 1, bias = use_bias),
                # nn.BatchNorm2d(c_out),
                norm_layer(c_out) if norm != 'none' else None,
                nn.ReLU()
            ]
            c_in = c_out
            c_out *= 2

        shape_encode_layers = [l for l in shape_encode_layers if l is not None]
        self.shape_encode = nn.Sequential(*shape_encode_layers)

        c_shape_code = c_in
        d1 = min(max_nf, feat_nc*2)
        d2 = min(max_nf, feat_nc*4)
        if self.reduce_type == 'conv':
            encode_layers = [
                    nn.Conv2d(c_shape_code+feat_nc, d1, kernel_size=3, stride=2, bias=use_bias),
                    # nn.BatchNorm2d(d1),
                    norm_layer(d1) if norm != 'none' else None,
                    nn.ReLU(),
                    nn.Conv2d(d1, d2, kernel_size=3, stride=2, bias=use_bias),
                    # nn.BatchNorm2d(d2),
                    norm_layer(d2) if norm=='batch' else None,
                    nn.ReLU(),
                ]
        elif self.reduce_type == 'pool':
            encode_layers = [
                nn.Conv2d(c_shape_code+feat_nc, d1, kernel_size=3, stride=1, padding=1, bias=use_bias),
                # nn.BatchNorm2d(d1),
                norm_layer(d1) if norm != 'none' else None,
                nn.ReLU(),
                nn.Conv2d(d1, d2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                # nn.BatchNorm2d(d2),
                norm_layer(d2) if norm=='batch' else None,
                nn.AvgPool2d(kernel_size=7),
                nn.ReLU(),
                ]
        encode_layers = [l for l in encode_layers if l is not None]
        self.encode = nn.Sequential(*encode_layers)

        decode_layers = [
            nn.Conv2d(c_shape_code+d2, d1, kernel_size=3, stride=1, padding=1, bias=use_bias),
            # nn.BatchNorm2d(d1),
            norm_layer(d1) if norm != 'none' else None,
            nn.ReLU(),
            nn.Conv2d(d1, feat_nc, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        ]
        decode_layers = [l for l in decode_layers if l is not None]
        self.decode =nn.Sequential(*decode_layers)

    def forward(self, feat_input, shape_src, shape_tar, single_device = False):
        if len(self.gpu_ids) > 1 and (not single_device):
            return nn.parallel.data_parallel(self, (feat_input, shape_src, shape_tar), module_kwargs = {'single_device': True})
        else:
            shape_code_src = self.shape_encode(shape_src)
            shape_code_tar = self.shape_encode(shape_tar)
            feat_nonspatial = self.encode(torch.cat((feat_input, shape_code_src), dim=1))

            b, c = feat_nonspatial.size()[0:2]
            h, w = feat_input.size()[2:4]
            feat_tile = feat_nonspatial.expand(b,c,h,w)
            feat_output = self.decode(torch.cat((feat_tile, shape_code_tar), dim=1))
            return feat_output

###############################################################################
# General Image Encoder
###############################################################################
def define_image_encoder(opt, which_encoder='edge'):
    norm_layer = get_norm_layer(opt.norm)
    activation = nn.ReLU
    
    if which_encoder == 'edge':
        if opt.edge_shape_guided:
            input_nc = 1 + opt.shape_nc
        else:
            input_nc = 1
        nf = opt.edge_nf
        output_nc = opt.edge_nof
        num_downs = opt.edge_ndowns
        encoder_type = opt.encoder_type if opt.edge_encoder_type == 'default' else opt.edge_encoder_type
        encoder_block = opt.encoder_block if opt.edge_encoder_block == 'default' else opt.edge_encoder_block
    elif which_encoder == 'color':
        if opt.color_patch and opt.color_patch!='single':
            input_nc = 6
        else:
            input_nc = 3
        if opt.color_shape_guided:
            input_nc += opt.shape_nc
        nf = opt.color_nf
        output_nc = opt.color_nof
        num_downs = opt.color_ndowns
        encoder_type = opt.encoder_type if opt.color_encoder_type == 'default' else opt.color_encoder_type
        encoder_block = opt.encoder_block if opt.color_encoder_block == 'default' else opt.color_encoder_block
    elif which_encoder == 'shape':
        input_nc = opt.shape_nc
        nf = opt.shape_nf
        output_nc = opt.shape_nof
        num_downs = opt.shape_ndowns
        encoder_type = opt.encoder_type if opt.shape_encoder_type == 'default' else opt.shape_encoder_type
        encoder_block = opt.encoder_block if opt.shape_encoder_block =='default' else opt.shape_encoder_block
    else:
        raise NotImplementedError('invalid encoder type %s'%which_encoder)
    
    if encoder_type == 'normal':
        image_encoder = ImageEncoder(block=encoder_block, input_nc=input_nc, output_nc=output_nc, nf=nf, num_downs=num_downs, norm_layer=norm_layer, activation=activation, gpu_ids=opt.gpu_ids)
    elif encoder_type == 'pool':
        image_encoder = PoolingImageEncoder(block=encoder_block, input_nc=input_nc, output_nc=output_nc, nf=nf, num_downs=num_downs, norm_layer=norm_layer, activation=activation, use_attention=opt.encoder_attention, gpu_ids=opt.gpu_ids)
    elif encoder_type == 'fc':
        image_encoder = FCImageEncoder(block=encoder_block, input_nc=input_nc, output_nc=output_nc, nf=nf, num_downs=num_downs, norm_layer=norm_layer, activation=activation, gpu_ids=opt.gpu_ids)
    elif encoder_type == 'st':
        if opt.tar_guided:
            image_encoder = STImageEncoder(block=encoder_block, input_nc=input_nc, output_nc=output_nc, guide_nc=opt.shape_nc, nf=nf, num_downs=num_downs, norm_layer=norm_layer, activation=activation, gpu_ids=opt.gpu_ids)
        else:
            image_encoder = STImageEncoder(block=encoder_block, input_nc=input_nc, output_nc=output_nc, guide_nc=0, nf=nf, num_downs=num_downs, norm_layer=norm_layer, activation=activation, gpu_ids=opt.gpu_ids)
    if len(opt.gpu_ids) > 0:
        image_encoder.cuda()
    init_weights(image_encoder, init_type=opt.init_type)
    return image_encoder

class DownsampleEncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer, activation, use_bias):
        super(DownsampleEncoderBlock, self).__init__()
        layers = []
        layers.append(nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=use_bias))
        if norm_layer is not None:
            layers.append(norm_layer(output_nc))
        if activation is not None:
            if not isinstance(activation, nn.Module):
                activation = activation()
            layers.append(activation)
        self.model = nn.Sequential(*layers)
    def forward(self, input):
        return self.model(input)

class ResidualEncoderBlock(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer, activation, use_bias, stride=2):
        super(ResidualEncoderBlock, self).__init__()
        if not isinstance(activation, nn.Module):
            activation = activation()
        self.conv = nn.Sequential(
            nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=stride, padding=1, bias=use_bias),
            norm_layer(output_nc),
            activation,
            nn.Conv2d(output_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(output_nc))
        if input_nc == output_nc and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_nc, output_nc, kernel_size=1, stride=stride, bias=use_bias),
                norm_layer(output_nc))
        self.activation = activation

    def forward(self, input):
        out = self.conv(input)
        if self.downsample is None:
            residual = input
        else:
            residual = self.downsample(input)
        return self.activation(out + residual)

class ImageEncoder(nn.Module):
    def __init__(self, block, input_nc, output_nc=-1, nf=64, num_downs=5, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, gpu_ids=[]):
        super(ImageEncoder, self).__init__()
        max_nf = 512
        n_innermost = 7 #feat_map at this level has 1x1 size
        self.input_nc = input_nc
        self.output_nc = output_nc if output_nc>0 else min(nf*2**(num_downs), max_nf)
        self.nf = nf
        self.num_downs = num_downs
        self.gpu_ids = gpu_ids
        self.block = block

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        layers = [
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(nf),
            activation()]

        for n in range(num_downs):
            c_in = min(max_nf, nf*2**n)
            c_out = min(max_nf, nf*2**(n+1)) if n < num_downs -1 else self.output_nc
            if block == 'downsample' or n == n_innermost:
                layers.append(nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=use_bias))
                if n < n_innermost or norm_layer.func == nn.BatchNorm2d:
                    layers.append(norm_layer(c_out))
                layers.append(activation())
            elif block == 'residual':
                layers.append(ResidualEncoderBlock(c_in, c_out, norm_layer, activation, use_bias, stride=2))

        self.net = nn.Sequential(*layers)

    def forward(self, img):
        if len(self.gpu_ids) > 1:
            return nn.parallel.data_parallel(self.net, img)
        else:
            return self.net(img)

class PoolingImageEncoder(nn.Module):
    def __init__(self, block, input_nc, output_nc=-1, nf=64, num_downs=5, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, use_attention=False, gpu_ids=[]):
        super(PoolingImageEncoder, self).__init__()
        max_nf = 512
        self.input_nc = input_nc
        self.output_nc = output_nc if output_nc > 0 else min(nf*2**(num_downs), max_nf)
        self.nf = nf
        self.num_downs = num_downs
        self.use_attention = use_attention
        self.gpu_ids = gpu_ids
        self.block = block # downsample or residual

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        layers = [
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(nf),
            activation()]

        for n in range(num_downs):
            c_in = min(max_nf, nf*2**n)
            c_out = min(max_nf, nf*2**(n+1)) if n < num_downs -1 else self.output_nc
            if block == 'downsample':
                layers.append(DownsampleEncoderBlock(c_in, c_out, norm_layer, activation, use_bias))
            elif block == 'residual':
                layers.append(ResidualEncoderBlock(c_in, c_out, norm_layer, activation, use_bias, stride=2))
        
        self.conv = nn.Sequential(*layers)
        self.activation = activation()
        if use_attention:
            self.attention_cls = nn.Sequential(
                nn.Conv2d(c_out, 1, kernel_size=1),
                nn.Softmax2d()
            )
        else:
            self.attention_cls = None
        
    def forward(self, img):
        if len(self.gpu_ids) > 1:
            feat_map = nn.parallel.data_parallel(self.conv, img)
        else:
            feat_map = self.conv(img)
        
        if self.use_attention:
            attention = self.attention_cls(feat_map)
            feat_map = feat_map * attention
            feat_map = feat_map.sum(dim=3, keepdim=True).sum(dim=2, keepdim=True)
        else:
            feat_map = feat_map.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
        return self.activation(feat_map)
    
class FCImageEncoder(nn.Module):
    def __init__(self, block, input_nc, output_nc=-1, nf=64, num_downs=5, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, gpu_ids=[]):
        super(FCImageEncoder, self).__init__()
        max_nf = 512
        input_size = 256
        self.input_nc = input_nc
        self.output_nc = output_nc if output_nc > 0 else min(nf*2**(num_downs), max_nf)
        self.nf = nf
        self.num_downs = num_downs
        self.gpu_ids = gpu_ids
        self.block = block # downsample or residual
        self.feat_size = input_size // 2**num_downs

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        layers = [
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(nf),
            activation()]

        for n in range(num_downs):
            c_in = min(max_nf, nf*2**n)
            c_out = min(max_nf, nf*2**(n+1))
            if block == 'downsample':
                layers.append(DownsampleEncoderBlock(c_in, c_out, norm_layer, activation, use_bias))
            elif block == 'residual':
                layers.append(ResidualEncoderBlock(c_in, c_out, norm_layer, activation, use_bias, stride=2))
        
        layers += [
            nn.Conv2d(c_out, self.output_nc, kernel_size=self.feat_size),
            activation()
            ]
        self.net = nn.Sequential(*layers)
        
    def forward(self, img):
        if len(self.gpu_ids) > 1:
            return nn.parallel.data_parallel(self.net, img)
        else:
            return self.net(img)

class STImageEncoder(nn.Module):
    def __init__(self, block, input_nc, output_nc=-1, guide_nc=0, nf=64, num_downs=5, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, gpu_ids=[]):
        super(STImageEncoder,self).__init__()
        max_nf = 512
        input_size = 256
        self.input_nc = input_nc
        self.output_nc = output_nc if output_nc > 0 else min(nf*2**(num_downs), max_nf)
        self.nf = nf
        self.num_downs = num_downs
        self.gpu_ids = gpu_ids
        self.block = block # downsample or residual
        self.feat_size = input_size // 2**num_downs
        self.guide_nc = guide_nc

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        layers = [
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(nf),
            activation()]

        for n in range(num_downs):
            c_in = min(max_nf, nf*2**n)
            c_out = min(max_nf, nf*2**(n+1)) if n < num_downs-1 else self.output_nc
            if block == 'downsample':
                layers.append(DownsampleEncoderBlock(c_in, c_out, norm_layer, activation, use_bias))
            elif block == 'residual':
                layers.append(ResidualEncoderBlock(c_in, c_out, norm_layer, activation, use_bias, stride=2))

        self.conv = nn.Sequential(*layers)
        self.stn = SpatialTransformerNetwork(input_nc=self.output_nc+guide_nc, size=self.feat_size)

    def forward(self, img, guide = None, single_device=False):
        if len(self.gpu_ids) > 1 and not single_device:
            if guide is None:
                return nn.parallel.data_parallel(self, img, module_kwargs={'guide':None, 'single_device':True})
            else:
                return nn.parallel.data_parallel(self, (img, guide), module_kwargs={'single_device':True})
        else:
            feat = self.conv(img)
            if self.guide_nc>0:
                guide = F.upsample(guide, size=(self.feat_size, self.feat_size), mode='bilinear')
                feat = torch.cat((feat, guide), dim=1)

            return self.stn(feat)[:,0:self.output_nc]
        


class SpatialTransformerNetwork(nn.Module):
    def __init__(self, input_nc, size=8):
        super(SpatialTransformerNetwork, self).__init__()
        self.input_nc = input_nc
        self.size = size

        self.loc_conv = nn.Sequential(
            nn.Conv2d(input_nc, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32,32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True)
            )
        self.loc_fc = nn.Sequential(
            nn.Linear(32*size*size//16, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
            )

        self.loc_fc[2].weight.data.fill_(0)
        self.loc_fc[2].bias.data = torch.FloatTensor([1,0,0,0,1,0])

    def forward(self, x):
        xs = self.loc_conv(x)
        xs = xs.view(x.size(0),-1)
        theta = self.loc_fc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        return F.grid_sample(x, grid)


def define_image_decoder_from_params(input_nc, output_nc, nf=64, num_ups=5, norm='batch', output_activation=nn.Tanh, gpu_ids=[], init_type='normal'):
    # Todo: from option
    norm_layer = get_norm_layer(norm)
    activation=nn.ReLU
    image_decoder = ImageDecoder(input_nc, output_nc, nf, num_ups, norm_layer, activation, output_activation, gpu_ids)
    if len(gpu_ids)>0:
        image_decoder.cuda()
    init_weights(image_decoder, init_type)
    return image_decoder

class ImageDecoder(nn.Module):
    def __init__(self, input_nc, output_nc, nf=64, num_ups=5, norm_layer=nn.BatchNorm2d, activation=nn.ReLU, output_activation=nn.Tanh, gpu_ids=[]):
        super(ImageDecoder, self).__init__()
        max_nf=512
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.nf = nf
        self.num_ups=num_ups
        self.gpu_ids = gpu_ids

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        c_out = min(max_nf, nf*2**num_ups)
        layers = []
        
        layers.append(nn.Conv2d(input_nc, c_out, kernel_size=1, stride=1,padding=0))
        if num_ups < 8 or (not use_bias):
            # not InstanceNorm + vector
            layers.append(norm_layer(c_out))
        layers.append(activation())

        for n in range(num_ups, 0, -1):
            c_in = min(max_nf, nf*2**n)
            c_out = min(max_nf, nf*2**(n-1))
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(c_out),
                activation()
            ]

        layers += [nn.ReflectionPad2d(3)]
        layers += [nn.Conv2d(nf, output_nc, kernel_size=7)]
        if output_activation is not None:
            layers += [output_activation()]

        self.model = nn.Sequential(*layers)

    def forward(self, input):
        if len(self.gpu_ids) > 1:
            return nn.parallel.data_parallel(self.model, input)
        else:
            return self.model(input)



###############################################################################
# General Image Encoder V2
###############################################################################
def load_encoder_v2(opt, which_model):
    from argparse import Namespace
    reserve_opt = ['gpu_ids']

    opt_dict = io.load_json(os.path.join('checkpoints', which_model, 'train_opt.json'))
    opt_load = Namespace()
    for k, v in opt_dict.iteritems():
        opt_load.__setattr__(k, v)
    
    for k in reserve_opt:
        opt_load.__setattr__(k, opt.__getattribute__(k))

    model = define_encoder_v2(opt_load)
    model.load_state_dict(torch.load(os.path.join('checkpoints', which_model, 'latest_net_encoder.pth')))
    return model, opt_load

def define_encoder_v2(opt):
    norm_layer = get_norm_layer(opt.norm)
    activation = nn.ReLU(True)

    # input_nc
    if opt.input_type == 'image':
        input_nc = 3
    elif opt.input_type == 'seg':
        input_nc = 7
    elif opt.input_type == 'edge':
        input_nc = 1
    elif opt.input_type == 'shape':
        input_nc = 7 + 18
    else:
        raise NotImplementedError()

    encoder = Encoder_V2(block=opt.block, input_nc=input_nc, output_nc=opt.nof, input_size=opt.fine_size, nf=opt.nf, max_nf=opt.max_nf, nf_increase=opt.nf_increase, ndowns=opt.ndowns, final_fc=opt.encode_fc, norm_layer=norm_layer, activation=activation, gpu_ids=opt.gpu_ids)
    if opt.gpu_ids:
        encoder.cuda()
    init_weights(encoder, init_type=opt.init_type)
    return encoder

def define_decoder_v2(opt):
    norm_layer = get_norm_layer(opt.norm)
    activation = nn.ReLU(True)

    # output_nc
    if opt.output_type == 'image':
        output_activation = nn.Tanh()
        output_nc = 3
    elif opt.output_type == 'seg':
        output_activation = None
        output_nc = 7
    elif opt.output_type == 'edge':
        output_activation = nn.Sigmoid()
        output_nc = 1
    else:
        raise NotImplementedError()

    # input_nc
    if opt.decode_guide:
        input_nc = opt.nof + opt.decode_gf
    else:
        input_nc = opt.nof

    
    fc_output_size = opt.fine_size // (2**opt.ndowns)

    decoder = Decoder_V2(input_nc=input_nc, output_nc=output_nc, fc_output_size=fc_output_size, nf=opt.nf, max_nf=opt.max_nf, nf_increase=opt.nf_increase, nups=opt.ndowns, start_fc=opt.decode_fc, norm_layer=norm_layer, activation=activation, output_activation=output_activation, gpu_ids=opt.gpu_ids)
    if opt.gpu_ids:
        decoder.cuda()
    init_weights(decoder, init_type=opt.init_type)

    return decoder

class Encoder_V2(nn.Module):
    def __init__(self, block, input_nc, output_nc, input_size=256, nf=64, max_nf=512, nf_increase='exp', ndowns=5, final_fc=False, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True), gpu_ids=[]):
        super(Encoder_V2, self).__init__()
        assert nf <= max_nf
        assert 2**ndowns < input_size, '%dx%d input image can not be reduced by stride-2 convolution for ndowns=%d times' % (input_size, input_size, ndowns)
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.input_size = input_size
        self.gpu_ids = gpu_ids


        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        layers = [
            nn.Conv2d(input_nc, nf, kernel_size=7, padding=3, bias=use_bias),
            norm_layer(nf),
            activation]

        c_out = nf
        for n in range(ndowns):
            c_in = c_out
            if n == ndowns - 1 and not final_fc:
                c_out = output_nc
            elif nf_increase == 'exp':
                c_out = min(max_nf, nf*2**(n+1))
            else:
                c_out = min(max_nf, nf*(n+2))

            if block == 'conv':
                layers.append(DownsampleEncoderBlock(c_in, c_out, norm_layer, activation, use_bias))
            elif block == 'residual':
                layers.append(ResidualEncoderBlock(c_in, c_out, norm_layer, activation, use_bias, stride=2))

        if final_fc:
            feat_size = input_size // (2**ndowns)
            layers += [
                nn.Conv2d(c_out, output_nc, kernel_size=feat_size),
                activation]

        self.net = nn.Sequential(*layers)

    def forward(self, img):
        if len(self.gpu_ids) > 1:
            return nn.parallel.data_parallel(self.net, img)
        else:
            return self.net(img)

class Decoder_V2(nn.Module):
    def __init__(self, input_nc, output_nc, fc_output_size=8, nf=64, max_nf=512, nf_increase='exp', nups=5, start_fc=False, norm_layer=nn.BatchNorm2d, activation=nn.ReLU(True), output_activation=nn.Tanh(), gpu_ids=[]):
        super(Decoder_V2, self).__init__()
        assert nf <= max_nf
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.gpu_ids = gpu_ids

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        layers = []
        if start_fc:
            # input feature (1x1) will pass through a fc layer to create (input_size x input_size) feature map
            if nf_increase == 'exp':
                c_out = min(max_nf, nf*2**(nups))
            else:
                c_out = min(max_nf, nf*(nups+1))
            layers += [
                nn.ConvTranspose2d(input_nc, c_out, kernel_size=fc_output_size),
                norm_layer(c_out),
                activation
            ]
        else:
            c_out = input_nc

        for n in range(nups):
            c_in = c_out
            if nf_increase == 'exp':
                c_out = min(max_nf, nf*2**(nups-n-1))
            else:
                c_out = min(max_nf, nf*(nups-n))

            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                norm_layer(c_out),
                activation
            ]

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(c_out, output_nc, kernel_size=7)
        ]

        if output_activation is not None:
            if not isinstance(output_activation, nn.Module):
                output_activation = output_activation
            layers += [output_activation]

        self.net = nn.Sequential(*layers)

    def forward(self, feat):
        if len(self.gpu_ids) > 1:
            return nn.parallel.data_parallel(self.net, feat)
        else:
            return self.net(feat)

def define_DFN_from_params(nf, ng, nmid, feat_size, local_size, nblocks, norm, gpu_ids, init_type):
    dfn = DFNModule(nf, ng, nmid, feat_size, local_size, nblocks, norm, gpu_ids)
    init_weights(dfn, init_type)
    dfn.net[-2].bias.data.fill_(-1)
    dfn.net[-2].bias.data[local_size*local_size//2] = 1
    if gpu_ids:
        dfn.cuda()
    return dfn

class DFNModule(nn.Module):
    def __init__(self, nf, ng, nmid = 128, feat_size=8, local_size=3, nblocks=0, norm='instance', gpu_ids=[]):
        super(DFNModule, self).__init__()
        self.nf = nf
        self.ng = ng
        self.nmid = nmid
        self.feat_size = feat_size
        self.local_size = local_size
        self.gpu_ids = gpu_ids
        self.nblocks = nblocks

        self.net = nn.Sequential(
            nn.Conv2d(ng*2 + local_size*local_size, nmid, kernel_size=local_size, stride=1, padding=(local_size-1)//2),
            nn.ReLU(True),
            nn.Conv2d(nmid, local_size*local_size, kernel_size=local_size, stride=1, padding=(local_size-1)//2),
            nn.Softmax(dim=1),            
            )
        if nblocks > 0:
            blocks = []
            norm_layer = get_norm_layer(norm)
            use_bias = not (norm == 'batch')
            for i in range(nblocks):
                blocks += [ResidualEncoderBlock(nf, nf, norm_layer, nn.ReLU(True), use_bias, 1)]
            self.res_blocks = nn.Sequential(*blocks)

    def compute_correlation(self, g1, g2):

        ng1 = F.normalize(g1)
        ng2 = F.normalize(g2)

        max_shift = (self.local_size - 1) // 2
        pad = [max_shift] * 4
        ng2p = F.pad(ng2, pad, 'constant')
        corr = []

        for dh in range(-max_shift, max_shift+1):
            for dw in range(-max_shift, max_shift+1):
                ng2s = ng2p[:,:,(dh+max_shift):(dh+max_shift+self.feat_size), (dw+max_shift):(dw+max_shift+self.feat_size)]
                # corr += F.cosine_similarity(g1, g2s)
                corr.append((ng1 * ng2s).sum(dim=1))
        corr = torch.stack(corr, dim=1)
        return corr

    def apply_filter(self, x, coef):
        max_shift = (self.local_size-1)//2
        pad = [max_shift] * 4
        xp = F.pad(x, pad, 'constant')
        output = 0

        n = 0
        for dh in range(-max_shift, max_shift+1):
            for dw in range(-max_shift, max_shift+1):
                xs = xp[:,:,(dh+max_shift):(dh+max_shift+self.feat_size), (dw+max_shift):(dw+max_shift+self.feat_size)]
                c = coef[:,n:(n+1)]
                output += xs * c
                n += 1

        return output
        
    def forward(self, x, g1, g2, single_device=False):
        if len(self.gpu_ids) > 1 and not single_device:
            assert g1.is_same_size(g2)
            assert g1.size()[2] == g1.size(3) == self.feat_size
            return nn.parallel.data_parallel(self, (x, g1, g2), module_kwargs={'single_device': True})
        else:
            corr = self.compute_correlation(g1, g2)
            guide = torch.cat((g1, g2, corr), dim=1)
            coef = self.net(guide)
            output = self.apply_filter(x, coef)
            if self.nblocks > 0:
                output = self.res_blocks(output)
            return output, coef


