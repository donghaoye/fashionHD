from __future__ import division, print_function

import torch
import torchvision

import os
import time
import util.io as io
import util.image as image
from util.pavi import PaviClient
from options.base_options import opt_to_str


class BaseVisualizer(object):

    def __init__(self, opt):

        self.opt = opt
        self.expr_dir = os.path.join('checkpoints', opt.id)
        self.f_log = None

        # load pavi
        if opt.pavi:
            self.pavi_client = PaviClient(username = 'ly015', password = '123456')
            self.pavi_client.connect(model_name = opt.id, info = {'session_text': opt_to_str(opt)})
        else:
            self.pavi_client = None

        self.clock = time.time()
        self.step_counter = 0

        print('create visualizer')

    def __del__(self):
        if self.f_log:
            self.f_log.close()

    def print_train_error(self, iter_num, epoch, num_batch, lr, error):
        '''
        Display training log information on screen and output it to log file.

        Input:
            iter_num:   current iteration
            epoch:      current epoch
            num_batch: number of minibatch in each epoch
            lr:         current learning rate
            error:      error information
        '''

        if self.f_log is None:
            self.f_log = open(os.path.join(self.expr_dir, 'train_log.txt'), 'w')

        t_per_step = (time.time() - self.clock) / (iter_num - self.step_counter)
        epoch_step = (iter_num-1) % num_batch + 1


        log = '[%s] Train [Iter: %d, Epoch: %d, Prog: %d/%d (%.2f%%)] t_cost: %.2f, lr: %.3e]  ' % \
            (self.opt.id, iter_num, epoch, epoch_step, num_batch, 100.*epoch_step/num_batch, t_per_step, lr)
        log += '  '.join(['%s: %.6f' % (k,v) for k,v in error.iteritems()])

        print(log)
        print(log, file = self.f_log)

        self.clock = time.time()
        self.step_counter = iter_num

    def print_test_error(self, iter_num, epoch, result):
        '''
        Display testing log information during training
            iter_num:   current iteration
            epoch:      current epoch
            result:     test information
        '''

        if self.f_log is None:
            self.f_log = open(os.path.join(self.expr_dir, 'train_log.txt'), 'w')

            
        log = '[%s] Test [Iter: %d, Epoch %d]\n' % (self.opt.id, iter_num, epoch)
        log += '\n'.join(['%s: %.6f' % (k,v) for k,v in result.iteritems()])

        log = '\n'.join(['', '#'*50, log, '#'*50, '']) 

        print(log)
        print(log, file = self.f_log)
        self.clock = time.time()

    def print_error(self, result):
        '''
        Display result info on screen
        '''
        log = '[%s] Test [Epoch: %s]\n' % (self.opt.id, self.opt.which_epoch)
        log += '\n'.join(['%s: %.6f' % (k,v) for k,v in result.iteritems()])
        log = '\n'.join(['', '#'*50, log, '#'*50, ''])
        print(log)


    def pavi_log(self, phase, iter_num, outputs):

        assert self.pavi_client is not None, 'No pavi client (opt.pavi == False)'
        self.pavi_client.log(phase, iter_num, outputs)
        self.clock = time.time()



class AttributeVisualizer(BaseVisualizer):
    def __init__(self, opt):
        super(AttributeVisualizer, self).__init__(opt)

    # Todo: add sample visualization methods.

