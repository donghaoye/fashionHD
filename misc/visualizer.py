from __future__ import division, print_function

import torch
import torchvision

import os
import time
import util.io as io
import util.image as image
from util.pavi import PaviClient
from options.base_options import opt_to_str
import numpy as np


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
        self.data_loaded = False

    # Todo: add sample visualization methods.
    def load_attr_data(self):
        opt = self.opt
        self.samples = io.load_json(os.path.join(opt.data_root, opt.fn_sample))
        self.attr_label = io.load_data(os.path.join(opt.data_root, opt.fn_label))
        self.attr_entry = io.load_json(os.path.join(opt.data_root, opt.fn_entry))
        self.data_loaded = True

    def visualize_attr_pred(self, model, num_top_attr = 5):
        if not self.data_loaded:
            self.load_attr_data()

        opt = self.opt
        dir_output = os.path.join('checkpoints', opt.id, 'vis_attr')
        io.mkdir_if_missing(dir_output)

        for idx, s_id in enumerate(model.input['id']):
            prob = model.output['prob'][idx].data.cpu().numpy().flatten()
            if 'map' in model.output:
                prob_map = model.output['map'][idx].data.cpu().numpy()
            else:
                prob_map = None

            top_pred_attr = (-prob).argsort()[0:num_top_attr]
            gt_attr = [i for i, l in enumerate(self.attr_label[s_id]) if l == 1]

            img = image.imread(self.samples[s_id]['img_path'])

            if prob_map is None:
                img_out = img
            else:
                img_out = [img]            
                h, w = img.shape[0:2]
                for i_att in top_pred_attr:
                    p = prob[i_att]
                    m = prob_map[i_att]
                    m = (m - m.min()) / (m.max() - m.min())
                    m = image.resize(m, (w, h))[:,:,np.newaxis]

                    img_out.append(img * m)

                img_out = image.stitch(img_out, 0)

            tag_list = [s_id, self.samples[s_id]['img_path_org']]
            tag_list.append('prediction: ' + ', '.join(['%s (%.3f)' % (self.attr_entry[i]['entry'], prob[i]) for i in top_pred_attr]))
            tag_list.append('annotation: ' + ', '.join([self.attr_entry[i]['entry'] for i in gt_attr]))

            img_out = image.add_tag(img_out, tag_list, ['k', 'k', 'b', 'r'])
            fn_output = os.path.join(dir_output, s_id + '.jpg')
            image.imwrite(img_out, fn_output)





                









