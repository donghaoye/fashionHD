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
from collections import OrderedDict


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


    def _open_log_file(self):
        self.f_log = open(os.path.join(self.expr_dir, 'train_log.txt'), 'w')
        log = 'pytorch version: %s\n' % torch.__version__
        print(log, file = self.f_log)

    def print_train_error(self, iter_num, epoch, num_batch, lr, errors):
        '''
        Display training log information on screen and output it to log file.

        Input:
            iter_num:   current iteration
            epoch:      current epoch
            num_batch: number of minibatch in each epoch
            lr:         current learning rate
            errors:      error information
        '''

        if self.f_log is None:
            self._open_log_file()


        epoch_step = (iter_num-1) % num_batch + 1
        t_per_step = (time.time() - self.clock) / (iter_num - self.step_counter)
        eta = ((self.opt.niter + self.opt.niter_decay) * num_batch - iter_num) * t_per_step / 3600


        log = '[%s] Train [Iter: %d, Epoch: %d, Prog: %d/%d (%.2f%%)] t_cost: %.2f, ETA: %.1fh, lr: %.3e]  ' % \
            (self.opt.id, iter_num, epoch, epoch_step, num_batch, 100.*epoch_step/num_batch, t_per_step, eta, lr)
        log += '  '.join(['%s: %.6f' % (k,v) for k,v in errors.iteritems()])

        print(log)
        print(log, file = self.f_log)

        self.clock = time.time()
        self.step_counter = iter_num

    def print_test_error(self, iter_num, epoch, errors):
        '''
        Display testing log information during training
            iter_num:   current iteration
            epoch:      current epoch
            errors:     test information
        '''

        if self.f_log is None:
            self._open_log_file()

            
        log = '[%s] Test [Iter: %d, Epoch %d]\n' % (self.opt.id, iter_num, epoch)
        log += '\n'.join(['%s: %.6f' % (k,v) for k,v in errors.iteritems()])

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
        '''
        This method visualize attribute prediction result of each sample in an image:
        - original image
        - top-k predicted attributes
        - annotated attributes
        - top-k predicted attribute spatial maps (if available)

        Input:
            model: AttributeEncoderModel instance (to get model output)
            num_top_attr
        '''

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


    def show_attr_pred_statistic(self, result):
        '''
        Save attribute predction statistic information into json and txt file.
        Input:
            result (dict): test result with fields
        '''

        if hasattr(self, 'attr_entry'):
            attr_entry = self.attr_entry
        else:
            attr_entry = io.load_json(os.path.join(self.opt.data_root, self.opt.fn_entry))


        # add results of interest
        type_entry = {1:'texture', 2:'fabric', 3:'shape', 4:'part', 5:'style'}
        result['type_info'] = []
        for t in range(1, 6):
            idx_list = [i for i,att in enumerate(attr_entry) if att['type'] == t]
            ap = (np.array(result['AP_list'])[idx_list]).mean().tolist()
            rec3 = (np.array(result['rec3_list'])[idx_list]).mean().tolist()
            rec5 = (np.array(result['rec5_list'])[idx_list]).mean().tolist()
            result['type_info'].append({
                'type': '%d-%s' % (t, type_entry[t]),
                'ap': ap,
                'rec3': rec3,
                'rec5': rec5
            })

        ap_order = np.argsort(result['AP_list'])
        result['top_attr_list'] = [(attr_entry[i]['entry'], attr_entry[i]['type'], result['AP_list'][i]) \
            for i in ap_order[::-1][0:20]]
        result['worst_attr_list'] = [(attr_entry[i]['entry'], attr_entry[i]['type'], result['AP_list'][i]) \
            for i in ap_order[0:20]]


        # display
        result_disp = OrderedDict()
        for k, v in result.iteritems():
            if isinstance(v, float):
                result_disp[k] = v
        self.print_error(result_disp)


        # save json
        dir_test = os.path.join('checkpoints', self.opt.id, 'test')
        io.mkdir_if_missing(dir_test)
        fn_json = os.path.join(dir_test, 'test_result.json')
        for k, v in result.iteritems():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()

        io.save_json(result, fn_json)

        # save txt summary
        str_output = ['AttributeType\tAP\trec3\trec5']
        str_output += ['%s\t%f\t%f\t%f' % (t['type'], t['ap'], t['rec3'], t['rec5']) for t in result['type_info']]
        str_output += ['', 'Top Attribute']
        str_output += ['%s(%d)\t%f' % a for a in result['top_attr_list']]
        str_output += ['', 'Worst Attribute']
        str_output += ['%s(%d)\t%f' % a for a in result['worst_attr_list']]

        fn_txt = os.path.join('checkpoints', self.opt.id, 'test', 'test_result_summary.txt')
        io.save_str_list(str_output, fn_txt)

        # save txt detail
        str_output = ['%s\t%d\t%f\t%.2f\t%.2f\t%.2f' % (att['entry'],att['type'], att['pos_rate'], ap, rec3, rec5) 
            for (att, ap, rec3, rec5) in zip(attr_entry, result['AP_list'], result['rec5_list'], result['rec5_list'])]
        fn_txt = os.path.join('checkpoints', self.opt.id, 'test', 'test_result_detail.txt')

        io.save_str_list(str_output, fn_txt)



class GANVisualizer(BaseVisualizer):
    def __init__(self, opt):
        super(GANVisualizer, self).__init__(opt)

    def _seg_map_to_img(self, seg_map):
        # normalize to [-1, 1]
        seg_map = (seg_map - seg_map.min())/(seg_map.max() - seg_map.min())*2 - 1
        if seg_map.size(1) == 1:
            sz = list(seg_map.size())
            sz[1] = 3
            seg_map = seg_map.expand(sz)
        return seg_map


    def visualize_image(self, epoch, subset, visuals):
        opt = self.opt
        vis_dir = os.path.join('checkpoints', opt.id, 'vis')
        io.mkdir_if_missing(vis_dir)
        print('[%s] visualizing %s images' % (opt.id, subset))

        # post-process masks and landmark heatmaps
        visuals['seg_map'] = self._seg_map_to_img(visuals['seg_map'])
        visuals['landmark_heatmap'] = visuals['landmark_heatmap'].max(dim=1, keepdim=True)[0].expand_as(visuals['img_real'])
        visuals['edge_map'] = visuals['edge_map'].expand_as(visuals['img_real'])

        num_vis = min(opt.max_n_vis, visuals['img_real'].size(0))
        imgs = []
        for i in range(num_vis):
            imgs += [
                visuals['img_real'][i],
                visuals['img_fake'][i],
                # visuals['img_real_raw'][i],
                visuals['img_fake_raw'][i],
                visuals['seg_map'][i],
                visuals['landmark_heatmap'][i],
                visuals['edge_map'][i]
            ]

        imgs = torch.stack(imgs)
        fn_img = os.path.join(vis_dir, '%s_epoch%d.jpg' % (subset, epoch))
        torchvision.utils.save_image(imgs, fn_img, nrow = 6, normalize = True)

    def visualize_image_matrix(self, imgs, imgs_title = None, label = 'default', vis_dir = 'vis'):
        '''
        Input:
            imgs (tensor): image matrix, tensor of size n_row*n_col*C*H*W
            imgs_title (tensor): title images, tensor of size n*C*H*W (must have n==n_row==n_col)
            label (str): output filename

        '''
        vis_dir = os.path.join('checkpoints', self.opt.id, vis_dir)
        io.mkdir_if_missing(vis_dir)

        n_row, n_col, c, h, w = imgs.size()
        if imgs_title is not None:
            assert imgs_title.size(0) == n_row == n_col
            # insert title images at the head of each row
            imgs = torch.cat((imgs_title.view(n_row, 1, c, h, w), imgs), 1)
            # add a title row
            img_blank = torch.zeros([1]+list(imgs_title.size()[1::]))
            imgs_title = torch.cat((img_blank, imgs_title), 0)
            imgs = torch.cat((imgs_title.view(1, n_col+1, c, h, w), imgs), 0)

            n_col += 1
            n_row += 1

        imgs = imgs.view(n_row*n_col, c, h, w)
        fn_img = os.path.join(vis_dir, label+'.jpg')
        torchvision.utils.save_image(imgs, fn_img, nrow = n_col, normalize = True)



    def pavi_log(self, phase, iter_num, outputs):
        # upper_list = ['D_real', 'D_fake', '']
        upper_list = ['grad_G_GAN', 'grad_G_L1', 'grad_G_VGG', 'grad_G_attr']
        lower_list = ['D_GAN', 'G_GAN', 'G_L1', 'G_VGG', 'G_attr', 'G_sa']

        new_outputs = {}
        for k,v in outputs.iteritems():
            if k in upper_list:
                new_outputs[k+'_upper'] = v
            elif k in lower_list:
                new_outputs[k] = v

        super(GANVisualizer, self).pavi_log(phase, iter_num, new_outputs)

