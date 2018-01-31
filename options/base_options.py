from __future__ import division, print_function

import torch
import argparse
import os
import util.io as io


def opt_to_str(opt):
        return '\n'.join(['%s: %s' % (str(k), str(v)) for k, v in sorted(vars(opt).items())])


class BaseOptions(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):

        parser = self.parser
        # basic options
        parser.add_argument('--id', type = str, default = 'default', help = 'model ID. the experiment dir will be set as "./checkpoint/id/"')
        parser.add_argument('--gpu_ids', type = str, default = '0', help = 'gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # model options
        parser.add_argument('--init_type', type = str, default = 'normal', help = 'network initialization method [normal|xavier|kaiming|orthogonal]')


        # data options
        parser.add_argument('--dataset_mode', type = str, default = 'gan_self', help = 'type of dataset',
            choices = ['attribute', 'attribute_exp', 'gan_self'])
        parser.add_argument('--data_root', type = str, default = './datasets/DeepFashion/Fashion_design/', help = 'data root path')
        parser.add_argument('--nThreads', type = int, default = 12, help = 'number of workers to load data')
        parser.add_argument('--max_dataset_size', type = int, default = float('inf'), help = 'maximum number of samples')
        parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
        parser.add_argument('--load_size', type = int, default = 256, help = 'scale input image to this size')
        parser.add_argument('--fine_size', type = int, default = 224, help = 'crop input image to this size')
        parser.add_argument('--resize_or_crop', type = str, default = 'resize_and_crop', help = 'scaling and cropping of images at load time',
            choices = ['resize_and_crop', 'resize'])
        parser.add_argument('--image_normalize', type = str, default = 'imagenet', help = 'image normalization mode [imagenet|standard]',
            choices = ['imagenet', 'standard'])

        self.initialized = True

    
    def auto_set(self):
        '''
        options that will be automatically set
        '''
        # set training status
        self.opt.is_train = self.is_train

        # set gpu_ids
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            g_id = int(str_id)
            if g_id >= 0:
                self.opt.gpu_ids.append(g_id)


    def parse(self, ord_str = None, save_to_file = True, display = True, set_gpu = True):

        if not self.initialized:
            self.initialize()

        if ord_str is None:
            self.opt = self.parser.parse_args()
        else:
            ord_list = ord_str.split()
            self.opt = self.parser.parse_args(ord_list)
            
        self.auto_set()

        if len(self.opt.gpu_ids) > 0 and set_gpu:
            if torch.__version__.startswith('0.2.'):
                # for pytorch v0.2 
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in self.opt.gpu_ids])
                self.opt.gpu_ids = range(len(self.opt.gpu_ids))
                torch.cuda.set_device(0)
            else:
                # for pytorch v0.3 and above
                torch.cuda.set_device(self.opt.gpu_ids[0])
        args = vars(self.opt)
        # display options
        if display:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to disk        
        if save_to_file:
            expr_dir = os.path.join('checkpoints', self.opt.id)
            io.mkdir_if_missing(expr_dir)
            if self.opt.is_train:
                fn_out = os.path.join(expr_dir, 'train_opt.json')
            else:
                fn_out = os.path.join(expr_dir, 'test_opt.json')
            io.save_json(args, fn_out)

        return self.opt

    def load(self, fn):
        args = io.load_json(fn)
        return argparse.Namespace(**args)
