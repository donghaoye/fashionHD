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
        parser.add_argument('--dataset_mode', type = str, default = 'attribute', help = 'type of dataset',
            choices = ['attribute'])
        parser.add_argument('--data_root', type = str, default = './datasets/DeepFashion/Fashion_design/', help = 'data root path')
        parser.add_argument('--nThreads', type = int, default = 2, help = 'number of workers to load data')
        parser.add_argument('--shuffle', type = int, default = 0, help = 'shuffle dataset [1:True|-1:False|0:Auto]',
            choices = [0,1,-1])
        parser.add_argument('--flip', type = int, default = 0, help = 'flip images [1:True|-1:False|0:Auto]',
            choices = [0,1,-1])
        parser.add_argument('--max_dataset_size', type = int, default = float('inf'), help = 'maximum number of samples')
        parser.add_argument('--batch_size', type = int, default = 128, help = 'batch size')
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

        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        # set dataset options
        if self.opt.shuffle == 0:
            self.opt.shuffle = 1 if self.is_train else -1
        if self.opt.flip == 0:
            self.opt.flip = 1 if self.is_train else -1



    def parse(self):

        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
        self.auto_set()       

        args = vars(self.opt)

        # display options
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to disk
        expr_dir = os.path.join('checkpoints', self.opt.id)
        io.mkdir_if_missing(expr_dir)
        
        if self.opt.is_train:
            fn_out = os.path.join(expr_dir, 'train_opt.json')
        else:
            fn_out = os.path.join(expr_dir, 'test_opt.json')
        io.save_json(args, fn_out)

        return self.opt
