from __future__ import division, print_function

import torch
import os

class BaseModel(object):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join('checkpoints', opt.id)

        self.input = {}
        self.output = {}

    def set_input(self, data):
        self.input = data

    def forward(self):
        pass

    # used in test time, no backprob
    def test(self):
        pass

    def optimize_parameters(self):
        pass

    
    def get_current_visuals(self):
        return self.input

    
    def get_current_errors(self):
        return {}

    def train(self):
        pass

    def eval(self):
        pass

    def save(self, label):
        pass


    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)

        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

        # lr = self.optimizers[0].param_groups[0]['lr']
        # print('learning rate = %.7f' % lr)