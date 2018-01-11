from __future__ import division, print_function
import torch
from util.timer import Timer

def test_AttributeEncoder():
    from attribute_encoder import AttributeEncoder
    from data.data_loader import CreateDataLoader
    from options.attribute_options import TrainAttributeOptions, TestAttributeOptions

    
    mode = 'train'

    timer = Timer()
    timer.tic()

    if mode == 'train':
        opt = TrainAttributeOptions().parse()
        model = AttributeEncoder()
        model.initialize(opt)

        timer.tic()
        loader = CreateDataLoader(opt)
        loader_iter = iter(loader)
        print('cost %.3f sec to create data loader.' % timer.toc())

        data = loader_iter.next()

        model.set_input(data)
        model.optimize_parameters()
        error = model.get_current_errors()
        print(error)

        model.save_network(model.net, 'AE', 'ep0', model.opt.gpu_ids)
        

if __name__ == '__main__':
    test_AttributeEncoder()
