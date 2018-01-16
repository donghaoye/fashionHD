from __future__ import division, print_function

import torch    
from util.timer import Timer

def test_AttributeDataset():
    from data_loader import CreateDataLoader
    from options.attribute_options import TrainAttributeOptions, TestAttributeOptions

    timer = Timer()
    
    timer.tic()
    opt = TrainAttributeOptions().parse()
    loader = CreateDataLoader(opt)
    print('cost %.3f sec to create data loader.' % timer.toc())

    loader_iter = iter(loader)
    data = loader_iter.next()

    for k, v in data.iteritems():
        print('data["%s"]: %s' % (k, type(v)))


if __name__ == '__main__':
    test_AttributeDataset()
