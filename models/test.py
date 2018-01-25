from __future__ import division, print_function
import torch
from torch.autograd import Variable
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

def test_DesignerGAN():
    # from designer_gan_model import DesignerGAN
    from designer_gan_model import load_attribute_encoder_net
    net, opt = load_attribute_encoder_net(id = '1.5', gpu_ids = [0,1,2,3], is_train = False)
    
    for k, v in sorted(opt.__dict__.items()):
        print('%s: %s' % (str(k), str(v)))

def test_patchGAN():
    from networks import NLayerDiscriminator

    input_size = 256
    input = Variable(torch.zeros(1,3,input_size,input_size))
    for n in range(1, 7):
        netD = NLayerDiscriminator(input_nc = 3, ndf = 64, n_layers = n)
        output = netD(input)
        print('n_layers=%d, in_size: %d, out_size: %d' % (n, input_size, output.size(2)))


if __name__ == '__main__':
    # test_AttributeEncoder()
    # test_DesignerGAN()
    test_patchGAN()
