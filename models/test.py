from __future__ import division, print_function
import torch
import torch.nn as nn
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

def test_patchGAN_output_size():
    from networks import NLayerDiscriminator

    input_size = 224
    input = Variable(torch.zeros(1,3,input_size,input_size))
    for n in range(1, 7):
        netD = NLayerDiscriminator(input_nc = 3, ndf = 64, n_layers = n)
        output = netD(input)
        print('n_layers=%d, in_size: %d, out_size: %d' % (n, input_size, output.size(2)))

def test_Unet_size():
    from networks import UnetGenerator
    model = UnetGenerator(input_nc = 3, output_nc = 3, num_downs = 7, ngf=64)
    x = Variable(torch.rand(1,3,224,224))
    y = model(x)

def test_scheduler():
    from options.gan_options import TrainGANOptions
    from models.networks import get_scheduler

    model = nn.Linear(10,10)
    optim = torch.optim.Adam(model.parameters(), lr = 1e-4)
    opt = TrainGANOptions().parse('--gpu_ids -1 --benchmark debug', False, False, False)
    scheduler = get_scheduler(optim, opt)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        scheduler.step()
        print('epoch: %d, lr: %f' % (epoch, optim.param_groups[0]['lr']))

def test_feature_spatial_transformer():
    from options.feature_spatial_transformer_options import TrainFeatureSpatialTransformerOptions
    from data.data_loader import CreateDataLoader
    from feature_transform_model import FeatureSpatialTransformer

    opt = TrainFeatureSpatialTransformerOptions().parse('--benchmark debug --batch_size 10 --gpu_ids 0')
    loader = CreateDataLoader(opt)
    data = iter(loader).next()

    model = FeatureSpatialTransformer()
    model.initialize(opt)

    model.set_input(data)
    model.forward()

    errors = model.get_current_errors()
    for k, v in errors.iteritems():
        print('%s: (%s) %s' % (k, type(v), v.size()))

def test_DesignerGAN():
    # from designer_gan_model import DesignerGAN
    from designer_gan_model import DesignerGAN
    from data.data_loader import CreateDataLoader
    from options.gan_options import TrainGANOptions

    opt = TrainGANOptions().parse('--benchmark debug --batch_size 10 --gpu_ids -1')

    loader = CreateDataLoader(opt)
    loader_iter = iter(loader)
    data = loader_iter.next()

    model = DesignerGAN()
    model.initialize(opt)

    model.set_input(data)
    model.forward()

    visuals = model.get_current_visuals()
    for k, v in visuals.iteritems():
        print('%s: (%s) %s' % (k, type(v), v.size()))

def test_MultiModalDesignerGAN():
    from multimodal_designer_gan_model import MultimodalDesignerGAN
    from data.data_loader import CreateDataLoader
    from options.multimodal_gan_options import TrainMMGANOptions
    
    opt = TrainMMGANOptions().parse('--benchmark debug --batch_size 8 --gpu_ids -1 --use_edge --use_color')
    loader = CreateDataLoader(opt)
    loader_iter = iter(loader)
    data = loader_iter.next()

    model = MultimodalDesignerGAN()
    model.initialize(opt)

    model.set_input(data)
    model.forward()

    visuals = model.get_current_visuals()
    for k, v in visuals.iteritems():
        print('%s: (%s) %s' % (k, type(v), v.size()))

def test_decoder_generator():
    import networks
    model = networks.DecoderGenerator(input_nc_1=512, input_nc_2=128, output_nc=3, nblocks_1=1, nups_1=3, nblocks_2=1, nups_2=2, norm='instance', use_dropout=True, gpu_ids=[0])
    x = Variable(torch.rand(1,512,8,8))
    x2 = Variable(torch.rand(1,128,64,64))
    y = model(x, x2)
    print(model)
    print(y.size())

def test_MultiModalDesignerGAN_V2():
    from multimodal_designer_gan_model_v2 import MultimodalDesignerGAN_V2
    from options.multimodal_gan_options_v2 import TrainMMGANOptions_V2
    from data.data_loader import CreateDataLoader

    def count_param(model):
        c = 0
        for module in model.__dict__.values():
            if isinstance(module, nn.Module):
                for p in module.parameters():
                    c += p.numel()
        return c
    # 8x8 feature map, without FTN
    # args = '--benchmark debug --batch_size 8 --G_nblocks_1 3 --gpu_ids 0 --ftn_model none'
    ###### decoder G ######
    # 1x1 feature, without FTN
    # args = '--benchmark debug --batch_size 8 --gpu_ids 0 --encoder_type fc --feat_size_lr 1 --G_nblocks_1 0 --G_nups_1 6  --G_nblocks_2 0 --G_nups_2 2'

    # 8x8 shape feature, 1x1 edge/color feature, without FTN
    # args = '--benchmark debug --batch_size 8 --gpu_ids 0 --encoder_type fc --shape_encoder_type normal --feat_size_lr 8 --G_nblocks_1 0 --G_nups_1 3 --G_nblocks_2 0 --G_nups_2 2'
    
    # 8x8 feature map, with FTN
    # args = '--benchmark debug --batch_size 1 --gpu_ids 0 --encoder_type normal --feat_size_lr 8 --ftn_model reduce --G_nblocks_1 2 --G_nups_1 3 --G_nblocks_2 6 --G_nups_2 2'


    ###### unet G ######
    # args = '--benchmark debug --batch_size 4 --gpu_ids 0,1 --which_model_netG unet --feat_size_lr 8 --G_ndowns 5 --G_nblocks 5 --ftn_model trans --ftn_nblocks 1 --G_block residual'

    

    opt = TrainMMGANOptions_V2().parse(args)
    loader = CreateDataLoader(opt)
    loader_iter = iter(loader)
    data = loader_iter.next()

    model = MultimodalDesignerGAN_V2()
    model.initialize(opt)

    model.set_input(data)
    model.forward()
    model.optimize_parameters()

    print('\n############## model ##############\n')
    for name, net in model.modules.iteritems():
        print(name)
        print(net)
    print('\nparameter size: %d' % count_param(model))
    print('\n############## data ##############\n')
    for k, v in model.output.iteritems():
        print('%s: %s' % (k, v.size()))

def test_UnetResidualGenerator():
    from networks import UnetResidualGenerator

    model = UnetResidualGenerator(input_nc_1=7, input_nc_2=128, output_nc=3, nf=32, nof=128, ndowns=5, nblocks=5, norm='instance', use_dropout=True)
    if model.gpu_ids:
        model.cuda()
    input_1 = Variable(torch.rand(2, 7, 256, 256))
    input_2 = Variable(torch.rand(2, 128, 8, 8))
    y, feat = model(input_1, input_2, 'full')
    y.sum().backward(retain_graph=True)
    feat.sum().backward()
    print('test full')
    print(y.size())
    print(feat.size())

    print('test encode')
    input_1 = Variable(torch.rand(2, 7, 256, 256))
    feat = model(input_1, None, 'encode')
    feat.sum().backward()
    print(feat.size())


def test_V3Model():
    from data.data_loader import CreateDataLoader
    from options.multimodal_gan_options_v3 import TrainMMGANOptions_V3 as Option
    from models.multimodal_designer_gan_model_v3 import MultimodalDesignerGAN_V3 as Model
    from misc.visualizer import GANVisualizer_V3
    
    opt = Option().parse('--debug --batch_size 1 --nThreads 1 --shape_encode flx_seg --shape_with_face 1')
    data = iter(CreateDataLoader(opt)).next()

    visualizer = GANVisualizer_V3(opt)
    
    model = Model()
    model.initialize(opt)
    model.set_input(data)
    model.forward()

    for k, v in model.output.iteritems():
        try:
            print('%s: %s'%(k, v.size()))
        except:
            print('%s' % k)

    visuals = model.get_current_visuals(mode = 'input')
    visualizer.visualize_image(1, subset='input', visuals=visuals)

def test_TwoStagePoseTrasferModel():
    from data.data_loader import CreateDataLoader
    from options.pose_transfer_options import TrainPoseTransferOptions
    from models.two_stage_pose_transfer_model import TwoStagePoseTrasferModel
    opt = TrainPoseTransferOptions().parse()
    model = TwoStagePoseTrasferModel()
    model.initialize(opt)
    # data = iter(CreateDataLoader(opt)).next()

if __name__ == '__main__':
    # test_AttributeEncoder()
    # test_patchGAN_output_size()
    # test_DesignerGAN()
    # test_scheduler()
    # test_Unet_size()
    # test_feature_spatial_transformer()
    # test_MultiModalDesignerGAN()
    # test_upsample_generator()
    # test_MultiModalDesignerGAN_V2()
    # test_UnetResidualGenerator()
    # test_V3Model()
    test_TwoStagePoseTrasferModel()
