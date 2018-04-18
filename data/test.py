from __future__ import division, print_function

import torch    
import torchvision
import util.image as image
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


def test_EXPAttributeDataset():
    from data_loader import CreateDataLoader
    from options.attribute_options import TrainAttributeOptions


    opt = TrainAttributeOptions().parse('--dataset_mode attribute_exp --benchmark debug --batch_size 10')
    loader = CreateDataLoader(opt)
    loader_iter = iter(loader)
    data = loader_iter.next()

    for k, v in data.iteritems():
        print('data["%s"], type: %s' % (k, type(v)))
        try:
            print(v.size())
        except:
            pass

    for idx in range(opt.batch_size):
        # show image
        img = torchvision.utils.make_grid(data['img'], nrow=5, normalize = True).cpu().numpy()
        img = img.transpose([1,2,0])
        img = img[:,:,[2,1,0]] # from RGB to BGR
        image.imshow(img)

        # show heat map
        img = data['img'][idx] # 3xHxW
        lm_maps = data['landmark_heatmap'][idx] # 18xHxW

        print(lm_maps.min())
        print(lm_maps.max())

        img_maps = []
        
        for i in range(lm_maps.size(0)):
            img_map = img * lm_maps[i]
            img_maps.append(img_map)
        img_maps = torch.stack(img_maps)
        img_maps = torchvision.utils.make_grid(img_maps, nrow=9, normalize = True).cpu().numpy()
        img_maps = img_maps.transpose([1,2,0])
        img_maps = img_maps[:,:,[2,1,0]] # from RGB to BGR
        image.imshow(img_maps)

def test_GANDataset():
    from data_loader import CreateDataLoader
    from options.gan_options import TrainGANOptions

    opt = TrainGANOptions().parse('--benchmark debug --batch_size 1')
    loader = CreateDataLoader(opt)
    loader_iter = iter(loader)
    data = loader_iter.next()

    for k, v in data.iteritems():
        if isinstance(v, torch.Tensor):
            print('[%s]: (%s), %s' % (k,type(v), v.size()))
        else:
            print('[%s]: %s' % (k, type(v)))

    img = torchvision.utils.make_grid(data['img'], nrow=5, normalize = True).cpu().numpy()
    img = img.transpose([1,2,0])
    img = img[:,:,[2,1,0]] # from RGB to BGR
    if data['seg_mask'].size(1) > 1:
        data['seg_mask'] = data['seg_mask'].max(dim=1, keepdim=True)
    image.imshow(img)
    for idx in range(opt.batch_size):
        
        # show samples
        img = data['img'][idx] # 3xHxW
        lm_maps = data['lm_map'][idx] # 18xHxW

        img_maps = []
        
        img_maps.append(img)
        img_maps.append(img * data['seg_mask'][idx,0])
        for i in range(lm_maps.size(0)):
            img_map = img * lm_maps[i]
            img_maps.append(img_map)

        img_maps = torch.stack(img_maps)
        img_maps = torchvision.utils.make_grid(img_maps, nrow=10, normalize = True).cpu().numpy()
        img_maps = img_maps.transpose([1,2,0])
        img_maps = img_maps[:,:,[2,1,0]] # from RGB to BGR
        image.imshow(img_maps)

def test_AlignedGANDataset():
    from data_loader import CreateDataLoader
    from options.multimodal_gan_options_v3 import TrainMMGANOptions_V3

    opt = TrainMMGANOptions_V3().parse('--debug --batch_size 1 --nThreads 1')
    loader = CreateDataLoader(opt, 'train')
    loader_iter = iter(loader)
    data = loader_iter.next()

    for k, v in data.iteritems():
        if isinstance(v, torch.Tensor):
            print('[%s]: (%s), %s' % (k,type(v), v.size()))
        else:
            print('[%s]: %s' % (k, type(v)))


def test_GANDataset_V2():
    from data_loader import CreateDataLoader
    import argparse
    ################
    # set opt
    ################
    opt = argparse.Namespace()
    opt.debug = False
    opt.is_train = True
    opt.dataset_mode = 'gan_v2'
    opt.batch_size = 4
    opt.nThreads = 1
    # path
    opt.data_root = 'datasets/Zalando/'
    opt.img_dir = 'Img/img_zalando_256/'
    opt.seg_dir = 'Img/seg_zalando_256/'
    opt.edge_dir = 'Img/edge_zalando_256_cloth/'
    opt.fn_split = 'Split/zalando_split.json'
    opt.fn_pose = 'Label/zalando_pose_label_256.pkl'
    # shape
    opt.seg_bin_size = 16
    # edge
    opt.edge_threshold = 0
    # color
    opt.color_gaussian_ksz = 15
    opt.color_gaussian_sigma = 10.0
    opt.color_bin_size = 16
    opt.color_jitter = True
    # pose
    opt.pose_size = 11
    # deformatin
    opt.shape_deformation_scale = 0.1
    opt.shape_deformation_flip = True

    ################
    # create dataset
    ################
    loader = CreateDataLoader(opt, 'train')
    loader_iter = iter(loader)

    ################
    # visualize one mini batch
    ################
    data = loader_iter.next()
    for k, v in data.iteritems():
        if isinstance(v, torch.Tensor):
            print('[%s]: (%s), %s' % (k,type(v), v.size()))
        else:
            print('[%s]: %s' % (k, type(v)))

    torch.save(data, 'temp.pth')


def test_PoseTransferDataset():
    from data_loader import CreateDataLoader
    import argparse
    ################
    # set opt
    ################
    opt = argparse.Namespace()
    opt.debug = False
    opt.is_train = True
    opt.dataset_mode = 'pose_transfer'
    opt.batch_size = 4
    opt.nThreads = 1
    # path
    opt.data_root = 'datasets/DF_Pose/'
    opt.fn_split = 'Label/pair_split.json'
    opt.img_dir = 'Img/img_df/'
    opt.sge_dir = 'Img/seg_df/'
    opt.fn_pose = 'Label/pose_label.pkl'
    # pose
    opt.pose_radius = 5
    
    ################
    # create dataset
    ################
    loader = CreateDataLoader(opt, 'train')
    loader_iter = iter(loader)
    ################
    # visualize one mini batch
    ################
    data = loader_iter.next()
    for k, v in data.iteritems():
        if isinstance(v, torch.Tensor):
            print('[%s]: (%s), %s' % (k,type(v), v.size()))
        else:
            print('[%s]: %s' % (k, type(v)))




if __name__ == '__main__':
    # test_AttributeDataset()
    # test_EXPAttributeDataset()
    # test_GANDataset()
    # test_AlignedGANDataset()
    # test_GANDataset_V2()
    test_PoseTransferDataset()
