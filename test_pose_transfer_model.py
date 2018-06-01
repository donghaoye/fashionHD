from __future__ import division, print_function

import torch
from data.data_loader import CreateDataLoader
from options.pose_transfer_options import TestPoseTransferOptions
from misc.visualizer import GANVisualizer_V3
from misc.loss_buffer import LossBuffer

import util.io as io
import os
import sys
import time
import numpy as np
import cv2
from collections import OrderedDict


opt = TestPoseTransferOptions().parse()
train_opt = io.load_json(os.path.join('checkpoints', opt.id, 'train_opt.json'))
preserved_opt = {'gpu_ids', 'batch_size', 'is_train'}
for k, v in train_opt.iteritems():
    if k in opt and (k not in preserved_opt):
        setattr(opt, k, v)
# create model
if opt.which_model_T in {'unet', 'resnet'}:
    from models.supervised_pose_transfer_model import SupervisedPoseTransferModel
    model = SupervisedPoseTransferModel()
elif opt.which_model_T == 'vunet':
    from models.vunet_pose_transfer_model import VUnetPoseTransferModel
    model = VUnetPoseTransferModel()
elif opt.which_model_T == '2stage':
    from models.two_stage_pose_transfer_model import TwoStagePoseTransferModel
    model = TwoStagePoseTransferModel()
else:
    raise NotImplementedError()

model.initialize(opt)
# create data loader
# train_loader = CreateDataLoader(opt, split = 'train')
val_loader = CreateDataLoader(opt, split = 'test')
# create visualizer
visualizer = GANVisualizer_V3(opt)

pavi_upper_list = ['PSNR', 'SSIM']
pavi_lower_list = ['loss_L1', 'loss_content', 'loss_style', 'loss_G', 'loss_D', 'loss_pose', 'loss_kl']

# visualize
if opt.nvis > 0:
    print('visulizing first %d samples' % opt.nvis)
    num_vis_batch = int(np.ceil(1.0*opt.nvis/opt.batch_size))
    val_visuals = None
    for i, data in enumerate(val_loader):
        if i == num_vis_batch:
            break
        model.set_input(data)
        model.test(compute_loss=False)
        visuals = model.get_current_visuals()
        if val_visuals is None:
            val_visuals = visuals
        else:
            for name, v in visuals.iteritems():
                val_visuals[name][0] = torch.cat((val_visuals[name][0], v[0]),dim=0)
    visualizer.visualize_image(epoch = opt.which_epoch, subset = 'test', visuals = val_visuals)
        
if opt.vis_only:
    exit()


# test
loss_buffer = LossBuffer(size=len(val_loader))
if opt.save_output:
    img_dir = os.path.join(model.save_dir, 'test')
    io.mkdir_if_missing(img_dir)

for i, data in enumerate(val_loader):
    if opt.nbatch >= 0 and i == opt.nbatch:
        break

    model.set_input(data)
    model.test(compute_loss=False)
    loss_buffer.add(model.get_current_errors())
    print('\rTesting %d/%d (%.2f%%)' % (i, len(val_loader), 100.*i/len(val_loader)), end = '')
    sys.stdout.flush()
    # save output
    if opt.save_output:
        id_list = model.input['id']
        images = model.output['img_out'].cpu().numpy().transpose(0,2,3,1)
        images = ((images + 1.0) * 127.5).clip(0,255).astype(np.uint8)
        for (id1, id2), img in zip(id_list, images):
            img = img[:,:,[2,1,0]] # convert to BGR channel order for cv2
            cv2.imwrite(os.path.join(img_dir,'%s_%s.jpg' % (id1, id2)), img)
print('\n')

test_error = loss_buffer.get_errors()
visualizer.print_error(test_error)
