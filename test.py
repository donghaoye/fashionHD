from __future__ import division

from options.pose_transfer_options import TestPoseTransferOptions
from data.data_loader import CreateDataLoader
import numpy as np
import imageio

output_dir = 'temp/vunet_limbs/'

opt = TestPoseTransferOptions().parse()
opt.appearance_type = 'limb'
opt.batch_size = 8

loader = iter(CreateDataLoader(opt, 'test'))
data = next(loader)

img = data['img_1'].numpy().transpose(0,2,3,1)
img = (img * 127.5 + 127.5).clip(0,255).astype(np.uint8)

limb = data['limb_1'].numpy().transpose(0,2,3,1)
limb = (limb * 127.5 + 127.5).clip(0,255).astype(np.uint8)


for i in range(opt.batch_size):
    imageio.imwrite(output_dir + '%d.jpg'%i, img[i])
    for j in range(8):
        imageio.imwrite(output_dir + '%d_%d.jpg'%(i,j), limb[i,:,:,(j*3):(j*3+3)])
	
