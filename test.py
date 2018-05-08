from __future__ import division

import torch
from options.domain_transfer_options import TestDomainTransferOptions
from data.data_loader import CreateDataLoader
import util.image as image

opt = TestDomainTransferOptions().parse()
loader = iter(CreateDataLoader(opt, 'test'))

data = next(loader)
print(data.keys())

for k, d in(data.iteritems()):
    if isinstance(d, torch.Tensor):
        print('%s: %s'%(k, str(d.size())))

img2 = data['img_2'].numpy().transpose(0,2,3,1)[:,:,:,[2,1,0]]
img2 = (img2 + 1)/2

image.imshow(img2[0])
