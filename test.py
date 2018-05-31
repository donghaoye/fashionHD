import torch
import time
from misc.color_space import *

rgb = torch.rand(16,3,256,256)
rgb = (rgb-0.5)/0.5
rgb = rgb.cuda(0)

t = time.time()
for i in range(10):
    lab = rgb2lab(rgb)
print('rgb to lab: %f' % ((time.time()-t)/10.))

t = time.time()
for i in range(10):
    rgb = lab2rgb(lab)
print('lab to rgb: %f' % ((time.time()-t)/10.))
