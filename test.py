from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import util.inception_score
import util.io as io
import util.image as image

design_root = 'datasets/DeepFashion/Fashion_design/'

id_list = io.load_json(design_root + 'Split/ca_gan_split_trainval_upper.json')['train']
imgs = [image.imread(design_root + 'Img/img_ca_256/' + s_id + '.jpg') for s_id in id_list[0:1000]]

s,v = util.inception_score.get_inception_score(imgs)
print s, v