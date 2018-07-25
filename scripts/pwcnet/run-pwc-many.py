import sys
import cv2
import torch
import numpy as np
from math import ceil
from torch.autograd import Variable
from scipy.ndimage import imread
import tqdm
pwc_dir = '/data2/ynli/patch_matching_and_flow/PWC-Net/PyTorch/'
sys.path.append(pwc_dir)
import models
"""
Contact: Deqing Sun (deqings@nvidia.com); Zhile Ren (jrenzhile@gmail.com)
"""
def writeFlowFile(filename,uv):
	"""
	According to the matlab code of Deqing Sun and c++ source code of Daniel Scharstein  
	Contact: dqsun@cs.brown.edu
	Contact: schar@middlebury.edu
	"""
	TAG_STRING = np.array(202021.25, dtype=np.float32)
	if uv.shape[2] != 2:
		sys.exit("writeFlowFile: flow must have two bands!");
	H = np.array(uv.shape[0], dtype=np.int32)
	W = np.array(uv.shape[1], dtype=np.int32)
	with open(filename, 'wb') as f:
		f.write(TAG_STRING.tobytes())
		f.write(W.tobytes())
		f.write(H.tobytes())
		f.write(uv.tobytes())


# load file list
pair_list = []
with open(sys.argv[1]) as f:
	for line in f.readlines():
		if line.strip() != '':
			pair_list.append(line.split())


# load model
pwc_model_fn = pwc_dir + 'pwc_net.pth.tar';
net = models.pwc_dc_net(pwc_model_fn)
net = net.cuda()
net.eval()

for im1_fn, im2_fn, flow_fn in tqdm.tqdm(pair_list):
	im_all = [imread(img) for img in [im1_fn, im2_fn]]
	im_all = [im[:, :, :3] for im in im_all]

	# rescale the image size to be multiples of 64
	divisor = 64.
	H = im_all[0].shape[0]
	W = im_all[0].shape[1]

	H_ = int(ceil(H/divisor) * divisor)
	W_ = int(ceil(W/divisor) * divisor)
	for i in range(len(im_all)):
		im_all[i] = cv2.resize(im_all[i], (W_, H_))

	for _i, _inputs in enumerate(im_all):
		im_all[_i] = im_all[_i][:, :, ::-1]
		im_all[_i] = 1.0 * im_all[_i]/255.0
		
		im_all[_i] = np.transpose(im_all[_i], (2, 0, 1))
		im_all[_i] = torch.from_numpy(im_all[_i])
		im_all[_i] = im_all[_i].expand(1, im_all[_i].size()[0], im_all[_i].size()[1], im_all[_i].size()[2])	
		im_all[_i] = im_all[_i].float()
	    
	im_all = torch.autograd.Variable(torch.cat(im_all,1).cuda(), volatile=True)

	flo = net(im_all)
	flo = flo[0] * 20.0
	flo = flo.cpu().data.numpy()

	# scale the flow back to the input size 
	flo = np.swapaxes(np.swapaxes(flo, 0, 1), 1, 2) # 
	u_ = cv2.resize(flo[:,:,0],(W,H))
	v_ = cv2.resize(flo[:,:,1],(W,H))
	u_ *= H/ float(H_)
	v_ *= W/ float(W_)
	flo = np.dstack((u_,v_))

	writeFlowFile(flow_fn, flo)
