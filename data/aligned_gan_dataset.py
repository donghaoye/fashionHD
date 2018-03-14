from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *

import cv2
import PIL
import numpy as np
import os
import util.io as io


class AlignedGANDataset(BaseDataset):
    '''
    Dataset for GAN modeel train/test/visualize. Samples are aligned as [shape_id(target), edge_src, color_src]
    '''

    def name(self):
        return 'AlignedGANDataset'

    def initialize(self, opt, split):
        '''
        split in {'train', 'test', 'debug', 'vis'}
        '''
        self.opt = opt
        self.root = opt.data_root
        if opt.debug:
            split = 'debug'
        self.split = split

        #############################
        # load data
        #############################
        print('loading data ...')
        # data split
        data_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))
        # aligned index
        if split != 'vis':
            aligned_index = io.load_json(os.path.join(opt.data_root, 'Label', 'ca_gan_trainval_upper_aligned_index.json'))
        else:
            aligned_index = io.load_json(os.path.join(opt.data_root, 'Label', 'ca_gan_vis_upper_aligned_index.json'))
            opt.edge_warp_dir = 'Img/edge_ca_256_cloth_tps_vis/'
        # load label
        # self.lm_label = io.load_data(os.path.join(opt.data_root, opt.fn_landmark)) #remove later
        # set data dir
        self.img_dir = os.path.join(opt.data_root, opt.img_dir)
        self.seg_dir = os.path.join(opt.data_root, opt.seg_dir)
        self.flx_seg_dir = os.path.join(opt.data_root, opt.flx_seg_dir)
        self.edge_dir = os.path.join(opt.data_root, opt.edge_dir)
        self.edge_warp_dir = os.path.join(opt.data_root, opt.edge_warp_dir)
        
        #############################
        # create index list
        #############################
        self.id_list = []
        for s_id in data_split[split]:
            edge_ids = aligned_index[s_id]['edge_ids']
            color_ids = aligned_index[s_id]['color_ids']
            if split == 'vis':
                edge_ids = [s_id] + edge_ids
                color_ids = [s_id] + color_ids
            for edge_id in edge_ids:
                for color_id in color_ids:
                    self.id_list.append((s_id, edge_id, color_id))
        print('dataset created (%d anchor, %d samples)' % (len(data_split[split]), len(self.id_list)))
        #############################
        # other
        #############################
        self.to_tensor = transforms.ToTensor()
        self.tensor_normalize_std = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        self.to_pil_image = transforms.ToPILImage()

    def __len__(self):
        return len(self.id_list)

    def read_seg(self, s_id, flx = False):
        try:
            seg_dir = self.flx_seg_dir if flx else self.seg_dir
            fn = os.path.join(seg_dir, s_id + '.bmp')
            seg_map = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float32)[:,:,np.newaxis]
            return seg_map
        except:
            raise Exception('fail to load image %s' % fn)

    def read_image(self, s_id):
        try:
            fn = os.path.join(self.img_dir, s_id + '.jpg')
            img = cv2.imread(fn).astype(np.float32) / 255.
            img = img[:,:,[2,1,0]]
            return img
        except:
            raise Exception('fail to load image %s' % fn)


    def read_edge(self, s_id, src_id = None, warp = False):
        try:
            if warp and (s_id!=src_id):
                fn = os.path.join(self.edge_warp_dir, '%s_%s.jpg' % (s_id, src_id))
            else:
                fn = os.path.join(self.edge_dir, '%s.jpg' % s_id)

            edge_map = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
            edge_map = (edge_map >= self.opt.edge_threshold) * edge_map / 255.
            edge_map = edge_map[:,:,np.newaxis]
            return edge_map
        except:
            raise Exception('fail to load image %s' % fn)

    def __getitem__(self, index):
        s_id, edge_id, color_id = self.id_list[index]
        ######################
        # load seg map
        ######################
        seg_map = self.read_seg(s_id)
        flx_seg_map = self.read_seg(s_id, flx=True)
        # seg_map_edge = self.read_seg(edge_id)
        seg_map_color = self.read_seg(color_id)
        ######################
        # load image
        ######################
        img = self.read_image(s_id)
        img_edge = self.read_image(edge_id)
        img_color = self.read_image(color_id)

        ######################
        # color jitter
        ######################
        if self.opt.color_jitter and self.split == 'train' and self.opt.is_train:
            img_j = self.to_pil_image((img_color*255).astype(np.uint8))
            img_j = self.color_jitter(img_j)
            img_j = self.to_tensor(img_j).numpy().transpose([1,2,0])
            mask = ((seg_map_color==3) | (seg_map_color==4)).astype(np.float32)
            img_color = img_j * mask + img_color * (1-mask)

        ######################
        # load edge
        ######################
        edge_map = self.read_edge(s_id)
        edge_map_src = self.read_edge(s_id, src_id=edge_id, warp=True)
        
        ######################
        # load land mark
        ######################
        # h, w = img.shape[0:2]
        # lm_map = landmark_to_heatmap(img_sz=(w, h), lm_label=self.lm_label[s_id], cloth_type=1)
        
        ######################
        # random flip
        ######################
        if self.split == 'train' and self.opt.is_train:
            coin = np.random.rand()
            seg_map = trans_random_horizontal_flip(seg_map, coin)
            flx_seg_map = trans_random_horizontal_flip(flx_seg_map, coin)
            img = trans_random_horizontal_flip(img, coin)
            img_edge = trans_random_horizontal_flip(img_edge, coin)
            edge_map = trans_random_horizontal_flip(edge_map, coin)
            edge_map_src = trans_random_horizontal_flip(edge_map_src, coin)
            # lm_map = trans_random_horizontal_flip(lm_map, coin)
        ######################
        # get color map
        ######################
        color_map = cv2.GaussianBlur(img, (self.opt.color_gaussian_ksz, self.opt.color_gaussian_ksz), self.opt.color_gaussian_sigma)
        color_map_src = cv2.GaussianBlur(img_color, (self.opt.color_gaussian_ksz, self.opt.color_gaussian_ksz), self.opt.color_gaussian_sigma)
        ######################
        # convert to tensor
        ######################
        t_seg_mask = torch.Tensor(segmap_to_mask_v2(seg_map, nc=7).transpose([2,0,1]))
        t_seg_map = torch.Tensor(seg_map.transpose([2,0,1]))
        t_flx_seg_mask = torch.Tensor(segmap_to_mask_v2(flx_seg_map, nc=7).transpose([2,0,1]))
        # t_flx_seg_map = torch.Tensor(flx_seg_map.transpose([2,0,1]))
        # t_seg_mask_edge = torch.Tensor(segmap_to_mask_v2(seg_map_edge, nc=7).transpose([2,0,1]))
        # t_seg_map_edge = torch.Tensor(seg_map_edge.transpose([2,0,1]))
        # t_seg_mask_color = torch.Tensor(segmap_to_mask_v2(seg_map_color, nc=7).transpose([2,0,1]))
        # t_seg_map_color = torch.Tensor(seg_map_color.transpose([2,0,1]))
        t_edge_map = torch.Tensor(edge_map.transpose([2,0,1]))
        t_edge_map_src = torch.Tensor(edge_map_src.transpose([2,0,1]))
        t_img = self.tensor_normalize_std(self.to_tensor(img))
        t_img_edge = self.tensor_normalize_std(self.to_tensor(img_edge))
        t_img_color = self.tensor_normalize_std(self.to_tensor(img_color))
        # t_lm_map = torch.Tensor(lm_map.transpose([2,0,1]))
        t_color_map = self.tensor_normalize_std(self.to_tensor(color_map))
        t_color_map_src = self.tensor_normalize_std(self.to_tensor(color_map_src))

        if self.opt.color_patch:
            patches = get_color_patch(color_map, seg_map, self.opt.color_patch_mode)
            patches_src = get_color_patch(color_map_src, seg_map_color, self.opt.color_patch_mode)
            t_color_map = torch.cat([self.to_tensor(p) for p in patches], dim=0)
            t_color_map_src = torch.cat([self.to_tensor(p) for p in patches_src], dim=0)
        else:
            # t_color_map = self.tensor_normalize_std(self.to_tensor(color_map))
            # t_color_map_src = self.tensor_normalize_std(self.to_tensor(color_map_src))
            pass

        ######################
        # output dict
        ######################
        data = {
            # from image_tar
            'img': t_img,
            'seg_map': t_seg_map,
            'seg_mask': t_seg_mask,
            'flx_seg_mask': t_flx_seg_mask,
            'edge_map': t_edge_map,
            'color_map': t_color_map,
            # from image_edge
            'img_edge': t_img_edge,
            'edge_map_src': t_edge_map_src,
            # from image_color
            'img_color': t_img_color,
            'color_map_src': t_color_map_src,
            # other
            'id': [s_id, edge_id, color_id]
        }

        # for k, v in data.iteritems():
        #     if k != 'id':
        #         print('%s: %f' % (k, v.mean().item()))

        return data

            



