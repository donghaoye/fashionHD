from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *

import cv2
import PIL
import numpy as np
import os
import util.io as io


class GANDataset(BaseDataset):
    '''
    Dataset for GAN model training and testing
    '''

    def name(self):
        return 'GANDataset'

    def initialize(self, opt, split):
        self.opt = opt
        self.root = opt.data_root
        self.split = split

        print('loading data ...')
        samples = io.load_json(os.path.join(opt.data_root, opt.fn_sample))
        # attr_label = io.load_data(os.path.join(opt.data_root, opt.fn_label))
        # attr_entry = io.load_json(os.path.join(opt.data_root, opt.fn_entry))
        data_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))
        lm_label = io.load_data(os.path.join(opt.data_root, opt.fn_landmark))
        seg_paths = io.load_json(os.path.join(opt.data_root, opt.fn_seg_path))
        edge_paths = io.load_json(os.path.join(opt.data_root, opt.fn_edge_path))
        # color_paths = io.load_json(os.path.join(opt.data_root, opt.fn_color_path))

        self.id_list = data_split[split]
        # self.attr_entry = attr_entry
        if opt.max_dataset_size != float('inf'):
            self.id_list = self.id_list[0:opt.max_dataset_size]
        self.sample_list = [samples[s_id] for s_id in self.id_list]
        # self.attr_label_list = [attr_label[s_id] for s_id in self.id_list]
        self.lm_list = [lm_label[s_id] for s_id in self.id_list]
        self.seg_path_list = [seg_paths[s_id] for s_id in self.id_list]
        self.edge_path_list = [edge_paths[s_id] for s_id in self.id_list]
        # self.color_path_list = [color_paths[s_id] for s_id in self.id_list]


        # check data
        # assert len(self.attr_entry) == len(self.attr_label_list[0]) == opt.n_attr, 'Attribute number not match!'
        print('dataset created (%d samples)' % len(self))

        # get transform
        self.to_tensor = transforms.ToTensor()

        # use standard normalization, which is different from attribute dataset
        # image will be normalized again (under imagenet distribution) before fed into attribute encoder in GAN model
        self.tensor_normalize_std = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.tensor_normalize_imagenet = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        if self.opt.color_jitter:
            self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
            self.to_pil_image = transforms.ToPILImage()

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        s_id = self.id_list[index]

        # load segmentation map
        seg_map = cv2.imread(self.seg_path_list[index], cv2.IMREAD_GRAYSCALE)

        # load image
        img_uint = cv2.imread(self.sample_list[index]['img_path'])
        if img_uint.ndim == 3:
            # convert BRG to RBG
            img_uint = img_uint[:,:,[2,1,0]]
        img = img_uint.astype(np.float32) / 255.
        if self.opt.color_jitter and self.split == 'train':
            img_j = self.to_pil_image(img_uint)
            img_j = self.color_jitter(img_j)
            img_j = self.to_tensor(img_j).numpy().transpose([1,2,0])
            mask = ((seg_map==3) | (seg_map==4)).astype(np.float32)[:,:,np.newaxis]
            img = img_j * mask + img * (1-mask)

        # create landmark heatmap
        h, w = img.shape[0:2]
        lm_map = landmark_to_heatmap(
            img_sz = (w, h),
            lm_label = self.lm_list[index],
            cloth_type = self.sample_list[index]['cloth_type']
            )

        # load edge map
        edge_map = cv2.imread(self.edge_path_list[index], cv2.IMREAD_GRAYSCALE)
        edge_map = (edge_map >= self.opt.edge_threshold) * edge_map
        edge_map = edge_map.astype(np.float32)[:,:,np.newaxis] / 255.
        # load color map
        # color_map = cv2.imread(self.color_path_list[index]).astype(np.float32) / 255.
        # create color map on the fly
        color_map = cv2.GaussianBlur(img, (self.opt.color_gaussian_ksz, self.opt.color_gaussian_ksz), self.opt.color_gaussian_sigma)            
        nc_img, nc_lm, nc_edge, nc_color = img.shape[-1], lm_map.shape[-1], edge_map.shape[-1], color_map.shape[-1]
        mix = np.concatenate((img, lm_map, edge_map, color_map), axis = 2)

        # transform
        if self.opt.resize_or_crop == 'resize':
            # only resize
            mix = trans_resize(mix, size = (self.opt.fine_size, self.opt.fine_size))
            seg_map = trans_resize(seg_map, size =(self.opt.fine_size, self.opt.fine_size), interp = cv2.INTER_NEAREST)
            mix = np.concatenate((mix, seg_map[:,:,np.newaxis]), axis = 2)
        elif self.opt.resize_or_crop == 'resize_and_crop':
            mix = trans_resize(mix, size = (self.opt.load_size, self.opt.load_size))
            seg_map = trans_resize(seg_map, size =(self.opt.load_size, self.opt.load_size), interp = cv2.INTER_NEAREST)
            mix = np.concatenate((mix, seg_map[:,:,np.newaxis]), axis = 2)
            if self.split == 'train':
                mix = trans_random_crop(mix, size = (self.opt.fine_size, self.opt.fine_size))
                mix = trans_random_horizontal_flip(mix)
            else:
                mix = trans_center_crop(mix, size = (self.opt.fine_size, self.opt.fine_size))

        seg_map = mix[:,:,-1::]
        seg_mask = segmap_to_mask(seg_map, self.opt.input_mask_mode, self.sample_list[index]['cloth_type'])
        t_seg_mask = torch.Tensor(seg_mask.transpose([2, 0, 1]))
        t_seg_map = torch.Tensor(seg_map.transpose([2, 0, 1]))

        img = mix[:,:,0:nc_img]
        t_img = self.to_tensor(img)
        t_img = self.tensor_normalize_std(t_img)

        lm_map = mix[:,:,nc_img:(nc_img+nc_lm)]
        t_lm_map = torch.Tensor(lm_map.transpose([2, 0, 1])) # convert to CxHxW

        edge_map = mix[:,:,(nc_img+nc_lm):(nc_img+nc_lm+nc_edge)]
        t_edge_map = torch.Tensor(edge_map.transpose([2, 0, 1])) # convert to CxHxW

        color_map = mix[:,:,(nc_img+nc_lm+nc_edge):(nc_img+nc_lm+nc_edge+nc_color)]
        if self.opt.color_patch:
            t_color_map_full = self.tensor_normalize_std(self.to_tensor(color_map))
            patches = get_color_patch(color_map, seg_map, self.opt.color_patch_mode)
            color_map = np.concatenate(patches, axis=2)
            t_color_map = [self.to_tensor(p) for p in patches]
            t_color_map = torch.cat(t_color_map, dim=0)
        else:
            t_color_map = self.tensor_normalize_std(self.to_tensor(color_map))
            t_color_map_full = t_color_map.clone()

        # load label
        # att = np.array(self.attr_label_list[index], dtype = np.float32)

        data = {
            'img': t_img,
            'lm_map': t_lm_map,
            'seg_mask': t_seg_mask,
            'seg_map': t_seg_map,
            'edge_map': t_edge_map,
            'color_map': t_color_map,
            'color_map_full': t_color_map_full,
            # 'attr_label':att,
            'id': s_id
        }

        # affine augmentatin
        if self.opt.affine_aug:
            if self.split == 'train':
                if 'lm' in self.opt.shape_encode:
                    edge_map_affine, color_map_affine, seg_mask_affine, lm_map_affine = trans_random_affine([edge_map, color_map, seg_mask, lm_map], self.opt.affine_aug_scale)
                else:
                    edge_map_affine, color_map_affine, seg_mask_affine = trans_random_affine([edge_map, color_map, seg_mask], self.opt.affine_aug_scale)
                    lm_map_affine = lm_map #don't comput affine transformation of lm_map for efficiency

                data['edge_map_aug'] = torch.Tensor(edge_map_affine.transpose([2,0,1]))
                data['color_map_aug'] = self.to_tensor(color_map_affine)# has been normalized, no need to do it again
                data['seg_mask_aug'] = torch.Tensor(seg_mask_affine.transpose([2,0,1]))
                data['lm_map_aug'] = torch.Tensor(lm_map_affine.transpose([2,0,1]))
            else:
                data['edge_map_aug'] = data['edge_map']
                data['color_map_aug'] = data['color_map']
                data['seg_mask_aug'] = data['seg_mask']
                data['lm_map_aug'] = data['lm_map']
        return data

