from __future__ import division
import torch
import torchvision.transforms as transforms
from base_dataset import *
from misc.pose_util import get_joint_coord

import cv2
import PIL
import numpy as np
import os
import util.io as io

class DomainTransferDataset(BaseDataset):
    def name(self):
        return 'DomainTransferDataset'

    def initialize(self, opt, split):
        self.opt = opt
        self.root = opt.data_root
        if opt.debug:
            split = 'debug'
        self.split = split
        #############################
        # load data
        #############################
        print('loading data ...')
        data_split = io.load_json(os.path.join(opt.data_root, opt.fn_split))
        self.img_dir_1 = os.path.join(opt.data_root, opt.img_dir_1)
        self.img_dir_2 = os.path.join(opt.data_root, opt.img_dir_2)
        self.seg_dir = os.path.join(opt.data_root, opt.seg_dir)
        self.pose_label = io.load_data(os.path.join(opt.data_root, opt.fn_pose))
        #############################
        # create index list
        #############################
        self.id_list = data_split[split]
        #############################
        # output content
        #############################
        self.output_joint = True
        self.output_stickman = 'stickman' in opt.pose_type
        self.output_seg = False
        self.output_limb = False
        #############################
        # other
        #############################
        self.tensor_normalize_std = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        #############################
        # define body limbs
        #############################
        self.bparts = [
            ['rshoulder', 'rhip', 'lhip', 'lshoulder'],
            ['rshoulder', 'lshoulder', 'nose'],
            ['rshoulder', 'relbow'],
            ['relbow', 'rwrist'],
            ['lshoulder', 'lelbow'],
            ['lelbow', 'lwrist'],
            ['rhip', 'rknee'],
            ['lhip', 'lknee']]

    def __len__(self):
        return len(self.id_list)

    def to_tensor(self, img):
        return torch.Tensor(img.transpose((2, 0, 1)))

    def read_image(self, s_id, domain=1):
        if domain == 1:
            fn = os.path.join(self.img_dir_1, s_id + '.jpg')
        else:
            fn = os.path.join(self.img_dir_2, s_id + '.jpg')
        img = cv2.imread(fn).astype(np.float32) / 255.
        img = img[:,:,[2,1,0]]
        return img

    def read_seg(self, s_id):
        fn = os.path.join(self.seg_dir, s_id + '.bmp')
        seg = cv2.imread(fn, cv2.IMREAD_GRAYSCALE).astype(np.float32)[:,:,np.newaxis]
        return seg

    def get_limb_crop(self, img, coords, w, h, o_w, o_h):
        '''
        img: (h,w,3) np.ndarray
        coords: [[x0, y0], ..., [x17, y17]]
        '''

        trans_mats = []
        coords = np.array(coords, dtype=np.float32)
        for bpart in self.bparts:
            part_src = get_joint_coord(coords, bpart)
            # fall backs 
            if not (part_src>=0).all():
                fallback = True
                if bpart[0] == 'lhip' and bpart[1] == 'lknee':
                    bpart = ['lhip']
                elif bpart[0] == 'rhip' and bpart[1] == 'rknee':
                    bpart = ['rhip']
                elif bpart[0] == 'rshoulder' and bpart[1] == 'lshoulder' and bpart[2] == 'nose':
                    bpart = ['rshoulder', 'lshoulder', 'lshoulder']
                else:
                    fallback = False

                if fallback:
                    part_src = get_joint_coord(coords, bpart)

            if not (part_src>=0).all():
                trans_mats.append(None)
                continue

            if part_src.shape[0] == 1:
                # leg fallback
                a = part_src[0]
                b = np.float32([a[0], o_h-1])
                part_src = np.float32([a,b])

            if part_src.shape[0] == 4:
                pass
            elif part_src.shape[0] == 3:
                # lshoulder, rshoulder, nose/rshoulder
                if bpart[2] == 'lshoulder':
                    segment = part_src[1] - part_src[0]
                    normal = np.array([-segment[1], segment[0]])
                    if normal[1] > 0:
                        normal = -normal
                    a = part_src[0] + normal
                    b = part_src[0]
                    c = part_src[1]
                    d = part_src[1] + normal
                    part_src = np.float32([a,b,c,d])
                else:
                    assert bpart[2] == 'nose'
                    neck = 0.5*(part_src[0]+part_src[1])
                    neck_to_nose = part_src[2] - neck
                    part_src = np.float32([neck + 2*neck_to_nose, neck])

                    segment = part_src[1] - part_src[0]
                    normal = np.array([-segment[1], segment[0]])
                    # a = part_src[0] + 0.5*normal
                    # b = part_src[0] - 0.5*normal
                    # c = part_src[1] - 0.5*normal
                    # d = part_src[1] + 0.5*normal
                    # part_src = np.float32([b,c,d,a])
                    a = part_src[0] + 0.5*normal
                    b = part_src[1] + 0.5*normal
                    c = part_src[1] - 0.5*normal
                    d = part_src[0] - 0.5*normal
                    part_src = np.float32([a,b,c,d])
            else:
                assert part_src.shape[0] == 2
                segment = part_src[1] - part_src[0]
                normal = np.array([-segment[1], segment[0]])
                # a = part_src[0] + 0.25*normal
                # b = part_src[0] - 0.25*normal
                # c = part_src[1] - 0.25*normal
                # d = part_src[1] + 0.25*normal
                a = part_src[0] + 0.25*normal
                b = part_src[1] + 0.25*normal
                c = part_src[1] - 0.25*normal
                d = part_src[0] - 0.25*normal
                part_src = np.float32([a,b,c,d])

            part_dst = np.float32([[0,0], [0,h], [w, h], [w, 0]])
            M = cv2.getPerspectiveTransform(part_src, part_dst)
            trans_mats.append(M)

        # normalization
        crops = []
        for M in trans_mats:
            crop = np.zeros((h, w, 3))
            if M is not None:
                crop = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            crops.append(crop)
        crops = np.concatenate(crops, axis=2)
        return crops


    def __getitem__(self, index):
        s_id = self.id_list[index]
        ######################
        # load image
        ######################
        img_1 = self.read_image(s_id, 1)
        img_2 = self.read_image(s_id, 2)
        joint_c = self.pose_label[s_id]
        if self.output_seg:
            seg = self.read_seg(s_id)
        ######################
        # augmentation
        ######################
        if self.split == 'train' and self.opt.is_train:
            # flip person image
            coin = np.random.rand()
            img_1 = trans_random_horizontal_flip(img_1, coin)
            joint_c = trans_random_horizontal_flip_pose_c(joint_c, (img_1.shape[1], img_1.shape[0]), coin)
            if self.output_seg:
                seg = trans_random_horizontal_flip(seg, coin)
        ######################
        # create pose representation
        ######################
        if self.output_joint:
            joint = pose_to_map(img_sz=(img_1.shape[1], img_1.shape[0]), label=joint_c, mode=self.opt.joint_mode, radius=self.opt.joint_radius)
        if self.output_stickman:
            stickman = pose_to_stickman(img_sz=(img_1.shape[1], img_1.shape[0]), label=joint_c)
        ######################
        # create limb crops
        ######################
        if self.output_limb:
            box_factor = self.opt.vunet_box_factor if 'vunet_box_factor' in self.opt else 0
            o_h, o_w = img_1.shape[0:2]
            w = o_w // 2**pose_box_factor
            h = o_h // 2**pose_box_factor
            limb = self.get_limb_crop(img_1, joint_c, w, h, o_w, o_h)
            # normalize
            limb = (limb - 0.5)/0.5
        ######################
        # convert to tensor
        ######################
        data = {
            'img_1': self.tensor_normalize_std(self.to_tensor(img_1)),
            'img_2': self.tensor_normalize_std(self.to_tensor(img_2)),
            'joint_c': torch.Tensor(joint_c),
            'id': s_id
        }
        if self.output_joint:
            data['joint'] = self.to_tensor(joint)
        if self.output_stickman:
            data['stickman'] = self.to_tensor(stickman)
        if self.output_seg:
            data['seg'] = self.to_tensor(seg)
            data['seg_mask'] = self.to_tensor(segmap_to_mask_v2(seg, nc=7, bin_sizse=self.opt.seg_bin_size))
        if self.output_limb:
            data['limb'] = self.to_tensor(limb)

        return data


