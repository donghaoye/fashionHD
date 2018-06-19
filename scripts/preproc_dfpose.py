from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np
import util.io as io
import imageio
from collections import defaultdict, OrderedDict

root = 'datasets/DF_Pose/'

def create_image_index():
    ''' create image index, split and pose label from original index.p file '''
    index = io.load_data(root + 'Anno/index.pkl')

    split = {'train': [], 'test': []}
    pose_label = {}

    for img, joints, is_train in zip(index['imgs'], index['joints'], index['train']):
        s_id = img.split('/')[1][0:7]
        if is_train:
            split['train'].append(s_id)
        else:
            split['test'].append(s_id)

        for i in range(len(joints)):
            if not (joints[i][0] == -1 and joints[i][1] == -1):
                joints[i][0] *= 256
                joints[i][1] *= 256
        pose_label[s_id] = joints.tolist()

    io.save_json(split, root + 'Label/split.json')
    io.save_data(pose_label, root + 'Label/pose_label.pkl')


def create_pair_index():
    '''
    create pair index
    '''
    from itertools import combinations

    split = io.load_json(root + 'Label/split.json')
    pair_split = {'train': [], 'test': []}
    for subset in ['train', 'test']:
        cloth = defaultdict(lambda : [])
        for s_id in split[subset]:
            cloth[s_id[0:5]].append(s_id)

        for group in cloth.values():
            pair_split[subset] += combinations(group, 2)

    np.random.shuffle(pair_split['train'])
    np.random.shuffle(pair_split['test'])
    pair_split['debug'] = pair_split['train'][0:32]
    io.save_json(pair_split, root + 'Label/pair_split.json')
    print('train pair: %d' % len(pair_split['train']))
    print('test pair: %d' % len(pair_split['test']))


def load_test_pair_index():
    num_pair = 12800
    pair_index = io.load_data('datasets/DF_Pose/Anno/NIPS17-test/p_pair_test.p') # test index used in NIPS17 paper: Pose Guided Person Image Generation
    pair_split = io.load_json('datasets/DF_Pose/Label/pair_split.json')
    # store previous generated pairs
    pair_split['test_disordered_pair'] = pair_split['test']
    # use pair indexes provided in  NIPS17 paper
    pair_split['test'] = [[s1[0:-4], s2[0:-4]] for s1, s2 in pair_index[0:num_pair]]
    np.random.shuffle(pair_split['test'])

    io.save_json(pair_split, 'datasets/DF_Pose/Label/pair_split.json')


def merge_lip_segment_label():
    '''
    merge segment label of 20 classes (defined in LIP dataset).
    original label:
    0. Background
    1. Hat
    2. Hair
    3. Glove
    4. Sunglasses
    5. UpperClothes
    6. Dress
    7. Coat
    8. Socks
    9. Pants
    10. Jumpsuits
    11. Scarf
    12. Skirt
    13. Face
    14. Left-arm
    15. Right-arm
    16. Left-leg
    17. Right-leg
    18. Left-shoe
    19. Right-shoe
    '''

    parse_rst_dir = '/data2/ynli/Fashion/LIP_JPPNet/output/parsing/df_pose/'
    output_dir = 'datasets/DF_Pose/Img/seg-lip_df/'
    
    def mapping(x):
        if x == 0:
            # background
            return 0
        elif x in {1, 2}:
            # new hair
            return 1
        elif x in {4, 13}:
            # new face
            return 2
        elif x in {5, 6, 10, 11}:
            # new upperbody
            return 3
        elif x in {9, 12}:
            # new lowerbody
            return 4
        elif x in {8, 16, 17, 18, 19}:
            # new leg
            return 5
        elif x in {3, 14, 15}:
            # new arm
            return 6
        elif x in {7}:
            # coat
            return 7
        else:
            raise Exception('wrong label! %d'%x)

    mapping = np.vectorize(mapping)

    split = io.load_json('datasets/DF_Pose/Label/split.json')
    id_list = split['train'] + split['test']
    io.mkdir_if_missing(output_dir)

    for i, s_id in enumerate(id_list):
        fn_in = parse_rst_dir + '%s.png'%s_id
        fn_out = output_dir + '%s.bmp'%s_id
        print('%d/%d' % (i, len(id_list)))
        # print('%d / %d: %s -> %s'%(i, len(id_list), fn_in, fn_out))
        rst = imageio.imread(fn_in)
        new_rst = mapping(rst).astype(np.uint8)
        imageio.imwrite(fn_out, new_rst)

def revise_coat_label():
    '''
    Reivese the segment label of coat(7) and upperbody(3).
    '''
    import cv2
    
    img_dir = 'datasets/DF_Pose/Img/img_df/'
    seg_dir = 'datasets/DF_Pose/Img/seg-lip_df/'
    output_dir = 'datasets/DF_Pose/Img/seg-lip_df_revised/'
    split = io.load_json('datasets/DF_Pose/Label/split.json')
    id_list = split['train'] + split['test']
    pid_to_sids= defaultdict(lambda :[])
    for sid in id_list:
        pid = sid[0:5]
        pid_to_sids[pid].append(sid)
    print('find %d person ids'%len(pid_to_sids))

    n_revised = 0
    io.mkdir_if_missing(output_dir)
    for i, (pid, sids) in enumerate(pid_to_sids.items()):
        seg_0 = cv2.imread(seg_dir + pid + '_1.bmp', cv2.IMREAD_GRAYSCALE) # try to load frontal image
        if (seg_0 is not None) and (7 in seg_0) and (3 in seg_0):
            n_revised += 1
            img_0 = cv2.imread(img_dir + pid+ '_1.jpg')
            mask_u = (seg_0 == 3).astype(np.uint8)
            mask_c = (seg_0 == 7).astype(np.uint8)
            hist_u = cv2.calcHist([img_0], [0,1,2], mask_u, [8]*3, [0,256]*3)
            hist_c = cv2.calcHist([img_0], [0,1,2], mask_c, [8]*3, [0,256]*3)
            for sid in sids:
                if sid == pid+'_1':
                    shutil.copyfile(seg_dir+sid+'.bmp', output_dir+sid+'.bmp')
                else:
                    seg_i = cv2.imread(seg_dir+sid+'.bmp', cv2.IMREAD_GRAYSCALE)
                    img_i = cv2.imread(img_dir+sid+'.jpg')
                    mask_u_i = (seg_i==3).astype(np.uint8)
                    mask_c_i = (seg_i==7).astype(np.uint8)
                    for mask_i in [mask_u_i, mask_c_i]:
                        if mask_i.any():
                            hist_i = cv2.calcHist([img_i], [0,1,2], mask_i, [8]*3, [0,256]*3)
                            if cv2.compareHist(hist_i, hist_u, cv2.HISTCMP_CORREL) < cv2.compareHist(hist_i, hist_c, cv2.HISTCMP_CORREL):
                                seg_i[mask_i] = 3
                            else:
                                seg_i[mask_i] = 7
                    cv2.imwrite(output_dir+sid+'.bmp', seg_i)
        else:
            for sid in sids:
                shutil.copyfile(seg_dir+sid+'.bmp', output_dir+sid+'.bmp')
        print('%d / %d (%d revised)' % (i, len(pid_to_sids), n_revised))



def compare_segment():
    '''
    Compare segment result of two methods:
    1. segment model used in FashionGAN (Be Your Own Prada: Fashion Synthesis with Structural Coherence, ICCV 2017)
    2. segment model used in VITON (VITON: An Image-based Virtual Try-on Network, CVPR 2018)
        - Look into Person: Joint Body Parsing & Pose Estimation Network and A new Benchmark, T-PAMI 2018
    '''
    import torch
    import torchvision
    from misc.visualizer import GANVisualizer_V3
    
    pair_list = io.load_json('datasets/DF_Pose/Label/pair_split.json')['test'][0:64]
    img_dir = 'datasets/DF_Pose/Img/img_df/'
    seg_atr_dir = 'datasets/DF_Pose/Img/seg_df/'
    seg_lip_dir = 'datasets/DF_Pose/Img/seg-lip_df/'
    seg_lip_rv_dir = 'datasets/DF_Pose/Img/seg-lip_df_revised/'

    visuals = defaultdict(lambda :[])
    for (id_1, id_2) in pair_list:
        visuals['img_1'].append(imageio.imread(img_dir + id_1 + '.jpg'))
        visuals['seg_atr_1'].append(imageio.imread(seg_atr_dir + id_1 + '.bmp'))
        visuals['seg_lip_1'].append(imageio.imread(seg_lip_dir + id_1 + '.bmp'))
        visuals['seg_lip_rv_1'].append(imageio.imread(seg_lip_rv_dir + id_1 + '.bmp'))

        visuals['img_2'].append(imageio.imread(img_dir + id_2 + '.jpg'))
        visuals['seg_atr_2'].append(imageio.imread(seg_atr_dir + id_2 + '.bmp'))
        visuals['seg_lip_2'].append(imageio.imread(seg_lip_dir + id_2 + '.bmp'))
        visuals['seg_lip_rv_2'].append(imageio.imread(seg_lip_rv_dir + id_2 + '.bmp'))

    visuals = {k: np.stack(v, axis=0) for k,v in visuals.iteritems()}
    for name in ['img_1', 'img_2']:
        visuals[name] = (torch.Tensor(visuals[name].transpose(0,3,1,2)).div_(127.5).sub_(1), 'rgb')
    for name in ['seg_atr_1', 'seg_atr_2', 'seg_lip_1', 'seg_lip_2', 'seg_lip_rv_1', 'seg_lip_rv_2']:
        visuals[name] = (torch.Tensor(visuals[name][:,np.newaxis,:,:]), 'seg')

    visuals_ordered = OrderedDict()
    for k in ['img_1', 'seg_atr_1', 'seg_lip_1', 'seg_lip_rv_1', 'img_2', 'seg_atr_2', 'seg_lip_2', 'seg_lip_rv_2']:
        visuals_ordered[k] = visuals[k]
    imgs, vis_list = GANVisualizer_V3.merge_visual(visuals_ordered)
    print(vis_list)
    output_dir = 'temp/df_seg/'
    io.mkdir_if_missing(output_dir)
    torchvision.utils.save_image(imgs, output_dir+'compare_segment_atr-and-lip.jpg', nrow=8, normalize=True)

if __name__ == '__main__':
    # create_image_index()
    # create_pair_index()
    

    # merge_lip_segment_label()
    # revise_coat_label()
    compare_segment()
