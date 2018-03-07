from __future__ import division, print_function
import sys
import os
import shutil
import numpy as np
from collections import defaultdict

import util.io as io
import util.image as image
import cv2


design_root = 'datasets/DeepFashion/Fashion_design/'

def create_train_pair():
    np.random.rand(0)
    split = io.load_json(design_root + 'Split/ca_gan_split_trainval_upper.json')
    cat_label = io.load_data(design_root + 'Label/ca_cat_label.pkl')
    cat_entry = io.load_json(design_root + 'Label/cat_entry.json')
    
    # group samples by category label
    cat_to_ids = defaultdict(lambda :[])
    for s_id in split['train']:
        c = cat_label[s_id]
        cat_to_ids[c].append(s_id)

    n = 0
    pair_list = []
    for c, s_list in cat_to_ids.iteritems():
        print('[%d/%d] %s: %d samples...' % (n, len(cat_to_ids), cat_entry[c]['entry'], len(s_list)))
        n += 1
        s_list_org = [s_id for s_id in s_list]
        for i in range(len(s_list)-1):
            j = np.random.randint(i+1, len(s_list))
            temp = s_list[i]
            s_list[i] = s_list[j]
            s_list[j] = temp
        pair_list += zip(s_list_org, s_list)

    pair_dict = {s_tar:s_src for s_tar, s_src in pair_list}
    io.save_json(pair_dict, design_root + 'Temp/ca_train_tps_pair.json')

    io.save_str_list(pair_dict.keys(), design_root + 'Temp/ca_train_tps_tar.txt')
    io.save_str_list(pair_dict.values(), design_root + 'Temp/ca_train_tps_src.txt')

def create_test_pair():
    np.random.rand(0)
    split = io.load_json(design_root + 'Split/ca_gan_split_trainval_upper.json')
    cat_label = io.load_data(design_root + 'Label/ca_cat_label.pkl')
    cat_entry = io.load_json(design_root + 'Label/cat_entry.json')
    # group samples by category label
    cat_to_ids = defaultdict(lambda :[])
    for s_id in split['test']:
        c = cat_label[s_id]
        cat_to_ids[c].append(s_id)
    n = 0
    pair_list = []
    for c, s_list in cat_to_ids.iteritems():
        print('[%d/%d] %s: %d samples...' % (n, len(cat_to_ids), cat_entry[c]['entry'], len(s_list)))
        n += 1
        s_list_org = [s_id for s_id in s_list]
        for i in range(len(s_list)-1):
            j = np.random.randint(i+1, len(s_list))
            temp = s_list[i]
            s_list[i] = s_list[j]
            s_list[j] = temp
        pair_list += zip(s_list_org, s_list)

    pair_dict = {s_tar:s_src for s_tar, s_src in pair_list}
    io.save_json(pair_dict, design_root + 'Temp/ca_test_tps_pair.json')

    io.save_str_list(pair_dict.keys(), design_root + 'Temp/ca_test_tps_tar.txt')
    io.save_str_list(pair_dict.values(), design_root + 'Temp/ca_test_tps_src.txt')


def gather_tps_pair():
    split = io.load_json(design_root + 'Split/ca_gan_split_trainval_upper.json')
    tps_pair = io.load_json(design_root + 'Temp/ca_train_tps_pair.json')
    tps_pair.update(io.load_json(design_root + 'Temp/ca_test_tps_pair.json'))
    io.save_json(tps_pair, design_root + 'Label/ca_tps_pair.json')
    print(len(split))
    print(len(tps_pair))
    img_dir = design_root + 'Img/edge_ca_256_tps/'
    missing_list = []
    for i, (tar_id, src_id) in enumerate(tps_pair.items()):
        print('%d/%d' % (i, len(tps_pair)))
        fn_old = img_dir + tar_id + '.jpg'
        fn_new = img_dir + tar_id + '_' + src_id + '.jpg'
        if os.path.isfile(fn_old):
            shutil.move(fn_old, fn_new)
        else:
            missing_list.append(tar_id)

    print(missing_list)




def create_vis_pair():
    '''
    select a small group of targets from test set. prepare several edge source items for each target.
    '''
    np.random.rand(0)
    num_tar = 500
    num_src_per_tar = 6

    split = io.load_json(design_root + 'Split/ca_gan_split_trainval_upper.json')
    cat_label = io.load_data(design_root + 'Label/ca_cat_label.pkl')
    cat_entry = io.load_json(design_root + 'Label/cat_entry.json')
    id_list = split['test']
    # group samples by category label
    cat_to_ids = defaultdict(lambda :[])
    for s_id in split['test']:
        c = cat_label[s_id]
        cat_to_ids[c].append(s_id)
    n = 0
    # target list
    np.random.shuffle(id_list)
    tar_list = id_list[0:num_tar]
    # select src for each target
    group_dict = {}
    for tar_id in tar_list:
        c = cat_label[tar_id]
        src_list = [s_id for s_id in cat_to_ids[c] if s_id != tar_id]
        np.random.shuffle(src_list)
        group_dict[tar_id] = src_list[0:num_src_per_tar]

    io.save_json(group_dict, design_root + 'Temp/ca_vis_tps_group.json')

    output_src_list = []
    output_tar_list = []
    for tar_id, src_list in group_dict.iteritems():
        output_tar_list += [tar_id] * len(src_list)
        output_src_list += src_list
    io.save_str_list(output_tar_list, design_root + 'Temp/ca_vis_tps_tar.txt')
    io.save_str_list(output_src_list, design_root + 'Temp/ca_vis_tps_src.txt')



def create_cloth_edge_map():
    '''
    create edge map that only contains cloth edge (inside the cloth mask)
    '''
    # config
    mask_dilate = 5

    seg_dir = design_root + 'Img/seg_ca_syn_256/'
    edge_dir = design_root + 'Img/edge_ca_256/'
    output_dir = design_root + 'Img/edge_ca_256_cloth/'
    io.mkdir_if_missing(output_dir)

    split = io.load_json(design_root + 'Split/ca_gan_split_trainval_upper.json')
    id_list = split['train'] + split['test']

    for i, s_id in enumerate(id_list):
        print('%d/%d' % (i, len(id_list)))
        seg_map = image.imread(seg_dir + s_id + '.bmp', 'grayscale')
        edge_map = image.imread(edge_dir + s_id + '.jpg', 'grayscale')
        assert seg_map.shape == edge_map.shape
        mask = ((seg_map==3) | (seg_map==4)).astype(np.uint8)
        mask = cv2.dilate(mask, kernel = np.ones((mask_dilate, mask_dilate)))
        edge_map_cloth = edge_map * mask
        image.imwrite(edge_map_cloth, output_dir + s_id + '.jpg')


def gather_pose_estimation_result():
    '''
    We use the state-of-the-art human pose estimation method (https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation) to get key points
    This function is for gathering estimation results.
    '''
    num_key_p = 18
    rst_dir = 'datasets/DeepFashion/Fashion_design/Temp/pose_pkl/'
    split = io.load_json('datasets/DeepFashion/Fashion_design/Split/ca_gan_split_trainval_upper.json')
    id_list = split['train'] + split['test']

    pose_label = {}
    n_fail = 0
    for idx, s_id in enumerate(id_list):
        print('%d/%d : %s' % (idx, len(id_list), s_id))
        fn_pose = rst_dir + s_id + '.pkl'
        if not os.path.isfile(fn_pose):
            pose_label[s_id] = [[-1, -1] for _ in range(num_key_p)]
            n_fail += 1
        else:
            pose = io.load_data(fn_pose)
            assert len(pose) == num_key_p
            # p[i][j] = (x, y, score, id) is the j-th keypoints of i-th type.
            # we assume that j <= 1, because our image contains single person
            pose_label[s_id] = [[p[0][0], p[0][1]] if len(p) > 0 else [-1, -1] for p in pose]

    io.save_data(pose_label, 'datasets/DeepFashion/Fashion_design/Label/ca_gan_pose_label_256.pkl')
    print('%d (out of %d) samples failed' % (n_fail, len(id_list)))


def create_flexible_segmap():
    '''
    reset the region label of a segmap (orignal 7 channel, 0-background, 1-hair, 2-head, 3-upper, 4-lower, 5-arm, 6-leg):
        0-background
        1-head
        2-hair
        3-cloth (both upper and lower)
        4-cloth_flexible (boundary area of cloth region 3)
        5-arm_flexible (arm region + sleeve_points region)
        6-leg
    '''

    # config
    cloth_mefilt_size = 9
    cloth_dilate_size = 9
    cloth_dilate_neck_size = 21
    cloth_erode_size = 9
    lower_arm_width = 15
    upper_arm_width = 18
    img_size = 256

    # load data
    seg_dir = design_root + 'Img/seg_ca_syn_256/'
    output_dir = design_root + 'Img/seg_ca_syn_256_flexible/'
    io.mkdir_if_missing(output_dir)

    split = io.load_json(design_root + 'Split/ca_gan_split_trainval_upper.json')
    lm_label = io.load_data(design_root + 'Label/ca_landmark_label_256.pkl')
    pose_label = io.load_data(design_root + 'Label/ca_gan_pose_label_256.pkl')
    id_list = split['train'] + split['test']
    

    # create grid
    gx, gy = np.meshgrid(range(img_size), range(img_size))

    # subfunctions
    def _get_rect_region(p1, p2, width, gx, gy):
        '''return the mask of an overlap region between a rectangle and a cycle'''
        rect = np.abs((p2[1]-p1[1])*gx - (p2[0]-p1[0])*gy + p2[0]*p1[1] - p2[1]*p1[0]) / np.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2) <= width
        cycle = (gx - (p1[0]+p2[0])/2) ** 2 + (gy - (p1[1]+p2[1])/2)**2 <= (np.sqrt((p1[0]-p2[0])**2/4 + (p1[1]-p2[1])**2/4) + width) ** 2
        return rect & cycle
    
    def _get_cos_score(p0, p1, p2):
        '''should be positive if p1 and p2 are at the same side of p0'''
        v1 = p1 - p0
        v2 = p2 - p0
        return v1.dot(v2) / np.sqrt(v1.dot(v1) * v2.dot(v2))

    for i, s_id in enumerate(id_list):
        print('%d/%d: %s' % (i, len(id_list), s_id))
        seg_map = image.imread(seg_dir + s_id + '.bmp', 'grayscale')
        # unchanged part: head, hair and leg
        hair_mask = seg_map == 1
        head_mask = seg_map == 2
        leg_mask = seg_map == 5
        # cloth
        cloth_mask = ((seg_map==3) | (seg_map==4)).astype(np.uint8)
        base_cloth_mask = cv2.medianBlur(cloth_mask, ksize=cloth_mefilt_size)
        # cloth flexible
        cloth_flx_mask_outer = cv2.dilate(cloth_mask, kernel=np.ones((cloth_dilate_size, cloth_dilate_size),np.uint8))
        cloth_flx_mask_inner = cv2.erode(cloth_mask, kernel=np.ones((cloth_erode_size, cloth_erode_size),np.uint8))
        cloth_flx_mask_neck = cv2.dilate(cloth_flx_mask_outer, kernel=np.ones((cloth_dilate_neck_size, cloth_dilate_neck_size),np.uint8)) * (head_mask.astype(np.uint8))
        cloth_flx_mask = cloth_flx_mask_outer - cloth_flx_mask_inner + cloth_flx_mask_neck
        cloth_mask = cloth_mask.astype(np.bool)
        cloth_flx_mask = cloth_flx_mask.astype(np.bool)
        # arm flexible
        arm_flx_mask = (seg_map==6)
        pose = np.array(pose_label[s_id])
        p_sleeve = np.array(lm_label[s_id][2:4])[:,0:2] # sleeve points [[x_l, y_l], [x_r, y_r]]
        v_sleeve = np.array(lm_label[s_id][2:4])[:,2]
        p_hand = pose[[4,7]] # hand points [[x_l, y_l], [x_r, y_r]]. the same below
        p_elbow = pose[[3,6]]
        p_shoulder = pose[[2,5]]
        # clr_points = np.array(lm_label[s_id][0:2]) # collar
        for j in range(2):
            # if the sleeve point is visible: add arm region (defined by pose point, not fashion landmark) to arm_flx_mask
            # otherwise, only add original arm regsion into arm_flx_mask
            if v_sleeve[j] == 0 and (p_hand[j]!=-1).all() and (p_elbow[j]!=-1).all() and (p_shoulder[j]!=-1).all():
                # case one: sleeve point on lower arm
                if _get_cos_score(p_sleeve[j], p_hand[j], p_elbow[j]) < _get_cos_score(p_sleeve[j], p_elbow[j], p_shoulder[j]):
                    upper_arm = _get_rect_region(p_elbow[j], p_shoulder[j], upper_arm_width, gx, gy)
                    lower_arm = _get_rect_region(p_sleeve[j], p_elbow[j], lower_arm_width, gx, gy)
                    arm_flx_mask = arm_flx_mask | ((upper_arm|lower_arm) & cloth_flx_mask_outer.astype(np.bool))
                else:
                    upper_arm = _get_rect_region(p_sleeve[j], p_shoulder[j], upper_arm_width, gx, gy)
                    arm_flx_mask = arm_flx_mask | (upper_arm & cloth_flx_mask_outer.astype(np.bool))
                

        # combine channels
        seg_org = seg_map.copy()
        seg_map[:] = 0
        seg_map[hair_mask] = 1
        seg_map[head_mask] = 2
        seg_map[cloth_mask] = 3
        seg_map[cloth_flx_mask] = 4
        seg_map[arm_flx_mask] = 5
        seg_map[leg_mask] = 6

        # save
        # image.imwrite(np.concatenate((seg_org, arm_flx_mask.astype(np.uint8)),axis=0)*20, output_dir + s_id + '.bmp')
        # image.imwrite(np.concatenate((seg_org, seg_map, arm_flx_mask.astype(np.uint8)),axis=0)*10, output_dir + s_id + '.bmp')
        image.imwrite(seg_map, output_dir + s_id + '.bmp')







if __name__ == '__main__':
    #############################
    # TPS edge warpping 
    #############################
    # create_cloth_edge_map()
    # create_train_pair()
    # create_test_pair()
    gather_tps_pair()
    # create_vis_pair()
    #############################
    # flexible segmap
    #############################
    # gather_pose_estimation_result()
    # create_flexible_segmap()



