import numpy as np
from skimage.draw import circle, line_aa, polygon

joint2idx = {
    'nose': 0,
    'neck': 1,
    'rshoulder':2,
    'relbow': 3,
    'rwrist': 4,
    'lshoulder': 5,
    'lelbow': 6,
    'lwrist': 7,
    'rhip': 8,
    'rknee': 9,
    'rankle': 10,
    'lhip': 11,
    'lknee': 12,
    'lankle': 13,
    'reye': 14,
    'leye': 15,
    'rear': 16,
    'lear': 17,
}

def get_joint_coord(label, joint_list):
    indices = [joint2idx[j] for j in joint_list]
    if isinstance(label, list):
        label = np.float32(label)

    return label[indices, :]


##############################################################################################
# Derived from Deformable GAN (https://github.com/AliaksandrSiarohin/pose-gan)
##############################################################################################
LIMB_SEQ = [[1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [1,8], [8,9],
           [9,10], [1,11], [11,12], [12,13], [1,0], [0,14], [14,16],
           [0,15], [15,17], [2,1], [5,1]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
LABELS = ['nose', 'neck', 'Rsho', 'Relb', 'Rwri', 'Lsho', 'Lelb', 'Lwri',
               'Rhip', 'Rkne', 'Rank', 'Lhip', 'Lkne', 'Lank', 'Leye', 'Reye', 'Lear', 'Rear']
MISSING_VALUE = -1

def map_to_coords(pose_map, threshold=0.1):
    '''
    Input:
        pose_map: (h, w, channel)
    Output:
        coord:
    '''
    all_peaks = [[] for i in range(18)]
    pose_map = pose_map[...,:18]

    y,x,z = np.where(np.logical_and(pose_map==pose_map.max(axis=(0,1)), pose_map>threshold))
    for x_i, y_i, z_i in zip(x, y, z):
        all_peaks[z_i].append([x_i, y_i])

    x_values = []
    y_values = []

    for i in range(18):
        if len(all_peaks[i]) != 0:
            x_values.append(all_peaks[i][0][0])
            y_values.append(all_peaks[i][0][1])
        else:
            x_values.append(MISSING_VALUE)
            y_values.append(MISSING_VALUE)

    return np.concatenate([np.expand_dims(y_values, -1), np.expand_dims(x_values, -1)], axis=1)

def draw_pose_from_coords(pose_joints, img_size, radius=2, draw_joints=True):
    colors = np.zeros(shape=img_size + (3, ), dtype=np.uint8)
    mask = np.zeros(shape=img_size, dtype=bool)

    if draw_joints:
        for f, t in LIMB_SEQ:
            from_missing = pose_joints[f][0] == MISSING_VALUE or pose_joints[f][1] == MISSING_VALUE
            to_missing = pose_joints[t][0] == MISSING_VALUE or pose_joints[t][1] == MISSING_VALUE
            if from_missing or to_missing:
                continue
            yy, xx, val = line_aa(pose_joints[f][0], pose_joints[f][1], pose_joints[t][0], pose_joints[t][1])
            colors[yy, xx] = np.expand_dims(val, 1) * 255
            mask[yy, xx] = True
    for i, joint in enumerate(pose_joints):
        if pose_joints[i][0] == MISSING_VALUE or pose_joints[i][1] == MISSING_VALUE:
            continue
        yy, xx = circle(joint[0], joint[1], radius=radius, shape=img_size)
        colors[yy, xx] = COLORS[i]
        mask[yy, xx] = True

    return colors, mask

def draw_pose_from_map(pose_map, threshold=0.1, radius=2, draw_joints=True):
    img_size = pose_map.shape[0:2]
    coords = map_to_coords(pose_map, threshold)
    return draw_pose_from_coords(coords, img_size, radius, draw_joints)

