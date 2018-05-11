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
