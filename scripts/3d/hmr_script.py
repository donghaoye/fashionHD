from __future__ import division, print_function

import util.io as io
import numpy as np
import tqdm
from misc import flow_util

# Load definition of SMPL_faces
SMPL_Faces = np.load('scripts/3d/smpl_faces.npy')

def calc_correspondence_from_smpl():
    '''
    Compute pixel-wise correspondence between image pair by SMPL model (http://smpl.is.tue.mpg.de/). 
    The SMPL fit result is predicted by HMR(https://github.com/akanazawa/hmr), with following format:
        pred = {
            'id': sid,
            'theta': theta,
            'proc_param': proc_param,
            'verts2d': verts2d,
            'verts_z': verts_z,
            'visibility': visibility, # a map with same size of img, each pixel is its corresponding SMPL face index (or 4294967295 if it's corresponding to no face)
        }
    See '/data2/ynli/human3d/hmr/run_hmr.py' for more details
    '''

    num_pair = 64
    # load pair ids
    pairs = io.load_json('datasets/DF_Pose/Label/pair_split.json')
    pair_list = pairs['test'] + pairs['train']
    pair_list = pair_list[:num_pair]
    
    # directory where HMR predictions are stored.
    hmr_pred_dir = 'temp/3d_hmr/hmr_df_openpose/pred/'

    # output dir
    output_dir = 'temp/3d_hmr/corr_flow/'
    io.mkdir_if_missing(output_dir)

    for id_1, id_2 in tqdm.tqdm(pair_list):
        pred_1 = io.load_data(hmr_pred_dir + id_1 + '.pkl')
        pred_2 = io.load_data(hmr_pred_dir + id_2 + '.pkl')

        flow_2to1, mask_2 = calc_correspondence_from_smpl_internal(pred_2, pred_1)
        flow_util.write_flow_and_mask(output_dir + '%s_%s.fam'%(id_2, id_1), flow_2to1, mask_2)

        flow_1to2, mask_1 = calc_correspondence_from_smpl_internal(pred_1, pred_2)
        flow_util.write_flow_and_mask(output_dir + '%s_%s.fam'%(id_1, id_2), flow_1to2, mask_1)


def calc_correspondence_from_smpl_internal(pred_1, pred_2, faces=SMPL_Faces):
    '''
    Compute for each pixel (x,y) in img_1 the corresponding pixel (u,v) in img_2.
    Input:
        pred_1: HMR prediction for image 1. see calc_correspondence_from_smpl.
        pred_2: HMR prediction for image 2
        faces: list, each element is a triangle face represented by 3 vertex indices (i_a, i_b, i_c)
    Output:
        corr_map: (img_size, img_size, 2). corr_map[y,x] = (u,v)
        corr_mask: (img_size, img_size). The value is one of:
            0: human pixel with correspondence in img_1
            1: human pixel without correspondece in img_1
            2: background pixel
    '''
    invisible = 4294967295
    # informations from predictions
    verts_1 = pred_1['verts2d']
    verts_2 = pred_2['verts2d']
    vis_1 = pred_1['visibility']
    h, w = vis_1.shape[:2]

    # common visible face indices
    visible_face_1 = np.unique(pred_1['visibility'])
    visible_face_2 = np.unique(pred_2['visibility'])
    common_face = np.intersect1d(visible_face_1, visible_face_2, assume_unique=True)
    common_face = common_face[common_face!=invisible]

    # corr_map and corr_mask    
    corr_map = np.zeros((h, w, 2), dtype=np.float32)
    corr_mask = np.ones((h, w), dtype=np.uint8)
    corr_mask[vis_1==invisible] = 2


    xx, yy = np.meshgrid(range(w), range(h), indexing='xy')
    for face_id in common_face:
        vis_mask = (vis_1==face_id)
        pts_1 = np.stack([xx[vis_mask], yy[vis_mask]]).T #(N, 2)

        # barycentric coordinate transformation
        vert_ids = faces[face_id] #[i_a, i_b, i_c]
        tri_1 = verts_1[vert_ids] #(3, 2)
        tri_2 = verts_2[vert_ids]
        pts_bc = get_barycentric_coords(pts_1, tri_1) #(N, 3)
        pts_2 = pts_bc.dot(tri_2) #(N, 2)

        corr_map[vis_mask] = pts_2
        corr_mask[vis_mask] = 0
    return corr_map, corr_mask


def get_barycentric_coords(pts, triangle):
    '''
    Compute the barycentric coordinates of a set of points respect to a given triangle.
    Input:
        pts: (N, 2), points coordinates in original space
        triangle: (3, 2), triangle vertices
    Output:
        pts_bc: (N, 3), barycentirc coordinates
    '''
    a, b, c = triangle
    v0 = b-a
    v1 = c-a
    v2 = pts - a
    d00 = v0.dot(v0) # scalar
    d01 = v0.dot(v1) # scalar
    d11 = v1.dot(v1) # scalar
    d20 = v2.dot(v0) # (N,)
    d21 = v2.dot(v1) # (N,)
    denom = d00*d11 - d01*d01
    v = (d11*d20 - d01*d21) / denom
    w = (d00*d21 - d01*d20) / denom
    u = 1. - v - w
    return np.stack([u,v,w]).T




if __name__ == '__main__':
    calc_correspondence_from_smpl()