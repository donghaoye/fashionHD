from __future__ import print_function, division
import util.io as io
import scipy.io
import numpy as np
import imageio
import tqdm
import cv2

####################################
# TPS
####################################
def mask_guided_tps():
    print('check "df_tps_warp.m"')

def keypoint_guided_tps():
    
    num_sample = 64
    pair_list = io.load_json('datasets/DF_Pose/Label/pair_split.json')['test'][0:num_sample]
    pose_label = io.load_data('datasets/DF_Pose/Label/pose_label.pkl')
    image_dir = 'datasets/DF_Pose/Img/img_df/'
    seg_dir = 'datasets/DF_Pose/Img/seg-lip_df_revised/'
    output_dir = 'temp/patch_matching/output/tps_keypoint/'
    io.mkdir_if_missing(output_dir)
    tps = cv2.createThinPlateSplineShapeTransformer()

    for i, (id_1, id_2) in enumerate(tqdm.tqdm(pair_list)):
        kp_1 = np.array(pose_label[id_1][1:14], dtype=np.float64).reshape(1,-1,2)
        kp_2 = np.array(pose_label[id_2][1:14], dtype=np.float64).reshape(1,-1,2)
        kp_matches = []
        for j in range(kp_1.shape[1]):
            if (kp_1[0,j]>=0).all() and (kp_2[0,j]>=0).all():
                kp_matches.append(cv2.DMatch(j,j,0))
        if len(kp_matches) == 0:
            continue

        tps.estimateTransformation(kp_2, kp_1, kp_matches)
        img_1 = cv2.imread(image_dir + id_1 + '.jpg')
        img_2 = cv2.imread(image_dir + id_2 + '.jpg')
        
        img_w = tps.warpImage(img_1)
        seg = cv2.imread(seg_dir + id_2 + '.bmp', cv2.IMREAD_GRAYSCALE)
        mask = ((seg==3) | (seg==7)).astype(img_w.dtype)[:,:,np.newaxis]
        img_out = img_w * mask + img_2 * (1-mask)

        cv2.imwrite(output_dir+'%d_%s_%s.jpg'%(i, id_1, id_2), img_out)
        cv2.imwrite(output_dir+'w%d_%s_%s.jpg'%(i, id_1, id_2), img_w)


if __name__ == '__main__':

    ###########################
    # TPS functions
    ###########################
    keypoint_guided_tps()