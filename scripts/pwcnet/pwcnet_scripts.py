from __future__ import division, print_function

import util.io as io
import numpy as np
import tqdm
import os
from misc import flow_util
import imageio
from skimage.measure import compare_ssim, compare_psnr


pwcnet_dir = '/data2/ynli/patch_matching_and_flow/PWC-Net/PyTorch/'

def warp_image_by_flow():
    ################################
    # config
    ################################
    num_sample = 64
    # mode = 'real2real'
    # mode = 'real2fake'
    mode = 'fake2fake'

    if mode == 'real2real':
        output_dir = 'temp/flow/PWC_real2real/'
        io.mkdir_if_missing(output_dir)
        pairs = io.load_json('datasets/DF_Pose/Label/pair_split.json')['test'][0:num_sample]
        # flow direction: img_flow2 ->  img_flow1
        img_flow1_dir = 'datasets/DF_Pose/Img/img_df/'
        img_flow1_namefmt = 'id_1'
        img_flow2_dir = 'datasets/DF_Pose/Img/img_df/'
        img_flow2_namefmt = 'id_2'
        # warping direction: img_warp1 -> img_warp2
        img_warp1_dir = 'datasets/DF_Pose/Img/img_df/'
        img_warp1_namefmt = 'id_1'
        img_warp2_dir = 'datasets/DF_Pose/Img/img_df/'
        img_warp2_namefmt = 'id_2'
        # segmentation map as mask
        seg_dir = 'datasets/DF_Pose/Img/seg-lip_df/'
        seg_namefmt = 'id_2'
        # target image
        img_tar_dir = 'datasets/DF_Pose/Img/img_df/'
        img_tar_namefmt = 'id_2'
    elif mode == 'real2fake':
        output_dir = 'temp/flow/PWC_real2fake/'
        io.mkdir_if_missing(output_dir)
        pairs = io.load_json('datasets/DF_Pose/Label/pair_split.json')['test'][0:num_sample]
        # flow direction: img_flow2 ->  img_flow1
        img_flow1_dir = 'datasets/DF_Pose/Img/img_df/'
        img_flow1_namefmt = 'id_1'
        img_flow2_dir = 'checkpoints/PoseTransfer_7.5/test/'
        img_flow2_namefmt = 'pair'
        # warping direction: img_warp1 -> img_warp2
        img_warp1_dir = 'datasets/DF_Pose/Img/img_df/'
        img_warp1_namefmt = 'id_1'
        img_warp2_dir = 'checkpoints/PoseTransfer_7.5/test/'
        img_warp2_namefmt = 'pair'
        # segmentation map as mask
        seg_dir = 'checkpoints/PoseTransfer_7.5/test_seg/'
        seg_namefmt = 'pair'
        # target image
        img_tar_dir = 'datasets/DF_Pose/Img/img_df/'
        img_tar_namefmt = 'id_2'
    elif mode == 'fake2fake':
        output_dir = 'temp/flow/PWC_fake2fake/'
        io.mkdir_if_missing(output_dir)
        pairs = io.load_json('datasets/DF_Pose/Label/pair_split.json')['test'][0:num_sample]
        # flow direction: img_flow2 ->  img_flow1
        img_flow1_dir = 'checkpoints/PoseTransfer_7.5/test_ref/'
        img_flow1_namefmt = 'pair'
        img_flow2_dir = 'checkpoints/PoseTransfer_7.5/test/'
        img_flow2_namefmt = 'pair'
        # warping direction: img_warp1 -> img_warp2
        img_warp1_dir = 'datasets/DF_Pose/Img/img_df/'
        img_warp1_namefmt = 'id_1'
        img_warp2_dir = 'checkpoints/PoseTransfer_7.5/test/'
        img_warp2_namefmt = 'pair'
        # segmentation map as mask
        seg_dir = 'checkpoints/PoseTransfer_7.5/test_seg/'
        seg_namefmt = 'pair'
        # target image
        img_tar_dir = 'datasets/DF_Pose/Img/img_df/'
        img_tar_namefmt = 'id_2'
    else:
        raise Exception('invalid mode!')
    
    

    def _get_name(id_1, id_2=None, idx=None, fmt='id_1', ext='jpg'):
        if fmt =='id_1':
            return '%s.%s'%(id_1, ext)
        elif fmt =='id_2':
            return '%s.%s'%(id_2, ext)
        elif fmt == 'pair':
            return '%s_%s.%s'%(id_1, id_2, ext)
        elif fmt == 'ipair':
            return '%s_%s_%s.%s' % (idx, id_1, id_2, ext)
        else:
            raise Exception('wrong name format %s' % fmt)

    ################################
    # compute flow
    ################################
    # create pair list file for pwc-net
    pair_list = []
    flow_file_list = []
    for idx, (id_1, id_2) in enumerate(pairs):
        fn_1 = os.path.abspath(img_flow1_dir + _get_name(id_1, id_2, idx, img_flow1_namefmt))
        fn_2 = os.path.abspath(img_flow2_dir + _get_name(id_1, id_2, idx, img_flow2_namefmt))
        fn_out = os.path.abspath(output_dir + _get_name(id_1, id_2, idx, 'ipair', 'flo'))
        flow_file_list.append(fn_out)
        pair_list.append(' '.join([fn_2, fn_1, fn_out])) # note that the flow is from img_2(target pose) to img_1(reference pose)

    # run pwc-net script to compute optical flow
    if True:
        fn_pair_list = os.path.abspath(output_dir + 'pair_list.txt')
        io.save_str_list(pair_list, fn_pair_list)
        cwd = os.getcwd()
        os.chdir('./scripts/pwcnet/')
        os.system('python run-pwc-many.py %s' % fn_pair_list)
        os.chdir(cwd)

    ################################
    # compute flow
    ################################
    ssim_score = []
    psnr_score = []
    for idx, (id_1, id_2) in enumerate(tqdm.tqdm(pairs, desc='warping image')):
        fn_1 = img_warp1_dir + _get_name(id_1, id_2, idx, img_warp1_namefmt)
        fn_2 = img_warp2_dir + _get_name(id_1, id_2, idx, img_warp2_namefmt)
        fn_seg = seg_dir + _get_name(id_1, id_2, idx, seg_namefmt, 'bmp')
        fn_flow = flow_file_list[idx]

        img_1 = imageio.imread(fn_1)
        img_2 = imageio.imread(fn_2)
        seg = imageio.imread(fn_seg)
        mask = ((seg==3)|(seg==4)|(seg==7)).astype(np.uint8)[..., np.newaxis]
        flow_2to1 = flow_util.readFlow(fn_flow)

        img_2_warp_raw = flow_util.warp_image(img_1, flow_2to1)
        img_2_warp = img_2_warp_raw * mask + img_2 * (1 - mask)
        img_flow = flow_util.flow_to_rgb(flow_2to1)

        fn_warp = output_dir + _get_name(id_1, id_2, idx, 'ipair')
        fn_warp_raw = output_dir + _get_name(id_1, id_2, idx, 'ipair', 'raw.jpg')
        fn_visflow = output_dir + _get_name(id_1, id_2, idx, 'ipair', 'flow.jpg')

        imageio.imwrite(fn_warp, img_2_warp)
        imageio.imwrite(fn_warp_raw, img_2_warp_raw)
        imageio.imwrite(fn_visflow, img_flow)        


        img_tar = imageio.imread(img_tar_dir + _get_name(id_1, id_2, idx, img_tar_namefmt))
        ssim_score.append(compare_ssim(img_tar, img_2_warp, multichannel=True))
        psnr_score.append(compare_psnr(img_tar, img_2_warp))

    ################################
    # output result
    ################################
    print('Output Dir: %s' % output_dir)
    print('Result:')
    print('psnr: %f' % np.mean(psnr_score))
    print('ssim: %f' % np.mean(ssim_score))


if __name__ == '__main__':
    warp_image_by_flow()

