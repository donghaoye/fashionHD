clear all;
addpath('../../VITON/shape_context');
addpath('./parfor_progress');

%% load data
%for training set
% fn_src_list = '../datasets/DeepFashion/Fashion_design/Temp/ca_train_tps_src.txt';
% fn_tar_list = '../datasets/DeepFashion/Fashion_design/Temp/ca_train_tps_tar.txt';
% seg_dir = '../datasets/DeepFashion/Fashion_design/Img/seg_ca_syn_256/';
% edge_dir = '../datasets/DeepFashion/Fashion_design/Img/edge_ca_256_cloth/';
% output_dir = '../datasets/DeepFashion/Fashion_design/Img/edge_ca_256_tps/';

%for vis set
% fn_src_list = '../datasets/DeepFashion/Fashion_design/Temp/ca_vis_tps_src.txt';
% fn_tar_list = '../datasets/DeepFashion/Fashion_design/Temp/ca_vis_tps_tar.txt';
% seg_dir = '../datasets/DeepFashion/Fashion_design/Img/seg_ca_syn_256/';
% edge_dir = '../datasets/DeepFashion/Fashion_design/Img/edge_ca_256_cloth/';
% output_dir = '../datasets/DeepFashion/Fashion_design/Img/edge_ca_256_tps_vis/';

% for DF_Pose
fn_src_list = '../temp/patch_matching/label/image_src_list.txt';
fn_tar_list = '../temp/patch_matching/label/image_tar_list.txt';
seg_dir = '../datasets/DF_Pose/Img/seg-lip_df_revised/';
img_dir = '../datasets/DF_Pose/Img/img_df/';
output_dir = '../temp/patch_matching/output/tps/';

src_list = textread(fn_src_list, '%s');
tar_list = textread(fn_tar_list, '%s');

if ~exist(output_dir)
    mkdir(output_dir)
end
%% config
n_control = 10;

%% perform TPS
% parpool(16)
start_idx = 1;
%N = 20000
N = length(tar_list)
parfor_progress(N);
for i = start_idx:(N+start_idx-1)
% for i = start_idx:(N+start_idx-1)
%     fprintf('%s\n', tar_list{i});
    parfor_progress;
    fn_out = sprintf('%s/%d_%s_%s.jpg', output_dir, i, src_list{i}, tar_list{i});
    fn_out_warp = sprintf('%s/w%d_%s_%s.jpg', output_dir, i, src_list{i}, tar_list{i});
%     if exist(fn_out)
%         disp('existing file');
%         continue
%     end
    seg_src = imread([seg_dir src_list{i} '.bmp']);
    mask_src = double(seg_src == 3 | seg_src == 7);
    
    seg_tar = imread([seg_dir tar_list{i} '.bmp']);    
    mask_tar = double(seg_tar == 3 | seg_tar == 7);
    
    img_src = double(imread([img_dir src_list{i} '.jpg']))/255.;
    img_tar = double(imread([img_dir tar_list{i} '.jpg']))/255.;
    
%     imshow([mask_src edge_src mask_tar]);
%     [keypoints1, keypoints2, warp_points0, edge_warp] = tps_main(mask_src, mask_tar, n_control, edge_src, 0);
    try
        [keypoints1, keypoints2, warp_points0, img_warp] = tps_main(mask_src, mask_tar, n_control, img_src, 0);
    catch
        disp('skipping...');
        continue
    end
    
    img_warp(isnan(img_warp))=0;
    mask_tar = mask_tar(:,:,[1,1,1]);
    img_out = img_warp .* mask_tar + img_tar .* (1-mask_tar);
    
    imwrite(img_out, fn_out);
    imwrite(img_warp, fn_out_warp);
    
end
