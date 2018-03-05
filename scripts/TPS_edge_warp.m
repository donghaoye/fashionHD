clear all;
addpath('../../VITON/shape_context');
%addpath('/data2/ynli/download/ParforProgMon');
addpath('./parfor_progress');

%% load data
fn_src_list = '../datasets/DeepFashion/Fashion_design/Temp/ca_tps_src.txt';
fn_tar_list = '../datasets/DeepFashion/Fashion_design/Temp/ca_tps_tar.txt';
seg_dir = '../datasets/DeepFashion/Fashion_design/Img/seg_ca_syn_256/';
edge_dir = '../datasets/DeepFashion/Fashion_design/Img/edge_ca_256_cloth/';
output_dir = '../datasets/DeepFashion/Fashion_design/Img/edge_ca_256_tps/';

src_list = textread(fn_src_list, '%s');
tar_list = textread(fn_tar_list, '%s');

if ~exist(output_dir)
    mkdir(output_dir)
end
%% config
n_control = 10;

%% perform TPS
parpool(16)
range = 1:20000;
%ppm = ParforProgMon('TPS', length(tar_list));
parfor_progress(length(tar_list));
%parfor i = 1:length(tar_list)
parfor i = range
    %fprintf('%s\n', tar_list{i});
    %ppm.increment();
    parfor_progress;

    fn_out = [output_dir tar_list{i} '.jpg'];
    if exist(fn_out)
   	disp('existing file');
   	continue
    end
    seg_src = rgb2gray(imread([seg_dir src_list{i} '.bmp']));
    mask_src = double(seg_src>0);
    mask_src = medfilt2(imfill(mask_src), [10,10]);
    
    seg_tar = rgb2gray(imread([seg_dir tar_list{i} '.bmp']));
    mask_tar = double(seg_tar>0);
    mask_tar = medfilt2(imfill(mask_tar), [10,10]);
    
    edge_src = double(imread([edge_dir src_list{i} '.jpg']))/255.;
    edge_src = edge_src(:,:,[1,1,1]);
    
%     imshow([mask_src edge_src mask_tar]);
%     [keypoints1, keypoints2, warp_points0, edge_warp] = tps_main(mask_src, mask_tar, n_control, edge_src, 0);
    try
        [keypoints1, keypoints2, warp_points0, edge_warp] = tps_main(mask_src, mask_tar, n_control, edge_src, 0);
    catch
        disp('skipping...');
        continue
    end
    
    edge_warp = uint8(edge_warp(:,:,1)*255);
    imwrite(edge_warp, fn_out);
    
end
