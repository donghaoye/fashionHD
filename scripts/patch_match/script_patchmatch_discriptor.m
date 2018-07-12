% PatchMatch with given descriptor
% clear all;
warning off;
%% patchmatch path
addpath(genpath('/data2/ynli/patch_matching_and_flow/patchmatch-2.1/'));
scnet_root = '/data2/ynli/patch_matching_and_flow/SCNet/';% SCNet root
run([scnet_root 'utils/vlfeat-0.9.20/toolbox/vl_setup.m']);
addpath(genpath([scnet_root 'utils/']));
matconvnet_root = '/home/ynli/matconvnet/';% MatConvNet Root
run([matconvnet_root 'matlab/vl_setupnn.m']);
addpath([scnet_root 'SCNet_Baselines/']);

%-------------------------------------
%% config
% basic
num_sample = 64;
image_size = 256;
src_1 = 'gt'; % 'gt'
src_2 = 'gen'; % 'gt' or 'gen'
% test_id = sprintf('%s_seg+vgg_h1+h2_pw1(7)', src_2);
test_id = sprintf('%s_vunet_feat2_pw15', src_2);
% patchmatch setting
patch_w = 15; % patch_size at base level
patch_w_vote = 15;% patch_size for voting 
normalize_desc = false; % normalize each descriptor searately
seg_guided = false; % segmentation guided descriptor
seg_penalty = 2; % distance penalty for pixels with mismatched segmentation. set to value range of each channel. (2 for rgb)
descriptor_pyramid_scales = []; % compute a descriptor pyramid at each pixel
use_image_pyramid = false; % use [0.75, 1, 1.25] image pyramid of img_1
use_bds = false; % use bidirectional similarity in voting

% descriptor file list
dir_desc = '/data2/ynli/Fashion/fashionHD/temp/patch_matching/descriptor/';
desc_list = {};
desc_list = [desc_list, {sprintf('%s_vunet_feat2', src_2)}];
% desc_list = [desc_list, {sprintf('%s_vunet_feat1', src_2)}];
% desc_list = [desc_list, {sprintf('%s_vunet_img', src_2)}];
% desc_list = [desc_list, {sprintf('%s_inhomoseg', src_2)}];
% desc_list = [desc_list, {sprintf('%s_rgb', src_2)}];
% desc_list = [desc_list, {sprintf('%s_seg', src_2)}];
% desc_list = [desc_list, {sprintf('%s_vgg_h1', src_2)}];
% desc_list = [desc_list, {sprintf('%s_vgg_h2', src_2)}];
% desc_list = [desc_list, {sprintf('%s_vgg_h3', src_2)}];
% desc_list = [desc_list, {sprintf('%s_vgg_h4', src_2)}];
% desc_list = [desc_list, {sprintf('%s_vgg_h5', src_2)}];
%% load image
image_info = load('/data2/ynli/Fashion/fashionHD/temp/patch_matching/label/image_info.mat');
% data_dict = {
%         'id_1': np.array(id_1, dtype=np.object),
%         'id_2': np.array(id_2, dtype=np.object),
%         'image_1': np.array(image_1, dtype=np.object),
%         'image_2': np.array(image_2, dtype=np.object),
%         'image_gen': np.array(image_gen, dtype=np.object),
%         'model_id': model_id,
%     }

images.images_1 = uint8(zeros(num_sample, image_size, image_size, 3));
images.images_2 = uint8(zeros(num_sample, image_size, image_size, 3));
images.images_gen = uint8(zeros(num_sample, image_size, image_size, 3));
for i=1:num_sample
   images.images_1(i,:,:,:) = imread(image_info.image_1{i});
   images.images_2(i,:,:,:) = imread(image_info.image_2{i});
   images.images_gen(i,:,:,:) = imread(image_info.image_gen{i});
end

images_1 =images.images_1;
if strcmp(src_2, 'gt')
    images_2 = images.images_2;
elseif strcmp(src_2, 'gen')
    images_2 = images.images_gen;
else
    fprintf('invalid src_2');
    return;
end

%% load descriptor
desc_1 = [];
desc_2 = [];
for i=1:length(desc_list)
    fn_desc = [dir_desc, 'desc_', desc_list{i}, '.mat'];
    desc_data = load(fn_desc);
    fprintf('loading descriptor "%s" from %s\n', desc_data.name, fn_desc);
    
    d1 = desc_data.desc_1(1:num_sample,:,:,:);
    d2 = desc_data.desc_2(1:num_sample,:,:,:);
    if normalize_desc
        norm_1 = sqrt(sum(d1 .* d1, 4));
        norm_2 = sqrt(sum(d2 .* d2, 4));
        d1 = d1 ./ (repmat(norm_1, [1, 1, 1, size(d1,4)]) + 1e-8);
        d2 = d2 ./ (repmat(norm_2, [1, 1, 1, size(d1,4)]) + 1e-8);
    end
    
    if size(d1, 1) ~= image_size || size(d1, 2) ~= image_size
        new_d1 = zeros(num_sample, image_size, image_size, size(d1, 4));
        new_d2 = zeros(num_sample, image_size, image_size, size(d2, 4));
        for j=1:num_sample
            new_d1(j,:,:,:) = imresize(squeeze(d1(j,:,:,:)), [image_size, image_size], 'bilinear');
            new_d2(j,:,:,:) = imresize(squeeze(d2(j,:,:,:)), [image_size, image_size], 'bilinear');
        end
        d1 = new_d1;
        d2 = new_d2;
    end
    
    if isempty(desc_1)
        desc_1 = d1;
        desc_2 = d2;
    else
        desc_1 = cat(4, desc_1, d1);
        desc_2 = cat(4, desc_2, d2);
    end
end

%% create segmentation guided descriptor
desc_seg = load([dir_desc, sprintf('desc_%s_seg.mat', src_2)]);
if size(desc_seg.desc_2, 4) == 8
    mask_tar = desc_seg.desc_2(:,:,:,4) + desc_seg.desc_2(:,:,:,8);
else
    mask_tar = desc_seg.desc_2(:,:,:,4);
end

if seg_guided
    desc_1 = expand_descriptor_by_seg(desc_1, desc_seg.desc_1, seg_penalty);
    desc_2 = expand_descriptor_by_seg(desc_2, desc_seg.desc_2, seg_penalty);
end
%% compute descriptor pyramid
if ~isempty(descriptor_pyramid_scales)
    pyramid_1 = desc_1;
    pyramid_2 = desc_2;
    
    for s=descriptor_pyramid_scales
        k = ones(s,s)/s/s;
        l_1 = desc_1;
        l_2 = desc_2;
        for i=1:num_sample
            l_1(i,:,:,:) = imfilter(squeeze(l_1(i,:,:,:)), k);
            l_2(i,:,:,:) = imfilter(squeeze(l_2(i,:,:,:)), k);
        end
        pyramid_1 = cat(4, pyramid_1, l_1);
        pyramid_2 = cat(4, pyramid_2, l_2);
    end
    
    desc_1 = pyramid_1;
    desc_2 = pyramid_2;
end

%% image pyramid
if use_image_pyramid
%     img_1 = generate_flat_pyramid(img_1);
%     desc_1 = generate_flat_pyramid(desc_1);
    [images_1, mask_ref] = generate_flat_pyramid(images_1, 0);
    [desc_1,~] = generate_flat_pyramid(desc_1, -10000);
else
    mask_ref = [];
end
%% apply patchmatch and reconstruction
images_2_vote = cell(1, num_sample);
nn_1to2 = cell(1, num_sample);
nn_2to1 = cell(1, num_sample);
% parfor i=1:num_sample
parfor i=1:num_sample
    fprintf('processing image %d/%d\n', i, num_sample);
%     img_1 = images_1{i};
%     img_2 = images_2{i};
%     nn_1to2 = nnmex(img_1, img_2, 'cputiled', patch_w, 16);
%     nn_2to1 = nnmex(img_2, img_1, 'cputiled', patch_w, 16);
    img_1 = squeeze(images_1(i,:,:,:));
    d1 = squeeze(desc_1(i,:,:,:));
    d2 = squeeze(desc_2(i,:,:,:));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % normal
    nn_1to2{i} = nnmex(d1, d2, 'cpu', patch_w, 16);
    nn_2to1{i} = nnmex(d2, d1, 'cpu', patch_w, 16);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % with nn_prev
%     [XX,YY] = meshgrid(1:size(d2,2), 1:size(d2,1));
%     nn_prev = zeros(size(d2,1), size(d2,2), 3);
%     nn_prev(:,:,1)=XX;
%     nn_prev(:,:,2)=YY;
%     nn_prev = int32(nn_prev);
%     nn_1to2{i} = nnmex(d1, d2, 'cpu', patch_w, 16, [], [], [], [], [], [], [], nn_prev);
%     nn_2to1{i} = nnmex(d2, d1, 'cpu', patch_w, 16, [], [], [], [], [], [], [], nn_prev);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % with rotation+scale
%     nn_1to2{i} = nnmex(d1, d2, 'rotscale', patch_w, 16, [], [], [], [], [], [], [], [], [], [], [], [], [], 2);
%     nn_2to1{i} = nnmex(d2, d1, 'rotscale', patch_w, 16, [], [], [], [], [], [], [], [], [], [], [], [], [], 2);
    
    if use_bds
        images_2_vote{i} = votemex(img_1, nn_2to1{i}, nn_1to2{i}, 'cpu', patch_w_vote);
    else
        images_2_vote{i} = votemex(img_1, nn_2to1{i}, [], 'cpu', patch_w_vote);
    end
end
%% evaluation and visualization
psnr_scores = zeros(1, num_sample);
ssim_scores = zeros(1, num_sample);
parfor i=1:num_sample
    fprintf('testing %d/%d\n', i, num_sample);
    psnr_scores(i) = psnr(images_2_vote{i}, squeeze(images.images_2(i,:,:,:)));
    ssim_scores(i) = ssim(images_2_vote{i}, squeeze(images.images_2(i,:,:,:)));
end
mean_psnr = mean(psnr_scores);
mean_ssim = mean(ssim_scores);

desc_names = '';
for i=1:length(desc_list)
    desc_names = sprintf('%s %s', desc_names, desc_list{i});
end

fprintf('######### Information #########\n');
fprintf('id: %s\n', test_id);
fprintf('descriptor names:%s\n', desc_names);
fprintf('descriptor dimension: %d (pyramid level is %d)\n', size(desc_1, 4), length(descriptor_pyramid_scales)+1);
fprintf('######### Evaluation ##########\n');
fprintf('psnr: %f\n', mean_psnr);
fprintf('ssim: %f\n', mean_ssim);
fprintf('###############################\n');

fprintf('saving images ...\n');
dir_out = ['/data2/ynli/Fashion/fashionHD/temp/patch_matching/output/' test_id '/'];
mkdir(dir_out);
for i=1:num_sample
    fn_out = sprintf('%s/%d_%s_%s.jpg', dir_out, i, image_info.id_1{i}, image_info.id_2{i});
    imwrite(images_2_vote{i}, fn_out);
    fn_match = sprintf('%s/%d_%s_%s_m.jpg', dir_out, i, image_info.id_1{i}, image_info.id_2{i});
    save_nn_match(nn_2to1{i}, squeeze(images_1(i,:,:,:)), images_2_vote{i}, 15, squeeze(mask_tar(i,:,:,:)), fn_match);
    
end

