% compute patch matching and dense flow using HOG feature
clear;
global conf; % needed by load_view function
%% config
num_img = 64;
num_proposal = 1000;
img_sz = [256, 256];
%-------------------------------------
% matching algorithm
% algorithm = 'PHM';
algorithm = 'LOM';
%-------------------------------------
% warping method
warp_method = 'warp';
% warp_method = 'vote'; % use voting method from PatchMatch
%-------------------------------------
% set path
% note that img_A is the target, img_B is the reference.
output_dir = 'output/LOM_HOG_GT_64/';
img_dir_A = '../../datasets/DF_Pose/Img/img_df/';
img_dir_B = '../../datasets/DF_Pose/Img/img_df/';
seg_dir_A = '../../datasets/DF_Pose/Img/seg-lip_df_revised/';
proposal_A_fn = './cache/proposal/proposal_A_GT_64.mat';
proposal_B_fn = './cache/proposal/proposal_B_GT_64.mat';
scnet_root = '/data2/ynli/patch_matching_and_flow/SCNet/';% SCNet root
addpath(genpath([scnet_root 'utils/']));
% addpath([scnet_root 'SCNet_Baselines/']);
%-------------------------------------
% flags
b_show_match = true;
b_AisGT = strcmp(img_dir_B, '../../datasets/DF_Pose/Img/img_df/');
%-------------------------------------
% load image list
pid_list = load('../../temp/patch_matching/label/pair_split.mat', 'test');
img_fns_A = cell(1, num_img);
img_fns_B = cell(1, num_img);
for i=1:num_img
    if b_AisGT
        img_fns_A{i} = sprintf('%s/%s.jpg', img_dir_A, pid_list.test{i, 2});
    else
        img_fns_A{i} = sprintf('%s/%s_%s.jpg', img_dir_A, pid_list.test{i,1}, pid_list.test{i,2});
    end
    img_fns_B{i} = sprintf('%s/%s.jpg', img_dir_B, pid_list.test{i,1});
end
%% compute proposal
% use random prim (RP) to generate proposals
if exist(proposal_A_fn, 'file')
    fprintf('loading proposals_A from %s\n', proposal_A_fn);
    load(proposal_A_fn);% load proposals_A
else
    proposals_A = compute_proposal(img_fns_A,num_proposal, false);
    try
        save(proposal_A_fn, 'proposals_A');
    catch ME
        fprintf('fail to save proposals to file');
    end
end

if exist(proposal_B_fn, 'file')
    fprintf('loading proposals_B from %s\n', proposal_B_fn);
    load(proposal_B_fn);% load proposals_A
else
    proposals_B = compute_proposal(img_fns_B, num_proposal, false);
    try
        save(proposal_B_fn, 'proposals_B');
    catch ME
        fprintf('fail to save proposals to file');
    end
end
%% compute patch matching and flow
% options for matching
opt.bDeleteByAspect = true;
opt.bDensityAware = false;
opt.bSimVote = true;
opt.bVoteExp = true;
opt.feature = 'HOG';

% options for flow computation
sdf.nei= 0;                 % 0: 4-neighbor 1: 8-neighbor
sdf.lambda = 20;            % smoothness parameter
sdf.sigma_g = 30;           % bandwidth for static guidance
sdf.sigma_u = 15;           % bandwidth for dynamic guidance
sdf.itr=2;                  % number of iterations
sdf.issparse=true;          % is the input sparse or not
bPost = true;
% loop
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
flow_out = cell(1,num_img);
for i=1:num_img
    fprintf('computing matching and flow: %d/%d\n', i, num_img);
    img_A = imread(img_fns_A{i});
    img_B = imread(img_fns_B{i});
    op_A.coords = proposals_A{i};
    op_B.coords = proposals_B{i};
    idx_for_active_opA = 1:size(op_A, 1);
    %extract feature
    feat_A = extract_segfeat_hog(img_A, op_A);
    feat_B = extract_segfeat_hog(img_B, op_B);
    % load view
    view_A = load_view(img_A, op_A, feat_A, 'conf', conf);
    view_B = load_view(img_B, op_B, feat_B, 'conf', conf);
    % compute confidence
    if strcmp(algorithm, 'PHM')
        confidenceMap = PHM(view_A, view_B, opt);
    elseif strcmp(algorithm, 'LOM')
        confidenceMap = LOM(view_A, view_B, opt);
    else
        error('invalid algorithm');
    end
    % compute flow
    flow = flow_field_generation(view_A, view_B, confidenceMap, sdf, bPost);
    flow_out{i} = flow;
    if strcmp(warp_method, 'warp')
        warped_image = warpImage(im2double(img_B), flow.vx, flow.vy);
    elseif strcmp(warp_method, 'vote')
        nn_A2B = flow2nn(flow, size(img_B));
        warped_image = votemex(img_B, nn_A2B, [], 'cputiled', 7);
    end
    % save output image
    % save output
    seg = imread(sprintf('%s/%s.bmp', seg_dir_A, pid_list.test{i,2}));
    mask = uint8((seg==3)|(seg==7));
    mask = cat(3, mask, mask, mask);
    warped_image = uint8(min(max(warped_image, 0),1)*255);
    fused_image = mask.*warped_image + (1-mask).*img_A;
    
    fn_warp = sprintf('%s/w%d_%s_%s.jpg', output_dir, i-1, pid_list.test{i,1}, pid_list.test{i,2});
    imwrite(warped_image, fn_warp);
    fn_out = sprintf('%s/%d_%s_%s.jpg', output_dir, i-1, pid_list.test{i,1}, pid_list.test{i,2});
    imwrite(fused_image, fn_out);
    
    % show match
    if b_show_match
        num_to_show = 10;
        [ confidenceA, max_id ] = max(confidenceMap,[],2);
        match = [1:numel(max_id); max_id'];
        if num_to_show > 0
            [confidenceA, I] = sort(confidenceA, 'descend');
            confidenceA = confidenceA(1:num_to_show);
            match = match(:,I(1:num_to_show));
        end
        h=figure(1);
        clf;
        imgInput = appendimages(img_A, img_B, 'h');
        imshow(rgb2gray(imgInput)); hold on;
        showColoredMatches(view_A.frame, view_B.frame, match, confidenceA, 'offset', [size(img_A,2) 0], 'mode', 'box');
        fn_match = sprintf('%s/m%d_%s_%s.jpg', output_dir, i-1, pid_list.test{i,1}, pid_list.test{i,2});
        saveas(h, fn_match);
        %         pause;
    end
end

% save flow
flow_output_fn = sprintf('%s/flow.mat', output_dir);
save(flow_output_fn, 'flow_out');
