% compute patch matching and dense flow using SCNet feature
clear;
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
output_dir = 'output/LOM_SCN_GT_64/';
img_dir_A = '../../datasets/DF_Pose/Img/img_df/';
img_dir_B = '../../datasets/DF_Pose/Img/img_df/';
seg_dir_A = '../../datasets/DF_Pose/Img/seg-lip_df_revised/';
proposal_A_fn = './cache/proposal/proposal_A_GT_64.mat';
proposal_B_fn = './cache/proposal/proposal_B_GT_64.mat';


scnet_root = '/data2/ynli/patch_matching_and_flow/SCNet/';% SCNet root
matconvnet_root = '/home/ynli/matconvnet/';% MatConvNet Root
% scnet_model_path = [scnet_root 'data/trained_models/exp_SCNet_AGplus_bsz8_lrdecay_PF-PASCAL-RP-500/net-epoch-20.mat'];
% scnet_model_path = [scnet_root 'data/trained_models/PASCAL-RP/SCNet-AGplus.mat'];% model path
scnet_model_path = [scnet_root 'data/trained_models/PASCAL-RP/SCNet-A.mat'];% model path
run([scnet_root 'utils/vlfeat-0.9.20/toolbox/vl_setup.m']);
addpath(genpath([scnet_root 'utils/']));
run([matconvnet_root 'matlab/vl_setupnn.m']);
addpath([scnet_root 'SCNet_Baselines/']);
%-------------------------------------
% flags
use_gpu = true;
use_postMatch = true; % set "true" to use PHM/LOM matching with SCNet feature as descriptor, set "false" to use SCNet matching directly.
b_show_match = true;
b_AisGT = strcmp(img_dir_B, '../../datasets/DF_Pose/Img/img_df/');
% load pair IDs
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
opt.feature = 'LPF';

% options for flow computation
sdf.nei= 0;                 % 0: 4-neighbor 1: 8-neighbor
sdf.lambda = 20;            % smoothness parameter
sdf.sigma_g = 30;           % bandwidth for static guidance
sdf.sigma_u = 15;           % bandwidth for dynamic guidance
sdf.itr=2;                  % number of iterations
sdf.issparse=true;          % is the input sparse or not
bPost = true;

% load SCNet model
if use_gpu
    gpuDevice(1);
end
load(scnet_model_path);
net = dagnn.DagNN.loadobj(net);
if use_gpu
    net.move('gpu');
end
net.conserveMemory = false;

load('cache/image_statistics.mat', 'img_stat');
image_mean = single(repmat(mean(mean(img_stat.image_mean,1),2), img_sz));
image_std = single(repmat(mean(mean(img_stat.image_std, 1),2), img_sz));

% loop
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

flow_out = cell(1,num_img);
% imdb = load('/data2/ynli/patch_matching_and_flow/SCNet/data/PF-PASCAL-RP-500.mat');
for i=1:num_img
    fprintf('computing matching and flow: %d/%d\n', i, num_img);
    % prepare batch
    img_A = imread(img_fns_A{i});
    img_B = imread(img_fns_B{i});
    img_A_norm = (single(img_A) - image_mean)./image_std;
    img_B_norm = (single(img_B) - image_mean)./image_std;
    op_A = proposals_A{i};
    op_B = proposals_B{i};
    op_A_ext = [ones(size(op_A,1),1).*1 op_A];
    op_B_ext = [ones(size(op_B,1),1).*2 op_B];
    idx_for_active_opA = 1:size(op_A, 1);
    IoU2GT = ones(size(op_A, 1), size(op_B, 1));
    % for test
%     img_A = imdb.data.images{i,1};
%     img_B = imdb.data.images{i,2};
%     op_A = imdb.data.proposals{i,1};
%     op_B = imdb.data.proposals{i,2};
%     idx_for_active_opA = imdb.data.idx_for_active_opA{i};
%     IoU2GT = imdb.data.IoU2GT{i};
%     image_mean = imdb.data.image_mean;
%     image_std = imdb.data.image_std;
%     
%     img_A_norm = (single(img_A) - image_mean)./image_std;
%     img_B_norm = (single(img_B) - image_mean)./image_std;
%     op_A = [ones(size(op_A,1),1).*1 op_A];
%     op_B = [ones(size(op_B,1),1).*2 op_B];
    
    if use_gpu
        batch = {'b1_input', gpuArray(im2single(img_A_norm)), 'b2_input', gpuArray(im2single(img_B_norm)),...
            'b1_rois', gpuArray(single(op_A_ext')), 'b2_rois', gpuArray(single(op_B_ext')), ...
            'idx_for_active_opA', gpuArray(single(idx_for_active_opA)), 'IoU2GT', gpuArray(single(IoU2GT))};
    else
        batch = {'b1_input', im2single(img_A_norm), 'b2_input', im2single(img_B_norm), ...
            'b1_rois', single(op_A_ext'), 'b2_rois', single(op_B_ext'), ...
            'idx_for_active_opA', single(idx_for_active_opA), 'IoU2GT', single(IoU2GT)};
    end
    % network forward
    net.eval(batch);
    if use_postMatch
        % -------------------------------------------
        % compute matching using PHM
        feat_b1 = net.vars(net.getVarIndex('b1_L2NM')).value;
        feat_b1 = reshape(feat_b1, size(feat_b1,3), size(feat_b1,4));
        feat_b1 = feat_b1';
        feat_b2 = net.vars(net.getVarIndex('b2_L2NM')).value;
        feat_b2 = reshape(feat_b2, size(feat_b2,3), size(feat_b2,4));
        feat_b2 = feat_b2';
        if use_gpu
            feat_b1 = gather(feat_b1);
            feat_b2 = gather(feat_b2);
        end
        % get view
        view_A = vl_getView(op_A, img_A);
        view_A.desc = feat_b1';
        view_A.img = img_A;
        view_B = vl_getView(op_B, img_B);
        view_B.desc = feat_b2';
        view_B.img = img_B;    
        % confidence map
        if strcmp(algorithm, 'PHM')
            confidenceMap = PHM(view_A, view_B, opt);
        elseif strcmp(algorithm, 'LOM')
            confidenceMap = LOM(view_A, view_B, opt);
        end
        % -------------------------------------------
    else
        % -------------------------------------------
        % use SCNet condifence map output
        confidenceMap = net.vars(net.getVarIndex('AG_out')).value;
        if use_gpu
            confidenceMap = gather(confidenceMap);
        end
        % get view
        view_A = vl_getView(op_A, img_A);
        view_A.img = img_A;
        view_B = vl_getView(op_B, img_B);
        view_B.img = img_B;
        % -------------------------------------------
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
%         pause;
        fn_match = sprintf('%s/m%d_%s_%s.jpg', output_dir, i-1, pid_list.test{i,1}, pid_list.test{i,2});
        saveas(h, fn_match);
    end
end

% save flow
flow_output_fn = sprintf('%s/flow.mat', output_dir);
save(flow_output_fn, 'flow_out');