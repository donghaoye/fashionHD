% script for setting path

% SCNet root
scnet_root = '/data2/ynli/patch_matching_and_flow/SCNet/';
% MatConvNet Root
matconvnet_root = '/home/ynli/matconvnet/';

% setup
run([scnet_root 'utils/vlfeat-0.9.20/toolbox/vl_setup.m']);
run([matconvnet_root 'matlab/vl_setupnn.m']);
addpath(genpath([scnet_root 'utils/']));

% model path
scnet_model_path = [scnet_root 'data/trained_models/PASCAL-RP/SCNet-AGplus.mat'];
