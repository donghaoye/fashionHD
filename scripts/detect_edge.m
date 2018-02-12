%% config
edge_dir='/data2/ynli/download/edges/';
addpath(genpath(edge_dir));

img_dir='/data2/ynli/datasets/DeepFashion/Fashion_design/Img/img_ca_256/';
output_dir='/data2/ynli/datasets/DeepFashion/Fashion_design/Img/edge_ca_256/';


%% load model
opts=edgesTrain();                % default options (good settings)
opts.modelDir= [edge_dir  'models/'];          % model will be in models/forest
opts.modelFnm='modelBsds';        % model name
opts.nPos=5e5; opts.nNeg=5e5;     % decrease to speedup training
opts.useParfor=0;                 % parallelize if sufficient memory

tic, model=edgesTrain(opts); toc; % will load model if already trained

model.opts.multiscale=0;          % for top accuracy set multiscale=1
model.opts.sharpen=2;             % for top speed set sharpen=0
model.opts.nTreesEval=4;          % for top speed set nTreesEval=1
model.opts.nThreads=4;            % max number threads for evaluation
model.opts.nms=0;                 % set to true to enable nms

%% edge detections
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
img_list = dir([img_dir '*.jpg']);
parfor i = 1:length(img_list)
    fprintf('%d/%d\n', i, length(img_list));
    img_src = imread([img_dir img_list(i).name]);
    E = edgesDetect(img_src, model);
    imwrite(E, [output_dir img_list(i).name]);
end