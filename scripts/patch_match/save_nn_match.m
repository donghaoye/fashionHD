function save_nn_match(nn, img_ref, img_tar, density, mask_tar, fn_out)
% Input:
%     nn: (h,w,3) mat, nn from img_tar to img_ref.
%           channel 1: x coord
%           channel 2: y coord
%           channel 3: distance
%     img_ref:
%     img_tar:
%     patch_size:
%     mask_tar:
if nargin < 6
    fn_out = '';
end

if nargin < 5
    mask_tar = 1;
end

% sample patch
    nn = double(nn);
    [H,W,~] = size(nn);
    sample_mask = zeros(H,W);
    sample_mask(1:density:end, 1:density:end)=1;
    sample_mask = sample_mask .* mask_tar;
    [y_tar, x_tar] = find(sample_mask);
    sample_indices = find(sample_mask);
    x_ref = nn(:,:,1);
    y_ref = nn(:,:,2);
    
    x_ref = x_ref(sample_indices);
    y_ref = y_ref(sample_indices);
%     confidence = nn(:,:,3);
%     confidence = 1 ./ (1+confidence(sample_indices));
%     confidence = exp(-confidence(sample_indices));
    confidence = y_tar;
% get view
    d = density/2;
    op_ref = [x_ref-d, y_ref-d, x_ref+d, y_ref+d];
    op_tar = [x_tar-d, y_tar-d, x_tar+d, y_tar+d];
    view_ref = vl_getView(op_ref, img_ref);
    view_tar = vl_getView(op_tar, img_tar);
% show match
    match = [1:length(sample_indices); 1:length(sample_indices)];
    h = figure(1);
    clf;
    img = appendimages(img_tar, img_ref, 'h');
    imshow(rgb2gray(img)); hold on;
    showColoredMatches(view_tar.frame, view_ref.frame, match, confidence, 'offset', [size(img_tar, 2) 0], 'mode', 'box');
    
    if fn_out
        saveas(h, fn_out);
    end
       
    
end