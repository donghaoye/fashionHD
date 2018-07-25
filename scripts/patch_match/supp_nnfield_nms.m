function mask_supp = supp_nnfield_nms(score, patch_w, overlap, mask)
% create sparse nearest neighboor fielde by non-maximal suppression.
    if nargin < 4
        mask = ones(size(score));
    end
    [yy, xx, ~] = find(mask>0);
    s = score(mask>0);
    boxes = [xx-floor(patch_w/2), yy-floor(patch_w/2), xx+floor(patch_w/2), yy+floor(patch_w/2), s];
    pick = nms(boxes, overlap);
    mask_supp = zeros(size(mask));
    mask_supp(sub2ind(size(mask), yy(pick), xx(pick))) = 1;
end