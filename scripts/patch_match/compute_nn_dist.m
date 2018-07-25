function D = compute_nn_dist(desc_1, desc_2, nn_1to2, patch_w)
    [h1, w1, c1] = size(desc_1);
    [h2, w2, c2] = size(desc_2);
    assert(c1==c2);
    assert(size(nn_1to2, 1) == h1);
    assert(size(nn_1to2, 2) == w1);
    D = zeros(h1*w1, 1);
    
%     [xx1, yy1] = meshgrid(1:w1, 1:h1); % start from 1
%     ind1 = sub2ind([h1, w1], yy1(:), xx1(:));
    xx2 = double(nn_1to2(:,:,1)); % start from 0
    yy2 = double(nn_1to2(:,:,2)); % start from 0
    
    d1 = reshape(desc_1, [h1*w1, c1]);
    d2 = reshape(desc_2, [h2*w2, c2]);
    for dy = 1:patch_w
        for dx = 1:patch_w
            ind2 = sub2ind([h2, w2], yy2(:)+dy, xx2(:)+dx);
            d2_offset = d2(ind2,:);
            
            D = D + sum((d1-d2_offset).*(d1-d2_offset),2);
        end
    end
    D = reshape(D, [h1, w1]);
end