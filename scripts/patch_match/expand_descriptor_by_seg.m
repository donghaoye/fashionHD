function CE = expand_descriptor_by_seg(C, S, seg_penalty)
    [n,h,w,c] = size(C);
    s = size(S, 4);
    assert(all(size(S)==[n,h,w,s]));
    CE = zeros(n,h,w,c+s);
    CE(:,:,:,1:c) = C;
    CE(:,:,:,(c+1):end) = S*seg_penalty*sqrt(c/2.0);
end

% function desc_exp = expand_descriptor_by_seg(desc, seg, fillvalue)
%     [N,H,W,C] = size(desc);
%     N_seg = size(seg, 4);
%     assert(all(size(seg) == [N,H,W,N_seg]));
%     desc_exp = cell(1, N_seg);
%     
%     for i=1:N_seg
%        mask = repmat(seg(:,:,:,i), [1,1,1,C]);
%        desc_exp{i} = desc .* mask + fillvalue*(1-mask);
%     end
%     
%     desc_exp = cat(4, desc_exp{:}); 
% end

% function [CE_1, CE_2] = expand_descriptor_by_seg(C_1, C_2, S, seg_penalty)
%     % With c-channel descriptor C, and s-channel segmentation map S, the
%     % expanded feature map CE is (c*s)-channel. For each pixel in CE, the
%     % corresponding s channels to its segmentation class is filled with C,
%     % while other channels are filled with CF. CF is computed from C1 and
%     % C2 that meets (C1-CF).^2 + (C2-CF).^2 == seg_penalty^2.
%     
%     CF = 0.5*(C_1 + C_2 + sqrt(2*seg_penalty*seg_penalty - (C_1-C_2).*(C_1-C_2)));
%     
%     
% end