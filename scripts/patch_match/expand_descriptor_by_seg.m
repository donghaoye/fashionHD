function desc_exp = expand_descriptor_by_seg(desc, seg, fillvalue)
    [N,H,W,C] = size(desc);
    N_seg = size(seg, 4);
    assert(all(size(seg) == [N,H,W,N_seg]));
    desc_exp = cell(1, N_seg);
    
    for i=1:N_seg
       mask = repmat(seg(:,:,:,i), [1,1,1,C]);
       desc_exp{i} = desc .* mask + fillvalue*(1-mask);
    end
    
    desc_exp = cat(4, desc_exp{:}); 
end