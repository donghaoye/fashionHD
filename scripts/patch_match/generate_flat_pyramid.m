function [flat_pyramid, mask] = generate_flat_pyramid(x, default_value)
    
    scale_list = [1, 0.75, 1.25];
    [N,H,W,C] = size(x);
    x = reshape(permute(x, [2,1,3,4]), [N*H,W,C]);
    pyramid = cell(1, length(scale_list));
    H_out = 0;
    W_out = 0;
    for i=1:length(scale_list)
        s = scale_list(i);
        h = int32(H*s);
        w = int32(W*s);
        H_out = max(H_out, h);
        W_out = W_out + w;
        p = imresize(x,[N*h, w], 'nearest');
        p = permute(reshape(p,[h,N,w,C]), [2,1,3,4]);
        pyramid{i} = p;
    end
    
    flat_pyramid = ones(N,H_out,W_out,C) * default_value;
    mask = ones(N,H_out,W_out,C);
    w_left = 1;
    for i=1:length(pyramid)
        p = pyramid{i};
        [~,h,w,~] = size(p);
        flat_pyramid(:,1:h,w_left:(w_left+w-1),:)= p;
        mask(:,1:h,w_left:(w_left+w-1),:) = 0;
        w_left = w_left+w;
    end
    
    if isa(x, 'uint8')
        flat_pyramid = uint8(flat_pyramid);
    end
end