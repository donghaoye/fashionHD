function rgb = seg_to_rgb(seg)
    color_map = [73,0,255;255,0,0;255,0,219;255, 219,0;0,255,146;0,146,255;0,146,255;255,127,80];
    rgb = zeros(size(seg,1), size(seg, 2), 3);
    for i=1:size(seg, 3)
        for c = 1:3
            rgb(:,:,c) = rgb(:,:,c) + seg(:,:,i)*color_map(i, c);
            
        end
    end
    rgb = uint8(rgb);
end