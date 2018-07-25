function nn = flow2nn(flow, size_2)
    h2 = size_2(1);
    w2 = size_2(2);
    [h1, w1] = size(flow.vx);
    [XX, YY] = meshgrid(1:w1, 1:h1);
    XX = XX + flow.vx;
    YY = YY + flow.vy;
    
    XX = min(max(XX,1), w2);
    YY = min(max(YY,1), h2);
    score = zeros(h1, w1, 1);
    
    nn = int32(cat(3, XX, YY, score));
    
end