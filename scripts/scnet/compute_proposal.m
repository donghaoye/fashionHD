function proposals = compute_proposal(img_fns, num_proposal, parallel)
    num_img = length(img_fns);
    proposals = cell(1, num_img);
    if ~parallel
        textprogressbar('computing proposals: ');
        for i=1:num_img
            img = imread(img_fns{i});
            [proposal, ~] = RP(img, num_proposal);
            proposals{i} = proposal;
            textprogressbar(i/num_img*100);
        end
        textprogressbar('done');
    else
        parfor i=1:num_img
            img = imread(img_fns{i});
            [proposal, ~] = RP(img, num_proposal);
            proposals{i} = proposal;
        end
    end
    
end