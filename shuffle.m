function v=shuffle(v, seed)
    if exist('seed', 'var')
        rng(seed);
    end
    
    v=v(randperm(length(v)));
end