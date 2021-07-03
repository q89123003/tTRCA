function [W, V, elapsed_time_avg] = original_trca(eeg, iterN)

if ~exist('iterN', 'var')
    iterN = 1;
end

[num_chans, num_smpls, num_trials]  = size(eeg);

% start measuring
start = tic;

for iter = 1 : iterN

    S = zeros(num_chans);
    for trial_i = 1:1:num_trials
        for trial_j = trial_i+1:1:num_trials
            x1 = squeeze(eeg(:,:,trial_i));
            %x1 = bsxfun(@minus, x1, mean(x1,2));
            x2 = squeeze(eeg(:,:,trial_j));
            %x2 = bsxfun(@minus, x2, mean(x2,2));
            %S = S + x1*x2' + x2*x1';
            S = S + x1*x2';
        end % trial_j
    end % trial_i
    S = S + S.';
    UX = reshape(eeg, num_chans, num_smpls*num_trials);
    %UX = bsxfun(@minus, UX, mean(UX,2));
    Q = UX*UX';
end

% stop measureing
elapsed_time = toc(start);
elapsed_time_avg = elapsed_time / iterN;

[W,V] = eigs(S, Q);
