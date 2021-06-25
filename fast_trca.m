function [W, V, elapsed_time_avg] = fast_trca(eeg, iterN)

if ~exist('iterN', 'var')
    iterN = 1;
end

[num_chans, num_smpls, num_trials]  = size(eeg);

% start measuring
start = tic;

for iter = 1 : iterN
    UX = reshape(eeg, num_chans, num_smpls*num_trials);
    SX = sum(eeg, 3);
    S = SX*SX.';
    Q = UX*UX';
end

% stop measureing
elapsed_time = toc(start);
elapsed_time_avg = elapsed_time / iterN;

[W, V] = eigs(S-Q, Q);