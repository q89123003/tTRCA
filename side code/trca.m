
function [W, V] = trca(eeg) % Origial
% Task-related component analysis (TRCA). This script was written based on
% the reference paper [1].
%
% function W = trca(eeg)
%
% Input:
%   eeg         : Input eeg data 
%                 (# of channels, Data length [sample], # of trials)
%
% Output:
%   W           : Weight coefficients for electrodes which can be used as 
%                 a spatial filter.
%   
% Reference:
%   [1] H. Tanaka, T. Katura, H. Sato,
%       "Task-related component analysis for functional neuroimaging and 
%        application to near-infrared spectroscopy data",
%       NeuroImage, vol. 64, pp. 308-327, 2013.
%
% Masaki Nakanishi, 22-Dec-2017
% Swartz Center for Computational Neuroscience, Institute for Neural
% Computation, University of California San Diego
% E-mail: masaki@sccn.ucsd.edu

[num_chans, num_smpls, num_trials]  = size(eeg);
S = zeros(num_chans);
for trial_i = 1:1:num_trials
    x1 = squeeze(eeg(:,:,trial_i));
    S = S + x1 * x1.';
    for trial_j = trial_i+1:1:num_trials
        x2 = squeeze(eeg(:,:,trial_j));
        S = S + x1*x2' + x2*x1';
    end % trial_j
end % trial_i

% if num_trials == 1
%     S = eye(num_chans);
% end
UX = reshape(eeg, num_chans, num_smpls*num_trials);

Q = UX*UX';
[W,V] = eigs(S, Q);