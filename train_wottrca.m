function model = train_wottrca(template, supplements, fs, num_fbs)
% Training stage of the task-related component analysis (TRCA)-based 
% steady-state visual evoked potentials (SSVEPs) detection [1].
%
% function model = train_trca(eeg, fs, num_fbs)
%
% Input:
%   eeg         : Input eeg data 
%                 (# of targets, # of channels, Data length [sample])
%   fs          : Sampling rate
%   num_fbs     : # of sub-bands
%
% Output:
%   model       : Learning model for tesing phase of the ensemble 
%                 TRCA-based method
%     - traindata   : Training data decomposed into sub-band components 
%                     by the filter bank analysis
%                     (# of targets, # of sub-bands, # of channels, 
%                      Data length [sample])
%     - W           : Weight coefficients for electrodes which can be 
%                     used as a spatial filter.
%     - num_fbs     : # of sub-bands
%     - fs          : Sampling rate
%     - num_targs   : # of targets
%
% See also:
%   test_trca.m
%
% Reference:
%   [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
%       "Enhancing detection of SSVEPs for a high-speed brain speller using 
%        task-related component analysis",
%       IEEE Trans. Biomed. Eng, 65(1):104-112, 2018.
%
% Masaki Nakanishi, 22-Dec-2017
% Swartz Center for Computational Neuroscience, Institute for Neural
% Computation, University of California San Diego
% E-mail: masaki@sccn.ucsd.edu

if nargin < 2
    error('stats:train_trca:LackOfInput', 'Not enough input arguments.'); 
end

if ~exist('num_fbs', 'var') || isempty(num_fbs), num_fbs = 3; end

[num_targs, num_chans, num_smpls, ~] = size(template);
trains = zeros(num_targs, num_fbs, num_chans, num_smpls);
W = zeros(num_fbs, num_targs, num_chans);
V = zeros(num_fbs, num_targs);
V_ratio = zeros(num_fbs, num_targs);
for targ_i = 1:1:num_targs
    supplement_targ = squeeze(supplements(targ_i, :, :, :));
    template_targ = squeeze(template(targ_i, :, :, :));
    for fb_i = 1:1:num_fbs
        supplement_tmp = filterbank(supplement_targ, fs, fb_i);
        
        template_tmp = filterbank(template_targ, fs, fb_i);
        template_tmp_mean = squeeze(mean(template_tmp, 3));
        pooled_eeg_tmp = cat(3, template_tmp, supplement_tmp);
        
        tmpl(targ_i, fb_i, :, :) = template_tmp_mean;
        trains(targ_i,fb_i,:,:) = template_tmp_mean;
        [w_tmp, v_tmp] = trca(pooled_eeg_tmp);
        W(fb_i, targ_i, :) = w_tmp(:,1);
        V(fb_i, targ_i) = v_tmp(1, 1);
        %V_ratio(fb_i, targ_i) = v_tmp(1, 1) / v_tmp(2, 2);
        V_ratio(fb_i, targ_i) = abs(v_tmp(1, 1)) / trace(abs(v_tmp(2:end, 2:end)));
    end % fb_i
end % targ_i
model = struct('trains', trains, 'W', W, 'V', V, 'V_ratio', V_ratio,...
    'num_fbs', num_fbs, 'fs', fs, 'num_targs', num_targs, 'template_mean', tmpl);


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
    for trial_j = trial_i+1:1:num_trials
        x1 = squeeze(eeg(:,:,trial_i));
        x1 = bsxfun(@minus, x1, mean(x1,2));
        x2 = squeeze(eeg(:,:,trial_j));
        x2 = bsxfun(@minus, x2, mean(x2,2));
        S = S + x1*x2' + x2*x1';
    end % trial_j
end % trial_i
UX = reshape(eeg, num_chans, num_smpls*num_trials);
UX = bsxfun(@minus, UX, mean(UX,2));
Q = UX*UX';
[W,V] = eigs(S, Q);
