function model = train_utrca(templates, supplements, fs, num_fbs, pad_length)
% including original TRCA, LST, Transfer-TRCA

if nargin < 3
    error('stats:train_trca:LackOfInput', 'Not enough input arguments.'); 
end

if ~exist('num_fbs', 'var') || isempty(num_fbs), num_fbs = 3; end

if ~exist('pad_length', 'var') || isempty(pad_length), pad_length = 0; end

[num_targs, num_chans, num_smpls, ~] = size(templates);
num_smpls = num_smpls - pad_length;
trains = zeros(num_targs, num_fbs, num_chans, num_smpls);
W = zeros(num_fbs, num_targs, num_chans);

lst_trains = zeros(num_targs, num_fbs, num_chans, num_smpls);
lst_W = zeros(num_fbs, num_targs, num_chans);

ttrca_template = zeros(num_targs, num_fbs, num_chans, num_smpls, size(templates, 4));
ttrca_supplement_cell = cell(size(supplements));
ttrca_W = zeros(num_fbs, num_targs, num_chans);
ttrca_V_cell = cell(size(supplements));

for i_c = 1 : size(ttrca_supplement_cell, 1)
    ttrca_supplement_cell{i_c} = zeros(num_targs, num_fbs, size(supplements{i_c}, 2), num_smpls, size(supplements{i_c}, 4));   
    ttrca_V_cell{i_c} = zeros(num_fbs, num_targs, size(supplements{i_c}, 2));
end

for targ_i = 1:1:num_targs
    template_targ = squeeze(templates(targ_i, :, :, :));

    for fb_i = 1:1:num_fbs
        %template_tmp = filterbank(template_targ, fs, fb_i);
        template_tmp = filterbank_pad(template_targ, fs, pad_length, fb_i);
        template_tmp = template_tmp - mean(template_tmp, 2);
        supplement_cat = zeros(num_chans, num_smpls, 0);   
        supplement_tmp = cell(size(supplements));
        for i_c = 1 : size(ttrca_supplement_cell, 1)
            sup_targ_temp = squeeze(supplements{i_c}(targ_i, :, :, :));
            %sup_tmp = filterbank(sup_targ_temp, fs, fb_i);
            sup_tmp = filterbank_pad(sup_targ_temp, fs, pad_length, fb_i);
            sup_tmp = sup_tmp - mean(sup_tmp, 2);
            
            supplement_cat = cat(3, supplement_cat, sup_tmp);
            
            supplement_tmp{i_c} = sup_tmp;
            ttrca_supplement_cell{i_c}(targ_i, fb_i, :, :, :) = sup_tmp;
        end

        template_tmp_mean = squeeze(mean(template_tmp, 3));
        
        % baseline trca
        trains(targ_i,fb_i,:,:) = template_tmp_mean;
        [w_tmp, ~] = trca(template_tmp);
        W(fb_i, targ_i, :) = w_tmp(:,1);
        
        % lst
        Y = template_tmp_mean;
        transferred_eeg_tmp = zeros(num_chans, num_smpls, size(supplement_cat, 3));
        
        for trialIdx = 1 : size(supplement_cat, 3)
            
            single_trial_eeg_tmp = squeeze(supplement_cat(:, :, trialIdx));
            
            X = [ones(1, size(Y, 2)); single_trial_eeg_tmp];
            b = Y * X.' / (X * X.');
            transferred_eeg_tmp(:, :, trialIdx) = (b * X);
        end
        
        transferred_eeg_tmp = cat(3, template_tmp, transferred_eeg_tmp);
        lst_trains(targ_i,fb_i,:,:) = squeeze(mean(transferred_eeg_tmp, 3));
        [w_tmp, ~] = trca(transferred_eeg_tmp);
        lst_W(fb_i, targ_i, :) = w_tmp(:,1);
        
        % ttrca  
        [w_tmp, v_tmp_cell, ~] = ttrca(template_tmp, supplement_tmp);
        ttrca_template(targ_i, fb_i, :, :, :) = template_tmp;             
        
        ttrca_W(fb_i, targ_i, :) = w_tmp(:,1);
        for i_c = 1 : size(ttrca_V_cell, 1)
            ttrca_V_cell{i_c}(fb_i, targ_i, :) = v_tmp_cell{i_c};
        end
        
    end % fb_i
    
end % targ_i

model = struct('num_fbs', num_fbs, 'fs', fs, 'num_targs', num_targs, 'pad_length', pad_length, ... % Basic parameters
    'W', W, 'trains', trains, ... % original TRCA 
    'lst_W', lst_W, 'lst_trains', lst_trains, ... % lst
    'ttrca_W', ttrca_W, 'ttrca_V_cell', {ttrca_V_cell}, ... % ttrca
    'ttrca_template', ttrca_template, 'ttrca_supplement_cell', {ttrca_supplement_cell}); % ttrca

function [W, Vs, eigv] = ttrca(template, supplement)
[num_ch0, num_smpls, num_t0]  = size(template);

total_channel_num = num_ch0;

for i_c = 1 : size(supplement, 1)
    total_channel_num = total_channel_num + size(supplement{i_c}, 1);
end

S = zeros(total_channel_num);
Q = zeros(total_channel_num);

S_0 = zeros(num_ch0);
for trial_i = 1:1:num_t0
    for trial_j = trial_i+1:1:num_t0
        x1 = squeeze(template(:,:,trial_i));
        x2 = squeeze(template(:,:,trial_j));
        S_0 = S_0 + x1*x2' + x2*x1';
    end % trial_j
end % trial_i

S(1 : num_ch0, 1 : num_ch0) = S_0;

UX_0 = reshape(template, num_ch0, num_smpls*num_t0);
Q_0 = UX_0*UX_0';

Q(1 : num_ch0, 1 : num_ch0) = Q_0;
num_ch_cml = num_ch0;

for i_c = 1 : size(supplement, 1)
    sup_i = supplement{i_c};
    [num_chi, ~, num_ti] = size(sup_i);

    S_i = zeros(num_chi);
    for trial_i = 1:1:num_ti
        for trial_j = trial_i+1:1:num_ti
            x1 = squeeze(sup_i(:,:,trial_i));
            x2 = squeeze(sup_i(:,:,trial_j));
            S_i = S_i + x1*x2' + x2*x1';
        end % trial_j
    end % trial_i

    S(num_ch_cml + 1 : num_ch_cml + num_chi, ...
        num_ch_cml + 1 : num_ch_cml + num_chi) = S_i;
        
    S_0i = zeros(num_ch0, num_chi);
    for trial_i = 1:1:num_t0
        for trial_j = 1:1:num_ti
            x1 = squeeze(template(:,:,trial_i));
            x2 = squeeze(sup_i(:,:,trial_j));
            S_0i = S_0i + x1*x2';
        end % trial_j
    end % trial_i
    
    S(1 : num_ch0, num_ch_cml + 1 : num_ch_cml + num_chi) = S_0i;
    S(num_ch_cml + 1 : num_ch_cml + num_chi, 1 : num_ch0) = S_0i.';   

%     % S_ij
%     num_ch_cmlj = num_ch_cml + num_chi;
%     for j_c = (i_c + 1) : size(supplement, 1)
%         sup_j = supplement{j_c};
%         [num_chj, ~, num_tj] = size(sup_j);
%         S_ij = zeros(num_chi, num_chj);
%         for trial_i = 1:1:num_ti
%             for trial_j = 1:1:num_tj
%                 x1 = squeeze(sup_i(:,:,trial_i));
%                 x2 = squeeze(sup_j(:,:,trial_j));
%                 S_ij = S_ij + x1*x2';
%             end % trial_j
%         end % trial_i
% 
%         S(num_ch_cml + 1 : num_ch_cml + num_chi, num_ch_cmlj + 1 : num_ch_cmlj + num_chj) = S_ij;
%         S(num_ch_cmlj + 1 : num_ch_cmlj + num_chj, num_ch_cml + 1 : num_ch_cml + num_chi) = S_ij.';
% 
%         num_ch_cmlj = num_ch_cmlj + num_chj;
%     end
    
    UX_i = reshape(sup_i, num_chi, num_smpls * num_ti);
    Q_i = UX_i*UX_i.';

    Q(num_ch_cml + 1 : num_ch_cml + num_chi, ...
        num_ch_cml + 1 : num_ch_cml + num_chi) = Q_i;

    num_ch_cml = num_ch_cml + num_chi;
end

[E, eigv] = eigs(S, Q);
W = E(1 : num_ch0, :);

Vs = cell(size(supplement));

num_ch_cml = num_ch0;
for i_c = 1 : size(supplement, 1)
    num_chi = size(supplement{i_c}, 1);
    Vs{i_c} = E(num_ch_cml + 1 : num_ch_cml + num_chi, 1, 1);
    
    num_ch_cml = num_ch_cml + num_chi;
end


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
        x2 = squeeze(eeg(:,:,trial_j));
        S = S + x1*x2' + x2*x1';
    end % trial_j
end % trial_i
UX = reshape(eeg, num_chans, num_smpls*num_trials);

Q = UX*UX';
[W,V] = eigs(S, Q);

