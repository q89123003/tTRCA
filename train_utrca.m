function model = train_utrca(templates, supplements, fs, num_fbs, pad_length, list_freqs, num_harms)
% including original TRCA, LST, Transfer-TRCA

if nargin < 3
    error('stats:train_trca:LackOfInput', 'Not enough input arguments.'); 
end

if ~exist('num_fbs', 'var') || isempty(num_fbs), num_fbs = 3; end

if ~exist('pad_length', 'var') || isempty(pad_length), pad_length = 0; end

if ~exist('list_freqs', 'var') || ~exist('num_harms', 'var') || isempty(list_freqs) || isempty(num_harms)
    add_cca_template = false;
else
    add_cca_template = true;
end

[num_targs, num_chans, num_smpls, ~] = size(templates);
num_smpls = num_smpls - pad_length;
num_suppls = size(supplements, 1);

trains = zeros(num_targs, num_fbs, num_chans, num_smpls);
W = zeros(num_fbs, num_targs, num_chans);

lst_trains = zeros(num_targs, num_fbs, num_chans, num_smpls);
lst_W = zeros(num_fbs, num_targs, num_chans);

ttrca_template = zeros(num_targs, num_fbs, num_chans, num_smpls, size(templates, 4));
ttrca_supplement_cell = cell(size(supplements));
ttrca_W = zeros(num_fbs, num_targs, num_chans);
ttrca_V_cell = cell(size(supplements));

for i_c = 1 : num_suppls
    ttrca_supplement_cell{i_c} = zeros(num_targs, num_fbs, size(supplements{i_c}, 2), num_smpls, size(supplements{i_c}, 4));   
    ttrca_V_cell{i_c} = zeros(num_fbs, num_targs, size(supplements{i_c}, 2));
end

if add_cca_template
    y_ref = cca_reference(list_freqs, fs, num_smpls, num_harms);
    ttrca_supplement_cell = [ttrca_supplement_cell; zeros(num_targs, num_fbs, size(y_ref, 2), num_smpls, 1)];
    ttrca_V_cell = [ttrca_V_cell; zeros(num_fbs, num_targs, size(y_ref, 2))];
end


for targ_i = 1:1:num_targs
    template_targ = squeeze(templates(targ_i, :, :, :));

    for fb_i = 1:1:num_fbs
        %template_tmp = filterbank(template_targ, fs, fb_i);
        template_tmp = filterbank_pad(template_targ, fs, pad_length, fb_i);
        template_tmp = template_tmp - mean(template_tmp, 2);
        supplement_tmp = cell(size(supplements));
        total_sup_num = 0;
        for i_c = 1 : size(supplements, 1)
            sup_targ_temp = squeeze(supplements{i_c}(targ_i, :, :, :));
            %sup_tmp = filterbank(sup_targ_temp, fs, fb_i);
            sup_tmp = filterbank_pad(sup_targ_temp, fs, pad_length, fb_i);
            sup_tmp = sup_tmp - mean(sup_tmp, 2);
                        
            supplement_tmp{i_c} = sup_tmp;
            ttrca_supplement_cell{i_c}(targ_i, fb_i, :, :, :) = sup_tmp;
            
            total_sup_num = total_sup_num + size(sup_tmp, 3);
        end
        
        if add_cca_template
            tmp_cca_template = repmat(squeeze(y_ref(targ_i, :, :)), 1, 1, 1);
            supplement_tmp = [supplement_tmp; tmp_cca_template];
            ttrca_supplement_cell{end}(targ_i, fb_i, :, :, :) = repmat(squeeze(y_ref(targ_i, :, :)), 1, 1, 1);
        end

        template_tmp_mean = squeeze(mean(template_tmp, 3));
        
        % baseline trca
        trains(targ_i,fb_i,:,:) = template_tmp_mean;
        [w_tmp, ~] = trca(template_tmp);
        W(fb_i, targ_i, :) = w_tmp(:,1);
        
        % lst
        Y = template_tmp_mean;
        transferred_eeg_tmp = zeros(num_chans, num_smpls, total_sup_num);
        
        sup_count = 0;
        for i_c = 1 : size(supplement_tmp, 1)
            sup_tmp = supplement_tmp{i_c};
            sup_tmp_mean = mean(sup_tmp, 3);
            X = [ones(1, size(Y, 2)); sup_tmp_mean];
            b = Y * X.' / (X * X.');
            
            for trialIdx = 1 : size(sup_tmp, 3)
                sup_count = sup_count + 1;
                single_trial_eeg_tmp = squeeze(sup_tmp(:, :, trialIdx));
                X_trial = [ones(1, size(Y, 2)); single_trial_eeg_tmp];
                transferred_eeg_tmp(:, :, sup_count) = (b * X_trial);
            end
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

template_sum = sum(template, 3);
S_0 = template_sum*template_sum.';
S_0 = S_0 / num_t0 ^ 2;

S(1 : num_ch0, 1 : num_ch0) = S_0;

UX_0 = reshape(template, num_ch0, num_smpls*num_t0);
Q_0 = UX_0*UX_0';
Q_0 = Q_0 / num_t0;

Q(1 : num_ch0, 1 : num_ch0) = Q_0;
num_ch_cml = num_ch0;

for i_c = 1 : size(supplement, 1)
    sup_i = supplement{i_c};
    [num_chi, ~, num_ti] = size(sup_i);
    
    sup_i_sum = sum(sup_i, 3);
    S_i = sup_i_sum*sup_i_sum.';
    S_i = S_i / num_ti ^ 2;

    S(num_ch_cml + 1 : num_ch_cml + num_chi, ...
        num_ch_cml + 1 : num_ch_cml + num_chi) = S_i;
    
    
    S_0i = template_sum * sup_i_sum.';
    S_0i = S_0i / (num_t0 * num_ti);
    S(1 : num_ch0, num_ch_cml + 1 : num_ch_cml + num_chi) = S_0i;
    S(num_ch_cml + 1 : num_ch_cml + num_chi, 1 : num_ch0) = S_0i.';   

%     % S_ij
%     num_ch_cmlj = num_ch_cml + num_chi;
%     for j_c = (i_c + 1) : size(supplement, 1)
%         sup_j = supplement{j_c};
%         sup_j_sum = sum(sup_j, 3);
%         [num_chj, ~, num_tj] = size(sup_j);
% 
%         S_ij = sup_i_sum*sup_j_sum.';
%         S_ij = S_ij / (num_ti * num_tj);
% 
%         S(num_ch_cml + 1 : num_ch_cml + num_chi, num_ch_cmlj + 1 : num_ch_cmlj + num_chj) = S_ij;
%         S(num_ch_cmlj + 1 : num_ch_cmlj + num_chj, num_ch_cml + 1 : num_ch_cml + num_chi) = S_ij.';
% 
%         num_ch_cmlj = num_ch_cmlj + num_chj;
%     end
    
    UX_i = reshape(sup_i, num_chi, num_smpls * num_ti);
    Q_i = UX_i*UX_i.';
    Q_i = Q_i / num_ti;

    Q(num_ch_cml + 1 : num_ch_cml + num_chi, ...
        num_ch_cml + 1 : num_ch_cml + num_chi) = Q_i;

    num_ch_cml = num_ch_cml + num_chi;
end

[E, eigv] = eigs(S \ Q);
W = E(1 : num_ch0, :);

Vs = cell(size(supplement));

num_ch_cml = num_ch0;
for i_c = 1 : size(supplement, 1)
    num_chi = size(supplement{i_c}, 1);
    Vs{i_c} = E(num_ch_cml + 1 : num_ch_cml + num_chi, 1, 1);
    
    num_ch_cml = num_ch_cml + num_chi;
end

function [W, V] = trca(eeg) % Origial

[num_chans, num_smpls, num_trials]  = size(eeg);
UX = reshape(eeg, num_chans, num_smpls*num_trials);
SX = sum(eeg, 3);
S = SX*SX.';
Q = UX*UX';
[W,V] = eigs(S \ Q);


function [ y_ref ] = cca_reference(list_freqs, fs, num_smpls, num_harms)

if nargin < 3 
    error('stats:cca_reference:LackOfInput',...
        'Not enough input arguments.');
end

if ~exist('num_harms', 'var') || isempty(num_harms), num_harms = 3; end

num_freqs = length(list_freqs);
tidx = (1:num_smpls)/fs;
for freq_i = 1:1:num_freqs
    tmp = [];
    for harm_i = 1:1:num_harms
        stim_freq = list_freqs(freq_i);
        tmp = [tmp;...
            sin(2*pi*tidx*harm_i*stim_freq);...
            cos(2*pi*tidx*harm_i*stim_freq)];
    end % harm_i
    y_ref(freq_i, 1:2*num_harms, 1:num_smpls) = tmp;
end % freq_i

% function [W, Vs, eigv] = ttrca(template, supplement)
% 
% [num_ch0, num_smpls, num_t0]  = size(template);
% 
% total_channel_num = num_ch0;
% 
% for i_c = 1 : size(supplement, 1)
%     total_channel_num = total_channel_num + size(supplement{i_c}, 1);
% end
% 
% S = zeros(total_channel_num);
% Q = zeros(total_channel_num);
% 
% S_0 = zeros(num_ch0);
% for trial_i = 1:1:num_t0
%     x1 = squeeze(template(:,:,trial_i));
%     S_0 = S_0 + x1 * x1.';
%     for trial_j = trial_i+1:1:num_t0
%         x2 = squeeze(template(:,:,trial_j));
%         S_0 = S_0 + x1*x2' + x2*x1';
%     end % trial_j
%     
% end % trial_i
% 
% 
% S_0 = S_0 / num_t0 ^ 2;
% 
% S(1 : num_ch0, 1 : num_ch0) = S_0;
% 
% UX_0 = reshape(template, num_ch0, num_smpls*num_t0);
% Q_0 = UX_0*UX_0';
% Q_0 = Q_0 / num_t0;
% 
% Q(1 : num_ch0, 1 : num_ch0) = Q_0;
% num_ch_cml = num_ch0;
% 
% for i_c = 1 : size(supplement, 1)
%     sup_i = supplement{i_c};
%     [num_chi, ~, num_ti] = size(sup_i);
%     
%     S_i = zeros(num_chi);
%     for trial_i = 1:1:num_ti
%         x1 = squeeze(sup_i(:,:,trial_i));
%         S_i = S_i + x1 * x1.';
%         for trial_j = trial_i+1:1:num_ti
%             x2 = squeeze(sup_i(:,:,trial_j));
%             S_i = S_i + x1*x2' + x2*x1';
%         end % trial_j
%     end % trial_i
% 
%     S_i = S_i / num_ti ^ 2;
% 
%     S(num_ch_cml + 1 : num_ch_cml + num_chi, ...
%         num_ch_cml + 1 : num_ch_cml + num_chi) = S_i;
%         
%     S_0i = zeros(num_ch0, num_chi);
%     for trial_i = 1:1:num_t0
%         for trial_j = 1:1:num_ti
%             x1 = squeeze(template(:,:,trial_i));
%             x2 = squeeze(sup_i(:,:,trial_j));
%             S_0i = S_0i + x1*x2';
%         end % trial_j
%     end % trial_i
%     
%     S_0i = S_0i / (num_t0 * num_ti);
%     S(1 : num_ch0, num_ch_cml + 1 : num_ch_cml + num_chi) = S_0i;
%     S(num_ch_cml + 1 : num_ch_cml + num_chi, 1 : num_ch0) = S_0i.';   
% 
% %     % S_ij
% %     num_ch_cmlj = num_ch_cml + num_chi;
% %     for j_c = (i_c + 1) : size(supplement, 1)
% %         sup_j = supplement{j_c};
% %         [num_chj, ~, num_tj] = size(sup_j);
% %         S_ij = zeros(num_chi, num_chj);
% %         for trial_i = 1:1:num_ti
% %             for trial_j = 1:1:num_tj
% %                 x1 = squeeze(sup_i(:,:,trial_i));
% %                 x2 = squeeze(sup_j(:,:,trial_j));
% %                 S_ij = S_ij + x1*x2';
% %             end % trial_j
% %         end % trial_i
% % 
% %         S(num_ch_cml + 1 : num_ch_cml + num_chi, num_ch_cmlj + 1 : num_ch_cmlj + num_chj) = S_ij;
% %         S(num_ch_cmlj + 1 : num_ch_cmlj + num_chj, num_ch_cml + 1 : num_ch_cml + num_chi) = S_ij.';
% % 
% %         num_ch_cmlj = num_ch_cmlj + num_chj;
% %     end
%     
%     UX_i = reshape(sup_i, num_chi, num_smpls * num_ti);
%     Q_i = UX_i*UX_i.';
%     Q_i = Q_i / num_ti;
% 
%     Q(num_ch_cml + 1 : num_ch_cml + num_chi, ...
%         num_ch_cml + 1 : num_ch_cml + num_chi) = Q_i;
% 
%     num_ch_cml = num_ch_cml + num_chi;
% end
% 
% [E, eigv] = eigs(S \ Q);
% W = E(1 : num_ch0, :);
% 
% Vs = cell(size(supplement));
% 
% num_ch_cml = num_ch0;
% for i_c = 1 : size(supplement, 1)
%     num_chi = size(supplement{i_c}, 1);
%     Vs{i_c} = E(num_ch_cml + 1 : num_ch_cml + num_chi, 1, 1);
%     
%     num_ch_cml = num_ch_cml + num_chi;
% end
% 
% 
% function [W, V] = trca(eeg) % Origial
% 
% [num_chans, num_smpls, num_trials]  = size(eeg);
% S = zeros(num_chans);
% for trial_i = 1:1:num_trials
%     x1 = squeeze(eeg(:,:,trial_i));
%     S = S + x1 * x1.';
%     for trial_j = trial_i+1:1:num_trials
%         x2 = squeeze(eeg(:,:,trial_j));
%         S = S + x1*x2' + x2*x1';
%     end % trial_j
% end % trial_i
% 
% % if num_trials == 1
% %     S = eye(num_chans);
% % end
% UX = reshape(eeg, num_chans, num_smpls*num_trials);
% 
% Q = UX*UX';
% [W,V] = eigs(S \ Q);
