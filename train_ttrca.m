function model = train_ttrca(templates, supplements, fs, num_fbs)

if nargin < 2
    error('stats:train_trca:LackOfInput', 'Not enough input arguments.'); 
end

if ~exist('num_fbs', 'var') || isempty(num_fbs), num_fbs = 3; end

[num_targs, num_chans, num_smpls, ~] = size(templates);
trains = zeros(num_targs, num_fbs, num_chans, num_smpls);
template = zeros(num_targs, num_fbs, num_chans, num_smpls, size(templates, 4));

supplement_cell = cell(size(supplements));
V_cell = cell(size(supplements));

for i_c = 1 : size(supplement_cell, 1)
    supplement_cell{i_c} = zeros(num_targs, num_fbs, num_chans, num_smpls, size(supplements{i_c}, 4));   
end

for i_c = 1 : size(V_cell, 1)
    V_cell{i_c} = zeros(num_fbs, num_targs, num_chans);
end
 
W = zeros(num_fbs, num_targs, num_chans);
V_ratio = zeros(num_fbs, num_targs);

for targ_i = 1:1:num_targs
    %supplement_targ = squeeze(supplements(targ_i, :, :, :));
    template_targ = squeeze(templates(targ_i, :, :, :));

    for fb_i = 1:1:num_fbs
        template_tmp = filterbank(template_targ, fs, fb_i);
                
        supplement_cat = zeros(num_chans, num_smpls, 0);   
        supplement_tmp = cell(size(supplements));
        for i_c = 1 : size(supplement_cell, 1)
            sup_targ_temp = squeeze(supplements{i_c}(targ_i, :, :, :));
            supplement_tmp{i_c} = filterbank(sup_targ_temp, fs, fb_i);
            
            supplement_cell{i_c}(targ_i, fb_i, :, :, :) = supplement_tmp{i_c};
            supplement_cat = cat(3, supplement_cat, supplement_tmp{i_c});
        end
        %supplement_tmp = filterbank(supplement_targ, fs, fb_i);
        
                
        % LST to get better mean of trials
        template_tmp_mean = squeeze(mean(template_tmp, 3));
        Y = template_tmp_mean;
        
        transferred_eeg_tmp = zeros(num_chans, num_smpls, size(supplement_cat, 3));
        
        for trialIdx = 1 : size(supplement_cat, 3)
            
            single_trial_eeg_tmp = squeeze(supplement_cat(:, :, trialIdx));
            
            X = [ones(1, size(Y, 2)); single_trial_eeg_tmp];
            b = Y * X.' / (X * X.');
            transferred_eeg_tmp(:, :, trialIdx) = (b * X);
            
        end
        
        transferred_eeg_tmp = cat(3, template_tmp, transferred_eeg_tmp);
        trains(targ_i,fb_i,:,:) = squeeze(mean(transferred_eeg_tmp, 3));
        
        %        
        [w_tmp, v_tmp_cell, eigv] = ttrca(template_tmp, supplement_tmp);
        template(targ_i, fb_i, :, :, :) = template_tmp;
        %supplement(targ_i, fb_i, :, :, :) = supplement_tmp;
               
        
        W(fb_i, targ_i, :) = w_tmp(:,1);
        for i_c = 1 : size(V_cell, 1)
            V_cell{i_c}(fb_i, targ_i, :) = v_tmp_cell{i_c};
        end
        
        %V_ratio(fb_i, targ_i) = v_tmp(1, 1) / v_tmp(2, 2);
        V_ratio(fb_i, targ_i) = abs(eigv(1, 1)) / trace(abs(eigv(2:end, 2:end)));
    end % fb_i
    
end % targ_i

model = struct('W', W, 'V_cell', {V_cell}, 'V_ratio', V_ratio,...
    'num_fbs', num_fbs, 'fs', fs, 'num_targs', num_targs, ...
    'template', template, 'supplement_cell', {supplement_cell}, 'trains', trains);

function [W, Vs, eigv] = ttrca(template, supplement)
[num_ch0, num_smpls, num_t0]  = size(template);

S = zeros(num_ch0 * size(supplement, 1));
Q = zeros(num_ch0 * size(supplement, 1));

S_0 = zeros(num_ch0);
for trial_i = 1:1:num_t0
    for trial_j = trial_i+1:1:num_t0
        x1 = squeeze(template(:,:,trial_i));
        x1 = bsxfun(@minus, x1, mean(x1,2));
        x2 = squeeze(template(:,:,trial_j));
        x2 = bsxfun(@minus, x2, mean(x2,2));
        S_0 = S_0 + x1*x2' + x2*x1';
    end % trial_j
end % trial_i

S(1 : num_ch0, 1 : num_ch0) = S_0;

UX_0 = reshape(template, num_ch0, num_smpls*num_t0);
UX_0 = bsxfun(@minus, UX_0, mean(UX_0,2));
Q_0 = UX_0*UX_0';

Q(1 : num_ch0, 1 : num_ch0) = Q_0;


for i_c = 1 : size(supplement, 1)
    sup = supplement{i_c};

    [~, ~, num_tn] = size(sup);

    S_n = zeros(num_ch0);
    for trial_i = 1:1:num_tn
        for trial_j = trial_i+1:1:num_tn
            x1 = squeeze(sup(:,:,trial_i));
            x1 = bsxfun(@minus, x1, mean(x1,2));
            x2 = squeeze(sup(:,:,trial_j));
            x2 = bsxfun(@minus, x2, mean(x2,2));
            S_n = S_n + x1*x2' + x2*x1';
        end % trial_j
    end % trial_i

    S(num_ch0 + (i_c - 1) * num_ch0 + 1 : num_ch0 + i_c * num_ch0, ...
        num_ch0 + (i_c - 1) * num_ch0 + 1 : num_ch0 + i_c * num_ch0) = S_n;
        
    S_0n = zeros(num_ch0, num_ch0);
    for trial_i = 1:1:num_t0
        for trial_j = 1:1:num_tn
            x1 = squeeze(template(:,:,trial_i));
            x1 = bsxfun(@minus, x1, mean(x1,2));
            x2 = squeeze(sup(:,:,trial_j));
            x2 = bsxfun(@minus, x2, mean(x2,2));
            S_0n = S_0n + x1*x2';
        end % trial_j
    end % trial_i
    
    S(1 : num_ch0, num_ch0 + (i_c - 1) * num_ch0 + 1 : num_ch0 + i_c * num_ch0) = S_0n;
    S(num_ch0 + (i_c - 1) * num_ch0 + 1 : num_ch0 + i_c * num_ch0, 1 : num_ch0) = S_0n.';   


    UX_n = reshape(sup, num_ch0, num_smpls*num_tn);
    UX_n = bsxfun(@minus, UX_n, mean(UX_n,2));
    Q_n = UX_n*UX_n';

    Q(num_ch0 + (i_c - 1) * num_ch0 + 1 : num_ch0 + i_c * num_ch0, ...
    num_ch0 + (i_c - 1) * num_ch0 + 1 : num_ch0 + i_c * num_ch0) = Q_n;

end

[E, eigv] = eigs(S, Q);
W = E(1 : num_ch0, :);

Vs = cell(size(supplement));

for i_c = 1 : size(supplement, 1)
    Vs{i_c} = E(num_ch0 + (i_c - 1) * num_ch0 + 1 : num_ch0 + i_c * num_ch0, 1);
end

