function results = test_utrca(eeg, model, is_ensemble, sf_method, template_method)
% Test phase of the task-related component analysis (TRCA)-based
% steady-state visual evoked potentials (SSVEPs) detection [1].
%
% function results = test_trca(eeg, model, is_ensemble)
%
% Input:
%   eeg             : Input eeg data 
%                     (# of targets, # of channels, Data length [sample])
%   model           : Learning model for tesing phase of the ensemble 
%                     TRCA-based method
%   is_ensemble     : 0 -> TRCA-based method, 
%                     1 -> Ensemble TRCA-based method (defult: 1)
%
% Output:
%   results         : The target estimated by this method
%
% See also:
%   train_trca.m
%
% Reference:
%   [1] M. Nakanishi, Y. Wang, X. Chen, Y. -T. Wang, X. Gao, and T.-P. Jung,
%       "Enhancing detection of SSVEPs for a high-speed brain speller using 
%        task-related component analysis",
%       IEEE Trans. Biomed. Eng, 65(1): 104-112, 2018.
%
% Masaki Nakanishi, 22-Dec-2017
% Swartz Center for Computational Neuroscience, Institute for Neural
% Computation, University of California San Diego
% E-mail: masaki@sccn.ucsd.edu

if ~exist('is_ensemble', 'var') || isempty(is_ensemble)
    is_ensemble = 1; end

if ~exist('model', 'var')
    error('Training model based on TRCA is required. See train_trca().'); 
end

if sf_method <= 2 && template_method == 3
    warning('Using ttrca templates with other spatial filters will lead to poor performance.');
end

fb_coefs = [1:model.num_fbs].^(-1.25)+0.25;
supplement_num = size(model.ttrca_V_cell, 2);

for targ_i = 1:1:model.num_targs
    testdata = squeeze(eeg(targ_i, :, :));
    for fb_i = 1:1:model.num_fbs
        % testdata_tmp = filterbank(testdata, model.fs, fb_i);
        testdata_tmp = filterbank_pad(testdata, model.fs, model.pad_length, fb_i);
        for class_i = 1:1:model.num_targs
                        
            switch sf_method
                case 1 % original TRCA
                    W = model.W;
                case 2 % LST
                    W = model.lst_W;
                case 3 % tTRCA
                    W = model.ttrca_W;
            end
            
            if ~is_ensemble
                w = squeeze(W(fb_i, class_i, :));
            else
                w = squeeze(W(fb_i, :, :)).';
            end
        
            switch template_method
                case 1
                    train = squeeze(model.trains(class_i, fb_i, :, :));
                    train_proj = train.' * w;
                    r_tmp = corrcoef(normalize(testdata_tmp'*w, 1), normalize(train_proj, 1));
                    r(fb_i,class_i) = r_tmp(1,2);
                case 2
                    train = squeeze(model.lst_trains(class_i, fb_i, :, :));
                    train_proj = train.' * w;
                    r_tmp = corrcoef(normalize(testdata_tmp'*w, 1), normalize(train_proj, 1));
                    r(fb_i,class_i) = r_tmp(1,2);
                case 3
                    template = squeeze(model.ttrca_template(class_i, fb_i, :, :, :));
                    
% ------------------------- old -----------------------------------------%
                    train_size = size(template, 3);
                    train_proj_sum = train_size * squeeze(mean(template, 3)).' * w;

                    for i_sup = 1 : supplement_num
                        tmp_supplement = squeeze(model.ttrca_supplement_cell{i_sup}(class_i, fb_i, :, :, :));
                        tmp_sup_size = size(tmp_supplement, 3);

                        if ~is_ensemble
                            tmp_v = squeeze(model.ttrca_V_cell{i_sup}(fb_i, class_i, :));
                        else
                            tmp_v = squeeze(model.ttrca_V_cell{i_sup}(fb_i, :, :)).';
                        end

                        train_size = train_size + tmp_sup_size;
                        train_proj_sum = train_proj_sum + tmp_sup_size * squeeze(mean(tmp_supplement, 3)).' * tmp_v; 
                    end

                    train_proj = train_proj_sum / train_size; 
                    r_tmp = corrcoef(normalize(testdata_tmp'*w, 1), normalize(train_proj, 1));
                    r(fb_i,class_i) = r_tmp(1,2);
% ------------------------- old -----------------------------------------%

% ------------------------- new -----------------------------------------%
%                     template_mean = mean(template, 3);
%                     r_tmp = cov(testdata_tmp.'*w, template_mean.'*w);
%                     r_sum = r_tmp(1, 2);
%                     
%                     for i_sup = 1 : supplement_num
%                         tmp_supplement = squeeze(model.ttrca_supplement_cell{i_sup}(class_i, fb_i, :, :, :));
%                         tmp_supplement_mean = mean(tmp_supplement, 3);
% 
%                         if ~is_ensemble
%                             tmp_v = squeeze(model.ttrca_V_cell{i_sup}(fb_i, class_i, :));
%                         else
%                             tmp_v = squeeze(model.ttrca_V_cell{i_sup}(fb_i, :, :)).';
%                         end
% 
%                         r_tmp = cov(testdata_tmp.'*w, tmp_supplement_mean.'*tmp_v);
%                         r_sum = r_sum + r_tmp(1, 2);
% %                         r_tmp = cov(tmp_supplement_mean.'*tmp_v, tmp_supplement_mean.'*tmp_v);
% %                         r_sum = r_sum + r_tmp(1, 2);
%                     end
%                     r(fb_i,class_i) = r_sum;
% ------------------------- new -----------------------------------------%
            end
        end % class_i
    end % fb_i
    
    rho = fb_coefs*r;
    [~, tau] = max(rho);
    results(targ_i) = tau;
end % targ_i