clear, close all;

fs = 250;
nFBs = 5;
isEnsemble = true;
channels = [53 : 59 61 : 63];
delay = round(0.65 * fs);
length = 0.5 * fs;

subjects = 1 : 35;
subject_eeg = cell(1, size(subjects, 2));

for subject = subjects
    load(sprintf('./data/S%d.mat', subject));
    
    eeg = data(channels, delay : delay + length - 1, :, :);
    eeg = permute(eeg, [3, 1, 2, 4]);
    subject_eeg{subject} = eeg;
end

%%
ONE_SUPPLEMENT = false;

target_subjects = 21 : 35;
existing_subjects = 1 : 20;

template_sizes = [2 4];
max_template_size = max(template_sizes);

subject_accs = zeros(4, size(target_subjects, 2), size(template_sizes, 2));
subject_models = cell(4, size(target_subjects, 2), size(template_sizes, 2));

for i_subject = 1 : size(target_subjects, 2)
    subject = target_subjects(i_subject);
        
    if ONE_SUPPLEMENT == true
        sup_eeg = cell(1, 1);
        sup_subject = subject + 1;
        sup_eeg{1} = subject_eeg{sup_subject};    
    else
        sup_subjects = existing_subjects;
        sup_eeg = cell(size(sup_subjects, 2), 1);
        
        for i_sup = 1 : size(sup_subjects, 2)
            sub_subject = sup_subjects(i_sup);
            sup_eeg{i_sup} = subject_eeg{sub_subject};
        end
    end
    
    sup_eeg_cat = zeros(size(sup_eeg{1}, 1), size(sup_eeg{1}, 2), size(sup_eeg{1}, 3), 0);
    for i_c = 1 : size(sup_eeg, 1)
        sup_eeg_cat = cat(4, sup_eeg_cat, sup_eeg{i_c});
    end
    
    eeg = subject_eeg{subject};
    valTrue = 1 : size(eeg, 1);
    
    test = eeg(:, :, :, max_template_size + 1 : end); 
    for i_ts = 1 : size(template_sizes, 2)
        template_size = template_sizes(i_ts);
        template = eeg(:, :, :, 1 : template_size);

        % Baseline
        baseline_model = train_trca(template, fs, nFBs);

        % woLST
        woLST_train = cat(4, template, sup_eeg_cat);
        woLST_model = train_trca(woLST_train, fs, nFBs);
        %woLST_model = train_wottrca(template, sup_eeg_cat, fs, nFBs);

        % LST
        LST_model = train_trca_LST(template, sup_eeg_cat, fs, nFBs);

        % tTRCA
        tTrca_model = train_ttrca(template, sup_eeg, fs, nFBs);
            
        for loocv_i = 1:size(test, 4)
            % Get single trial testing data.
            test_trial = squeeze(test(:, :, :, loocv_i));

            % Test phase ---------------------------------------
            baseline_estimated = test_trca(test_trial, baseline_model, isEnsemble);
            woLST_estimated = test_trca(test_trial, woLST_model, isEnsemble);
            LST_estimated = test_trca(test_trial, LST_model, isEnsemble);
            tTrca_estimated = test_ttrca(test_trial, tTrca_model, isEnsemble);

            % Evaluation ----------------------------------------

            baseline_isCorrect = (baseline_estimated == valTrue);
            baseline_accs(loocv_i) = mean(baseline_isCorrect);

            woLST_isCorrect = (woLST_estimated==valTrue);
            woLST_accs(loocv_i) = mean(woLST_isCorrect);

            LST_isCorrect = (LST_estimated==valTrue);
            LST_accs(loocv_i) = mean(LST_isCorrect);

            tTrca_isCorrect = (tTrca_estimated==valTrue);
            tTrca_accs(loocv_i) = mean(tTrca_isCorrect);

        end % loocv_i

        fprintf('Subject %d, template size: %d\n', subject,  template_size);
        fprintf('Baseline: Averaged accuracy = %2.2f%c%2.2f%%\n\n', ...
                    mean(baseline_accs)*100, char(177), std(baseline_accs)*100);
        fprintf('woLST: Averaged accuracy = %2.2f%c%2.2f%%\n\n', ...
                mean(woLST_accs)*100, char(177), std(woLST_accs)*100);
        fprintf('LST: Averaged accuracy = %2.2f%c%2.2f%%\n\n', ...
                mean(LST_accs)*100, char(177), std(LST_accs)*100);
        fprintf('tTRCA: Averaged accuracy = %2.2f%c%2.2f%%\n\n', ...
                mean(tTrca_accs)*100, char(177), std(tTrca_accs)*100);

        subject_accs(1, i_subject, i_ts) = mean(baseline_accs);
        subject_accs(2, i_subject, i_ts) = mean(woLST_accs);
        subject_accs(3, i_subject, i_ts) = mean(LST_accs);
        subject_accs(4, i_subject, i_ts) = mean(tTrca_accs);

        subject_models{1, i_subject, i_ts} = baseline_model;
        subject_models{2, i_subject, i_ts} = woLST_model;
        subject_models{3, i_subject, i_ts} = LST_model;
        subject_models{4, i_subject, i_ts} = tTrca_model;
    end
end

%% find the largest difference
acc_diffs = zeros(1, size(subject_accs, 2));

for i_s = 1 : size(subject_accs, 2)
    acc_diffs(i_s) = subject_accs(4, i_s, 1) - subject_accs(1, i_s, 1);
end

[~, ind] = sort(acc_diffs);
ind = ind(end:-1:1);

figure();

for i_p = 1 : 3
    
    subplot(3, 1, i_p);
    p_ind = ind(i_p);
    
    hold on
    baseline_proj = squeeze(subject_models{1, p_ind, 1}.trains(1, 1, :, :)).' ...
        * squeeze(subject_models{1, p_ind, 1}.W(1, 1, :));
    baseline_proj = baseline_proj - mean(baseline_proj);
    baseline_proj = normalize(baseline_proj, 1, 'range');

    inverse = sign(mean(subject_models{1, p_ind, 1}.W(1, 1, :)) ...
        * mean(subject_models{4, p_ind, 1}.W(1, 1, :)));
    
    tTrca_proj = squeeze(mean(subject_models{4, p_ind, 1}.template(1, 1, :, :, :), 5)).' ...
        * squeeze(subject_models{4, p_ind, 1}.W(1, 1, :));
    tTrca_proj = tTrca_proj - mean(tTrca_proj);
    tTrca_proj = normalize(tTrca_proj * inverse, 1, 'range');

    plot((1 : size(baseline_proj, 1)) / fs, baseline_proj);
    plot((1 : size(tTrca_proj, 1)) / fs, tTrca_proj);
    hold off

    legend({'Baseline', 'tTrca'});
    ylabel('normalize signals');
    title(sprintf('Example subject %d\nBaseline: %.2f%%, tTRCA: %.2f%%', ...
        i_p, subject_accs(1, p_ind, 1) * 100, subject_accs(4, p_ind, 1) * 100))
    xlabel('Time (s)');
end

%%

[~, n_class, n_chan] = size(subject_models{1, 1, 1}.W);

figure();

for i_p = 1 : 3
    
    subplot(3, 1, i_p);
    p_ind = ind(i_p);
    
    hold on
    baseline_W = squeeze(subject_models{1, p_ind, 1}.W(1, :, :)).';
    %baseline_W = normalize(baseline_W, 1, 'range', [-1 1]);
    baseline_W(:) = normalize(baseline_W(:), 'range', [-1 1]);
    [~, m_w_ind] = max(abs(baseline_W), [], 1);
    
    inverse = zeros(1, n_class);
    for i_class = 1 : n_class
        inverse(i_class) = sign(baseline_W(m_w_ind(i_class), i_class));
    end
    
    baseline_W = baseline_W .* inverse;
    
    tTrca_W = squeeze(subject_models{4, p_ind, 1}.W(1, :, :)).';
    %tTrca_W = normalize(tTrca_W, 1, 'range', [-1 1]);
    tTrca_W(:) = normalize(tTrca_W(:), 'range', [-1 1]);

    [~, m_w_ind] = max(abs(tTrca_W), [], 1);
    
    inverse = zeros(1, n_class);
    for i_class = 1 : n_class
        inverse(i_class) = sign(tTrca_W(m_w_ind(i_class), i_class));
    end
    
    tTrca_W = tTrca_W .* inverse;
    
    image = cat(1, tTrca_W, baseline_W);
    imagesc(image)
    colorbar();
    caxis([-1 1]);
    
    line([0.5 n_class + 0.5], [n_chan + 0.5 n_chan + 0.5], 'Color', 'red', 'LineWidth', 1.5);
    
    title(sprintf('Example subject %d\nBaseline: %.2f%%, tTRCA: %.2f%%', ...
        i_p, subject_accs(1, p_ind, 1) * 100, subject_accs(4, p_ind, 1) * 100))
    ylabel(sprintf('weights\ntTRCA        Baseline'));
    xlabel('class');
    xlim([0.5 n_class + 0.5]);
    ylim([0.5 2 * n_chan + 0.5]);
end

%%
select_subject = 1;
figure();

for i_p = 1 : 2
    
    subplot(2, 1, i_p);

    hold on
    baseline_W = squeeze(subject_models{1, select_subject, i_p}.W(1, :, :)).';
    %baseline_W = normalize(baseline_W, 1, 'range', [-1 1]);
    baseline_W(:) = normalize(baseline_W(:), 'range', [-1 1]);
    [~, m_w_ind] = max(abs(baseline_W), [], 1);
    
    inverse = zeros(1, n_class);
    for i_class = 1 : n_class
        inverse(i_class) = sign(baseline_W(m_w_ind(i_class), i_class));
    end
    
    baseline_W = baseline_W .* inverse;
    
    tTrca_W = squeeze(subject_models{4, select_subject, i_p}.W(1, :, :)).';
    %tTrca_W = normalize(tTrca_W, 1, 'range', [-1 1]);
    tTrca_W(:) = normalize(tTrca_W(:), 'range', [-1 1]);

    [~, m_w_ind] = max(abs(tTrca_W), [], 1);
    
    inverse = zeros(1, n_class);
    for i_class = 1 : n_class
        inverse(i_class) = sign(tTrca_W(m_w_ind(i_class), i_class));
    end
    
    tTrca_W = tTrca_W .* inverse;
    
    image = cat(1, tTrca_W, baseline_W);
    imagesc(image)
    colorbar();
    caxis([-1 1]);
    
    line([0.5 n_class + 0.5], [n_chan + 0.5 n_chan + 0.5], 'Color', 'red', 'LineWidth', 1.5);
    
    title(sprintf('Example subject %d, %d calibration trials\nBaseline: %.2f%%, tTRCA: %.2f%%', ...
        select_subject, template_sizes(i_p), ...
        subject_accs(1, select_subject, i_p) * 100, subject_accs(4, select_subject, i_p) * 100))
    ylabel(sprintf('weights\ntTRCA        Baseline'));
    xlabel('class');
    xlim([0.5 n_class + 0.5]);
    ylim([0.5 2 * n_chan + 0.5]);
end

%%

results = zeros(size(subject_accs, 3), size(target_subjects, 2), 4);
for i_m = 1 : 4
    results(:, :, i_m) = squeeze(subject_accs(i_m, :, :)).';
end

colors = [0.3 0.3 0.3; 0 0 1; 1 0 0; 1 0 1];
pairs = [1 3; 1 4; 3 4; 2 3; 2 4];
legends = {'(1) BASELINE', '\color{blue}(2) wo/LST', '\color{red}(3) w/LST', ...
['\color[rgb]{' num2str(colors(4, :)) '}(4) tTRCA']};
f = bar_error_pval(results, colors, pairs, legends, [10 10 460 610]);
ylabel('Accuracy');
yticks(0 : 0.2 : 1);
xticks(1 : size(template_sizes, 2));
xticklabels(template_sizes);
xlabel('Number of calibration trials per class');

