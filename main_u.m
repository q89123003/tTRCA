clear, close all;

method_legends = {'original', 'LST', 'trans'};
method_num = 3;

fs = 250;
nFBs = 5;
nHarms = 3;
isEnsemble = true;
channels = [53 : 59 61 : 63];
delay = round(0.65 * fs);
length = 1 * fs;

load('./data/Freq_Phase.mat');
list_freqs = freqs;

subjects = 1 : 35;
subject_eeg = cell(1, size(subjects, 2));

for subject = subjects
    load(sprintf('./data/S%d.mat', subject));
    
    %eeg = data(channels, delay : delay + length - 1, :, :);
    eeg = data(channels, 1 : delay + length - 1, :, :);
    eeg = permute(eeg, [3, 1, 2, 4]);
    subject_eeg{subject} = eeg;
end

%%
target_subjects = 21 : 35;
existing_subjects = 1 : 20;

max_template_size = size(subject_eeg{1}, 4) - 1;
template_sizes = 1 : max_template_size;
permutations = [1 2 3 4 5 6; 6 5 4 3 2 1];
%permutations = perms(1 : size(subject_eeg{1}, 4));

subject_accs = zeros(method_num, method_num, size(target_subjects, 2), size(template_sizes, 2), size(permutations, 1));

for i_subject = 1 : size(target_subjects, 2)
    subject = target_subjects(i_subject);
        
    sup_subjects = existing_subjects;
    sup_eeg = cell(size(sup_subjects, 2), 1);

    for i_sup = 1 : size(sup_subjects, 2)
        sub_subject = sup_subjects(i_sup);
        sup_eeg{i_sup} = subject_eeg{sub_subject};
    end
  
    eeg = subject_eeg{subject};
    valTrue = 1 : size(eeg, 1);
    
    accs = zeros(size(template_sizes, 2), size(permutations, 1));
    
    for i_perm = 1 : size(permutations, 1)
        tmp_perm = permutations(i_perm, :);
        train = eeg(:, :, :, tmp_perm(1 : end - 1));
        test = squeeze(eeg(:, :, :, tmp_perm(end))); 
        
        for i_ts = 1 : size(template_sizes, 2)
            template_size = template_sizes(i_ts);
            template = train(:, :, :, 1 : template_size);

            fprintf('Subject %d, perm %d, template size: %d\n', subject, i_perm, template_size);

            %model = train_utrca(template, sup_eeg, fs, nFBs, delay, list_freqs, nHarms);
            model = train_utrca(template, sup_eeg, fs, nFBs, delay);

            for i_sf_m = 1 : method_num
                for i_template_m = 1 : method_num
                    if i_sf_m <= 2 && i_template_m == 3
                        continue;
                    end

                    % Test phase ---------------------------------------
                    tmp_estimated = test_utrca(test, model, isEnsemble, i_sf_m, i_template_m);

                    % Evaluation ----------------------------------------
                    tmp_isCorrect = (tmp_estimated == valTrue);
                    tmp_acc = mean(tmp_isCorrect);

                    fprintf('Spatial fileter - %s; Template - %s: Averaged accuracy = %2.2f%%\n\n', ...
                       method_legends{i_sf_m}, method_legends{i_template_m}, tmp_acc*100);
                   
                    subject_accs(i_sf_m, i_template_m, i_subject, i_ts, i_perm) = tmp_acc;
                end
            end
        end
    end
    
    subject_acc = subject_accs(:, :, i_subject, :, :);
    save(sprintf('./results/Len%d_S%d.mat', length / fs * 1000, subject), 'subject_acc');
end

%%

results = zeros(size(template_sizes, 2), size(target_subjects, 2), 0);
model_names = {};
model_count = 0;
for i_sf_m = 1 : method_num
    for i_template_m = 1 : method_num
        if i_sf_m <= 2 && i_template_m == 3
            continue;
        end
    
        model_count = model_count + 1;
        results(:, :, model_count) = squeeze(mean(subject_accs(i_sf_m, i_template_m, :, :, :), 5)).';
        
        model_names = [model_names sprintf('Spatial fileter - %s; Template - %s', ...
            method_legends{i_sf_m}, method_legends{i_template_m})];
    end
end

colors = jet(model_count);

legends = {};
for i_model = 1 : model_count
    tmp_legend = sprintf('%s(%d) %s', ['\color[rgb]{' num2str(colors(i_model, :)) '}'], i_model, model_names{i_model});
    legends = [legends tmp_legend];
end

pairs = nchoosek(1 : model_count, 2);

f = bar_error_pval(results, colors, pairs, legends, [10 10 460 610]);
ylabel('Accuracy');
yticks(0 : 0.2 : 1);
xticks(1 : size(template_sizes, 2));
xticklabels(template_sizes);
xlabel('Number of calibration trials per class');

