method_legends = {'original', 'LST', 'trans'};
method_num = 3;

datasets = 2;
length = 1;

seed = 0;
%%
for dataset = datasets
    if dataset == 1
        subjects = 1 : 35;

    elseif dataset == 2
        
        subjects = 1 : 70;
    end
    
    suffle_subjects = shuffle(subjects, seed);
    
    if dataset == 1    
        target_subjects = suffle_subjects(1:15);
        existing_subjects = suffle_subjects(16:end);
    elseif dataset == 2
        target_subjects = suffle_subjects(1:30);
        existing_subjects = suffle_subjects(31:end);
    end
end

initial_flag = true;
for target_subject = target_subjects
    fileName = sprintf('./results/dataset%d/Len%d_S%d.mat', ...
        dataset, length * 1000, target_subject);
    
    load(fileName);
    
    if initial_flag == true
        initial_flag = false;
        subject_accs = subject_acc;
    else
        subject_accs = cat(3, subject_accs, subject_acc);
    end
end

colors = jet(7);

figure();
for i_template_size = 1 : size(subject_acc, 4)
    subplot(1, size(subject_acc, 4), i_template_size);
    
    hold on;
    for i_subject = 1 : size(target_subjects, 2)
        i_model = 0;
        for i_sp_m = 1 : method_num
            for i_tp_m = 1 : method_num
                if i_tp_m == 3 && i_sp_m <= 2
                    continue;
                end
                i_model = i_model + 1;
                scatter(i_model, mean(squeeze(subject_accs(i_sp_m, i_tp_m, i_subject, i_template_size, :))), 20, colors(i_model, :), 'filled');
            end
        end
    end
    hold off;
end

