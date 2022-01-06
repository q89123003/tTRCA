clear, close all;
fs = 500;
trial_num_sweep = round(exp(log(2) :  log(2) : log(1024)));
channel_nums = [4 16 64];
wave_iter_num = 1;
iterN = 10000;
durations = [2 4 8 16];
wave_num_range = [3 10];
freq_range = [0 60];

dt = 1 / fs;

[duration_grid, channel_num_grid] = meshgrid(durations, channel_nums);
duration_grid = duration_grid(:);
channel_num_grid = channel_num_grid(:);
n_condition = length(duration_grid);

elapsed_times_all = zeros(n_condition, 2, length(trial_num_sweep));
component_diffs_all = zeros(n_condition, length(trial_num_sweep));
parfor i_condition = 1 : n_condition
    duration = duration_grid(i_condition);
    channel_num = channel_num_grid(i_condition);
    % fprintf('Duration: %d, %d channels\n', duration, channel_num);
    
    t = 0 : dt : duration;
%         wave_nums = randi(wave_num_range, 1, wave_iter_num);
%         elapsed_times = zeros(2, length(trial_num_sweep), wave_iter_num);
%         component_diffs = zeros(length(trial_num_sweep), wave_iter_num);
    elapsed_time_orig = zeros(1, length(trial_num_sweep));
    elapsed_time_fast = zeros(1, length(trial_num_sweep));
    component_diffs = zeros(1, length(trial_num_sweep));

        for i_trial_num = length(trial_num_sweep) : -1 : 1
%     for i_trial_num = 1 : length(trial_num_sweep)
        trial_num = trial_num_sweep(i_trial_num);
        fprintf('Duration %d, %d channels, %d trials\n', duration, channel_num, trial_num);
%             for iter = 1 : wave_iter_num
%                 wave_num = wave_nums(iter);
            wave_num = randi(wave_num_range);
            x = zeros(channel_num, length(t), trial_num);

            for i_wave = 1 : wave_num
                tmp_f = 60 * rand(1, 1);
                tmp_p = 2 * rand(1, 1) * pi;
                tmp_amp = 1 + rand(1, 1);

                tmp_x = tmp_amp * sin(2 * pi * tmp_f * repmat(t, channel_num, 1) + tmp_p) .* (0.5 * rand(channel_num, 1) + 0.5);

                for i_trial = 1 : trial_num
                    x(:, :, i_trial) = x(:, :, i_trial) + tmp_x + rand(channel_num, length(t));
                end

            end
            x = x - mean(x, 2);

            [w_fast, v_fast, elapsed_time_avg] = fast_trca(x, iterN);
%                 elapsed_times(2, i_trial_num) = elapsed_time_avg;
            elapsed_time_fast(i_trial_num) = elapsed_time_avg;

            if trial_num <= 8
                origIterN = 10000;
            elseif trial_num <= 64
                origIterN = 100;
            else
                origIterN = 10;
            end
            [w, v, elapsed_time_avg] = original_trca(x, origIterN);
            elapsed_time_orig(i_trial_num) = elapsed_time_avg;

            component_diffs(i_trial_num) = norm(w_fast(:, 1) - w(:, 1));
    end

    elapsed_times = cat(1, elapsed_time_orig, elapsed_time_fast);
    elapsed_times_all(i_condition, :, :) = elapsed_times;
    component_diffs_all(i_condition, :) = component_diffs;
    
%     save(sprintf('./elapsed time results/elapsed_times_dur%d_chn%d.mat', duration, channel_num), 'elapsed_times');
%     save(sprintf('./elapsed time results/component_diff_dur%d_chn%d.mat', duration, channel_num), 'component_diffs');
end

%%
for i_duration = 1 : length(durations)
    duration = durations(i_duration);
    for i_channel_num = 1 : length(channel_nums)
        channel_num = channel_nums(i_channel_num);
        i_condition = (i_duration - 1) * length(channel_nums) + i_channel_num;
        elapsed_times = squeeze(elapsed_times_all(i_condition, :, :));
        component_diffs = squeeze(component_diffs_all(i_condition, :));
        
        save(sprintf('./elapsed time results/elapsed_times_dur%d_chn%d.mat', duration, channel_num), 'elapsed_times');
        save(sprintf('./elapsed time results/component_diff_dur%d_chn%d.mat', duration, channel_num), 'component_diffs');
    end
end
%%
% figure();
% hold on;
% for i_alg = 1 : 2
%     plot(log2(trial_num_sweep), log10(squeeze(mean(elapsed_times(i_alg, :, :), 3))));
% end
% hold off;
% 
% %%
% figure();
% hold on;
% for i_alg = 1 : 2
%     plot(trial_num_sweep, squeeze(mean(elapsed_times(i_alg, :, 2:end), 3)), 'LineWidth', 2);
% end
% hold off;