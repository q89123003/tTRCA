clear, close all;
fs = 500;
trial_num_sweep = round(exp(log(2) :  log(2) : log(1024)));
channel_nums = [4 16];
wave_iter_num = 5;
iterN = 10;
durations = [2];
wave_num_range = [3 10];
freq_range = [0 60];

dt = 1 / fs;

for i_duration = 1 : length(durations)
    duration = durations(i_duration);
    t = 0 : dt : duration;
    for i_channel_num = 1 : length(channel_nums)
        channel_num = channel_nums(i_channel_num);

        wave_nums = randi(wave_num_range, 1, wave_iter_num);
        elapsed_times = zeros(2, length(trial_num_sweep), wave_iter_num);
        component_diffs = zeros(length(trial_num_sweep), wave_iter_num);
        
        for i_trial_num = length(trial_num_sweep) : -1 : 1
            trial_num = trial_num_sweep(i_trial_num);
            for iter = 1 : wave_iter_num
                wave_num = wave_nums(iter);

                x = zeros(channel_num, length(t), trial_num);

                for i_wave = 1 : wave_num
                    tmp_f = freq_range(1) + (freq_range(2) - freq_range(1)) * rand(1, 1);
                    tmp_p = 2 * rand(1, 1) * pi;
                    tmp_amp = 1 + rand(1, 1);
                    sign = rand(1, 1) > 0.5;
                    if sign == true
                        tmp_x = tmp_amp * sin(2 * pi * tmp_f * repmat(t, channel_num, 1) + tmp_p);
                    else
                        tmp_x = tmp_amp * cos(2 * pi * tmp_f * repmat(t, channel_num, 1) + tmp_p);
                    end

                    for i_trial = 1 : trial_num
                        x(:, :, i_trial) = x(:, :, i_trial) + tmp_x + rand(channel_num, length(t));
                    end

                end
                x = x - mean(x, 2);


                [w_fast, v_fast, elapsed_time_avg] = fast_trca(x, iterN);
                elapsed_times(2, i_trial_num, iter) = elapsed_time_avg;

                [w, v, elapsed_time_avg] = original_trca(x, iterN);
                elapsed_times(1, i_trial_num, iter) = elapsed_time_avg;
                
                component_diffs(i_trial_num, iter) = norm(w_fast(:, 1) - w(:, 1));
            end
        end

        save(sprintf('./elapsed time results/elapsed_times_dur%d_chn%d.mat', duration, channel_num), 'elapsed_times');
        save(sprintf('./elapsed time results/component_diff_dur%d_chn%d.mat', duration, channel_num), 'component_diffs');
    end
end
%%
figure();
hold on;
for i_alg = 1 : 2
    plot(log2(trial_num_sweep), log10(squeeze(mean(elapsed_times(i_alg, :, :), 3))));
end
hold off;

%%
figure();
hold on;
for i_alg = 1 : 2
    plot(trial_num_sweep, squeeze(mean(elapsed_times(i_alg, :, 2:end), 3)), 'LineWidth', 2);
end
hold off;